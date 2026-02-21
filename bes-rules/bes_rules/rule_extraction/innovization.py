import itertools
import json
import logging
import os
import pickle
import time
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Any, Type, Union

import numpy as np
import pandas as pd

from bes_rules.configs import StudyConfig
from bes_rules.plotting.utils import (
    get_result_from_input_analysis_for_input,
    PlotConfig,
    get_result_for_input
)

from bes_rules.rule_extraction import features as features_module
from bes_rules.rule_extraction import plotting as rule_plotting
from bes_rules.rule_extraction import surrogates
from bes_rules.rule_extraction.regression import objective_space
from bes_rules.rule_extraction.regression.regressors import (
    Regressor,
    LinearRegressor,
    PowerLawRegressor
)
from bes_rules.utils.multiprocessing_ import execute_function_in_parallel

from bes_rules import DATA_PATH
from bes_rules.objectives.annuity import TechnikkatalogAssumptions

logger = logging.getLogger(__name__)


def load_f_inv_biv():
    with open(DATA_PATH.joinpath("prices", "f_inv_biv.json"), "r") as file:
        f_inv_bivs = json.load(file)
        return {k: TechnikkatalogAssumptions(**v) for k, v in f_inv_bivs.items()}


def mesh_arrays(array_list: List[np.ndarray]):
    # Use meshgrid to create a grid of all combinations
    grid = np.meshgrid(*array_list, indexing='ij')
    # Stack the arrays and reshape
    combined = np.stack(grid, axis=-1)
    # Reshape to 2D array where each row is a combination
    result = combined.reshape(-1, len(array_list))
    return result


def _load_single_input_case(
        config: StudyConfig,
        input_config,
        objectives: Dict[str, str],
        surrogate_type: Type[surrogates.Surrogate],
        surrogate_kwargs: dict,
        plot_config: PlotConfig,
        create_surrogate_plots: bool,
        save_path: Path,
        custom_features: List[Dict],
        flat_design_variables: Dict[str, np.ndarray],
        f_inv_bivs: dict = Dict[float, TechnikkatalogAssumptions],
        calculate_features: bool = True,
        all_practical_features: dict = None,
):
    custom_metrics_to_predict = get_feature_metrics(custom_features)
    if calculate_features:
        features = features_module.get_practical_features(input_config, all_practical_features)
    else:
        features = {}

    if config.simulation.type == "Static":
        df = get_result_from_input_analysis_for_input(study_config=config, input_config=input_config)
    else:
        df = get_result_for_input(study_config=config, input_config=input_config, with_user=True)
    return_values = {}
    for f_inv_biv, annuity_obj in f_inv_bivs.items():
        # TODO: Maybe change order for bayes?
        # annuity does not change relevant df columns
        df = annuity_obj.calc(df, input_config=input_config)
        surrogate = surrogate_type(df, **surrogate_kwargs)
        plot_name = f"{input_config.get_name()}_{f_inv_biv}.png"
        save_path_plot = save_path.joinpath(
            "SurrogateQuality", plot_name
        ) if create_surrogate_plots else None

        objective_surrogates = surrogate.predict(
            design_variables=flat_design_variables,
            metrics=list(objectives.keys()),
            save_path_plot=save_path_plot,
            plot_config=plot_config
        )
        save_path_plot_features = save_path.joinpath(
            "SurrogateQualityClosedLoopFeatures", plot_name
        ) if create_surrogate_plots else None

        custom_features_for_all_designs = {}
        if custom_features:
            custom_features_surrogates = surrogate.predict(
                design_variables=flat_design_variables,
                metrics=custom_metrics_to_predict,
                save_path_plot=save_path_plot_features,
                plot_config=plot_config
            )
            for feature in custom_features:
                basis_function = feature["basis_function"]
                custom_features_for_all_designs[feature["name"]] = basis_function(
                    **{metric: custom_features_surrogates.loc[:, metric]
                       for metric in feature["metrics"]}
                )
        real_optima = {}
        all_objectives_surrogates = {}
        for objective in objective_surrogates.columns:
            objective_values = objective_surrogates.loc[:, objective]
            all_objectives_surrogates[objective] = objective_values
            optimal_variables = objective_values.index[objective_values.argmin()]
            if len(objective_values.index.names) == 1:
                optimal_variables = [optimal_variables]
            for design_variable, value in zip(objective_values.index.names, optimal_variables):
                real_optima[(objective, design_variable)] = value
            for feature, values in custom_features_for_all_designs.items():
                features[feature] = values[objective_values.argmin()]
        return_values[f_inv_biv] = [features, real_optima, all_objectives_surrogates, input_config]
    return return_values


def _load_data(
        configs: List[StudyConfig],
        objectives: Dict[str, str],
        design_variables: Dict[str, np.ndarray],
        surrogate_type: Type[surrogates.Surrogate],
        surrogate_kwargs: dict,
        plot_config: PlotConfig,
        create_surrogate_plots: bool,
        save_path: Path,
        custom_features: List[Dict],
        f_inv_bivs: Dict[float, TechnikkatalogAssumptions],
        pre_calculated_features: Path = None
):
    logger.info("Loading data for innovization")
    t0 = time.time()
    input_feature_names = features_module.get_feature_names(configs[0].inputs)
    all_objectives_surrogates = {metric: [] for metric in objectives}
    if custom_features and len(objectives) > 1:
        logger.warning(
            "Custom features for multiple objectives are not yet supported, "
            "the design index of the last given objective optimum is used."
        )
    feature_values = {
        "f_inv_biv": [],
        **{feature: [] for feature in input_feature_names},
        **{feature["name"]: [] for feature in custom_features}
    }
    design_values = mesh_arrays(list(design_variables.values()))
    flat_design_variables = {var: design_values[:, idx] for idx, var in enumerate(design_variables)}
    real_optima = {(metric, var): [] for metric, var in itertools.product(objectives, design_variables)}
    kwargs_mp = []
    if len(configs) > 1 and isinstance(pre_calculated_features, Path):
        raise ValueError("pre_calculated_features is only supported for one config")
    for config in configs:
        input_configs = config.inputs.get_permutations()
        if pre_calculated_features is not None:
            df_pre_calculated_features = pd.read_excel(pre_calculated_features, index_col=0)
            all_practical_features = None
            calculate_features = False
        else:
            all_practical_features = features_module.load_practical_features_from_input_analysis()
            calculate_features = True
            df_pre_calculated_features = pd.DataFrame()

        for input_config in input_configs:
            kwargs_mp.append(dict(
                config=config,
                input_config=input_config,
                objectives=objectives,
                surrogate_type=surrogate_type,
                surrogate_kwargs=surrogate_kwargs,
                plot_config=plot_config,
                create_surrogate_plots=create_surrogate_plots,
                save_path=save_path,
                custom_features=custom_features,
                f_inv_bivs=f_inv_bivs,
                all_practical_features=all_practical_features,
                flat_design_variables=flat_design_variables,
                calculate_features=calculate_features
            ))

    use_mp = len(kwargs_mp) > 50000
    use_mp = False
    if use_mp:
        logger.info("Loading %s inputs with multiprocessing", len(kwargs_mp))
    results = execute_function_in_parallel(
        func=_load_single_input_case,
        n_cpu=5,
        func_kwargs=kwargs_mp,
        use_mp=use_mp,
        notifier=print,
        percentage_when_to_message=1
    )
    for result in results:
        for f_inv_biv, data in result.items():
            single_features, single_real_optima, single_objectives_surrogates, input_config = data
            if input_config.get_name() in df_pre_calculated_features.index:
                data = df_pre_calculated_features.loc[input_config.get_name()].to_dict()
                single_features = {
                    **single_features,
                    **data
                }
            for objective, objective_values in single_objectives_surrogates.items():
                all_objectives_surrogates[objective].append(objective_values)
            for _tuple, value in single_real_optima.items():
                real_optima[_tuple].append(value)
            for feature in input_feature_names:
                feature_values[feature].append(single_features[feature])
            feature_values["f_inv_biv"].append(float(f_inv_biv))

    # Filter empty lists
    real_optima = {k: v for k, v in real_optima.items() if v}

    def dict_list_to_np_array(dictionary):
        return {k: np.array(v) for k, v in dictionary.items()}

    logger.info(
        "Loading %s input configs took %s seconds",
        len(kwargs_mp), time.time() - t0
    )
    return {
        "objective_values": all_objectives_surrogates,
        "features": dict_list_to_np_array(feature_values),
        "optima": dict_list_to_np_array(real_optima)
    }


def build_basis_functions(features: List[Union[str, Dict]]):
    """
    Builds a consistent list with information on basis functions.
    If the given features are a string, the `default_basis_function`
    is applied.
    """
    features_clean = []
    for feature in features:
        if isinstance(feature, str):
            features_clean.append({
                "name": feature,
                "metrics": [feature],
                "basis_function": default_basis_function
            })
        else:
            features_clean.append(feature)
    return features_clean


def default_basis_function(**kwargs: Dict[str, pd.Series]) -> np.ndarray:
    """
    This function only gets one feature as input (a pd.Series)
    and returns the values of the series
    """
    assert len(kwargs) == 1, "Multiple metrics passed to default basis functions!"
    return next(iter(kwargs.values())).values


def get_feature_metrics(features: List[Dict]) -> List[str]:
    """
    Get a list of all metrics to predict in order to execute the basis functions.
    """
    metrics_to_predict = []
    for feature in features:
        metrics_to_predict.extend(feature["metrics"])
    return metrics_to_predict


def gradient(df, x: str, y: str):
    # Calculate the gradient dy/dx using numerical differentiation
    return np.gradient(df[y].values, df[x].values)


def gradient_series(df: pd.Series):
    # Calculate the gradient dy/dx using numerical differentiation
    return np.gradient(df.values, df.index)


def build_feature_clusters(feature_values: dict, discrete_features: List[str] = None):
    # Create an all-true mask with the first entry of features, as all have the same length
    clusters = {
        "all": {"all": np.ones(len(feature_values[list(feature_values.keys())[0]])) == 1}
    }
    if discrete_features is None:
        return clusters
    for discrete_feature in discrete_features:
        discrete_values = set(feature_values[discrete_feature])
        clusters[discrete_feature] = {
            str(discrete_value): feature_values[discrete_feature] == discrete_value
            for discrete_value in discrete_values
        }
    return clusters


def get_feature_combinations(
        features_to_consider: List[str],
        n_features: int = None,
):
    n_max = len(features_to_consider)
    if n_features is None:
        n_features = n_max
    elif n_features > n_max:
        raise ValueError(
            f"Maximal number of provided features is "
            f"{n_max}, can't combine more than that."
        )
    all_feature_names = []
    for i in range(1, n_features + 1):
        for combination in combinations(features_to_consider, i):
            all_feature_names.append(list(combination))
    return all_feature_names


def run_brute_force_innovization(
        configs: Union[List[StudyConfig], StudyConfig],
        objectives: Dict[str, str],
        design_variables: Dict[str, np.ndarray],
        regressors: List[Regressor],
        surrogate_type: Type[surrogates.Surrogate],
        surrogate_kwargs: dict,
        save_path: Path,
        plot_config: PlotConfig,
        discrete_features: List[str] = None,
        reload: bool = False,
        features_to_consider: List[str] = None,
        n_features: int = None,
        create_surrogate_plots: bool = True,
        custom_features: List[Union[str, Dict]] = None,
        pre_calculated_features: Path = None,
        plot_optimality_gap: bool = False,
        with_f_inv_bivs: bool = True
):
    if isinstance(configs, StudyConfig):
        configs = [configs]

    if features_to_consider is None and custom_features is None:
        features_to_consider = features_module.get_feature_names(configs[0].inputs)

    if custom_features is None:
        custom_features = []
    else:
        custom_features = build_basis_functions(custom_features)

    for custom_feature in custom_features:
        features_to_consider.append(custom_feature["name"])

    f_inv_bivs = load_f_inv_biv()
    if with_f_inv_bivs:
        features_to_consider.append("f_inv_biv")
    else:
        mid_item = list(f_inv_bivs.keys())[3]
        f_inv_bivs = {mid_item: f_inv_bivs[mid_item]}

    all_feature_names = get_feature_combinations(features_to_consider, n_features)

    logger.info("Loading data")
    load_args = dict(
        configs=configs,
        objectives=objectives,
        design_variables=design_variables,
        surrogate_type=surrogate_type,
        surrogate_kwargs=surrogate_kwargs,
        save_path=save_path,
        create_surrogate_plots=create_surrogate_plots,
        plot_config=plot_config,
        f_inv_bivs=f_inv_bivs,
        custom_features=custom_features,
        pre_calculated_features=pre_calculated_features
    )
    pickle_path = save_path.joinpath("loaded_data.pickle")

    if reload:
        data_struct = _load_data(**load_args)
    else:
        if not os.path.exists(pickle_path):
            data_struct = _load_data(**load_args)
        else:
            with open(pickle_path, "rb") as file:
                data_struct = pickle.load(file)
    os.makedirs(save_path, exist_ok=True)
    with open(pickle_path, "wb") as file:
        pickle.dump(data_struct, file)

    clusters = build_feature_clusters(
        discrete_features=discrete_features,
        feature_values=data_struct["features"]
    )
    all_results = []
    for (objective_name, design_variable, feature_names) in itertools.product(
            objectives.keys(),
            design_variables,
            all_feature_names
    ):
        logger.info(f"Finding rule for %s, %s and %s",
                    objective_name, design_variable, feature_names)
        if (objective_name, design_variable) not in data_struct["optima"]:
            if surrogate_type == surrogates.LinearInterpolationSurrogate:
                log_type = logger.info
            else:
                log_type = logger.error
            log_type(
                "No results for objective_name %s and design_variable %s",
                objective_name, design_variable
            )
            continue

        feature_values = np.array([
            data_struct["features"][feature_name]
            for feature_name in feature_names
        ])
        objective_values = data_struct["objective_values"][objective_name]
        optimal_design_values = data_struct["optima"][(objective_name, design_variable)]
        case_name = f"{design_variable}/{objective_name}/{'_'.join(feature_names)}"
        logger.error("Analyzing case %s", case_name)
        results = perform_regressions_for_all_clusters_and_regressors(
            objective_values=objective_values,
            optimal_design_values=optimal_design_values,
            feature_values=feature_values,
            design_variable=design_variable,
            feature_names=feature_names,
            regressors=regressors,
            clusters=clusters,
            save_path=save_path.joinpath(case_name),
            plot_config=plot_config,
            objective_name=objective_name,
            plot_optimality_gap=plot_optimality_gap
        )
        all_results.extend(
            [
                {
                    "design_variable": design_variable,
                    "objective": objective_name,
                    "features": '_'.join(feature_names),
                    **result
                }
                for result in results
            ]
        )
    pd.DataFrame(all_results).to_excel(save_path.joinpath("BruteForceInnovization.xlsx"))


def perform_regressions_for_all_clusters_and_regressors(
        objective_values: np.ndarray,
        optimal_design_values: np.ndarray,
        feature_values: np.ndarray,
        design_variable: str,
        feature_names: List[str],
        regressors: List[Regressor],
        save_path: Path,
        clusters: Dict[str, Any],
        plot_config: PlotConfig,
        objective_name: str,
        plot_optimality_gap: bool
):
    all_cluster_results = []
    all_cluster_parameters = {}
    for cluster_name, cluster_masks in clusters.items():
        for cluster_value, mask in cluster_masks.items():
            for regressor in regressors:
                case_name = "_".join([cluster_name, cluster_value, regressor.name])
                plot_path = save_path.joinpath(f"{case_name}.png")
                logger.debug("Analyzing regressor %s", case_name)
                if feature_names == ["f_inv_biv"]:
                    print("error")
                result = perform_single_regression(
                    regressor=regressor,
                    objective_values=[s for s, m in zip(objective_values, mask) if m],
                    optimal_design_values=optimal_design_values[mask],
                    feature_values=feature_values[:, mask],
                    design_variable=design_variable,
                    feature_names=feature_names,
                    save_path=plot_path,
                    plot_config=plot_config,
                    objective_name=objective_name,
                    plot_optimality_gap=plot_optimality_gap
                )
                all_cluster_parameters[case_name] = list(result["parameters"])
                all_cluster_results.append({
                    "cluster_name": cluster_name,
                    "cluster_value": cluster_value,
                    "Regression": regressor.name,
                    "mean": result["deviations"]["mean"],
                    "max": result["deviations"]["max"],
                    "RMSE": result["deviations"]["RMSE"],
                    "rule": result["rule"],
                    "parameters_json_key": case_name,
                    "plot_path": plot_path.as_posix()
                })
    with open(save_path.joinpath("rule_parameters.json"), "w") as file:
        json.dump(all_cluster_parameters, file, indent=2)
    return all_cluster_results


def perform_single_regression(
        regressor: Regressor,
        objective_values: list,  # pd.Series,
        optimal_design_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str],
        save_path: Path,
        design_variable: str,
        objective_name: str,
        plot_config: PlotConfig,
        plot_optimality_gap: bool = True
):
    parameters = regressor.get_parameters(x=feature_values, y=optimal_design_values)
    design_rule_string = regressor.get_equation_string(
        design_variable=design_variable,
        feature_names=feature_names,
        parameters=parameters,
        plot_config=plot_config
    )
    optimal_design_regressions = regressor.eval(x=feature_values, parameters=parameters)

    deviations, deviation_per_optimal_design = objective_space.get_deviation_from_optimum(
        optimal_design_regressions=optimal_design_regressions,
        objective_values=objective_values
    )
    if plot_optimality_gap:
        rule_plotting.plot_optimality_gap(
            optimal_design_regressions=optimal_design_regressions,
            objective_values=objective_values,
            plot_config=plot_config,
            feature_values=feature_values,
            feature_names=feature_names,
            design_variable=design_variable,
            save_path=save_path.with_stem(f"optimality_gap_{save_path.stem}"),
            objective_name=objective_name
        )
    rule_plotting.plot_single_regression(
        regressor=regressor,
        parameters=parameters,
        optimal_design_values=optimal_design_values,
        feature_values=feature_values,
        feature_names=feature_names,
        design_rule_string=design_rule_string,
        design_variable=design_variable,
        save_path=save_path,
        plot_config=plot_config,
        objective_name=objective_name,
        deviation_per_optimal_design=deviation_per_optimal_design
    )
    return {
        "deviations": deviations,
        "rule": design_rule_string,
        "parameters": parameters
    }


def analyze_convergence(
        innovization_results: Dict[str, Path],
        regressors: List[Regressor],
        save_path: Path,
        pickle_path: Path,
        objectives: Dict[str, str],
        plot_config: PlotConfig,
        design_variables: Dict[str, np.ndarray],
        discrete_features: List[str] = None,
        features_to_consider: List[str] = None,
        n_features: int = None,
):
    with open(pickle_path, "rb") as file:
        data_struct = pickle.load(file)
    os.makedirs(save_path, exist_ok=True)

    all_feature_names = get_feature_combinations(features_to_consider, n_features)

    clusters = build_feature_clusters(
        discrete_features=discrete_features,
        feature_values=data_struct["features"]
    )
    for (objective_name, design_variable, feature_names) in itertools.product(
            objectives.keys(),
            design_variables,
            all_feature_names
    ):
        feature_values = np.array([
            data_struct["features"][feature_name]
            for feature_name in feature_names
        ])
        outer_case_name_parts = [
            design_variable,
            objective_name,
            '_'.join(feature_names),
        ]
        objective_values = data_struct["objective_values"][objective_name]
        for cluster_name, cluster_masks in clusters.items():
            for cluster_value, mask in cluster_masks.items():
                for regressor in regressors:
                    objective_values = [s for s, m in zip(objective_values, mask) if m]
                    all_deviations = {
                        "mean": {},
                        "max": {},
                        # "RMSE": {},
                    }
                    for result, innovization_result_path in innovization_results.items():
                        with open(Path(innovization_result_path).parent.joinpath(
                                *outer_case_name_parts, "rule_parameters.json"
                        ), "r") as file:
                            loaded_parameters = json.load(file)

                        df = pd.read_excel(innovization_result_path, index_col=0)
                        idx = (
                                (df.loc[:, "features"] == "_".join(feature_names)) &
                                (df.loc[:, "objective"] == objective_name) &
                                (df.loc[:, "cluster_name"] == cluster_name) &
                                (df.loc[:, "cluster_value"] == cluster_value) &
                                (df.loc[:, "Regression"] == regressor.name)
                        )
                        parameters = np.array(loaded_parameters[df.loc[idx, "parameters_json_key"].values[0]])
                        optimal_design_regressions = regressor.eval(x=feature_values[:, mask], parameters=parameters)
                        deviations, _ = objective_space.get_deviation_from_optimum(
                            optimal_design_regressions=optimal_design_regressions,
                            objective_values=objective_values
                        )
                        for dev_metric, value in deviations.items():
                            all_deviations[dev_metric][result] = value
                    case_name = "_".join(outer_case_name_parts + [cluster_name, cluster_value, regressor.name]) + ".png"
                    rule_plotting.plot_convergence(
                        objective_name=objective_name,
                        all_deviations=all_deviations,
                        save_path=save_path.joinpath(case_name),
                        plot_config=plot_config
                    )


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    PLOT_CONFIG = PlotConfig.load_default(language="de")

    CONFIG = StudyConfig.from_json(r"R:\_Dissertationen\fwu\06_Diss\03_Ergebnisse\Test180Cases\study_config.json")
    CONFIG = StudyConfig.from_json(r"D:\00_temp\01_design_optimization\TestSimplified\study_config.json")

    run_brute_force_innovization(
        configs=CONFIG,
        reload=True,
        objectives={"costs_total": "min"},
        design_variables={
            "parameterStudy.TBiv": np.linspace(-16, 5, 100) + 273.15,
            "parameterStudy.VPerQFlow": np.array([12])
        },
        features_to_consider=[
            'GTZ_Ti_HT',
            #"Q_demand_total"
            "THyd_nominal"
            # 'QNomRed', 'THyd_nominal'
        ],
        regressors=[
            LinearRegressor(),
            PowerLawRegressor()
        ],
        #discrete_features=["year"],
        surrogate_type=surrogates.LinearInterpolationSurrogate,
        surrogate_kwargs={},
        save_path=CONFIG.study_path.joinpath("manual_innovization"),
        plot_config=PLOT_CONFIG,
        create_surrogate_plots=False,
        #custom_features=["costs_invest", "costs_operating"],
        with_f_inv_bivs=True
    )
