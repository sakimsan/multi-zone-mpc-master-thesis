import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bes_rules.input_analysis import oed, input_analysis, heat_pump_system
from bes_rules import RESULTS_FOLDER
from studies.use_case_1_design import simulate_oed_cases

from bes_rules import RESULTS_FOLDER, configs
from bes_rules.plotting.utils import PlotConfig
from bes_rules.plotting import utils
from bes_rules.rule_extraction.innovization import run_brute_force_innovization, analyze_convergence
from bes_rules.rule_extraction.regression.regressors import LinearRegressor, PowerLawRegressor
from bes_rules.rule_extraction.surrogates import LinearInterpolationSurrogate, BayesSurrogate

import numpy as np
import logging

PATH_INPUT_ANALYSIS = RESULTS_FOLDER.joinpath("input_analysis", "no_dhw_extended")
PATH_OED = PATH_INPUT_ANALYSIS.joinpath("OED")
PATH_SIMULATIONS = RESULTS_FOLDER.joinpath("UseCase_TBivAndV", "no_dhw")
FEATURES = [
    "Q_demand_total",
    "GTZ_Ti_HT",
    "THyd_nominal",
]


def get_innovization_default_kwargs():
    plot_config = PlotConfig.load_default(language="de")
    surrogate_type = LinearInterpolationSurrogate
    if surrogate_type == BayesSurrogate:
        with open(PATH_SIMULATIONS.joinpath("best_hyperparameters.json"), "r") as f:
            paras = json.load(f)
        surrogate_kwargs = {"metric_hyperparameters": paras}
    else:
        surrogate_kwargs = {}
    return dict(
        objectives={"costs_total": "min"},
        features_to_consider=FEATURES,
        regressors=[
            LinearRegressor(),
            PowerLawRegressor()
        ],
        discrete_features=[],
        surrogate_type=surrogate_type,
        surrogate_kwargs=surrogate_kwargs,
        plot_config=plot_config,
        reload=False,
        plot_optimality_gap=False,
        with_f_inv_bivs=True
        # custom_features=["costs_invest", "costs_operating"]
    )


def innovization_simplified(static_model: str, path_input_analysis: Path):
    config = input_analysis.create_study_config_for_analysis(
        save_path=path_input_analysis, name=static_model, n_to_load=None
    )
    run_brute_force_innovization(
        **get_innovization_default_kwargs(),
        configs=config,
        create_surrogate_plots=False,
        save_path=path_input_analysis.joinpath("manual_innovization", static_model),
        design_variables={
            "parameterStudy.TBiv": np.linspace(-16, 10, 100) + 273.15,
            # "parameterStudy.VPerQFlow": np.array([12])
        },
        pre_calculated_features=path_input_analysis.joinpath(static_model, "AllFeatures.xlsx")
    )


def innovization_simulations(oed_iter: int, path_simulations: Path = PATH_SIMULATIONS):
    study_name = f"OED_iter_{oed_iter}"

    config = configs.StudyConfig.from_json(path_simulations.joinpath(study_name, "study_config.json"))

    run_brute_force_innovization(
        **get_innovization_default_kwargs(),
        create_surrogate_plots=True,
        design_variables={
            "parameterStudy.TBiv": np.linspace(-16, 5, 100) + 273.15,
            "parameterStudy.VPerQFlow": np.linspace(12, 35, 100)
        },
        configs=config,
        save_path=path_simulations.joinpath(study_name, "manual_innovization")
    )


def perform_input_analysis():
    input_analysis.analyze_all(
        heat_pump=heat_pump_system.VitoCal250,
        save_path=PATH_INPUT_ANALYSIS,
        use_mp=True,
        dhw_profile="NoDHW",
        with_pv=False,
        save_plots=False,
        raise_errors=True,
        transfer_retrofit_to_maximum_heat_pump_temperature=True,
        # new_vent_rate_settings={"only_daytime": True, "winterReduction": [0.2, 273.15, 283.15]},
        # input_config_names=[
        #    'TRY2015_506745079707_Somm_B1859_standard_o0w2r0g0_SingleDwelling_NoDHW_0K-Per-IntGai',
        # ]
    )
    oed.create_all_practical_features_json(
       save_path=PATH_INPUT_ANALYSIS,
       save_path_excel=PATH_OED
    )


def create_features(path_input_analysis: Path = PATH_INPUT_ANALYSIS):
    for name in ["TEASER"]:
        oed.create_all_practical_features_json(
            save_path=path_input_analysis,
            save_path_excel=path_input_analysis.joinpath(name),
            name=name
        )


def perform_oed():
    plot_config = oed.load_plot_config(language="de")

    oed.perform_oed_for_all_inputs(
        save_path=PATH_OED.joinpath(f"2_n_exp=3"),
        features_to_consider=FEATURES,
        experiments_per_feature=3,
        save_path_all_features=PATH_OED.joinpath("AllFeatures.xlsx"),
        plot_config=plot_config,
        n_repeat=3,
        just_corner_points=False,
        with_cases_already_simulated=True
    )


def find_minimum_required_oed_examples(
        path_oed: Path = PATH_OED,
        path_input_analysis: Path = PATH_INPUT_ANALYSIS,
        dhw_profile: str = "NoDHW",
        heat_pump=heat_pump_system.VitoCal250,
        features: list = None,
        calc_oed: bool = True,
        calc_input_analysis: bool = True,
        calc_rules: bool = True,
        plot_oed: bool = False
):
    if features is None:
        features = FEATURES
    path_oed_verification = path_oed.joinpath("verification")
    plot_config = oed.load_plot_config(language="de")
    all_features_path = path_oed.joinpath("AllFeatures.xlsx")

    innovization_kwargs = dict(
        design_variables={"parameterStudy.TBiv": np.linspace(-16, 5, 100) + 273.15},
        pre_calculated_features=all_features_path,
        **get_innovization_default_kwargs(),
    )
    cases_simulated = []
    for i in range(0, 10):
        if i == 0:
            just_corner_points = True
            oed_name = "corner_points"
            with_cases_already_simulated = False
            file_name = "corner_results.xlsx"
        else:
            just_corner_points = False
            oed_name = f"iter_{i}"
            with_cases_already_simulated = True
            file_name = "OED_results_0.xlsx"
        path_oed_case = path_oed_verification.joinpath(oed_name)
        if calc_oed:
            oed.perform_oed_for_all_inputs(
                save_path=path_oed_case,
                features_to_consider=features,
                experiments_per_feature=1,
                save_path_all_features=all_features_path,
                plot_config=plot_config,
                n_repeat=1,
                just_corner_points=just_corner_points,
                with_cases_already_simulated=with_cases_already_simulated,
                cases_already_simulated=cases_simulated
            )
        if plot_oed:
            oed.plot_feature_design_space(
                save_path=path_oed_case,
                save_path_all_features=all_features_path,
                save_name=file_name.replace(".xlsx", ""),
                x_feature=features[0],
                plot_config=plot_config
            )
        df_oed = pd.read_excel(path_oed_case.joinpath(file_name), index_col=0)
        oed_cases_to_simulate = df_oed.index
        cases_simulated.extend(list(oed_cases_to_simulate))
        with open(path_oed_verification.joinpath(oed_name + ".json"), "w") as file:
            json.dump(cases_simulated, file, indent=2)

        if calc_input_analysis:
            input_analysis.analyze_all(
                heat_pump=heat_pump,
                save_path=path_oed_case.joinpath("input_analysis"),
                use_mp=True,
                dhw_profile=dhw_profile,
                with_pv=False,
                save_plots=False,
                raise_errors=True,
                transfer_retrofit_to_maximum_heat_pump_temperature=True,
                input_config_names=oed_cases_to_simulate
            )
        if calc_rules:
            config_for_oed_run = input_analysis.create_study_config_for_given_input_config_names(
                save_path=path_input_analysis,
                name="TEASER",
                input_config_names=cases_simulated,
            )
            run_brute_force_innovization(
                **innovization_kwargs,
                configs=config_for_oed_run,
                save_path=path_oed_case.joinpath("innovization")
            )


def plot_oed_convergence(full_innovization_path: Path, path_oed: Path = PATH_OED):
    plot_config = PlotConfig.load_default(language="de")
    pickle_path = full_innovization_path.joinpath("loaded_data.pickle")
    path_oed_verification = path_oed.joinpath("verification")
    innovization_results = {}
    total_sim = 0
    for i in range(0, 10):
        if i == 0:
            oed_name = "corner_points"
            file_name = "corner_results.xlsx"
        else:
            oed_name = f"iter_{i}"
            file_name = "OED_results_0.xlsx"
        df_oed = pd.read_excel(path_oed_verification.joinpath(oed_name, file_name), index_col=0)
        total_sim += len(df_oed.index)
        innovization_results[total_sim] = path_oed_verification.joinpath(oed_name, "innovization",
                                                                         "BruteForceInnovization.xlsx")
    innovization_results[total_sim * 2] = full_innovization_path.joinpath("BruteForceInnovization.xlsx")
    kwargs = get_innovization_default_kwargs()
    analyze_convergence(
        innovization_results=innovization_results,
        pickle_path=pickle_path,
        regressors=kwargs["regressors"],
        objectives=kwargs["objectives"],
        design_variables={"parameterStudy.TBiv": np.linspace(-16, 5, 100) + 273.15},
        discrete_features=kwargs["discrete_features"],
        features_to_consider=kwargs["features_to_consider"],
        save_path=path_oed_verification,
        plot_config=plot_config
    )


def run_simulations(
        oed_iter: int,
        path_oed: Path = PATH_OED,
        path_input_analysis: Path = PATH_INPUT_ANALYSIS
):
    study_name = f"OED_iter_{oed_iter}"
    with open(path_oed.joinpath("verification", f"iter_{oed_iter}" + ".json"), "r") as file:
        cases_to_simulate = json.load(file)

    simulate_oed_cases.run(
        study_name,
        base_path=PATH_SIMULATIONS,
        n_cpu=12,
        cases_to_simulate=cases_to_simulate,
        input_analysis_path=path_input_analysis
    )


def plot_SCOP_over_TBiv_for_cases(
        static_model: str,
        show=False,
        cases: dict = None,
        save_path: Path = None
):
    if cases is None:
        cases = {
            "BESMod": PATH_SIMULATIONS.joinpath("OED_1_3exp", "DesignOptimizationResults"),
            "Static": PATH_INPUT_ANALYSIS.joinpath(static_model),
            "VDI 4645": PATH_INPUT_ANALYSIS.joinpath(static_model + "_VDI4645")
        }
    if save_path is None:
        save_path = PATH_INPUT_ANALYSIS.joinpath("compare_to_VDI")
    plot_config = utils.load_plot_config()
    x_variable = "parameterStudy.TBiv"
    y_variables = [
        "SCOP_Sys",
        "HP_Coverage",
        # "eps_hp"
    ]
    plot_kwargs = {
        "VDI 4645": dict(color="red", marker="s"),
        "Static": dict(color="blue", marker="^"),
        "BESMod": dict(color="green", marker="o"),
    }
    all_comparisons = {case: [] for case in cases}
    for file in os.listdir(list(cases.values())[0]):
        fig, axes = utils.create_plots(
            plot_config=plot_config,
            x_variables=[x_variable],
            y_variables=y_variables
        )
        for case, path in cases.items():
            if case == "BESMod":
                xlsx_path = path.joinpath(file, "DesignOptimizerResults.xlsx")
            else:
                xlsx_path = path.joinpath(file + "_noPV_1h.xlsx")
            if not os.path.exists(xlsx_path):
                print(case, xlsx_path, "does not exist")
                continue
            df = pd.read_excel(xlsx_path, index_col=0)
            df = plot_config.scale_df(df)
            for _y_variable, _ax in zip(y_variables, axes[:, 0]):
                _ax.scatter(
                    df.loc[:, x_variable], df.loc[:, _y_variable],
                    label=case, **plot_kwargs[case]
                )
            mask = df.loc[:, x_variable] == df.loc[:, x_variable].min()
            all_comparisons[case].append(np.mean(df.loc[mask, "SCOP_Sys"].values))

        axes[0, 0].legend(bbox_to_anchor=(0, 1), loc="lower left")
        utils.save(
            fig=fig, axes=axes,
            save_path=save_path.joinpath(file.replace(".xlsx", "")),
            show=show, with_legend=False, file_endings=["png"]
        )
    cases_to_compare = ["Static", "BESMod"]
    fig, axes = plt.subplots(len(cases_to_compare), 1, figsize=utils.get_figure_size(n_columns=1, height_factor=1.5))
    if len(cases_to_compare) == 1:
        axes = [axes]
    for case, ax in zip(cases_to_compare, axes):
        ax.scatter(
            all_comparisons["VDI 4645"], all_comparisons[case],
            s=5, color="red"
        )
        ax.plot([2, 5], [2, 5], color="black")
        ax.set_ylabel(plot_config.get_label_and_unit("SCOP_Sys"))
        ax.set_xlabel("VDI-" + plot_config.get_label_and_unit("SCOP_Sys"))
        ax.set_title(case)
    utils.save(
        fig=fig, axes=axes,
        save_path=save_path.joinpath("compare_all"),
        show=show, with_legend=False, file_endings=["png"]
    )


if __name__ == '__main__':
    import faulthandler

    faulthandler.enable()  # start @ the beginning

    logging.basicConfig(level="INFO")
    # perform_input_analysis()
    # oed.create_all_feature_plots(PATH_OED)
    # create_features()
    innovization_simplified("TEASER", path_input_analysis=PATH_INPUT_ANALYSIS)

    # innovization_simulations(oed_iter=3)
    # simplified_models(static_model="TEASER")
    # find_minimum_required_oed_examples(calc_rules=True, calc_oed=True, calc_input_analysis=True, plot_oed=True)
    # plot_oed_convergence(full_innovization_path=PATH_INPUT_ANALYSIS.joinpath("manual_innovization", "TEASER"))
    # run_simulations(oed_iter=3)
    # plot_SCOP_over_TBiv_for_cases("StaticDemand")
