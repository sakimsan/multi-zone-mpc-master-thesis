"""
Module to build regression models to build design rules.
"""
import logging
import os

import pandas as pd

from bes_rules.configs import StudyConfig
from bes_rules.simulation_based_optimization.base import SurrogateBuilder
from bes_rules.utils import save_merge_dicts
from bes_rules.rule_extraction import features as features_module
logger = logging.getLogger(__name__)


def map_optimal_values_to_features(config: StudyConfig, percent_deviation: float):
    obj_vars = config.optimization.get_variable_names()
    obj_kpis = config.optimization.objective_names
    n_var = len(obj_vars)
    n_kpi = len(obj_kpis)
    if n_kpi == 0:
        raise ValueError("No optimization names / KPIs selected. Can't perform regression analysis")
    if n_var == 0:
        raise ValueError("No optimization variables selected. Can't perform regression analysis")
    feature_names = features_module.get_feature_names(config.inputs)
    df_kpis = [pd.DataFrame(columns=obj_vars + feature_names) for _ in obj_kpis]
    all_practical_features = features_module.load_practical_features_from_input_analysis()

    for idx, input_config in enumerate(config.inputs.get_permutations()):
        study_name = input_config.get_name()
        log_path = SurrogateBuilder.create_and_get_log_path(
            base_path=config.study_path, study_name=study_name
        )
        if not os.path.exists(log_path):
            logger.error("Can not process %s, no xlsx result file!", study_name)
            continue
        df = SurrogateBuilder.load_design_optimization_log(file_path=log_path)
        features = features_module.get_practical_features(input_config, all_practical_features)
        if list(features.keys()) != feature_names:
            raise KeyError("feature_names changed during inputs iteration!")
        for idx_kpi, obj_kpi in enumerate(obj_kpis):
            try:
                x_opt = df.loc[df.loc[:, obj_kpi].argmin(), obj_vars].to_dict()
                # Near optimal solutions:
                df_kpis[idx_kpi].loc[idx] = save_merge_dicts(x_opt, features)
                _add_percent_deviation(df_kpis[idx_kpi], df, idx, obj_kpi, obj_vars)
            except Exception as err:
                logger.error("Could not load %s, err: %s", study_name, err)

    for obj_kpi, df_kpi in zip(obj_kpis, df_kpis):
        df_kpi.to_excel(
            config.study_path.joinpath(f"FeatureOptimaMapping_{obj_kpi}.xlsx"), sheet_name=obj_kpi
        )


def _add_percent_deviation(df_kpi, df, idx, obj_kpi, obj_vars):
    for percent_deviation in [1, 3, 5, 10]:
        bound = df.loc[:, obj_kpi].min() * (1 + percent_deviation * 0.01)
        df_kpi.loc[
            idx,
            get_percent_deviation_names(obj_vars, percent_deviation, "min")
        ] = df.loc[df.loc[:, obj_kpi] < bound, obj_vars].min().values
        df_kpi.loc[
            idx,
            get_percent_deviation_names(obj_vars, percent_deviation, "max")
        ] = df.loc[df.loc[:, obj_kpi] < bound, obj_vars].max().values


def get_percent_deviation_names(obj_vars: list, percent_deviation: float, minmax: str):
    return [f"{obj_var} {percent_deviation}-percent {minmax}" for obj_var in obj_vars]


if __name__ == '__main__':
    CONFIG = StudyConfig.from_json(r"R:\_Dissertationen\fwu\06_Diss\03_Ergebnisse\Test180Cases\study_config.json")
    #for OBJ in ["costs_total", "costs_operating", "costs_invest", "emissions"]:
    #CONFIG.optimization.objective_names = [OBJ]
    #CONFIG.optimization.objective_names = ["costs_total"]
    CONFIG.optimization.objective_names = ["costs_total", "emissions"]
    map_optimal_values_to_features(config=CONFIG, percent_deviation=0.05)
