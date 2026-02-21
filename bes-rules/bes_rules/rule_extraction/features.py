import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bes_rules import RESULTS_FOLDER
from bes_rules.boundary_conditions.building import get_retrofit_temperatures
from bes_rules.configs import StudyConfig
from bes_rules.configs.inputs import InputConfig, InputsConfig
from bes_rules.utils.functions import get_heating_degree_days, get_heating_threshold_temperature_for_building


def load_practical_features_from_input_analysis() -> dict:
    with open(RESULTS_FOLDER.joinpath("input_analysis", "practical_features.json"), "r") as file:
        return json.load(file)


def get_practical_features(
        input_config: InputConfig,
        all_practical_features: dict = None,
        with_custom_features: bool = False
) -> dict:
    df = input_config.weather.get_hourly_weather_data()
    TOda = df.loc[:, "t"] + 273.15
    heating_threshold = get_heating_threshold_temperature_for_building(building=input_config.building)
    TZoneSet = input_config.user.room_set_temperature

    THyd_nominal, _, THydNoRet_nominal, _, QNoRet_flow_nominal, QRet_flow_nominal = get_retrofit_temperatures(
        building_config=input_config.building,
        TOda_nominal=input_config.weather.TOda_nominal,
        TRoom_nominal=input_config.user.room_set_temperature,
        retrofit_transfer_system_to_at_least=input_config.building.retrofit_transfer_system_to_at_least
    )
    if all_practical_features is None:
        q_demand_total = 0
        dhw_share = 0
    else:
        dhw_share = all_practical_features[input_config.get_name()]["dhw_share"]
        q_demand_total = all_practical_features[input_config.get_name()]["q_demand_total"]

    # praxisnahe Features
    practical_features = {
        "TOda_nominal": input_config.weather.TOda_nominal,
        "Q_demand_total": q_demand_total * input_config.building.net_leased_area,
        "q_demand_total": q_demand_total,
        "QHeaLoa_flow": QRet_flow_nominal,
        "qHeaLoa_flow": QRet_flow_nominal / (input_config.user.room_set_temperature - input_config.weather.TOda_nominal),
        #"TThr": get_heating_threshold_temperature_for_building(building=input_config.building),  # TODO: Maybe from input analysis?
        "GTZ_Ti_HT": get_heating_degree_days(TOda, TZoneSet, heating_threshold),
        #"TRoomSet": input_config.user.room_set_temperature,
        "THyd_nominal": THyd_nominal,
        #"TDHW_nominal": 273.15 + 50,
        "dhw_share": dhw_share
    }
    if with_custom_features:
        practical_features.update(
            {
                # "GTZ_Ti_Ti": get_heating_degree_days(TOda, TZoneSet, TZoneSet),
                # "area": input_config.building.net_leased_area,
                "year": input_config.building.year_of_construction,
                "QNomRed": QRet_flow_nominal / QNoRet_flow_nominal,
                # Custom / new features
                # "TMin": TOda.min(),
                # "TMean": TOda.mean(),
                # "TMeanSmaller20": df.loc[TOda < 20 + 273.15, "t"].mean(),
                # "phiHeatingToNominal": (df.loc[TOda < 20 + 273.15, "t"].mean()) / input_config.weather.TOda_nominal,
            }
        )
    return practical_features


def get_feature_names(inputs_config: InputsConfig) -> List[str]:
    return list(get_practical_features(InputConfig(
        weather=inputs_config.weathers[0],
        building=inputs_config.buildings[0],
        dhw_profile=inputs_config.dhw_profiles[0],
        user=inputs_config.users[0],
        evu_profile=inputs_config.evu_profiles[0]
    )).keys())


def add_features_to_surrogates(config: StudyConfig):
    path = config.study_path.joinpath(f"{config.name}_all_results.xlsx")

    all_practical_features = load_practical_features_from_input_analysis()

    df = pd.read_excel(path)
    for input_config in config.inputs.get_permutations():
        parts = input_config.get_name_parts()
        features = get_practical_features(input_config=input_config, all_practical_features=all_practical_features)
        mask = np.ones(len(df)) == 1
        for input_name, input_value in parts.items():
            mask = mask & (df.loc[:, input_name] == input_value)
        for feature, value in features.items():
            df.loc[mask, feature] = value

    save_path = config.study_path.joinpath(f"{config.name}_all_results_with_features.xlsx")
    df.to_excel(
        save_path,
        sheet_name="Results"
    )


def plot_feature_dependence(config: StudyConfig):
    from bes_rules.plotting.design_space import plot_scatter_for_x_over_multiple_y
    save_path = config.study_path.joinpath(f"{config.name}_all_results_with_features.xlsx")
    plot_path = config.study_path.joinpath("feature_plots")
    os.makedirs(plot_path, exist_ok=True)
    df = pd.read_excel(save_path, sheet_name="Results")
    variables = [
        "SCOP_Sys",
        "SCOP_HP",
        "HP_Coverage"
    ]
    for _df, name in zip([df, df.loc[df.loc[:, "parameterStudy.TBiv"] > 273.15 - 13]], ["oversize", "normal"]):
        mask = _df.loc[:, "parameterStudy.TBiv"] == _df.loc[:, "parameterStudy.TBiv"].min()
        for feature in [
            "THyd_nominal",
            #"QNomRed"
        ]:
            fig, axes = plt.subplots(3, 1, sharex=True)
            for var, ax in zip(variables, axes):
                ax.scatter(_df.loc[mask, feature] - 273.15, _df.loc[mask, var], label=var)
                ax.set_ylabel("$SCOP_\mathrm{WP}$ in -")
            axes[-1].set_xlabel("$T_\mathrm{VL}$ in °C")
            fig.savefig(plot_path.joinpath(f"{feature}_{name}.png"))
    plot_scatter_for_x_over_multiple_y(
        study_config=config,
        save_path=plot_path,
        x_variable="parameterStudy.TBiv",
        y_variables=variables
    )


def plot_SCOP_over_THyd(config: StudyConfig):
    from bes_rules.plotting.utils import load_plot_config, get_figure_size
    plot_config = load_plot_config()
    save_path = config.study_path.joinpath(f"{config.name}_all_results_with_features.xlsx")
    plot_path = config.study_path.joinpath("feature_plots")
    os.makedirs(plot_path, exist_ok=True)
    df = pd.read_excel(save_path, sheet_name="Results")
    df = df.loc[df.loc[:, "parameterStudy.TBiv"] > 273.15 - 13]
    mask = df.loc[:, "parameterStudy.TBiv"] == df.loc[:, "parameterStudy.TBiv"].min()
    x = df.loc[mask, "THyd_nominal"].values - 273.15
    y = df.loc[mask, "SCOP_HP"].values[x < 70]
    x = x[x < 70]
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=get_figure_size(1))
    ax.scatter(x, y)
    ax.plot(x, 5.08 - 0.031 * x, color="red")
    ax.set_ylabel("$SCOP_\mathrm{WP}$ in -")
    ax.set_xlabel("$T_\mathrm{VL}$ in °C")
    fig.tight_layout()
    fig.savefig(plot_path.joinpath(f"SCOP_THyd_nominal.png"))
    from bes_rules.rule_extraction.regression.regressors import LinearRegressor
    parameters = LinearRegressor.get_parameters(np.array([x]), y)
    print(f"SCOP = {parameters[-1]} + T_VL * {parameters[0]}")


if __name__ == '__main__':
    CONFIG = StudyConfig.from_json(RESULTS_FOLDER.joinpath("TBivOptimization", "OversizeRetrofitOptions", "study_config.json"))
    #add_features_to_surrogates(config=CONFIG)
    plot_SCOP_over_THyd(config=CONFIG)
