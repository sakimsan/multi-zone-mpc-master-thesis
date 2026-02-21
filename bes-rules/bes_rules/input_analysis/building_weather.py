import os
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bes_rules import configs, DATA_PATH
from bes_rules.boundary_conditions import building
from bes_rules.boundary_conditions.weather import get_all_weather_configs

from bes_rules.configs.inputs import WeatherConfig, BuildingConfig
from bes_rules.input_analysis import heat_load_estimation
from bes_rules.input_analysis.simulate_combinations import simulate_all_combinations
from bes_rules.utils.functions import get_heating_threshold_temperature_for_building
from bes_rules.input_analysis.utils import periodic_append_of_init_period


def estimate_thermal_demand(
        TOda_mean: pd.Series,
        building: BuildingConfig,
        weather_config: WeatherConfig,
        TRoom: float
):
    # Get interval span in hours:
    interval_length = TOda_mean.index.diff().dropna()[0].total_seconds() / 3600
    heat_load = building.get_heating_load(
        TOda_nominal=weather_config.TOda_nominal,
        TRoom_nominal=TRoom
    )
    UA = heat_load / (TRoom - weather_config.TOda_nominal)  # in W / K
    THeaThr = get_heating_threshold_temperature_for_building(building)
    Q_demand_building = UA * (THeaThr - TOda_mean) * interval_length  # to Wh
    Q_demand_building.name = "Q_demand_building_static"
    Q_demand_building[Q_demand_building < 0] = 0  # No cooling
    return Q_demand_building


def run_building_simulations(
        save_path: Path,
        n_cpu: int,
        all_retrofit_options: bool = True
):
    variable_names = [
        heat_load_estimation.heater_name,
        heat_load_estimation.outdoor_air_name,
        heat_load_estimation.TZone_name,
    ]
    if all_retrofit_options:
        buildings = building.get_all_tabula_sfh_buildings(
            construction_datas=["standard"], as_dict=False,
            use_led=False, modify_transfer_system=False,
            use_verboseEnergyBalance=False
        )
        buildings = building.generate_buildings_for_all_element_combinations(
            building_configs=buildings
        )
    else:
        buildings = building.get_all_tabula_sfh_buildings()
    # buildings = [buildings[0]]
    weathers = get_all_weather_configs()[:1]
    # weathers = weathers[:2]
    # Batches are used to get a feedback how long everything takes,
    # as 7 days with no feedback is too long.
    weathers_batch = 10  # Takes around 15 hours
    for idx_weather in range(0, len(weathers), weathers_batch):
        to_idx = min(idx_weather + weathers_batch, len(weathers))
        weathers_to_simulate = weathers[idx_weather:to_idx]
        inputs_config = configs.InputsConfig(
            weathers=weathers_to_simulate,
            buildings=buildings,
            users=[configs.inputs.UserProfile(use_stochastic_internal_gains=True)],
            dhw_profiles=[{"profile": "NoDHW"}],
        )
        simulate_all_combinations(
            inputs_config=inputs_config,
            save_path=save_path.joinpath(f"weather_batch_{idx_weather}_{to_idx}"),
            study_path=save_path,
            n_cpu=n_cpu,
            remove_mats=True,
            store_tsd_in_pickle=False,
            variable_names=variable_names,
            model_name="BESRules.InputAnalysis.PartialHeatDemand"
        )


def merge_results_in_one_json(save_path: Path):
    # all_results_json = []
    all_cases = []
    for folder in os.listdir(save_path):
        if not folder.startswith("weather_batch_"):
            continue
        save_path_batch = save_path.joinpath(folder)
        with open(save_path_batch.joinpath("HeatDemandSimulationResults.pickle"), "rb") as file:
            simulated_heat_demands = pickle.load(file)
        for simulated_heat_demand_case in simulated_heat_demands:
            input_config = simulated_heat_demand_case["input_config"]
            df_building_path = Path(simulated_heat_demand_case["df"])
            df_building_path = save_path_batch.joinpath(df_building_path.relative_to(df_building_path.parents[1]))
            # Adjust to new base in case of new device:
            # This was due to an error in simulate combinations
            if df_building_path.suffix == ".parqet":
                try:
                    os.rename(df_building_path, df_building_path.with_suffix(".parquet"))
                except FileNotFoundError:
                    pass
                df_building_path = df_building_path.with_suffix(".parquet")
            all_cases.append({
                "input_config": input_config,
                "df": df_building_path
            })
            # all_results_json.append({
            #    "input_config": input_config.model_dump(),
            #    "df": df_building_path
            # })
    with open(save_path.joinpath("AllHeatDemandSimulationResults.pickle"), "wb") as file:
        pickle.dump(all_cases, file)
    # Json requires 10 times disk usage, this is why pickle is used.
    # import json
    # from bes_rules.configs.study import PathEncoder
    # with open(save_path.joinpath("AllHeatDemandSimulationResults.json"), "w") as file:
    #    json.dump(all_results_json, file, indent=2, cls=PathEncoder)


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()


def calculate_dema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Double Exponential Moving Average
    DEMA = 2 * EMA(price) - EMA(EMA(price))
    """
    ema1 = calculate_ema(data, period)
    ema2 = calculate_ema(ema1, period)
    return (2 * ema1) - ema2


def get_vent_rate(
        TOda: pd.Series,
        df_internal_gains: pd.DataFrame,
        maxUserACH: float = 1.0,
        baseACH: float = 0.2,
        winterReduction: list = None,
        only_daytime: bool = False
):
    """
    Replica of AixLib.Controls.VentilationController.VentilationController
    """
    if winterReduction is None:
        winterReduction = [0.2, 273.15, 283.15]
    DEMA = calculate_dema(TOda, period=4 * 24)
    dTmin = (DEMA - winterReduction[1]) / (winterReduction[2] - winterReduction[1])
    redFac = np.ones(len(df_internal_gains)) * winterReduction[0]
    redFacDyn = np.minimum(dTmin * (1 - winterReduction[0]), 1 - winterReduction[0]) + winterReduction[0]
    redFac[dTmin > 0] = redFacDyn[dTmin > 0]
    if only_daytime:
        mask_daytime = (TOda.index.hour >= 7) & (TOda.index.hour < 22)
        redFac[~mask_daytime] = 0
    userACH = df_internal_gains.loc[:, "relOccupation"] * maxUserACH
    return baseACH + userACH * redFac


def modify_teaser_heat_demand_with_new_vent_rate(
        Q_demand_building: pd.Series,
        TOda: pd.Series,
        building: BuildingConfig,
        new_vent_rate_settings: dict,
        TZone: float
):
    """
    Modifies the given heat demand profiles with
    `new_vent_rate_settings`, with keys: maxUserACH, winterReduction.
    See `get_vent_rate` for infos.
    """
    assert TOda.index.freq == "h", "Frequency must be 1h for this function!"

    df_internal_gains = load_internal_gains()
    df_internal_gains = periodic_append_of_init_period(
        series=df_internal_gains,
        init_period=86400 * 2,
        first_day_of_year=TOda.index[0],
        last_day_of_year=TOda.index[-1]
    )

    ventRate_teaser = get_vent_rate(
        baseACH=building.base_infiltration,
        df_internal_gains=df_internal_gains,
        TOda=TOda
    )
    ventRate_new = get_vent_rate(
        baseACH=building.base_infiltration,
        df_internal_gains=df_internal_gains,
        TOda=TOda,
        **new_vent_rate_settings
    )
    VAir = building.net_leased_area * building.height_of_floors
    dQVen_flow = (ventRate_new - ventRate_teaser) * VAir * 1014.54 * 1.2 * (TOda - TZone) * (1 / 3600)
    Q_demand_building_new = Q_demand_building.copy()
    Q_demand_building_new += dQVen_flow
    Q_demand_building_new.loc[Q_demand_building_new < 0] = 0
    show_plot = False
    if show_plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, sharex=True)
        axes[0].plot(Q_demand_building, label="Q teaser")
        axes[0].plot(Q_demand_building_new, label="Q new")
        axes[1].plot(ventRate_teaser, label="ventRate_teaser")
        axes[1].plot(ventRate_new, label="ventRate_new")
        axes[2].plot(dQVen_flow, label="dQ")
        for ax in axes:
            ax.legend()
        plt.show()
    return Q_demand_building_new


def load_internal_gains():
    internal_gains_profile = DATA_PATH.joinpath("internal_gains", "InternalGains.txt")
    df = pd.read_csv(
        internal_gains_profile,
        skip_blank_lines=True, skiprows=[0, 1], sep="\t",
        index_col=0, header=None
    )
    df.loc[0] = [1, 0.1, 0]
    df.index.name = "Time"
    df.sort_index(inplace=True)
    df = df.rename(columns={1: "relOccupation", 2: "machines", 3: "lights"})
    return df


def analyze_building_capacities():
    from bes_rules.boundary_conditions.building import create_buildings
    buildings_configs = building.get_all_tabula_sfh_buildings(
        construction_datas=["standard"], as_dict=False,
        use_led=False, modify_transfer_system=False,
        use_verboseEnergyBalance=False
    )
    buildings_configs = building.generate_buildings_for_all_element_combinations(
        building_configs=buildings_configs
    )
    idx = 0
    df = pd.DataFrame()
    for t_bt in [1 / 24, 1, 2, 5, 7]:
        for buildings_config in create_buildings(
            name="cEffTest", building_configs=buildings_configs, export=False, t_bt=t_bt
        ):
            df.loc[idx, "name"] = buildings_config.name
            df.loc[idx, "VAir"] = buildings_config.building_parameters.volume_air
            df.loc[idx, "cEffInn"] = buildings_config.building_parameters.CEff / df.loc[idx, "VAir"] / 3600
            df.loc[idx, "cEff"] = buildings_config.building_parameters.CEff / df.loc[idx, "VAir"] / 3600 / 0.74
            df.loc[idx, "t_bt"] = t_bt
            idx += 1
    df.to_excel(RESULTS_FOLDER.joinpath("input_analysis", "c_eff.xlsx"))


def plot_ceff():
    from bes_rules import LATEX_FIGURES_FOLDER
    from bes_rules.configs.plotting import PlotConfig
    from bes_rules.plotting.utils import get_figure_size
    plot_config = PlotConfig.load_default()

    df = pd.read_excel(RESULTS_FOLDER.joinpath("input_analysis", "c_eff.xlsx"), index_col=0)
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=get_figure_size(n_columns=1))
    for t_bt in df["t_bt"].unique():
        df_sub = df.loc[df["t_bt"] == t_bt]
        ax.scatter(np.ones(len(df_sub)) * t_bt, df_sub["cEff"])
    ax.set_ylabel("$c_\mathrm{Eff}$ in Wh/m³K")
    ax.set_xlabel("$t_\mathrm{bt}$ in d")
    fig.tight_layout()
    fig.savefig(LATEX_FIGURES_FOLDER.joinpath("Appendix", "cEffTeaser_t_bt.png"))
    plt.show()


if __name__ == '__main__':
    from bes_rules import RESULTS_FOLDER

    SAVE_PATH = RESULTS_FOLDER.joinpath("input_analysis", "buildings_new")
    # merge_results_in_one_json(save_path=SAVE_PATH)
    run_building_simulations(save_path=SAVE_PATH, n_cpu=1)
    # analyze_building_capacities()
    # plot_ceff()
