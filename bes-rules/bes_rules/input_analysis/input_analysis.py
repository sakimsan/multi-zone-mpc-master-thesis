import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Type, Tuple, Union
import glob
import random

import numpy as np
import pandas as pd
from ebcpy import TimeSeriesData

from bes_rules.configs import InputConfig
from bes_rules import configs
from bes_rules.utils import multiprocessing_ as bes_rules_mp

from bes_rules.input_analysis import utils
from bes_rules.input_analysis import pv
from bes_rules.input_analysis import dhw
from bes_rules.input_analysis import heat_pump_system
from bes_rules.input_analysis import building_weather
from bes_rules.objectives import get_all_objectives
from bes_rules.boundary_conditions.building import get_retrofit_temperatures
from bes_rules import RESULTS_FOLDER

logger = logging.getLogger(__name__)


def get_all_pv_directions(with_pv: bool):
    if with_pv:
        return [[], ["South"], ["West"], ["North"], ["East", "West"], ["South", "North"]]
    return [[]]


def analyze_all(
        save_path,
        heat_pump: Type[heat_pump_system.PartialHeatPump],
        interval: str = "1h",
        use_mp: bool = True,
        transfer_retrofit_to_maximum_heat_pump_temperature: bool = False,
        with_pv: bool = False,
        with_users: bool = False,
        dhw_profile: str = "M",
        use_hybrid: bool = False,
        save_plots: bool = False,
        input_config_names: list = None,
        new_vent_rate_settings: dict = None,
        raise_errors: bool = False
):
    os.makedirs(save_path, exist_ok=True)

    if with_pv:
        with open(RESULTS_FOLDER.joinpath("input_analysis", "pv", "PVSimulations.json"), "r") as file:
            pv_cases = json.load(file)
    else:
        pv_cases = {}
    all_pv_directions = get_all_pv_directions(with_pv=with_pv)

    simulated_heat_demands = load_simulated_heat_demands()

    kwargs_mp = []
    manipulated_input_configs = []
    for input_config, df_path in simulated_heat_demands:
        if input_config_names is not None and input_config.get_name() not in input_config_names:
            continue
        input_config = manipulate_input_config(
            input_config=input_config,
            transfer_retrofit_to_maximum_heat_pump_temperature=transfer_retrofit_to_maximum_heat_pump_temperature,
            THeaPumMax=heat_pump.get_THeaPumMax(input_config.weather.TOda_nominal),
            dhw_profile=dhw_profile
        )
        manipulated_input_configs.append(input_config)
        kwargs_mp.append(
            {
                "df_pv_path": pv_cases.get(input_config.weather.get_name(), None),
                "input_config": input_config,
                "df_building_path": df_path,
                "interval": interval,
                "all_pv_directions": all_pv_directions,
                "use_hybrid": use_hybrid,
                "save_path": save_path,
                "save_plots": save_plots,
                "heat_pump": heat_pump,
                "new_vent_rate_settings": new_vent_rate_settings,
                "raise_errors": raise_errors
            }
        )
    with open(save_path.joinpath("manipulated_input_configs.pickle"), "wb") as file:
        pickle.dump(manipulated_input_configs, file)

    print("Going to analyze %s cases" % len(kwargs_mp))
    bes_rules_mp.execute_function_in_parallel(
        func=analyze_one_case_catch_error,
        func_kwargs=kwargs_mp,
        use_mp=use_mp,
        notifier=print,
        # n_cpu=5,
        percentage_when_to_message=5
    )
    merge_mp_logs(save_path=save_path, name_pattern="log_file_worker_*.log", merged_name="log_file.log")
    merge_mp_logs(save_path=save_path, name_pattern="errors_worker_*.json", merged_name="errors.json")
    if not os.path.exists(save_path.joinpath("errors.json")):
        return
    df = pd.read_json(save_path.joinpath("errors.json"), lines=True)
    for idx, row in df.iterrows():
        try:
            for col, value in json.loads(df.loc[idx, "error"]).items():
                df.loc[idx, col] = value
        except json.JSONDecodeError:
            df.loc[idx, "other_error"] = df.loc[idx, "error"]
    df = df.drop(columns=["error"])
    df.to_csv(save_path.joinpath("errors.csv"))
    os.remove(save_path.joinpath("errors.json"))


def create_study_config_for_analysis(save_path: Path, name: str, n_to_load: int = None):
    input_configs = load_input_configs(save_path=save_path)
    if n_to_load is not None:
        input_configs = random.sample(input_configs, n_to_load)
    return create_study_config_for_input_configs(
        save_path=save_path, name=name, input_configs=input_configs
    )


def create_study_config_for_given_input_config_names(save_path: Path, name: str, input_config_names: list):
    input_configs = load_input_configs(save_path=save_path)
    input_configs = [c for c in input_configs if c.get_name() in input_config_names]
    return create_study_config_for_input_configs(
        save_path=save_path, name=name, input_configs=input_configs
    )


def create_study_config_for_input_configs(save_path: Path, input_configs: list, name: str):
    from bes_rules.simulation_based_optimization.utils import constraints
    return configs.StudyConfig(
        name=name,
        base_path=save_path,
        n_cpu=1,
        simulation=configs.SimulationConfig(type="Static"),
        optimization=configs.OptimizationConfig(
            framework="doe",
            method="ffd",
            constraints=[constraints.BivalenceTemperatureGreaterNominalOutdoorAirTemperature()],
            variables=[
                configs.OptimizationVariable(
                    name="parameterStudy.TBiv",
                    lower_bound=273.15 - 18,
                    upper_bound=283.15,
                    discrete_steps=0.5
                )
            ]),
        inputs=get_inputs_config(input_configs),
        objectives=get_all_objectives()
    )


def get_inputs_config(
        input_configs: List[InputConfig]
):
    weathers = [input_config.weather for input_config in input_configs]
    buildings = [input_config.building for input_config in input_configs]
    users = [input_config.user for input_config in input_configs]
    dhw_profiles = [input_config.dhw_profile for input_config in input_configs]
    return configs.InputsConfig(
        full_factorial=True,
        weathers=weathers,
        buildings=buildings,
        users=users,
        dhw_profiles=dhw_profiles,
    )


def manipulate_input_config(
        input_config: InputConfig,
        transfer_retrofit_to_maximum_heat_pump_temperature: bool,
        THeaPumMax: float,
        dhw_profile: str
):
    if transfer_retrofit_to_maximum_heat_pump_temperature:
        # Either 15 or 10 K spread
        input_config.building.retrofit_transfer_system_to_at_least = (
            THeaPumMax, 15 if THeaPumMax > 273.15 + 55 else 10
        )
    input_config.dhw_profile.profile = dhw_profile
    return input_config


def merge_mp_logs(save_path: Path, name_pattern: str, merged_name: str):
    pattern = os.path.join(save_path, name_pattern)
    log_files = glob.glob(pattern)

    if not log_files:
        return
    log_files.sort()
    # Combine files
    with open(save_path.joinpath(merged_name), 'w') as outfile:
        for log_file in log_files:
            with open(log_file, 'r') as infile:
                outfile.write(infile.read())
            try:
                os.remove(log_file)
            except PermissionError:
                pass  # Sometimes the process is still alive, e.g. in no-mp case


def merge_excel_results(save_path: Path, interval: str = "1h", with_pv: bool = False):
    input_configs = load_input_configs(save_path=save_path)
    all_pv_directions = get_all_pv_directions(with_pv=with_pv)
    Q_demand_cases = {
        "StaticDemand": [],
        "TEASER": []
    }
    for input_config in input_configs:
        for name in Q_demand_cases.keys():
            for pv_directions in all_pv_directions:
                df = pd.read_excel(
                    save_path.joinpath(name, get_file_name(input_config, interval, pv_directions) + ".xlsx"),
                    index_col=0
                )
                df.loc[:, "pv_directions"] = get_pv_name(pv_directions)
                for input_name, input_value in input_config.get_name_parts().items():
                    df[input_name] = input_value

                Q_demand_cases[name].append(df)
    for name, all_dfs in Q_demand_cases.items():
        pd.concat(all_dfs).to_excel(save_path.joinpath(name, f"all_results_{interval}.xlsx"))


def load_simulated_heat_demands(
        save_path: Path = None
) -> List[Tuple[InputConfig, Path]]:
    if save_path is None:
        save_path = RESULTS_FOLDER.joinpath("input_analysis")
    save_path_pickle = save_path.joinpath("buildings_and_weathers", "AllHeatDemandSimulationResults.pickle")
    with open(save_path_pickle, "rb") as file:
        simulated_heat_demands = pickle.load(file)
    # model_dump to include new fields added after saving the pickle.
    return [
        (
            InputConfig(**simulated_heat_demand["input_config"].model_dump()),
            Path(simulated_heat_demand["df"])
        )
        for simulated_heat_demand in simulated_heat_demands
    ]


def load_input_configs(save_path: Path):
    with open(save_path.joinpath("manipulated_input_configs.pickle"), "rb") as file:
        input_configs = pickle.load(file)
    # model_dump to include new fields added after saving the pickle.
    return [
        InputConfig(**input_config.model_dump())
        for input_config in input_configs
    ]


def analyze_one_case_catch_error(**kwargs):
    raise_errors = kwargs.pop("raise_errors", False)
    try:
        analyze_one_case(**kwargs)
    except Exception as err:
        logger.error("Error in analysis of input_config %s: %s",
                     kwargs["input_config"].get_name(), err)
        error_infos = {"input_config": kwargs["input_config"].get_name(), "error": str(err)}
        save_path = kwargs["save_path"]
        with open(save_path.joinpath(f"errors_worker_{bes_rules_mp.get_worker_idx()}.json"), "a+") as file:
            file.write(f"{json.dumps(error_infos)}\n")
        if raise_errors:
            raise err


def analyze_one_case(
        input_config: InputConfig,
        df_building_path: Path,
        df_pv_path: Path,
        interval: str,
        all_pv_directions: List[List[str]],
        heat_pump: Type[heat_pump_system.PartialHeatPump],
        save_path: Path,
        dhw_storage_design: str = "part_storage",
        use_hybrid: bool = False,
        save_plots: bool = False,
        new_vent_rate_settings: dict = None
):
    logging.basicConfig(
        filename=save_path.joinpath(f"log_file_worker_{bes_rules_mp.get_worker_idx()}.log"),
        level=logging.WARNING,
        format='%(message)s'
    )

    df_building = TimeSeriesData(df_building_path)
    simulation_interval, simulation_interval_std = df_building.frequency
    if simulation_interval_std != 0:
        raise IndexError("Demand simulation is not equidistant")
    df_building = df_building.to_df()

    if df_pv_path is None:
        df_pv = pd.DataFrame(index=df_building.index)
    else:
        df_pv = TimeSeriesData(df_pv_path).to_df()

    dhw_daily_map = {  # Values from BESMod, hold at 60 °C
        "M": 100.679,
        "L": 200.756,
        "S": 36.17,
        "NoDHW": 0
    }
    # TODO: Maybe select profile for given daily demand instead and add circulation losses factor in BESMod?
    dhw_daily_per_person = dhw_daily_map[input_config.dhw_profile.profile] / input_config.building.number_of_occupants
    dhw_storage_size, QHeaPumDHW_flow_nominal = dhw.dhw_design_EN_15450(
        number_of_occupants=input_config.building.number_of_occupants,
        dhw_storage_design=dhw_storage_design,
        dhw_daily_per_person=dhw_daily_per_person
    )

    Q_demand_building, Q_demand_DHW, df_toda_mean, P_household = utils.get_sampled_demands(
        df_building=df_building,
        interval=interval,
        input_config=input_config,
        simulation_interval=simulation_interval,
        dhw_daily_per_person=dhw_daily_per_person
    )
    if new_vent_rate_settings is not None and interval == "1h":
        building_weather.modify_teaser_heat_demand_with_new_vent_rate(
            Q_demand_building=Q_demand_building,
            TOda=df_toda_mean,
            TZone=input_config.user.room_set_temperature,
            building=input_config.building,
            new_vent_rate_settings=new_vent_rate_settings
        )

    Q_demand_building_static = building_weather.estimate_thermal_demand(
        TOda_mean=df_toda_mean,
        building=input_config.building,
        weather_config=input_config.weather,
        TRoom=input_config.user.room_set_temperature,
    )
    Q_demand_cases = {
        "StaticDemand": Q_demand_building_static,
        "TEASER": Q_demand_building
    }
    # Further design
    THyd_nominal, dTHyd_nominal, THydSupOld_design, _, _, QDem_flow_nominal = get_retrofit_temperatures(
        building_config=input_config.building,
        TOda_nominal=input_config.weather.TOda_nominal,
        TRoom_nominal=input_config.user.room_set_temperature,
        retrofit_transfer_system_to_at_least=input_config.building.retrofit_transfer_system_to_at_least
    )

    for name, Q_demand_bui in Q_demand_cases.items():
        oversize_requirement = (
                (Q_demand_bui + Q_demand_DHW).max() /
                (QDem_flow_nominal + QHeaPumDHW_flow_nominal)
        )
        if oversize_requirement > 1:
            logger.warning(
                "Demand is greater than nominal design, would lead to discomfort. "
                "Oversize-requirement: %s percent. In case: %s, calculation method: %s",
                round(oversize_requirement * 100, 1), input_config.get_name(), name
            )

        simplified_design_optimization_given_demands(
            input_config=input_config,
            interval=interval,
            all_pv_directions=all_pv_directions,
            save_path=save_path.joinpath(name),
            use_hybrid=use_hybrid,
            df_toda_mean=df_toda_mean,
            Q_demand_building=Q_demand_bui,
            Q_demand_DHW=Q_demand_DHW,
            P_household=P_household,
            dhw_storage_size=dhw_storage_size,
            QHeaPumDHW_flow_nominal=QHeaPumDHW_flow_nominal,
            THyd_nominal=THyd_nominal,
            dTHyd_nominal=dTHyd_nominal,
            heat_pump=heat_pump,
            df_pv=df_pv,
            save_plots=save_plots
        )


def simplified_design_optimization_given_demands(
        input_config: InputConfig,
        df_pv: pd.DataFrame,
        df_toda_mean: pd.Series,
        Q_demand_building: pd.Series,
        Q_demand_DHW: pd.Series,
        P_household: pd.Series,
        dhw_storage_size: float,
        QHeaPumDHW_flow_nominal: float,
        THyd_nominal: float,
        dTHyd_nominal: float,
        heat_pump: Type[heat_pump_system.PartialHeatPump],
        interval: str,
        all_pv_directions: List[List[str]],
        save_path: Path,
        use_hybrid: bool = False,
        save_plots: bool = False
):
    # Dynamic simulations are also only run up to 5 °C
    TBivs = np.arange(input_config.weather.TOda_nominal, 273.15 + 10, 0.5)

    objectives = get_all_objectives(as_dict=True)

    save_path_vdi = save_path.with_name(save_path.name + "_VDI4645")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_vdi, exist_ok=True)

    for pv_directions in all_pv_directions:
        df_pv_sum = pv.get_sampled_pv_time_series(
            interval=interval,
            df_pv=df_pv,
            pv_directions=pv_directions
        )
        results = []
        results_vdi = []
        P_pv, power_pv_nominal = pv.estimate_pv_generation(
            df_pv_sum=df_pv_sum,
            building=input_config.building,
            n_roofs=len(pv_directions)
        )
        case_name = get_file_name(input_config, interval, pv_directions)
        if save_plots:
            save_path_plot = save_path.joinpath(case_name)
            os.makedirs(save_path_plot, exist_ok=True)
        else:
            save_path_plot = None
        for TBiv in TBivs:
            single_result, vdi_result = simplified_simulation_single_design(
                input_config=input_config,
                TBiv=TBiv,
                use_hybrid=use_hybrid,
                P_pv=P_pv,
                df_toda_mean=df_toda_mean,
                Q_demand_building=Q_demand_building,
                Q_demand_DHW=Q_demand_DHW,
                P_household=P_household,
                QHeaPumDHW_flow_nominal=QHeaPumDHW_flow_nominal,
                THyd_nominal=THyd_nominal,
                dTHyd_nominal=dTHyd_nominal,
                heat_pump=heat_pump,
                power_pv_nominal=power_pv_nominal,
                objectives=objectives,
                save_path_plot=save_path_plot
            )
            results.append({
                "parameterStudy.TBiv": TBiv,
                **single_result
            })
            results_vdi.append({
                "parameterStudy.TBiv": TBiv,
                **vdi_result
            })
        for _result, _path in zip([results, results_vdi], [save_path, save_path_vdi]):
            df = pd.DataFrame(_result)
            df.loc[:, "THyd_nominal"] = THyd_nominal
            df.loc[:, objectives["Annuity"].mapping.dhw_storage_size] = dhw_storage_size / 1000
            for obj in objectives.values():
                df = obj.calc(df=df, input_config=input_config)
            df.to_excel(_path.joinpath(case_name + ".xlsx"))


def get_file_name(input_config, interval, pv_directions):
    return input_config.get_name() + f"_{get_pv_name(pv_directions)}_{interval}"


def get_pv_name(pv_directions):
    if pv_directions:
        return "".join(pv_directions)
    return "noPV"


def simplified_simulation_single_design(
        input_config: InputConfig,
        P_pv: pd.Series,
        df_toda_mean: pd.Series,
        Q_demand_building: pd.Series,
        Q_demand_DHW: pd.Series,
        P_household: pd.Series,
        TBiv: float,
        QHeaPumDHW_flow_nominal: float,
        THyd_nominal: float,
        dTHyd_nominal: float,
        heat_pump: Type[heat_pump_system.PartialHeatPump],
        power_pv_nominal: float,
        objectives: dict,
        use_hybrid: bool = False,
        save_path_plot: Path = None
):
    if use_hybrid:
        TCutOff = TBiv
    else:
        TCutOff = - np.inf
    PEle_bui, PEle_dhw, results_hps, results_vdi = heat_pump_system.estimate_heat_pump_system_demands(
        Q_demand_DHW=Q_demand_DHW,
        Q_demand_building=Q_demand_building,
        TOda_mean=df_toda_mean,
        input_config=input_config,
        TBiv=TBiv,
        heat_pump=heat_pump,
        TCutOff=TCutOff,
        THyd_nominal=THyd_nominal,
        dTHyd_nominal=dTHyd_nominal,
        objectives=objectives,
        use_hybrid=use_hybrid,
        save_path_plot=save_path_plot,
        QHeaPumDHW_flow_nominal=QHeaPumDHW_flow_nominal,
        with_vdi4645=True
    )

    # Curtail PV which is bigger than demand of other producers (pro)
    P_pv_for_pro = P_pv.copy()
    mask_less_pro_demand_than_pv = P_household <= P_pv
    P_pv_for_pro[mask_less_pro_demand_than_pv] = P_household[mask_less_pro_demand_than_pv]
    # Curtail PV which is bigger than demand of dhw
    P_pv_without_pro_for_dhw = P_pv - P_pv_for_pro
    mask_less_dhw_demand_than_pv = PEle_dhw <= P_pv_without_pro_for_dhw
    P_pv_without_pro_for_dhw[mask_less_dhw_demand_than_pv] = PEle_dhw[mask_less_dhw_demand_than_pv]
    # Curtail PV which is bigger than demand of building
    P_pv_without_pro_and_dhw_for_bui = P_pv_for_pro - P_pv_without_pro_for_dhw
    mask_less_bui_demand_than_pv = PEle_bui <= P_pv_without_pro_and_dhw_for_bui
    P_pv_without_pro_and_dhw_for_bui[mask_less_bui_demand_than_pv] = PEle_bui[mask_less_bui_demand_than_pv]

    P_pv_grid_feed_in = P_pv - P_pv_for_pro - P_pv_without_pro_for_dhw - P_pv_without_pro_and_dhw_for_bui

    W_pv_self_usage = P_pv_for_pro.sum() + P_pv_without_pro_for_dhw.sum() + P_pv_without_pro_and_dhw_for_bui.sum()
    W_pv_self_usage_2 = P_pv.sum() - P_pv_grid_feed_in.sum()
    if not np.isclose(W_pv_self_usage, W_pv_self_usage_2, atol=1):
        raise ValueError("Check why those are not the same!")

    mapping_other = objectives["Miscellaneous"].mapping
    mapping_scop = objectives["SCOP"].mapping
    mapping_annuity = objectives["Annuity"].mapping

    WEle_demand_dhw = np.sum(PEle_dhw) * 3600  # In J
    WEle_demand_bui = np.sum(PEle_bui) * 3600  # In J
    WEle_demand_house = np.sum(P_household) * 3600  # In J
    if PEle_bui.min() < 0 or PEle_dhw.min() < 0:
        raise ValueError("Electrical demands are smaller than zero")
    if np.isnan(WEle_demand_bui) or np.isnan(WEle_demand_dhw):
        raise ValueError("Still NANs in data")

    WEle_demand = WEle_demand_bui + WEle_demand_dhw + WEle_demand_house
    W_pv_for_pro = P_pv_for_pro.sum() * 3600
    W_pv_for_dhw = P_pv_without_pro_for_dhw.sum() * 3600
    W_pv_for_bui = P_pv_without_pro_and_dhw_for_bui.sum() * 3600
    Q_demand_building_total = Q_demand_building.sum() * 3600
    Q_demand_DHW_total = Q_demand_DHW.sum() * 3600

    base_results = {
        mapping_annuity.electric_energy_demand: WEle_demand,
        mapping_scop.building_heat_supplied: Q_demand_building_total,  # In J
        "roof_area": input_config.building.building_parameters.roof_area,
        mapping_scop.dhw_heat_supplied: Q_demand_DHW_total,  # In J
    }
    pv_results = {
        **base_results,
        **results_hps,
        mapping_other.electric_energy_produced_by_pv: P_pv.sum() * 3600,  # In J
        mapping_annuity.power_pv_nominal: power_pv_nominal,
        mapping_annuity.electric_energy_feed_in: P_pv_grid_feed_in.sum() * 3600,  # In J
        "WEle_demand_dhw": WEle_demand_dhw,
        "WEle_demand_bui": WEle_demand_bui,
        "WEle_demand_house": WEle_demand_house,
        "W_pv_for_pro": W_pv_for_pro,
        "W_pv_for_dhw": W_pv_for_dhw,
        "W_pv_for_bui": W_pv_for_bui,
        "solar_share_dhw": W_pv_for_dhw / WEle_demand_dhw if WEle_demand_dhw > 0 else 0,
        "solar_share_bui": W_pv_for_bui / WEle_demand_bui,
        "solar_share_house": W_pv_for_pro / WEle_demand_house,
        "SCOP_bui": Q_demand_building_total / WEle_demand_bui,
        "SCOP_dhw": Q_demand_DHW_total / WEle_demand_dhw if WEle_demand_dhw > 0 else None,
    }
    vdi_results = {
        **base_results,
        **results_vdi,
        mapping_other.electric_energy_produced_by_pv: 0,  # In J
        mapping_annuity.power_pv_nominal: 0,
        mapping_annuity.electric_energy_feed_in: 0,  # In J
    }
    return pv_results, vdi_results


if __name__ == '__main__':
    # merge_excel_results(save_path=RESULTS_FOLDER.joinpath("input_analysis"))
    # raise Exception
    analyze_all(
        heat_pump=heat_pump_system.VitoCal250,
        save_path=RESULTS_FOLDER.joinpath("input_analysis", "no_dhw2"),
        use_mp=True,
        dhw_profile="NoDHW",
        with_pv=False,
        save_plots=False,
        raise_errors=True,
        transfer_retrofit_to_maximum_heat_pump_temperature=True,
        # new_vent_rate_settings={"only_daytime": True, "winterReduction": [0.2, 273.15, 283.15]},
        input_config_names=[
            'TRY2015_506745079707_Somm_B1859_standard_o0w2r0g0_SingleDwelling_NoDHW_0K-Per-IntGai',
        ]
    )
