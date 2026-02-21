import json
import logging
from pathlib import Path

import logging
from pathlib import Path

import pandas as pd

from bes_rules import configs
from bes_rules.boundary_conditions import building
from bes_rules.boundary_conditions.weather import get_all_weather_configs
from bes_rules.configs import BuildingConfig
from bes_rules.input_analysis.simulate_combinations import simulate_all_combinations

logger = logging.getLogger(__name__)


radTil = "radTil"
radTil_value = "radTil_value"


def estimate_pv_generation(building: BuildingConfig, df_pv_sum: pd.DataFrame, n_roofs: int):
    if n_roofs > 2:
        raise ValueError("More than two directions for pv not tested")
    logger.debug("Roof area (%s) to net leased area ratio: %s",
                 building.building_parameters.roof_area,
                 building.building_parameters.roof_area / building.net_leased_area
                 )
    A_pv = (
            n_roofs / 2 *  # two sides, if only one roof is used it is half the total area
            building.building_parameters.roof_area
    ) * 1  # 1 for 100 percent roof-side area usage
    P_mp0 = 285  # From record W/m2
    A_mod = 1.000 * 1.670  # From record
    power_pv_nominal = A_pv / A_mod * P_mp0  # In W
    return A_pv * df_pv_sum[radTil].values, power_pv_nominal


def load_and_filter_results(
        save_path: Path,
        interval: str,
        years_to_skip: list,
        filter_2045: bool = False
):
    df_results = pd.read_excel(
        save_path.joinpath(interval, f"Results_{interval}.xlsx"),
        index_col=0
    )

    # Filter results:
    for _year in years_to_skip:
        df_results = df_results[df_results.apply(
            lambda row: not row["building"].startswith(_year), axis=1)
        ]
    if filter_2045:
        df_results = df_results[df_results.apply(
            lambda row: "2045" not in row["weather"], axis=1)
        ]
    df_results = df_results.loc[df_results.loc[:, "case"] != "NorthSouth"]
    df_results = df_results.loc[df_results.loc[:, "case"] != "WestEast"]
    df_results.to_excel(save_path.joinpath(interval, f"Results_{interval}_filtered.xlsx"))

    return df_results


def run_pv_simulations(
        save_path: Path,
        n_cpu: int
):
    variable_names = [
        "electrical.generation.ARooSid"
    ]
    for direction_idx in range(1, 5):
        variable_names.append(f"electrical.outBusElect.gen.PElePVDir[{direction_idx}].value")
        variable_names.append(f"electrical.outBusElect.gen.PElePVDir[{direction_idx}].integral")
    buildings = building.get_all_tabula_sfh_buildings(
        construction_datas=["standard"], as_dict=False,
        use_led=False, modify_transfer_system=False, use_verboseEnergyBalance=False
    )
    buildings = [buildings[0]]  # Any building is enough
    weathers = get_all_weather_configs()
    inputs_config = configs.InputsConfig(
        weathers=weathers,
        buildings=buildings,
        users=[configs.inputs.UserProfile(use_stochastic_internal_gains=True)],
        dhw_profiles=[{"profile": "NoDHW"}],
    )

    simulate_all_combinations(
        inputs_config=inputs_config,
        save_path=save_path,
        study_path=save_path,
        n_cpu=n_cpu,
        remove_mats=True,
        store_tsd_in_pickle=False,
        variable_names=variable_names,
        model_name="BESRules.InputAnalysis.PartialHeatDemandAndPV",
        study_name="PVSimulationResults"
    )
    import pickle
    with open(save_path.joinpath("PVSimulationResults.pickle"), "rb") as file:
        pv_cases = pickle.load(file)
    pv_cases = {pv["input_config"].weather.get_name(): pv["df"].as_posix() for pv in pv_cases}
    with open(save_path.joinpath("PVSimulations.json"), "w") as file:
        json.dump(pv_cases, file, indent=2)


def get_sampled_pv_time_series(
        interval: str,
        df_pv: pd.DataFrame,
        pv_directions: list
):
    directions = {
        "West": 1,
        "South": 2,
        "East": 3,
        "North": 4
    }
    df_pv_new = pd.DataFrame(index=df_pv.index)
    df_pv_new.loc[:, radTil] = 0
    df_pv_new.loc[:, radTil_value] = 0

    if not pv_directions:
        return df_pv_new.resample(interval).sum()  # Avoids expensive groupby call later on

    for direction in pv_directions:
        direction_idx = directions[direction]
        df_pv_new.loc[:, radTil] += (
                df_pv.loc[:, f"electrical.outBusElect.gen.PElePVDir[{direction_idx}].integral"] /
                3600 /  # TODO: Is a factor for simulation_interval needed here
                df_pv.loc[:, "electrical.generation.ARooSid"]
        )
        df_pv_new.loc[:, radTil_value] += (
                df_pv.loc[:, f"electrical.outBusElect.gen.PElePVDir[{direction_idx}].value"] /
                df_pv.loc[:, "electrical.generation.ARooSid"]
        )

    def _get_last(x):
        return x.iloc[-1]

    df_pv_integral_over_interval = df_pv_new.groupby(pd.Grouper(freq=interval)).apply(_get_last)

    for col in df_pv_integral_over_interval.columns:
        df_pv_integral_over_interval.loc[:, col] = df_pv_integral_over_interval.loc[:, col].diff().fillna(
            df_pv_integral_over_interval.loc[:, col]  # Set first value to the first value without diff
        )
    return df_pv_integral_over_interval  # In Wh
    # Required?
    #df_pv_max_in_interval = df_pv.resample(interval).max()
    #return df_pv_integral_over_interval, df_pv_max_in_interval


if __name__ == '__main__':
    from bes_rules import RESULTS_FOLDER
    run_pv_simulations(
        save_path=RESULTS_FOLDER.joinpath("input_analysis", "pv"),
        n_cpu=10
    )
