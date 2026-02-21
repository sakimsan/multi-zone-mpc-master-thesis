import pickle
from pathlib import Path

import pandas as pd

from bes_rules import RESULTS_FOLDER
from bes_rules.configs.plotting import get_energy_balance_variables, PlotConfig
from bes_rules.input_analysis.heat_load_estimation import VARIABLE_NAMES
from bes_rules.input_analysis.simulate_combinations import simulate_all_combinations
from bes_rules import boundary_conditions
from bes_rules import configs


def compare_retrofit_options_in_potsdam(study_path: Path):
    buildings = boundary_conditions.building.get_all_tabula_sfh_buildings(
        construction_datas=["standard"], as_dict=False, use_led=True
    )
    buildings = boundary_conditions.building.generate_buildings_for_all_element_combinations(
        building_configs=buildings,
        elements=[
            "outer_walls",
            "windows",
            "rooftops",
        ],
        retrofit_choices=None
    )
    weathers = boundary_conditions.weather.get_all_weather_configs()
    weathers = boundary_conditions.weather.get_weather_configs_by_names(region_names=["Potsdam"])
    users = [
        #configs.inputs.UserProfile(room_set_temperature=292.15, night_set_back=0),
        #configs.inputs.UserProfile(room_set_temperature=292.15, night_set_back=3),
        configs.inputs.UserProfile(room_set_temperature=293.15, night_set_back=0),
        #configs.inputs.UserProfile(room_set_temperature=294.15, night_set_back=0),
        #configs.inputs.UserProfile(room_set_temperature=295.15, night_set_back=0),
        #configs.inputs.UserProfile(room_set_temperature=293.15, night_set_back=3),
        #configs.inputs.UserProfile(room_set_temperature=293.15, night_set_back=4),
    ]
    inputs_config = configs.InputsConfig(
        weathers=weathers,
        buildings=buildings,
        users=users,
        dhw_profiles=[{"profile": "NoDHW"}],
    )
    simulate_all_combinations(
        study_path=study_path,
        n_cpu=6,
        inputs_config=inputs_config,
        remove_mats=True,
        variable_names=VARIABLE_NAMES + list(get_energy_balance_variables().keys())
    )


def analyze_results(study_path: Path, plot_config: PlotConfig):
    with open(study_path.joinpath("ExtractedSimulationResults.pickle"), "rb") as file:
        results = pickle.load(file)
    # Filter building names:
    variables = get_energy_balance_variables()

    results_filtered = pd.DataFrame(
        columns=[
                    "name",
                    "A in m2",
                    "QHeaLoa_flow in kW",
                ] + [
                    plot_config.get_label_and_unit(variable)
                    for variable in variables
                ]
    )
    for idx, result in enumerate(results):
        input_config, df = result["input_config"], result["df"]
        for variable in variables:
            var_name = plot_config.get_label_and_unit(variable)
            results_filtered.loc[idx, var_name] = plot_config.scale(variable, df.iloc[-1][variable])
        results_filtered.loc[idx, "name"] = input_config.get_name()
        results_filtered.loc[idx, "A in m2"] = input_config.building.net_leased_area
        results_filtered.loc[idx, "QHeaLoa_flow in kW"] = df.iloc[0]["building.QRec_flow_nominal[1]"] / 1000
    results_filtered.to_excel(study_path.joinpath("results.xlsx"))


if __name__ == '__main__':
    PLOT_CONFIG = PlotConfig.load_default()
    STUDY_PATH = RESULTS_FOLDER.joinpath("compare_tabula_retrofits_led")
    compare_retrofit_options_in_potsdam(STUDY_PATH)
    analyze_results(STUDY_PATH, plot_config=PLOT_CONFIG)
