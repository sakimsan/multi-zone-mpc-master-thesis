from pathlib import Path
import logging

import pandas as pd

from bes_rules import configs, STARTUP_BESMOD_MOS, BESRULES_PACKAGE_MO, N_CPU
from bes_rules.input_variations import run_input_variations
from bes_rules.boundary_conditions import weather, building
from bes_rules.simulation_based_optimization.utils import constraints
from studies.use_case_2_pv.renewable_energies.plots import BASE_PATH


def get_simulation_config(model_name, test: bool = False):
    return configs.SimulationConfig(
        startup_mos=STARTUP_BESMOD_MOS,
        model_name=model_name,
        sim_setup=dict(stop_time=86400 * 365, output_interval=600),
        result_names=[],
        packages=[BESRULES_PACKAGE_MO],
        recalculate=False,
        show_window=True,
        debug=False,
        extract_results_during_optimization=True,
        convert_to_hdf_and_delete_mat=not test,
        dymola_api_kwargs={"time_delay_between_starts": 5}
    )


def get_config(study_name, model_name, test=False, with_elec_profile=True, n_cpu: int = None):
    logging.basicConfig(level="INFO")

    sim_config = get_simulation_config(model_name=model_name, test=test)
    cases_to_simulate = pd.read_excel(Path(__file__).absolute().parent.joinpath("SimulationCases.xlsx"), sheet_name="SimulationCases")
    all_weather_configs = weather.get_all_weather_configs()
    tabula_buildings = building.get_all_tabula_sfh_buildings(as_dict=True)
    weathers, buildings, modifiers = [], [], []
    import numpy as np
    for idx, row in cases_to_simulate.iterrows():
        for weather_config in all_weather_configs:
            if weather_config.get_name(pretty_print=True) == row["weather"]:
                weathers.append(weather_config)
                break
        else:
            raise KeyError(f'{row["weather"]} not found in data folder')

        buildings.append(tabula_buildings[row["building"].replace("-", "_")])
        direction, use_two_side = {
            "West": (90, "false"),
            "South": (0, "false"),
            "East": (-90, "false"),
            "North": (180, "false"),
            "EastWest": (90, "true"),
            "SouthNorth": (0, "true"),
        }[row["case"]]
        azi_rad = direction / 180 * np.pi
        inc_ele_pro = "true" if with_elec_profile else "false"
        modifiers.append({"name": row["case"], "modifier": f'useTwoRoo={use_two_side}, '
                                                           f'aziMaiRoo={azi_rad}, '
                                                           f'incElePro={inc_ele_pro}'})
    if n_cpu is None:
        n_cpu = N_CPU
    config = configs.StudyConfig(
        base_path=BASE_PATH,
        n_cpu=n_cpu,
        name=study_name,
        simulation=sim_config,
        test_only=test,
        optimization=configs.OptimizationConfig(
            framework="doe",
            method="ffd",
            constraints=[constraints.BivalenceTemperatureGreaterNominalOutdoorAirTemperature()],
            variables=[]
        ),
        inputs=configs.InputsConfig(
            weathers=weathers,
            buildings=buildings,
            dhw_profiles=[{"profile": "M"}],
            modifiers=modifiers,
            full_factorial=False,
            users=[{"use_stochastic_internal_gains": True}]
            # users={},  # Not supported yet
        )  # Default
    )
    return config


def run_design_optimization(config):
    is_control_optimization = config.simulation.model_name.endswith("PythonAPICtrlOpt")

    config.optimization.variables = [
        configs.OptimizationVariable(
            name="parameterStudy.TBiv",
            lower_bound=273.15 - 16,
            upper_bound=278.15,
            discrete_steps=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.VPerQFlow",
            lower_bound=5,
            upper_bound=150,
            levels=8
        ),
        configs.OptimizationVariable(
            name="parameterStudy.f_design",
            lower_bound=0 if not is_control_optimization else 1,
            upper_bound=1,
            levels=2 if not is_control_optimization else 1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.ShareOfPEleNominal",
            lower_bound=0.1 if is_control_optimization else 1,
            upper_bound=1,
            levels=4 if is_control_optimization else 1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.DHWOverheatTemp",
            lower_bound=333.15,
            upper_bound=333.15,
            levels=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.BufOverheatdT",
            lower_bound=15 if is_control_optimization else 15,
            upper_bound=15,
            levels=1 if is_control_optimization else 1
        )
    ]
    run_input_variations(config=config)


def run_control_optimization(test=False):
    config = get_config(study_name="BESCtrlOpt",
                        model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPICtrlOpt",
                        test=test)
    config.inputs.weathers = weather.get_weather_configs_by_names(region_names=["Bad Marienberg"])
    config.optimization.variables = [
        configs.OptimizationVariable(
            name="parameterStudy.TBiv",
            lower_bound=273.15 - 2,
            upper_bound=273.15 - 2,
            levels=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.VPerQFlow",
            lower_bound=23.5,
            upper_bound=23.5,
            levels=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.ShareOfPEleNominal",
            lower_bound=1,
            upper_bound=1,
            levels=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.f_design",
            lower_bound=1,
            upper_bound=1,
            levels=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.DHWOverheatTemp",
            lower_bound=323.15,
            upper_bound=343.15,
            levels=21
        ),
        configs.OptimizationVariable(
            name="parameterStudy.BufOverheatdT",
            lower_bound=0,
            upper_bound=20,
            levels=21
        )
    ]
    run_input_variations(config=config)


if __name__ == '__main__':
    run_design_optimization(get_config(
        study_name="BESCtrl",
        model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPICtrlOpt",
        test=False,
        with_elec_profile=True
    ))
    run_design_optimization(get_config(
        study_name="BESNoCtrl",
        model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPINoSupCtrl",
        test=False,
        with_elec_profile=True
    ))
    run_design_optimization(get_config(
        study_name="BESCtrlNoElec",
        model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPICtrlOpt",
        test=False,
        with_elec_profile=False
    ))
    run_design_optimization(get_config(
        study_name="BESNoCtrlNoElec",
        model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPINoSupCtrl",
        test=False,
        with_elec_profile=False
    ))
