import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bes_rules import RESULTS_FOLDER, configs
from bes_rules.input_analysis import oed, input_analysis, heat_pump_system
from bes_rules.plotting import utils
from bes_rules.rule_extraction.innovization import run_brute_force_innovization, analyze_convergence
from studies.use_case_1_design import simulate_oed_cases, no_dhw
from studies.use_case_1_design.no_dhw import innovization_simplified

PATH_INPUT_ANALYSIS = RESULTS_FOLDER.joinpath("input_analysis", "dhw")
PATH_OED = PATH_INPUT_ANALYSIS.joinpath("OED")
PATH_SIMULATIONS = RESULTS_FOLDER.joinpath("UseCase_TBivAndV", "dhw")
FEATURES = [
    "Q_demand_total",
    "GTZ_Ti_HT",
    "THyd_nominal",
    "dhw_share"
]


def innovization_simulations(oed_iter: int, with_no_dhw_results: bool = False):
    study_name = f"OED_iter_{oed_iter}"

    config = configs.StudyConfig.from_json(PATH_SIMULATIONS.joinpath(study_name, "study_config.json"))
    if with_no_dhw_results:
        from studies.use_case_1_design.no_dhw import PATH_SIMULATIONS as PATH_SIMULATIONS_NO_DHW
        config_no_dhw = configs.StudyConfig.from_json(PATH_SIMULATIONS_NO_DHW.joinpath(study_name, "study_config.json"))
        configs_to_include = [config, config_no_dhw]
        save_path = PATH_SIMULATIONS.joinpath(study_name, "manual_innovization_with_no_dhw")
    else:
        configs_to_include = config
        save_path = PATH_SIMULATIONS.joinpath(study_name, "manual_innovization")

    run_brute_force_innovization(
        **no_dhw.get_innovization_default_kwargs(),
        create_surrogate_plots=True,
        design_variables={
            "parameterStudy.TBiv": np.linspace(-16, 5, 100) + 273.15,
            "parameterStudy.VPerQFlow": np.linspace(12, 12, 100)
        },
        configs=configs_to_include,
        save_path=save_path
    )


def perform_input_analysis():
    input_analysis.analyze_all(
        heat_pump=heat_pump_system.VitoCal250,
        save_path=PATH_INPUT_ANALYSIS,
        use_mp=True,
        dhw_profile="M",
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


def find_minimum_required_oed_examples(calc_oed: bool = True, calc_input_analysis: bool = True, calc_rules: bool = True):
    no_dhw.find_minimum_required_oed_examples(
        calc_oed=calc_oed,
        calc_input_analysis=calc_input_analysis,
        calc_rules=calc_rules,
        dhw_profile="M",
        features=FEATURES,
        path_oed=PATH_OED,
        path_input_analysis=PATH_INPUT_ANALYSIS
    )


def run_simulations_from_no_dhw(oed_iter: int):
    from studies.use_case_1_design.no_dhw import PATH_OED as PATH_OED_NO_DHW
    study_name = f"OED_iter_{oed_iter}"
    with open(PATH_OED_NO_DHW.joinpath("verification", f"iter_{oed_iter}" + ".json"), "r") as file:
        cases_to_simulate = json.load(file)
    cases_to_simulate = [c.replace("NoDHW", "M") for c in cases_to_simulate]

    simulate_oed_cases.run(
        study_name,
        base_path=PATH_SIMULATIONS,
        n_cpu=12,
        cases_to_simulate=cases_to_simulate,
        input_analysis_path=PATH_INPUT_ANALYSIS
    )


def plot_SCOP_over_TBiv_for_cases(
        static_model: str,
        show=False,
):
    cases = {
        "BESMod": PATH_SIMULATIONS.joinpath("OED_1_3exp", "DesignOptimizationResults"),
        "Static": PATH_INPUT_ANALYSIS.joinpath(static_model),
        "VDI 4645": PATH_INPUT_ANALYSIS.joinpath(static_model + "_VDI4645")
    }
    no_dhw.plot_SCOP_over_TBiv_for_cases(
        static_model=static_model, show=show,
        save_path=PATH_INPUT_ANALYSIS.joinpath("compare_to_VDI"),
        cases=cases
    )


if __name__ == '__main__':
    import faulthandler
    #faulthandler.enable()  # start @ the beginning

    logging.basicConfig(level="INFO")
    #perform_input_analysis()
    #oed.create_all_feature_plots(PATH_OED)
    # no_dhw.create_features(path_input_analysis=PATH_INPUT_ANALYSIS)
    #plot_SCOP_over_TBiv_for_cases("StaticDemand")
    #find_minimum_required_oed_examples(False, False, True)
    #no_dhw.plot_oed_convergence(
    #    full_innovization_path=PATH_INPUT_ANALYSIS.joinpath("manual_innovization", "TEASER"),
    #    path_oed=PATH_OED,
    #)
    #run_simulations_from_no_dhw(oed_iter=3)
    #no_dhw.run_simulations(oed_iter=3, path_oed=PATH_OED, path_input_analysis=PATH_INPUT_ANALYSIS)
    innovization_simulations(oed_iter=3, with_no_dhw_results=True)
    #no_dhw.innovization_simplified(static_model="TEASER", path_input_analysis=PATH_INPUT_ANALYSIS)
