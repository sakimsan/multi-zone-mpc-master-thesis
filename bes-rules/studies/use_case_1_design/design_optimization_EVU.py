import logging
import socket

from bes_rules import configs, STARTUP_BESMOD_MOS, BESRULES_PACKAGE_MO
from bes_rules.input_variations import InputVariations
from bes_rules.boundary_conditions import weather, building
from bes_rules.simulation_based_optimization.utils import constraints


def get_config(test_only=True):
    n_days = 1 if test_only else 365

    if socket.gethostname() == "ESIM64":
        startup_mos_path = STARTUP_BESMOD_MOS
        base_path = r"D:\fwu-nmu\%s01_design_optimization" % ("00_TEST_" if test_only else "")
    else:
        startup_mos_path = r"D:\fwu-nmu\design_rule_syntheziser\design-rule-synthesizer\startup.mos"
        base_path = r"D:\fwu-nmu\00_temp\%s01_design_optimization" % ("00_TEST_" if test_only else "")

    sim_config = configs.SimulationConfig(
        startup_mos=startup_mos_path,
        model_name="BESRules.DesignOptimizationBANina.PythonAPI_GridInteraction_SG",
        sim_setup=dict(stop_time=86400 * n_days, output_interval=600),
        result_names=[
            "hydraulic.generation.heatPumpParameters.scalingFactor",
            "hydraulic.generation.heatPumpParameters.QSec_flow_nominal",
            "hydraulic.generation.heatPumpParameters.QPri_flow_nominal",
            "hydraulic.control.HP_EVU_Sperre.u2",
            "hydraulic.control.switchHP_SummerOrWinter.u2"
        ],
        packages=[BESRULES_PACKAGE_MO],
        type="Dymola",

        recalculate=False,
        show_window=True,
        debug=False,
        extract_results_during_optimization=True,
        convert_to_hdf_and_delete_mat=True
    )

    optimization_config = configs.OptimizationConfig(
        framework="doe",
        method="ffd",
        constraints=[constraints.BivalenceTemperatureGreaterNominalOutdoorAirTemperature()],
        variables=[
            configs.OptimizationVariable(
                name="parameterStudy.TBiv",
                lower_bound=273.15 - 16,
                upper_bound=278.15,
                levels=20 if not test_only else 15
            ),
            configs.OptimizationVariable(
                name="parameterStudy.VPerQFlow",
                lower_bound=12,
                upper_bound=100,
                levels=4
            )
        ],
    )

    weathers = weather.get_weather_configs_by_names(region_names=["Fichtelberg", "Bad Marienberg", "Bremerhaven"])
    buildings = building.get_building_configs_by_name(building_names=["Retrofit1918", "NoRetrofit1983"])

    if test_only:
        weathers = weathers[0]
        buildings = buildings[0]

    inputs_config = configs.InputsConfig(
        weathers=weathers,
        buildings=buildings,
        dhw_profiles=[{"profile": "M"}],
        profiles=[
            {"profile": "EVU_Sperre_EON"},
            {"profile": "EVU_Sperre_None"},
            {"profile": "EVU_Sperre_Enbw"},
            {"profile": "EVU_Sperre_Westnetz"}]
    )

    config = configs.StudyConfig(
        base_path=base_path,
        n_cpu=2 if test_only else 11,
        name="DesOptGridInteractionNewMap6hSum",
        simulation=sim_config,
        optimization=optimization_config,
        inputs=inputs_config  # Default
    )
    return config


def run_optimization(test_only=True):
    logging.basicConfig(level="INFO")
    config = get_config(test_only=test_only)
    DESOPT = InputVariations(config=config)
    DESOPT.run()


def run_optimization_different_heating_rod_control(test_only=True):
    logging.basicConfig(level="INFO")
    config = get_config(test_only=test_only)
    config.simulation.model_name = "BESRules.DesignOptimizationBANina.PythonAPI_GridInteraction_SG_HRC"
    config.name = "DesOptGridInteractionHRControl6h"
    configs.inputs.evu_profiles = [{"profile": "EVU_Sperre_EON"}, {"profile": "EVU_Sperre_None"}]
    DESOPT = InputVariations(config=config)
    DESOPT.run()


def run_optimization_no_summer_mode(test_only=True):
    logging.basicConfig(level="INFO")
    config = get_config(test_only=test_only)
    config.simulation.model_name = "BESRules.DesignOptimizationBANina.PythonAPI_GridInteraction_noSC"
    config.name = "DesOptGridInteractionNoSummer"
    configs.inputs.evu_profiles = [{"profile": "EVU_Sperre_EON"}, {"profile": "EVU_Sperre_None"}]
    DESOPT = InputVariations(config=config)
    DESOPT.run()


if __name__ == '__main__':
    run_optimization(test_only=False)
