import logging

from bes_rules import configs, STARTUP_BESMOD_MOS, BESRULES_PACKAGE_MO
from bes_rules.input_variations import InputVariations
from bes_rules.boundary_conditions import building, weather
from bes_rules import simulation_based_optimization
from bes_rules.simulation_based_optimization.utils import constraints


def run_optimization(test_only=False):
    sim_config = configs.SimulationConfig(
        startup_mos=STARTUP_BESMOD_MOS,
        model_name="BESRules.DesignOptimization.MonoenergeticVitoCal",
        sim_setup=dict(stop_time=86400 * 365, output_interval=600),
        result_names=[],
        packages=[BESRULES_PACKAGE_MO],
        type="Dymola",
        recalculate=True,
        show_window=True,
        debug=False,
        extract_results_during_optimization=True,
        convert_to_hdf_and_delete_mat=True,
        equidistant_output=True
    )

    ## Optimization
    optimization_config = configs.OptimizationConfig(
        framework="doe",
        method="ffd",
        constraints=[constraints.BivalenceTemperatureGreaterNominalOutdoorAirTemperature()],
        variables=[
            configs.OptimizationVariable(
                name="parameterStudy.TBiv",
                lower_bound=273.15 - 16,
                upper_bound=278.15,
                levels=10
            ),
            configs.OptimizationVariable(
                name="parameterStudy.VPerQFlow",
                lower_bound=12,
                upper_bound=12,
                levels=1
            )
        ],
    )

    weathers = weather.get_weather_configs_by_names(["Potsdam"])
    buildings = building.get_building_configs_by_name(["Retrofit1918"])

    inputs_config = configs.InputsConfig(
        weathers=weathers,
        buildings=buildings,
        dhw_profiles=[{"profile": "M"}],
    )
    config = configs.StudyConfig(
        base_path=r"D:\00_temp\01_design_optimization",
        n_cpu=5,
        name="TBivOnlyDetailed",
        simulation=sim_config,
        optimization=optimization_config,
        inputs=inputs_config,
        test_only=test_only,
    )
    DESOPT = InputVariations(
        config=config,
        surrogate_builder_class=simulation_based_optimization.BESMod
    )
    DESOPT.run()


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_optimization(test_only=False)
