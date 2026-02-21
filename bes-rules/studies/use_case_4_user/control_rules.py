from pathlib import Path
import logging

from bes_rules import configs, STARTUP_BESMOD_MOS
from bes_rules.input_variations import run_input_variations
from bes_rules.boundary_conditions import building


MODELS = {
    "Radiator": {"model": "BESMod.Utilities.TimeConstantEstimation.Radiator.SmartThermostat",
                 "x_variable": "timeInt", "bounds": [100, 1000, 10]},
    "UFH": {"model": "BESMod.Utilities.TimeConstantEstimation.UnderfloorHeating.SmartThermostat",
            "x_variable": "timeInt", "bounds": [100, 3600, 10]}
}


def run_optimization(typ: str, test_only=False):

    sim_config = configs.SimulationConfig(
        startup_mos=STARTUP_BESMOD_MOS,
        model_name=MODELS[typ]["model"],
        sim_setup=dict(stop_time=864000, output_interval=600),
        result_names=[],
        packages=[],
        type="Dymola",
        recalculate=False,
        show_window=True,
        debug=False,
        extract_results_during_optimization=True,
        convert_to_hdf_and_delete_mat=False
    )
    from bes_rules import DATA_PATH
    weather_cfg = configs.WeatherConfig(
        dat_file=DATA_PATH.joinpath("weather", "Potsdam", "TRY_523845130645", "TRY2015_523845130645_Jahr.dat"),
        TOda_nominal=-12.6
    )
    import numpy as np
    TOda_min = np.ceil(weather_cfg.TOda_nominal - 273.15)
    ## Optimization
    optimization_config = configs.OptimizationConfig(
        framework="doe",
        method="ffd",
        constraints=[],
        variables=[
            configs.OptimizationVariable(
                name=MODELS[typ]["x_variable"],
                lower_bound=MODELS[typ]["bounds"][0],
                upper_bound=MODELS[typ]["bounds"][1],
                levels=MODELS[typ]["bounds"][2]
            ),
            configs.OptimizationVariable(
                name="TOda_start",
                lower_bound=TOda_min + 273.15,
                upper_bound=15 + 273.15,
                levels=int((15 - TOda_min)/2) + 1
            )
        ],
    )

    buildings = building.get_building_configs_by_name(building_names=["Retrofit1918"], modify_transfer_system=False)

    inputs_config = configs.InputsConfig(
        weathers=[weather_cfg],
        buildings=buildings,
        dhw_profiles=[{"profile": "M"}],
        users=[{"use_stochastic_internal_gains": False}]
    )

    config = configs.StudyConfig(
        base_path=r"D:\fwu\07_Results",
        n_cpu=4,
        name="ControlEstimationRadiator",
        simulation=sim_config,
        optimization=optimization_config,
        inputs=inputs_config,
        test_only=test_only,
        objectives=[],
        time_series_dependent_objectives=[]
    )
    run_input_variations(config=config)



if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_optimization(test_only=False, typ="Radiator")
