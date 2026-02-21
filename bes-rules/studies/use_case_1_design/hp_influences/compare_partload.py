import pathlib

from bes_rules import configs, RESULTS_FOLDER, N_CPU
from bes_rules import boundary_conditions
from bes_rules.input_variations import run_input_variations
from studies.use_case_1_design import base_design_optimization
from studies.use_case_1_design.hp_influences import plotting


def get_config(model: str, study_name: str, sdf_path: pathlib.Path = None):
    sim_config = base_design_optimization.get_simulation_config(model=model)
    optimization_config = base_design_optimization.get_optimization_config(
        configs.OptimizationVariable(
            name="parameterStudy.TBiv",
            lower_bound=273.15 - 16,
            upper_bound=278.15,
            levels=24
        ),
        configs.OptimizationVariable(
            name="parameterStudy.VPerQFlow",
            lower_bound=12,
            upper_bound=12,
            levels=1
        )
    )
    weathers = boundary_conditions.weather.get_weather_configs_by_names(region_names=["Potsdam"])
    buildings = boundary_conditions.building.get_building_configs_by_name(
        building_names=["NoRetrofit1983"],
        modify_transfer_system=True
    )
    if sdf_path is None:
        modifiers = []
    else:
        modifiers = [{
            "name": sdf_path.stem.split("_")[-1],
            "modifier": f'filename_sdf="{sdf_path.as_posix()}"'
        }]
    inputs_config = configs.InputsConfig(
        weathers=weathers,
        buildings=buildings,
        dhw_profiles=[{"profile": "M"}],
        modifiers=modifiers
    )
    return configs.StudyConfig(
        base_path=RESULTS_FOLDER.joinpath("UseCase_TBivAndV"),
        n_cpu=12,
        name=study_name,
        simulation=sim_config,
        optimization=optimization_config,
        inputs=inputs_config,
        test_only=False
    )


def run_with_partload():
    partload_path = RESULTS_FOLDER.joinpath("vitocal_map", "ParameterCombination_2")
    config = get_config(
        model="MonoenergeticVitoCalPartLoad", study_name="partloadN_frosting",
        sdf_path=partload_path.joinpath("map_with_frosting.sdf")
    )
    run_input_variations(config=config, run_inputs_in_parallel=False)
    config = get_config(
        model="MonoenergeticVitoCalPartLoad", study_name="partloadN",
        sdf_path=partload_path.joinpath("Standard_Propane.sdf")
    )
    run_input_variations(config=config, run_inputs_in_parallel=False)


def run_without_partload():
    config = get_config(model="MonoenergeticVitoCal", study_name="no_partload")
    run_input_variations(config=config, run_inputs_in_parallel=False)


def compare_plots():
    plotting.compare_plots(
        y_variables=["costs_total", "SCOP_Sys", "outputs.hydraulic.gen.heaPum.numSwi", "QHeaPum_flow_A2W35"],
        x_variable="parameterStudy.TBiv",
        studies={
            "no_partload": {"color": "red"},
            "partload": {"color": "blue"},
            "partload_frosting": {"color": "gray"}
        },
        save_name="comparePartLoad"
    )


if __name__ == '__main__':
    # run_without_partload()
    run_with_partload()
    compare_plots()
