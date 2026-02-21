import json
import pathlib

from studies.use_case_1_design import base_design_optimization
from bes_rules import configs
from bes_rules.input_variations import run_input_variations
from bes_rules.input_analysis.input_analysis import load_input_configs, get_inputs_config


def get_inputs_config_to_simulate(
        cases_to_simulate: list,
        input_analysis_path: pathlib.Path
):
    input_configs = load_input_configs(save_path=input_analysis_path)
    inputs_to_simulate = []
    for input_config in input_configs:
        if input_config.get_name() in cases_to_simulate:
            input_config.building.modify_transfer_system = True  # Was not set in simulate combinations
            inputs_to_simulate.append(input_config)
    return get_inputs_config(inputs_to_simulate)


def run(
        study_name: str,
        cases_to_simulate: list,
        input_analysis_path: pathlib.Path,
        base_path: pathlib.Path,
        n_cpu: int = 1,
        time_step: int = 900,
        surrogate_builder_kwargs: dict = {},
        surrogate_builder_class=None,
        model: str = "MonoenergeticVitoCal",
):
    sim_config = base_design_optimization.get_simulation_config(
        model=model,
        time_step=time_step,
        convert_to_hdf_and_delete_mat=True,
        recalculate=False,
        equidistant_output=True
    )
    optimization_config = base_design_optimization.get_optimization_config(
        configs.OptimizationVariable(
            name="parameterStudy.TBiv",
            lower_bound=273.15 - 16,
            upper_bound=278.15,
            discrete_steps=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.VPerQFlow",
            lower_bound=12,
            upper_bound=35,
            levels=3
        )
    )

    inputs_config = get_inputs_config_to_simulate(
        cases_to_simulate=cases_to_simulate, input_analysis_path=input_analysis_path
    )
    config = configs.StudyConfig(
        base_path=base_path,
        n_cpu=n_cpu,
        name=study_name,
        simulation=sim_config,
        optimization=optimization_config,
        inputs=inputs_config,
        test_only=False
    )
    run_input_variations(
        config=config, run_inputs_in_parallel=False,
        surrogate_builder_class=surrogate_builder_class,
        **surrogate_builder_kwargs
    )
    add_cases_already_simulated(simulated_cases=cases_to_simulate, input_analysis_path=input_analysis_path)


def add_cases_already_simulated(simulated_cases: list, input_analysis_path: pathlib.Path):
    try:
        with open(input_analysis_path.joinpath("cases_already_simulated.json"), "r") as file:
            cases_already_simulated = json.load(file)
    except FileNotFoundError:
        cases_already_simulated = []
    with open(input_analysis_path.joinpath("cases_already_simulated.json"), "w") as file:
        json.dump(cases_already_simulated + list(simulated_cases), file, indent=2)
