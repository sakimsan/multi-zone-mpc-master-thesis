import json

from bes_rules import configs, RESULTS_FOLDER
from bes_rules.input_variations import run_input_variations
from studies.use_case_1_design import base_design_optimization
from studies.use_case_1_design.hp_influences import plotting
from studies.use_case_1_design.no_dhw import PATH_OED
from studies.use_case_1_design.dhw import PATH_INPUT_ANALYSIS
from studies.use_case_1_design.simulate_oed_cases import get_inputs_config_to_simulate


def run_all():
    cases = {
        "optihorst_2d": "MonoenergeticOptiHorst2DVCLibPy",
        "optihorst_4d": "MonoenergeticOptiHorst"
    }
    for study_name, model_name in cases.items():
        run(model_name=model_name, study_name=study_name)


def run(model_name: str, study_name: str, n_cpu: int = 12, ):
    sim_config = base_design_optimization.get_simulation_config(
        model=model_name,
        recalculate=False,
        convert_to_hdf_and_delete_mat=False
    )
    optimization_config = base_design_optimization.get_optimization_config(
        configs.OptimizationVariable(
            name="parameterStudy.TBiv",
            lower_bound=273.15 - 16,
            upper_bound=278.15,
            levels=24
        ),
        configs.OptimizationVariable(
            name="parameterStudy.VPerQFlow",
            lower_bound=5,
            upper_bound=5,
            levels=1
        )
    )
    with open(PATH_OED.joinpath("verification", f"corner_points" + ".json"), "r") as file:
        cases_to_simulate = json.load(file)
    cases_to_simulate = [c.replace("NoDHW", "M") for c in cases_to_simulate]
    cases_to_simulate = [cases_to_simulate[0], cases_to_simulate[2], cases_to_simulate[3]]

    inputs_config = get_inputs_config_to_simulate(
        cases_to_simulate=cases_to_simulate,
        input_analysis_path=PATH_INPUT_ANALYSIS
    )
    config = configs.StudyConfig(
        base_path=RESULTS_FOLDER.joinpath("UseCase_TBivAndV"),
        n_cpu=n_cpu,
        name=study_name,
        simulation=sim_config,
        optimization=optimization_config,
        inputs=inputs_config,
        test_only=False
    )
    run_input_variations(config=config, run_inputs_in_parallel=False)


def compare_plots():
    plotting.compare_plots(
        y_variables=[
            "costs_total",
            "SCOP_Sys",
            "outputs.hydraulic.gen.heaPum.numSwi",
            #"QHeaPum_flow_A2W35",
            "HP_Coverage",
            "outputs.THeaPumSinMean",
            #"outputs.building.dTComHea[1]"
        ],
        x_variable="parameterStudy.TBiv",
        studies={
            "optihorst_2d": {"color": "blue"},
            "optihorst_4d": {"color": "red"},
        },
        save_name="compare_optihorst"
    )


if __name__ == '__main__':
    #run_all()
    compare_plots()
