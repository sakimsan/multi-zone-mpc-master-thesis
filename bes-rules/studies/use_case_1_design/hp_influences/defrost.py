import pathlib
from typing import List

from bes_rules import configs, RESULTS_FOLDER, N_CPU
from bes_rules import boundary_conditions
from bes_rules.input_variations import run_input_variations

from studies.use_case_1_design import base_design_optimization
from studies.use_case_1_design.hp_influences import plotting


def get_config(model: str, study_name: str, sdf_path: pathlib.Path = None, modifiers: List[list] = None):
    sim_config = base_design_optimization.get_simulation_config(model=model)
    optimization_config = base_design_optimization.get_optimization_config(
        configs.OptimizationVariable(
            name="parameterStudy.TBiv",
            lower_bound=273.15 - 16,
            upper_bound=278.15,
            discrete_steps=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.VPerQFlow",
            lower_bound=5,
            upper_bound=50,
            levels=5
        )
    )
    weathers = boundary_conditions.weather.get_weather_configs_by_names(region_names=["Potsdam"])
    buildings = boundary_conditions.building.get_building_configs_by_name(building_names=["NoRetrofit1983"])
    if sdf_path is not None:
        sdf_modifier = {
            "name": sdf_path.stem.split("_")[-1],
            "modifier": f'filename_sdf="{sdf_path.as_posix()}"'
        }
    else:
        sdf_modifier = None
    if modifiers is None:
        if sdf_modifier:
            modifiers = [sdf_modifier]
        else:
            modifiers = [None]
    else:
        if sdf_modifier:
            for modifier in modifiers:
                modifier.append({
                    "name": sdf_path.stem.split("_")[-1],
                    "modifier": f'filename_sdf="{sdf_path.as_posix()}"'
                })

    inputs_config = configs.InputsConfig(
        weathers=weathers,
        buildings=buildings,
        dhw_profiles=[{"profile": "M"}],
        modifiers=modifiers
    )
    return configs.StudyConfig(
        base_path=RESULTS_FOLDER.joinpath("UseCase_TBivAndV", "influence_frosting"),
        n_cpu=12,
        name=study_name,
        simulation=sim_config,
        optimization=optimization_config,
        inputs=inputs_config,
        test_only=False
    )


def run_with_dynamic_frosting():
    config = get_config(
        model="MonoenergeticOptiHorstDefrost", study_name="dynamic_defrost"
    )
    run_input_variations(config=config, run_inputs_in_parallel=False)


def run_with_map_frosting():
    # TODO: Add Frosting to sdf data later on.
    sdf_path = pathlib.Path(r"E:\02_Paper\01_vclibpy\Results\MEN_MEN_ENTests_frosting.sdf")
    config = get_config(
        model="MonoenergeticOptiHorst", study_name="map",
        sdf_path=sdf_path
    )
    run_input_variations(config=config, run_inputs_in_parallel=False)


def run_with_no_frosting():
    sdf_path = pathlib.Path(r"E:\02_Paper\01_vclibpy\Results\MEN_MEN_ENTests.sdf")
    config = get_config(
        model="MonoenergeticOptiHorst", study_name="no_frost",
        sdf_path=sdf_path
    )
    run_input_variations(config=config, run_inputs_in_parallel=False)


def run_with_static_frosting():
    sdf_path = None  #pathlib.Path(r"E:\02_Paper\01_vclibpy\Results_3d\EN_MEN412_Linear\OptiHorst_R410A.sdf")
    static_frosting_approaches = {
        "Afjei": "AixLib.Fluid.HeatPumps.ModularReversible.RefrigerantCycle.Frosting.FunctionalIcingFactor("
                 "redeclare function icingFactor = "
                 "AixLib.Fluid.HeatPumps.ModularReversible.RefrigerantCycle.Frosting.Functions.wetterAfjei1997)",
        "Li": "BESRules.Components.Frosting.Li",
        "Roccatello": "BESRules.Components.Frosting.RoccatelloCOPCorrection",
        "No": "AixLib.Fluid.HeatPumps.ModularReversible.RefrigerantCycle.Frosting.NoFrosting"
    }
    modifiers = [[{
        "name": key, "modifier": f"redeclare model iceFacModel = {value}"
    }] for key, value in static_frosting_approaches.items()
    ]
    config = get_config(
        model="MonoenergeticOptiHorst", study_name="static_frost",
        sdf_path=sdf_path, modifiers=modifiers
    )
    run_input_variations(config=config, run_inputs_in_parallel=False)


def compare_plots():
    static_name = "influence_frosting/static_frost/DesignOptimizationResults"
    dyn_name = "influence_frosting/dynamic_defrost/DesignOptimizationResults"
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
            "dynamic": {"color": "red"},
            "map": {"color": "blue"},
            "no_frost": {"color": "gray"}
        },
        save_name="influence_frosting/compare_defrost"
    )


if __name__ == '__main__':
    run_with_static_frosting()
