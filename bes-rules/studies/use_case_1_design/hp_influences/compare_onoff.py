import logging

import pandas as pd
import numpy as np

from studies.use_case_1_design import base_design_optimization
from bes_rules import configs, RESULTS_FOLDER, N_CPU
from bes_rules import boundary_conditions
from bes_rules.input_variations import run_input_variations
from bes_rules.configs.inputs import custom_modifiers


def get_optimization_config(compare_to_mpc: bool, only_one_hp_size: bool = False):
    if compare_to_mpc:
        return base_design_optimization.get_optimization_config(
            configs.OptimizationVariable(
                name="parameterStudy.TBiv",
                lower_bound=273.15 - 6 if only_one_hp_size else 273.15 - 12,
                upper_bound=273.15 - 6 if only_one_hp_size else 273.15,
                levels=1 if only_one_hp_size else 3
            ),
            configs.OptimizationVariable(
                name="parameterStudy.VPerQFlow",
                lower_bound=12,
                upper_bound=58,
                levels=3
            )
        )
    return base_design_optimization.get_optimization_config(
        configs.OptimizationVariable(
            name="parameterStudy.TBiv",
            lower_bound=273.15 - 12,
            upper_bound=278.15,
            discrete_steps=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.VPerQFlow",
            lower_bound=12,
            upper_bound=35,
            levels=2
        )
    )


def get_inputs_config(
        inverter_uses_storage: bool,
        no_minimal_compressor_speed: bool,
        with_start_losses: bool,
        no_dhw: bool,
        only_inverter: bool = False,
):
    weathers = boundary_conditions.weather.get_weather_configs_by_names(region_names=["Potsdam"])
    buildings = boundary_conditions.building.get_building_configs_by_name(
        building_names=["NoRetrofit1983"],
        modify_transfer_system=True
    )
    if inverter_uses_storage:
        inverter_modifier = [custom_modifiers.NoModifier()]
    else:
        inverter_modifier = [custom_modifiers.HydraulicSeperatorModifier()]
    if no_minimal_compressor_speed:
        inverter_modifier.append(custom_modifiers.NoMinimalCompressorSpeed())

    if with_start_losses:
        modifiers = [
            [custom_modifiers.OnOffControlModifier(), custom_modifiers.StartLossModifier()],
            inverter_modifier + [custom_modifiers.StartLossModifier()]
        ]
    else:
        modifiers = [
            [custom_modifiers.OnOffControlModifier()],
            inverter_modifier
        ]

    return configs.InputsConfig(
        weathers=weathers,
        buildings=buildings,
        users=[configs.inputs.users.UserProfile(use_stochastic_internal_gains=True)],
        dhw_profiles=[{"profile": "NoDHW" if no_dhw else "M"}],
        modifiers=[modifiers[1]] if only_inverter else modifiers
    )


def run(
        study_name: str = "inverter_vs_onoff_hydSep",
        n_cpu: int = 1,
        time_step: int = 600,
        surrogate_builder_kwargs: dict = {},
        surrogate_builder_class=None,
        model: str = "MonoenergeticVitoCal",
        with_start_losses: bool = False,
        inverter_uses_storage: bool = False,
        compare_to_mpc: bool = False,
        no_minimal_compressor_speed: bool = False
):
    sim_config = base_design_optimization.get_simulation_config(
        model=model,
        time_step=time_step,
        convert_to_hdf_and_delete_mat=False,
        recalculate=False,
        equidistant_output=True,
        variables_to_save=dict(
            states=False,
            derivatives=False,
            inputs=True,
            outputs=True,
            auxiliaries=True,
        )
    )
    optimization_config = get_optimization_config(compare_to_mpc)
    inputs_config = get_inputs_config(
        no_dhw=compare_to_mpc and no_minimal_compressor_speed, inverter_uses_storage=inverter_uses_storage,
        no_minimal_compressor_speed=no_minimal_compressor_speed, with_start_losses=with_start_losses
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
    run_input_variations(
        config=config, run_inputs_in_parallel=False,
        surrogate_builder_class=surrogate_builder_class,
        **surrogate_builder_kwargs
    )


def run_mpc():
    variables = {
        "input_names": [
            "TDHWSet",
            "TBufSet",
            "actExtBufCtrl",
            "actExtDHWCtrl"
        ],
        "state_names": [
            "outputs.hydraulic.tra.opening[1]",
            "hydraulic.generation.eleHea.Q_flow_nominal"
            "scalingFactor",
            "QPriAtTOdaNom_flow_nominal",
            "hydraulic.generation.m_flow_nominal[1]",
        ],
        "output_names": [
            "hydraulic.control.sigBusDistr.TStoBufTopMea",
            "hydraulic.control.sigBusDistr.TStoDHWTopMea",
            "outputs.building.TZone[1]"
        ]
    }

    mapping_predictions = {
        "electrical.generation.outBusGen.PElePV.value": "P_el_pv_raw",
        "parameterStudy.f_design": "PVDesignSize",
        "outputs.weather.TDryBul": "T_Air",
        "hydraulic.control.buiAndDHWCtr.TSetBuiSup.TSet": "THeaCur",
        "outputs.DHW.Q_flow.value": "Q_DHW_Dem",
        "outputs.building.eneBal[1].traGain.value": "Q_Hou_Dem",
        "userProfiles.useProBus.absIntGaiRad": "internal_gains_radiative",
        "userProfiles.useProBus.absIntGaiConv": "internal_gains_convective"
    }

    def manipulate_predictions(df: pd.DataFrame):
        df.loc[:, "P_El_Dem"] = (
                df.loc[:, "internal_gains_convective"] +
                df.loc[:, "internal_gains_radiative"]
        )
        return df

    from bes_rules.simulation_based_optimization.milp import MILPBasedOptimizer, run_milp_model
    from bes_rules.boundary_conditions.prices import load_dynamic_electricity_prices
    time_step = 900
    run(
        model="MonoenergeticVitoCal_MPC",
        time_step=time_step,
        surrogate_builder_class=MILPBasedOptimizer,
        surrogate_builder_kwargs=dict(
            predictive_control_function=run_milp_model,
            predictive_control_options={
                "with_dhw": False,
                "control_horizon": 4,
                "minimal_part_load_heat_pump": 1,
                "closed_loop": True
            },
            variables=variables,
            manipulate_predictions=manipulate_predictions,
            mapping_predictions=mapping_predictions,
            mapping_prediction_defaults={
                "P_el_pv_raw": 0,
                "PVDesignSize": 1,
            },
            c_grid=load_dynamic_electricity_prices(year=2023, time_step=time_step, init_period=86400 * 2),
        ),
        study_name="on_off_milp"
    )


def calculate_with_dynamic_prices(study_name: str):
    from bes_rules.boundary_conditions.prices import calculate_operating_costs_with_dynamic_prices
    study_path = RESULTS_FOLDER.joinpath("UseCase_TBivAndV", study_name)
    study_config = configs.StudyConfig.from_json(study_path.joinpath("study_config.json"))
    calculate_operating_costs_with_dynamic_prices(
        study_config=study_config,
        year=2023
    )


if __name__ == '__main__':
    from studies.use_case_1_design.plotting import plot_onoff_inverter, plot_required_invest_difference_inverter_on_off
    logging.basicConfig(level="INFO")
    # STUDY_NAME = "inverter_vs_onoff_mpc_perfect"
    # run(STUDY_NAME, n_cpu=12, with_start_losses=False, compare_to_mpc=True, inverter_uses_storage=True, no_mininmal_compressor_speed=True)
    # calculate_with_dynamic_prices(STUDY_NAME)
    STUDY_NAME = "inverter_vs_onoff_mpc"
    run(STUDY_NAME, n_cpu=12, with_start_losses=False, compare_to_mpc=True, inverter_uses_storage=True,
        no_minimal_compressor_speed=False)
    calculate_with_dynamic_prices(STUDY_NAME)
    # run(STUDY_NAME, n_cpu=12, with_start_losses=True)
    # run_mpc()
    # plot_onoff_inverter(study_name=STUDY_NAME)
    # plot_required_invest_difference_inverter_on_off(study_name="inverter_vs_onoff_startLoss")
    # plot_required_invest_difference_inverter_on_off(study_name="inverter_vs_onoff_hydSepNew")
