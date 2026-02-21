import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from studies.use_case_1_design import base_design_optimization
from bes_rules import RESULTS_FOLDER
from bes_rules import configs
from bes_rules.input_variations import run_input_variations
from bes_rules.simulation_based_optimization import AgentLibMPC
from studies.use_case_1_design import MPC_UTILS_PATH
from studies.use_case_1_design.hp_influences import compare_onoff


def manipulate_predictions(df):
    df.loc[:, "P_el_demand"] = (
            df.loc[:, "internal_gains_convective"] +
            df.loc[:, "internal_gains_radiative"]
    )
    return df


def run_mpc(
        n_cpu,
        with_dynamic_prices: bool,
        no_minimal_compressor_speed: bool,
        inverter_uses_storage: bool
):
    time_step = 900

    study_name = "".join([
        _get_study_name(no_minimal_compressor_speed),
        #"_cDyn" if use_dynamic_prices else "_cCon",
        "" if inverter_uses_storage else "hydSep_test"
    ])

    mapping_predictions = {
        "electrical.generation.outBusGen.PElePV.value": "P_el_pv_raw",
        "parameterStudy.f_design": "PVDesignSize",
        "outputs.weather.TDryBul": "T_amb",
        "outputs.building.TZone[1]": "T_Air",
        "outputs.building.eneBal[1].traGain.value": "Q_demand",
        "userProfiles.useProBus.absIntGaiRad": "internal_gains_radiative",
        "userProfiles.useProBus.absIntGaiConv": "internal_gains_convective",
        "hydraulic.control.buiAndDHWCtr.TSetBuiSup.TSet": "THeaCur",
    }


    surrogate_builder_kwargs = dict(
        predictive_control_options=dict(
            mpc_module=MPC_UTILS_PATH.joinpath("agent_modules/mpc.json"),
            predictor_module=MPC_UTILS_PATH.joinpath("agent_modules/predictor.json"),
            simulator_module=MPC_UTILS_PATH.joinpath("agent_modules/simulator_fmu.json"),
            mpc_parameters={},
            save_mpc_results=False,
            save_mpc_stats=True,
            model_path=MPC_UTILS_PATH.joinpath("model_no_rom.py")
        ),
        mapping_prediction_defaults={
            "P_el_pv_raw": 0,
            "PVDesignSize": 1,
        },
        manipulate_predictions=manipulate_predictions,
        mapping_predictions=mapping_predictions
    )
    sim_config = base_design_optimization.get_simulation_config(
        model="MonoenergeticVitoCal_MPC",
        n_days=7,
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

    optimization_config = compare_onoff.get_optimization_config(
        compare_to_mpc=True, only_one_hp_size=inverter_uses_storage
    )
    inputs_config = compare_onoff.get_inputs_config(
        no_dhw=True, inverter_uses_storage=inverter_uses_storage,
        no_minimal_compressor_speed=no_minimal_compressor_speed, with_start_losses=False,
        only_inverter=True
    )
    if with_dynamic_prices:
        inputs_config.prices = [{"year": None}, {"year": 2023}, {"year": 2019}, {"year": 2023, "only_wholesale_price": True}]

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
        surrogate_builder_class=AgentLibMPC,
        **surrogate_builder_kwargs
    )
    calculate_with_dynamic_prices(study_name)


def plot_mpc_stats(study_name):
    from agentlib_mpc.utils.analysis import load_mpc, load_mpc_stats
    from agentlib_mpc.utils.plotting.interactive import show_dashboard
    mpc_results_path = RESULTS_FOLDER.joinpath(
        "UseCase_TBivAndV",
        study_name,
        "DesignOptimizationResults",
        "TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_M_0K-Per-IntGai"
    )
    mpc_results = load_mpc(mpc_results_path.joinpath("Design_0_mpc_agent.csv"))
    mpc_stats = load_mpc_stats(mpc_results_path.joinpath("stats_Design_0_mpc_agent.csv"))
    variables_to_plot = [
        # Not ok
        "TBufSet",
        "yValSet",
        "yEleHeaSet",
        "T_TES_1",
        # "T_TES_2",
        # "T_TES_3",
        "T_TES_4",
        # "TTraRet",
        "Qdot_hp",
        "Qdot_hp_max",
        "QTra_flow",
        # "c_feed_in_out",
        "c_grid_out",
        "P_el_feed_into_grid",
        "P_el_feed_from_grid",
        "PEleHeaPum",
        "PEleEleHea",
        # "PEleIntGai_out",
        # "PEleFee",
        # "P_pv",
        # "valve_actual",
        # "Tamb",
        "QTra_flow_slack",
        # "PEleHeaPum_slack",
        # "TTraSup_slack",
        "THeaPumSup_slack",
        # "QHeaPum_flow_slack",
        # "Q_storage_loss"
        # "mTra_flow",
    ]
    show_dashboard(data=mpc_results, stats=mpc_stats, scale="days", variables_to_plot=variables_to_plot)


def debug_casadi_model(design_config_path):
    from bes_rules.simulation_based_optimization.agentlib_mpc.simulate_mpc_model import run as simulate_casadi_and_fmu
    simulate_casadi_and_fmu(
        design_config_path=design_config_path,
        start_time=0,
        stop_time=86400 * 7,
        control_emulator_mapping={
            "hydraulic.control.buiAndDHWCtr.TSetBuiSup.TSet": "TBufSet",
            "outputs.hydraulic.tra.opening[1]": "yValSet",
            "hydraulic.generation.sigBusGen.uEleHea": "yEleHeaSet",
        }
    )


def plot_compare_casadi_fmu(design_config_path):
    mapping = {
        "mTra_flow": "hydraulic.transfer.portTra_in[1].m_flow",
        "TTraSup": "outputs.hydraulic.tra.TSup[1]",
        "TTraRet": "outputs.hydraulic.tra.TRet[1]",
        "QTra_flow": "outputs.building.eneBal[1].traGain.value",
        # "QTra_flow_slack": "outputs.building.eneBal[1].traGain.value",
        "Q_storage_loss": "outputs.hydraulic.dis.QBufLos_flow.value",
        "Qdot_hp": "outputs.hydraulic.gen.QHeaPum_flow.value",
        "PEleHeaPum": "outputs.hydraulic.gen.PEleHeaPum.value",
        "PEleEleHea": "outputs.hydraulic.gen.PEleEleHea.value",
        "P_el_feed_from_grid": "outputs.electrical.dis.PEleLoa.value",
        "T_TES_1": "hydraulic.distribution.stoBuf.layer[1].T",
        "T_TES_2": "hydraulic.distribution.stoBuf.layer[2].T",
        "T_TES_3": "hydraulic.distribution.stoBuf.layer[3].T",
        "T_TES_4": "hydraulic.distribution.stoBuf.layer[4].T",
        "TBufSet": "TBufSet",
        "yValSet": "yValSet",
        "valve_actual": "yValSet",
        "yEleHeaSet": "yEleHeaSet",
        "T_amb": "outputs.weather.TDryBul",
        "T_Air": "outputs.building.TZone[1]",
    }

    def _read_sim_result(path):
        return pd.read_csv(path, header=[0, 1, 2], index_col=0).droplevel(level=2, axis=1).droplevel(level=0, axis=1)

    df_fmu = _read_sim_result(design_config_path.parent.joinpath("Design_0_sim_agent_debug.csv"))
    df_casadi = _read_sim_result(design_config_path.parent.joinpath("Design_0_sim_agent_casadi_debug.csv"))
    n_batch = 5
    casadi_vars = list(mapping.keys())
    fmu_vars = list(mapping.values())
    for i in range(0, len(mapping), n_batch):
        if i + n_batch > len(mapping):
            n_batch = len(mapping) - i
        fig, axes = plt.subplots(n_batch, 1, sharex=True)
        for ax, casadi, fmu in zip(axes.flatten(), casadi_vars[i:i + n_batch], fmu_vars[i:i + n_batch]):
            ax.plot(df_fmu.index, df_fmu.loc[:, fmu], label="FMU", color="blue")
            ax.plot(df_fmu.index, df_casadi.loc[:, casadi], label="Casadi", color="red", linestyle="--")
            ax.set_ylabel(casadi)
        axes[-1].set_xlabel("Time")
    m_flow_var = "mTra_flow"
    T_ret_var = "TTraRet"
    T_supply_var = "TTraSup"
    T_room = "T_Air"
    UA_casadi = df_casadi.loc[:, m_flow_var] * 4184 * np.log(
        (df_casadi.loc[:, T_supply_var] - df_casadi.loc[:, T_room]) /
        (df_casadi.loc[:, T_ret_var] - df_casadi.loc[:, T_room])
    )
    m_flow_var = mapping[m_flow_var]
    T_ret_var = mapping[T_ret_var]
    T_supply_var = mapping[T_supply_var]
    T_room = mapping[T_room]
    UA_fmu = df_fmu.loc[:, m_flow_var] * 4184 * np.log(
        (df_fmu.loc[:, T_supply_var] - df_fmu.loc[:, T_room]) /
        (df_fmu.loc[:, T_ret_var] - df_fmu.loc[:, T_room])
    )
    plt.figure()
    plt.plot(df_fmu.index, UA_casadi, color="red", linestyle="--")
    plt.plot(df_fmu.index, UA_fmu, color="blue", )
    plt.show()


def calculate_with_dynamic_prices(study_name: str):
    from bes_rules.boundary_conditions.prices import calculate_operating_costs_with_dynamic_prices
    study_path = RESULTS_FOLDER.joinpath("UseCase_TBivAndV", study_name)
    study_config = configs.StudyConfig.from_json(study_path.joinpath("study_config.json"))
    calculate_operating_costs_with_dynamic_prices(
        study_config=study_config,
        year=2023
    )


def _get_study_name(no_minimal_compressor_speed: bool):
    return "mpc_perfect" if no_minimal_compressor_speed else "mpc_design"


def get_case_file_names(no_minimal_compressor_speed: bool):
    dhw = "M" if not no_minimal_compressor_speed else "NoDHW"
    _extra = "_perfect" if no_minimal_compressor_speed else ""
    mpc_name = _get_study_name(no_minimal_compressor_speed)

    def _get_path(study, extra_name):
        return RESULTS_FOLDER.joinpath(
            "UseCase_TBivAndV",
            study,
            "DesignOptimizationResults",
            f"TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_{dhw}_0K-Per-IntGai{extra_name}",
            "DesignOptimizationResultsWithDynamicCosts.xlsx"
        )

    cases = {
        "RBCOnOff": _get_path(f"inverter_vs_onoff_mpc{_extra}", "_OnOff"),
        "RBC": _get_path(f"inverter_vs_onoff_mpc{_extra}", "__100" if no_minimal_compressor_speed else "_"),
        "MPCCon": _get_path(f"{mpc_name}_cCon", "_100" if no_minimal_compressor_speed else ""),
        "MPCDyn": _get_path(f"{mpc_name}_cDyn", "_100" if no_minimal_compressor_speed else ""),
    }
    return cases


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    run_mpc(n_cpu=4, with_dynamic_prices=True, inverter_uses_storage=False, no_minimal_compressor_speed=False)

    # plot_mpc_stats(STUDY_NAME)

    #DESIGN_CONFIG_PATH = RESULTS_FOLDER.joinpath(
    #    "UseCase_TBivAndV", STUDY_NAME, "DesignOptimizationResults",
    #    "TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_M_0K-Per-IntGai", "generated_configs_Design_0"
    #)
    # debug_casadi_model(DESIGN_CONFIG_PATH)
    # plot_compare_casadi_fmu(DESIGN_CONFIG_PATH)
