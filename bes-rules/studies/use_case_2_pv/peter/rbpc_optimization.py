import pickle
from pathlib import Path
from bes_rules import configs
from bes_rules.input_variations import run_input_variations
from bes_rules.simulation_based_optimization import AgentLibMPC
from bes_rules.simulation_based_optimization.milp import MILPBasedOptimizer


from studies.use_case_2_pv.renewable_energies import pv_hp_design_optimization
import pandas as pd


MPC_PARAMETERS = {
    "standard": {
        "T_Air_ub": 297.15,
        "T_Air_lb": 293.15,
        "c_comf_upper": 1000,
        "c_comf_lower": 1000
    },
    "no_T_Air_ub": {
        "T_Air_ub": 100000,
        "c_comf_upper": 0
    },
    "constant_T_Air": {
        "T_Air_ub": 294.15,
        "T_Air_lb": 293.15,
        "c_comf_upper": 1000000000.0,
        "c_comf_lower": 1000000000.0
    }
}


VARIABLES = {
    "input_names": [
        "actExtDHWCtrl",
        "TDHWSet",
        "actExtBufCtrl",
        "TBufSet",
        "actExtVal",
        "actOverheat",
        "yValSet",
    ],
    "state_names": [
        "building.internalElectricalPin.PElecLoa",
        "outputs.weather.TDryBul",
        "hydraulic.control.buiAndDHWCtr.TSetBuiSup.TSet",
        "hydraulic.distribution.stoBuf.layer[1].T",
        "hydraulic.transfer.outBusTra.TSup[1]",
        "hydraulic.transfer.outBusTra.TRet[1]",
        "hydraulic.transfer.traControlBus.opening[1]",
        "hydraulic.transfer.rad[1].port_b.m_flow",
        "hydraulic.transfer.rad[1].Q_flow",
        "hydraulic.transfer.rad[1].QCon_flow",
        "hydraulic.transfer.rad[1].QRad_flow",
        "building.thermalZone[1].ROM.volAir.heatPort.T",
        "building.thermalZone[1].ROM.volAir.heatPort.Q_flow",
        "building.thermalZone[1].ROM.convIntWall.Q_flow",
        "building.thermalZone[1].ROM.convExtWall.Q_flow",
        "building.thermalZone[1].ROM.convRoof.Q_flow",
        "building.thermalZone[1].ROM.convFloor.Q_flow",
        "building.thermalZone[1].ROM.convWin.Q_flow",
        "building.thermalZone[1].ROM.intWallRC.port_a.T",
        "building.thermalZone[1].ROM.intWallRC.port_a.Q_flow",
        "building.thermalZone[1].ROM.intWallRC.thermCapInt[1].T",
        "building.thermalZone[1].ROM.intWallRC.thermCapInt[1].port.Q_flow",
        "building.thermalZone[1].ROM.extWallRC.port_a.T",
        "building.thermalZone[1].ROM.extWallRC.port_a.Q_flow",
        "building.thermalZone[1].ROM.extWallRC.port_b.T",
        "building.thermalZone[1].ROM.extWallRC.port_b.Q_flow",
        "building.thermalZone[1].ROM.extWallRC.thermCapExt[1].T",
        "building.thermalZone[1].ROM.extWallRC.thermCapExt[1].port.Q_flow",
        "building.thermalZone[1].ROM.roofRC.port_a.T",
        "building.thermalZone[1].ROM.roofRC.port_a.Q_flow",
        "building.thermalZone[1].ROM.roofRC.port_b.T",
        "building.thermalZone[1].ROM.roofRC.port_b.Q_flow",
        "building.thermalZone[1].ROM.roofRC.thermCapExt[1].T",
        "building.thermalZone[1].ROM.roofRC.thermCapExt[1].port.Q_flow",
        "building.thermalZone[1].ROM.floorRC.port_a.T",
        "building.thermalZone[1].ROM.floorRC.port_a.Q_flow",
        "building.thermalZone[1].ROM.floorRC.port_b.T",
        "building.thermalZone[1].ROM.floorRC.port_b.Q_flow",
        "building.thermalZone[1].ROM.floorRC.thermCapExt[1].T",
        "building.thermalZone[1].ROM.floorRC.thermCapExt[1].port.Q_flow",
        "building.thermalZone[1].ROM.radHeatSol[1].Q_flow",
        "building.thermalZone[1].ROM.radHeatSol[2].Q_flow",
        "building.thermalZone[1].ROM.radHeatSol[3].Q_flow",
        "building.thermalZone[1].ROM.radHeatSol[4].Q_flow",
        "building.outBusDem.QIntGain[1].value",
        "building.thermalZone[1].zoneParam.AZone",
        "building.thermalZone[1].lights.uRel",
        "building.thermalZone[1].zoneParam.lightingPowerSpecific",
        "building.thermalZone[1].machinesSenHea.uRel",
        "building.thermalZone[1].zoneParam.internalGainsMachinesSpecific",
        "building.thermalZone[1].humanSenHeaDependent.uRel",
        "building.thermalZone[1].zoneParam.specificPeople",
        "building.thermalZone[1].zoneParam.activityDegree",
        "building.thermalZone[1].airExc.Q_flow",
        "building.thermalZone[1].airExc.ventRate",
        "building.thermalZone[1].ventCont.relOccupation",
        "hydraulic.control.valCtrl.supervisoryControl.swi.u1",
        "hydraulic.control.valCtrl.supervisoryControl.swi.u2",
        "hydraulic.control.valCtrl.supervisoryControl.swi.u3",
        "hydraulic.control.valCtrl.supervisoryControl.swi.y",
        "scalingFactor",
        "QPriAtTOdaNom_flow_nominal",
        "hydraulic.generation.m_flow_nominal[1]",
        "hydraulic.distribution.parStoDHW.V",
        "hydraulic.distribution.parStoBuf.V",
        "hydraulic.generation.parEleHea.eta",
        "hydraulic.generation.dTTra_nominal[1]",
        "hydraulic.distribution.parStoBuf.QLoss_flow",
        "hydraulic.distribution.parStoDHW.QLoss_flow",
        "hydraulic.distribution.parStoBuf.T_m",
        "hydraulic.distribution.parStoBuf.TAmb",
        "hydraulic.distribution.parStoDHW.T_m",
        "hydraulic.distribution.parStoDHW.TAmb",
        "hydraulic.distribution.parStoBuf.sIns",
        "hydraulic.distribution.parStoDHW.sIns",
        "hydraulic.transfer.m_flow_nominal[1]",
        "hydraulic.transfer.dTTra_nominal[1]",
        "hydraulic.control.sigBusGen.yHeaPumSet",
        "hydraulic.control.weaBus.TDryBul",
        "building.internalElectricalPin.PElecLoa",
        "outputs.hydraulic.tra.opening[1]",
        "hydraulic.generation.eleHea.Q_flow_nominal"
    ],
    "output_names": [
        "outputs.electrical.dis.PEleLoa.value",
        "outputs.electrical.dis.PEleLoa.integral",
        "outputs.electrical.dis.PEleGen.value",
        "outputs.electrical.dis.PEleGen.integral",
        "outputs.hydraulic.gen.QHeaPum_flow.value",
        "outputs.hydraulic.gen.QHeaPum_flow.integral",
        "outputs.hydraulic.gen.PEleHeaPum.value",
        "outputs.hydraulic.gen.PEleHeaPum.integral",
        "outputs.hydraulic.gen.QEleHea_flow.value",
        "outputs.hydraulic.gen.QEleHea_flow.integral",
        "outputs.hydraulic.gen.PEleEleHea.value",
        "outputs.hydraulic.gen.PEleEleHea.integral",
        "outputs.electrical.gen.PElePV.value",
        "outputs.electrical.gen.PElePV.integral",
    ]
}


def load_rbpc_settings(save_path: Path, n_days: int, tree_depth: int):
    from bes_rules.rule_extraction.rbpc_development.clustering import load_results_from_pickle
    from bes_rules.rule_extraction.rbpc_development import utils
    clustering_results = load_results_from_pickle(save_path)
    cluster_map = utils.get_cluster_time_series_map(clustering_results[n_days])

    with open(save_path.joinpath("decision_tress.pickle"), "rb") as file:
        trees = pickle.load(file)
    kwargs = {
        "with_dhw": False,
        "solver_kwargs": {
            "cluster_map": cluster_map,
            "clf": trees[n_days][tree_depth],
            "control_horizon": 24
        }
    }
    return kwargs


def run_design_optimization(
        config: configs.StudyConfig,
        surrogate_builder_class,
        surrogate_builder_kwargs: dict = None
):
    if surrogate_builder_kwargs is None:
        surrogate_builder_kwargs = {}
    # South-facing building from 1994
    config.inputs.weathers = [config.inputs.weathers[-1]]
    config.inputs.buildings = [config.inputs.buildings[-1]]  # config.inputs.buildings[-2]
    modifiers = config.inputs.modifiers[-1]
    config.inputs.modifiers = [modifiers]
    config.simulation.sim_setup["output_interval"] = 60 * 15
    config.simulation.sim_setup["stop_time"] = 3600 * 24 * 365

    y_variables = {
        "$T_\mathrm{Oda}$ in °C": "outputs.weather.TDryBul",
        "$T_\mathrm{Room}$ in °C": ["outputs.building.TZone[1]"],
        "$y_\mathrm{Val}$ in %": "outputs.hydraulic.tra.opening[1]",
        #"$T_\mathrm{DHW}$ in °C": ["outputs.hydraulic.disCtrl.TStoDHWBotMea",
        #                           "outputs.hydraulic.disCtrl.TStoDHWTopMea"],
        "$T_\mathrm{Buf}$ in °C": ["hydraulic.distribution.stoBuf.layer[1].T", "TBufSet"],
        #"$T_\mathrm{HeaPum}$ in °C": ["outputs.hydraulic.genCtrl.THeaPumIn",
        #                              "outputs.hydraulic.genCtrl.THeaPumOut"],
        # "$COP$ in -": "outputs.hydraulic.genCtrl.COP",
        #"$y_\mathrm{HeaPum}$ in %": "outputs.hydraulic.genCtrl.yHeaPumSet",
        #"$\dot{Q}_\mathrm{DHW}$ in kW": "outputs.DHW.Q_flow.value",
        "$\dot{Q}_\mathrm{Bui}$ in kW": "outputs.building.eneBal[1].traGain.value",
        "$P_\mathrm{el,HeaPum}$": "outputs.hydraulic.gen.PEleHeaPum.value",
        "$P_\mathrm{el,EleHea}$": "outputs.hydraulic.gen.PEleEleHea.value",
        "$y_\mathrm{EleHea}$ in %": "yEleHeaSet"
    }
    from bes_rules.configs.plotting import PlotConfig
    plot_config = PlotConfig.load_default()
    custom_plot_config = {
        "outputs.building.TZone[1]": {"label": "$T_\mathrm{Room}$", "quantity": "Temperature"},
        "hydraulic.distribution.stoBuf.layer[1].T": {"label": "$T_\mathrm{Sup,Mea}$", "quantity": "Temperature"},
        "TBufSet": {"label": "$T_\mathrm{Sup,Set}$", "quantity": "Temperature"},
        "yEleHeaSet": {"label": "$y_\mathrm{EleHea}$", "quantity": "Percent"},
    }
    plot_config.update_config({"variables": custom_plot_config})

    plot_settings = dict(
        #x_vertical_lines=["parameterStudy.TBiv"],  # TODO Revert for new results
        plot_config=plot_config,
        y_variables=y_variables
    )
    config.simulation.plot_settings = plot_settings

    config.optimization.variables = [
        configs.OptimizationVariable(
            name="parameterStudy.TBiv",  # with TBiv changes Q_heaPum_max=QPriAtTOdaNom_flow_nominal * scalingFactor
            lower_bound=273.15 - 9,
            upper_bound=278.15,
            discrete_steps=4
        ),
        configs.OptimizationVariable(
            name="parameterStudy.VPerQFlow",  # QBui_flow_nominal=6378.3W, TOda,nominal=260.55K
            lower_bound=5,
            upper_bound=150,
            levels=3
        ),
        configs.OptimizationVariable(
            name="parameterStudy.f_design",  # in every new design the PV_size gets halved!
            lower_bound=1,
            upper_bound=1,
            levels=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.DHWOverheatTemp",
            lower_bound=293.15,
            upper_bound=293.15,
            levels=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.BufOverheatdT",
            lower_bound=15,
            upper_bound=15,
            levels=1
        )
    ]
    run_input_variations(
        config=config,
        surrogate_builder_class=surrogate_builder_class,
        run_inputs_in_parallel=False,
        **surrogate_builder_kwargs
    )


def run_rbpc(path, n_days, tree_depth):
    import rbpc
    predictive_control_options = load_rbpc_settings(save_path=Path(path), n_days=n_days, tree_depth=tree_depth)
    run_design_optimization(
        surrogate_builder_class=MILPBasedOptimizer,
        surrogate_builder_kwargs=dict(
            predictive_control_function=rbpc.run_rbpc,
            variables=VARIABLES,
            predictive_control_options=predictive_control_options
        ),
        config=pv_hp_design_optimization.get_config(
            n_cpu=1,
            study_name="BES_RBPC",
            model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPIExternalControlSmartThermostat",
            test=False
        ))
    from studies.use_case_2_pv.peter import plot_sim
    plot_sim(path)


def run_mpc():

    mapping_predictions = {
        "electrical.generation.outBusGen.PElePV.value": "P_el_pv_raw",
        "parameterStudy.f_design": "PVDesignSize",
        "outputs.weather.TDryBul": "T_amb",
        "hydraulic.control.buiAndDHWCtr.TSetBuiSup.TSet": "THeaCur",
        "building.thermalZone[1].ROM.radHeatSol[1].Q_flow": "Q_RadSol_or_1",
        "building.thermalZone[1].ROM.radHeatSol[2].Q_flow": "Q_RadSol_or_2",
        "building.thermalZone[1].ROM.radHeatSol[3].Q_flow": "Q_RadSol_or_3",
        "building.thermalZone[1].ROM.radHeatSol[4].Q_flow": "Q_RadSol_or_4",
        "building.thermalZone[1].lights.uRel": "schedule_light",
        "building.thermalZone[1].machinesSenHea.uRel": "schedule_dev",
        "building.thermalZone[1].humanSenHeaDependent.uRel": "schedule_human",
        "building.thermalZone[1].preTemWall.T": "T_preTemWall",
        "building.thermalZone[1].preTemWin.T": "T_preTemWin",
        "building.thermalZone[1].preTemRoof.T": "T_preTemRoof",
        "building.thermalZone[1].preTemFloor.T": "T_preTemFloor",
        "building.thermalZone[1].ventCont.redFac": "redFac",
    }

    def manipulate_predictions(df: pd.DataFrame):
        df.loc[:, "Q_RadSol"] = (
                df.loc[:, "Q_RadSol_or_1"] +
                df.loc[:, "Q_RadSol_or_2"] +
                df.loc[:, "Q_RadSol_or_3"] +
                df.loc[:, "Q_RadSol_or_4"]
        )
        df.drop(columns=["Q_RadSol_or_1", "Q_RadSol_or_2", "Q_RadSol_or_3", "Q_RadSol_or_4"], inplace=True)
        return df

    run_design_optimization(
        surrogate_builder_class=AgentLibMPC,
        surrogate_builder_kwargs=dict(
            predictive_control_options=dict(
                mpc_module="agent_modules/mpc.json",
                predictor_module="agent_modules/predictor.json",
                simulator_module="agent_modules/simulator_fmu.json",
                mpc_parameters=MPC_PARAMETERS["standard"]
            ),
            manipulate_predictions=manipulate_predictions,
            mapping_predictions=mapping_predictions
        ),
        config=pv_hp_design_optimization.get_config(
            n_cpu=12,
            study_name="BESMod_CASADI",
            model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPIExternalControlSmartThermostat",
            test=False
        )
    )


def run_reference():
    run_design_optimization(
        config=pv_hp_design_optimization.get_config(
            n_cpu=12,
            study_name="BES_RBC",
            model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPICtrlOpt",
            test=False
        ),
        surrogate_builder_class=None
    )


if __name__ == '__main__':
    #logging.basicConfig(level="DEBUG")
    run_mpc()
