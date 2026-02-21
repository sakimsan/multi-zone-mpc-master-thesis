import pickle
from pathlib import Path

from bes_rules import configs
from bes_rules.input_variations import run_input_variations

from bes_rules.simulation_based_optimization.milp import MILPBasedOptimizer
from studies.use_case_2_pv.renewable_energies import pv_hp_design_optimization


VARIABLES = {
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
    config.inputs.buildings = [config.inputs.buildings[-1]]
    modifiers = config.inputs.modifiers[-1]
    config.inputs.modifiers = [modifiers]

    config.simulation.sim_setup["output_interval"] = 900
    config.simulation.convert_to_hdf_and_delete_mat = False
    y_variables = {
        "$T_\mathrm{Oda}$ in °C": "outputs.weather.TDryBul",
        "$T_\mathrm{Room}$ in °C": ["outputs.building.TZone[1]", "outputs.user.TZoneSet[1]"],
        "$y_\mathrm{Val}$ in %": "outputs.hydraulic.tra.opening[1]",
        "$T_\mathrm{DHW}$ in °C": ["outputs.hydraulic.disCtrl.TStoDHWBotMea",
                                   "outputs.hydraulic.disCtrl.TStoDHWTopMea"],
        "$T_\mathrm{Buf}$ in °C": ["outputs.hydraulic.disCtrl.TStoBufBotMea",
                                   "outputs.hydraulic.disCtrl.TStoBufTopMea"],
        "$T_\mathrm{HeaPum}$ in °C": ["outputs.hydraulic.genCtrl.THeaPumIn",
                                      "outputs.hydraulic.genCtrl.THeaPumOut"],
        # "$COP$ in -": "outputs.hydraulic.genCtrl.COP",
        "$y_\mathrm{HeaPum}$ in %": "outputs.hydraulic.genCtrl.yHeaPumSet",
        "$\dot{Q}_\mathrm{DHW}$ in kW": "outputs.DHW.Q_flow.value",
        "$\dot{Q}_\mathrm{Bui}$ in kW": "outputs.building.eneBal[1].traGain.value",
        "$P_\mathrm{el,HeaPum}$": "outputs.hydraulic.gen.PEleHeaPum.value",
        "$P_\mathrm{el,EleHea}$": "outputs.hydraulic.gen.PEleEleHea.value"
    }
    from bes_rules.configs.plotting import PlotConfig
    plot_settings = dict(
        x_vertical_lines=["parameterStudy.TBiv"],
        plot_config=PlotConfig.load_default(),
        y_variables=y_variables
    )
    config.simulation.plot_settings = plot_settings
    config.base_path = Path(r"D:\zcbe")

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
        **surrogate_builder_kwargs
    )


def run_mpc():
    from bes_rules.simulation_based_optimization.milp.milp_model import run_milp_model
    run_design_optimization(
        surrogate_builder_class=MILPBasedOptimizer,
        surrogate_builder_kwargs=dict(
            predictive_control_function=run_milp_model,
            predictive_control_options={"with_dhw": False, "control_horizon": 4, "minimal_part_load_heat_pump": 0,
                                        "closed_loop": True}
        ),
        config=pv_hp_design_optimization.get_config(
            n_cpu=12,
            study_name="BES_MPC_NPL_OE",
            model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPIExternalControl",
            test=False
        ))


def run_rbpc():
    import rbpc
    save_path = Path(r"D:\zcbe\open_loop").joinpath(f"start=274_stop=120")
    predictive_control_options = load_rbpc_settings(save_path=save_path, n_days=4, tree_depth=4)

    run_design_optimization(
        surrogate_builder_class=MILPBasedOptimizer,
        surrogate_builder_kwargs=dict(
            predictive_control_function=rbpc.run_rbpc,
            predictive_control_options=predictive_control_options,
            variables=VARIABLES
        ),
        config=pv_hp_design_optimization.get_config(
            n_cpu=1,
            study_name="BES_RBPC",
            model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPIExternalControl",
            test=False
        ))


def run_reference():
    run_design_optimization(
        config=pv_hp_design_optimization.get_config(
            n_cpu=1,
            study_name="BES_RBC",
            model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPICtrlOpt",
            test=False
        ),
        surrogate_builder_class=None
    )
    run_design_optimization(
        config=pv_hp_design_optimization.get_config(
            n_cpu=1,
            study_name="BES_No_RBC",
            model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPINoSupCtrl",
            test=False
        ),
        surrogate_builder_class=None
    )


if __name__ == '__main__':
    run_reference()
    # run_mpc()
