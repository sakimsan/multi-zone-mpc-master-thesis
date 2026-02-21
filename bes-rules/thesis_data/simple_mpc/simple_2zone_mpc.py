import logging
import os
from pathlib import Path
from typing import List

import pandas as pd

from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.utils.analysis import load_mpc_stats
from agentlib_mpc.utils.plotting.interactive import show_dashboard

logger = logging.getLogger(__name__)

ub1 = 295.15
ub2 = 298.15

class MyCasadiModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        CasadiInput(name="mDot_1", value=0.02, unit="m³/s"),
        CasadiInput(name="mDot_2", value=0.02, unit="m³/s"),
        CasadiInput(name="load_1", value=100, unit="W"),
        CasadiInput(name="load_2", value=200, unit="W"),
        CasadiInput(name="T_in_1", value=290.15, unit="K"),
        CasadiInput(name="T_in_2", value=290.15, unit="K"),
        CasadiInput(name="T_upper_1", value=ub1, unit="K"),
        CasadiInput(name="T_upper_2", value=ub2, unit="K"),
    ]
    states: List[CasadiState] = [
        CasadiState(name="T1", value=293.15, unit="K"),
        CasadiState(name="T2", value=294.15, unit="K"),
        CasadiState(name="T_slack_1", value=0, unit="K"),
        CasadiState(name="T_slack_2", value=0, unit="K"),
    ]
    parameters: List[CasadiParameter] = [
        CasadiParameter(name="cp", value=1000, unit="J/kg*K"),
        CasadiParameter(name="C", value=100000, unit="J/K"),
        CasadiParameter(name="s_T", value=1, unit="-"),
        CasadiParameter(name="r_mDot", value=1, unit="-"),
    ]
    outputs: List[CasadiOutput] = [
        CasadiOutput(name="T_out_1", unit="K"),
        CasadiOutput(name="T_out_2", unit="K"),
    ]

class MyCasadiModel(CasadiModel):
    config: MyCasadiModelConfig

    def setup_system(self):
        self.T1.ode = self.cp * self.mDot_1 / self.C * (self.T_in_1 - self.T1) + self.load_1 / self.C
        self.T2.ode = self.cp * self.mDot_2 / self.C * (self.T_in_2 - self.T2) + self.load_2 / self.C

        self.T_out_1.alg = self.T1
        self.T_out_2.alg = self.T2

        self.constraints = [
            (0, self.T1 + self.T_slack_1, self.T_upper_1),
            (0, self.T2 + self.T_slack_2, self.T_upper_2),
        ]

        objective = sum([
            self.r_mDot * self.mDot_1,
            self.r_mDot * self.mDot_2,
            self.s_T * self.T_slack_1**2,
            self.s_T * self.T_slack_2**2,
        ])
        return objective

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 60}

AGENT_MPC = {
    "id": "myMPCAgent",
    "modules": [
        {"module_id": "Ag1Com", "type": "local_broadcast"},
        {
            "module_id": "myMPC",
            "type": "agentlib_mpc.mpc",
            "optimization_backend": {
                "type": "casadi",
                "model": {"type": {"file": __file__, "class_name": "MyCasadiModel"}},
                "discretization_options": {
                    "collocation_order": 2,
                    "collocation_method": "legendre",
                },
                "solver": {"name": "fatrop"},
                "results_file": "results//mpc.csv",
                "save_results": True,
                "overwrite_result_file": True,
            },
            "time_step": 300,
            "prediction_horizon": 15,
            "parameters": [
                {"name": "s_T", "value": 3},
                {"name": "r_mDot", "value": 1},
            ],
            "inputs": [
                {"name": "T_in_1", "value": 290.15},
                {"name": "T_in_2", "value": 290.15},
                {"name": "load_1", "value": 100},
                {"name": "load_2", "value": 200},
                {"name": "T_upper_1", "value": ub1},
                {"name": "T_upper_2", "value": ub2},
            ],
            "controls": [
                {"name": "mDot_1", "value": 0.02, "ub": 0.05, "lb": 0},
                {"name": "mDot_2", "value": 0.02, "ub": 0.05, "lb": 0},
            ],
            "outputs": [
                {"name": "T_out_1"},
                {"name": "T_out_2"},
            ],
            "states": [
                {"name": "T1", "value": 293.15, "ub": 303.15, "lb": 288.15, "alias": "T1", "source": "SimAgent"},
                {"name": "T2", "value": 294.15, "ub": 303.15, "lb": 288.15, "alias": "T2", "source": "SimAgent"},
            ],
        },
    ],
}

AGENT_SIM = {
    "id": "SimAgent",
    "modules": [
        {"module_id": "Ag1Com", "type": "local_broadcast"},
        {
            "module_id": "room",
            "type": "simulator",
            "model": {
                "type": {"file": __file__, "class_name": "MyCasadiModel"},
                "states": [
                    {"name": "T1", "value": 293.15},
                    {"name": "T2", "value": 293.15},
                ],
            },
            "t_sample": 10,
            "update_inputs_on_callback": False,
            "save_results": True,
            "outputs": [
                {"name": "T_out_1", "value": 293.15, "alias": "T1"},
                {"name": "T_out_2", "value": 293.15, "alias": "T2"},
            ],
            "inputs": [
                {"name": "mDot_1", "value": 0.02, "alias": "mDot_1"},
                {"name": "mDot_2", "value": 0.02, "alias": "mDot_2"},
            ],
        },
    ],
}

def run_example(with_plots=True, log_level=logging.INFO, until=10000, with_dashboard=False):
    os.chdir(Path(__file__).parent)
    logging.basicConfig(level=log_level)
    mas = LocalMASAgency(agent_configs=[AGENT_MPC, AGENT_SIM], env=ENV_CONFIG, variable_logging=False)
    mas.run(until=until)
    try:
        stats = load_mpc_stats("results/__mpc.csv")
    except Exception:
        stats = None
    results = mas.get_results(cleanup=False)
    mpc_results = results["myMPCAgent"]["myMPC"]
    sim_res = results["SimAgent"]["room"]

    if with_dashboard:
        show_dashboard(mpc_results, stats)

    if with_plots:
        plot(mpc_results, sim_res, until)

    return results

def plot(mpc_results: pd.DataFrame, sim_res: pd.DataFrame, until: float):
    import matplotlib.pyplot as plt
    from agentlib_mpc.utils.plotting.mpc import plot_mpc

    fig, ax = plt.subplots(3, 1, sharex=True)
    t_sample = sim_res["T_out_1"].index[1] - sim_res["T_out_1"].index[0]

    aie1 = (sim_res["T_out_1"] - ub1).abs().sum() * t_sample / 3600
    aie2 = (sim_res["T_out_2"] - ub2).abs().sum() * t_sample / 3600

    energy1 = ((sim_res["mDot_1"] * (sim_res["T_out_1"] - sim_res["T_in_1"])).sum() * t_sample) / 3600
    energy2 = ((sim_res["mDot_2"] * (sim_res["T_out_2"] - sim_res["T_in_2"])).sum() * t_sample) / 3600

    print(f"Raum 1: AIE = {aie1:.2f} Kh, Energie = {energy1:.2f} kWh")
    print(f"Raum 2: AIE = {aie2:.2f} Kh, Energie = {energy2:.2f} kWh")

    plot_mpc(series=mpc_results["variable"]["T1"] - 273.15, ax=ax[0], plot_actual_values=True, plot_predictions=True)
    plot_mpc(series=mpc_results["variable"]["T2"] - 273.15, ax=ax[1], plot_actual_values=True, plot_predictions=True)
    plot_mpc(series=mpc_results["variable"]["mDot_1"], ax=ax[2], plot_actual_values=True, plot_predictions=True)
    plot_mpc(series=mpc_results["variable"]["mDot_2"], ax=ax[2], plot_actual_values=True, plot_predictions=True)

    ax[0].axhline(ub1 - 273.15, color="grey", linestyle="--", label="upper boundary 1")
    ax[1].axhline(ub2 - 273.15, color="grey", linestyle="--", label="upper boundary 2")
    ax[0].set_ylabel("$T_{room1}$ / °C")
    ax[1].set_ylabel("$T_{room2}$ / °C")
    ax[2].set_ylabel("$\\dot{m}_{air}$ / kg/s")
    ax[2].set_xlabel("simulation time / s")
    ax[2].legend()
    ax[0].legend()
    ax[1].legend()
    ax[2].set_ylim([0, 0.06])
    ax[2].set_xlim([0, until])
    plt.show()

if __name__ == "__main__":
    run_example(with_plots=True, with_dashboard=True, until=7200, log_level=logging.INFO)

