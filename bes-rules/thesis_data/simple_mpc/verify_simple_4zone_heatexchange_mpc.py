# Szenario: Wärmeeintrag durch heiße Nachbarräume (mDot=0 in einem Raum)
# Verbesserte Version für garantierten mDot_1=0-Fall

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

# Ziel: Raum 1 wird nur durch heiße Nachbarräume erwärmt, mDot_1 kann auf 0 gesetzt werden
ub = [295.15, 340.15, 340.15, 340.15]  # Raum 1: 22°C, andere: 67°C

class MyCasadiModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        CasadiInput(name="mDot_1", value=0.0, unit="m³/s"),  # Startwert, MPC setzt optimal
        CasadiInput(name="mDot_2", value=0.02, unit="m³/s"),
        CasadiInput(name="mDot_3", value=0.02, unit="m³/s"),
        CasadiInput(name="mDot_4", value=0.02, unit="m³/s"),
        CasadiInput(name="load_1", value=0, unit="W"),
        CasadiInput(name="load_2", value=100, unit="W"),
        CasadiInput(name="load_3", value=100, unit="W"),
        CasadiInput(name="load_4", value=100, unit="W"),
        CasadiInput(name="T_in_1", value=295.15, unit="K"),    # gleich Upper Bound (neutral)
        CasadiInput(name="T_in_2", value=340.15, unit="K"),
        CasadiInput(name="T_in_3", value=340.15, unit="K"),
        CasadiInput(name="T_in_4", value=340.15, unit="K"),
        CasadiInput(name="T_upper_1", value=295.15, unit="K"),
        CasadiInput(name="T_upper_2", value=340.15, unit="K"),
        CasadiInput(name="T_upper_3", value=340.15, unit="K"),
        CasadiInput(name="T_upper_4", value=340.15, unit="K"),
    ]
    states: List[CasadiState] = [
        CasadiState(name="T1", value=293.15, unit="K"),    # 20°C
        CasadiState(name="T2", value=340.15, unit="K"),    # 67°C
        CasadiState(name="T3", value=340.15, unit="K"),
        CasadiState(name="T4", value=340.15, unit="K"),
        *[CasadiState(name=f"T_slack_{i+1}", value=0, unit="K") for i in range(4)],
        *[CasadiState(name=name, value=294.0 + j, unit="K") for j, name in enumerate([
            "T_wall_12_A", "T_wall_12_B",
            "T_wall_23_A", "T_wall_23_B",
            "T_wall_34_A", "T_wall_34_B",
            "T_wall_41_A", "T_wall_41_B"
        ])],
    ]
    parameters: List[CasadiParameter] = [
        CasadiParameter(name="cp", value=1000, unit="J/kg*K"),
        CasadiParameter(name="C", value=100000, unit="J/K"),
        CasadiParameter(name="s_T", value=1, unit="-"),
        CasadiParameter(name="r_mDot", value=1, unit="-"),
        CasadiParameter(name="C_wall", value=20000, unit="J/K"),
        CasadiParameter(name="U1", value=2.0, unit="W/K"),
        CasadiParameter(name="U3", value=0.5, unit="W/K"),
    ]
    outputs: List[CasadiOutput] = [
        *[CasadiOutput(name=f"T_out_{i+1}", unit="K") for i in range(4)],
    ]

class MyCasadiModel(CasadiModel):
    config: MyCasadiModelConfig

    def setup_system(self):
        I, S, O = self.config.inputs, self.config.states, self.config.outputs
        inp = {i.name: i for i in I}; st = {s.name: s for s in S}; out = {o.name: o for o in O}

        T = [st[f"T{i+1}"] for i in range(4)]
        T_in = [inp[f"T_in_{i+1}"] for i in range(4)]
        mDot = [inp[f"mDot_{i+1}"] for i in range(4)]
        load = [inp[f"load_{i+1}"] for i in range(4)]
        slack = [st[f"T_slack_{i+1}"] for i in range(4)]
        T_upper = [inp[f"T_upper_{i+1}"] for i in range(4)]
        T_out = [out[f"T_out_{i+1}"] for i in range(4)]

        pairs = [(0,1,"12"), (1,2,"23"), (2,3,"34"), (3,0,"41")]
        Q_wall_in = [0]*4

        for iA, iB, label in pairs:
            TwA = st[f"T_wall_{label}_A"]
            TwB = st[f"T_wall_{label}_B"]

            TwA.ode = (self.U1 * (T[iA] - TwA) + self.U3 * (TwB - TwA)) / self.C_wall
            TwB.ode = (self.U1 * (T[iB] - TwB) + self.U3 * (TwA - TwB)) / self.C_wall

            Q_wall_in[iA] += self.U1 * (TwA - T[iA])
            Q_wall_in[iB] += self.U1 * (TwB - T[iB])

        for i in range(4):
            T[i].ode = self.cp * mDot[i] / self.C * (T_in[i] - T[i]) + load[i] / self.C + Q_wall_in[i] / self.C
            T_out[i].alg = T[i]

        self.constraints = [(0, T[i] + slack[i], T_upper[i]) for i in range(4)]
        objective = sum([self.r_mDot * mDot[i] + self.s_T * slack[i]**2 for i in range(4)])
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
                "results_file": "results/mpc.csv",
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
                {"name": "T_in_1", "value": 295.15},
                {"name": "T_in_2", "value": 340.15},
                {"name": "T_in_3", "value": 340.15},
                {"name": "T_in_4", "value": 340.15},
                {"name": "load_1", "value": 0},
                {"name": "load_2", "value": 100},
                {"name": "load_3", "value": 100},
                {"name": "load_4", "value": 100},
                {"name": "T_upper_1", "value": 295.15},
                {"name": "T_upper_2", "value": 340.15},
                {"name": "T_upper_3", "value": 340.15},
                {"name": "T_upper_4", "value": 340.15},
            ],
            "controls": [
                {"name": "mDot_1", "value": 0.02, "ub": 0.05, "lb": 0},
                {"name": "mDot_2", "value": 0.02, "ub": 0.05, "lb": 0},
                {"name": "mDot_3", "value": 0.02, "ub": 0.05, "lb": 0},
                {"name": "mDot_4", "value": 0.02, "ub": 0.05, "lb": 0},
            ],
            "outputs": [
                {"name": "T_out_1"},
                {"name": "T_out_2"},
                {"name": "T_out_3"},
                {"name": "T_out_4"},
            ],
            "states": [
                {"name": "T1", "value": 293.15, "ub": 303.15, "lb": 288.15, "alias": "T1", "source": "SimAgent"},
                {"name": "T2", "value": 340.15, "ub": 350.15, "lb": 338.15, "alias": "T2", "source": "SimAgent"},
                {"name": "T3", "value": 340.15, "ub": 350.15, "lb": 338.15, "alias": "T3", "source": "SimAgent"},
                {"name": "T4", "value": 340.15, "ub": 350.15, "lb": 338.15, "alias": "T4", "source": "SimAgent"},
                *[{"name": name, "value": 294.0 + j} for j, name in enumerate([
                    "T_wall_12_A", "T_wall_12_B",
                    "T_wall_23_A", "T_wall_23_B",
                    "T_wall_34_A", "T_wall_34_B",
                    "T_wall_41_A", "T_wall_41_B"
                ])],
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
                    {"name": "T2", "value": 340.15},
                    {"name": "T3", "value": 340.15},
                    {"name": "T4", "value": 340.15},
                    *[{"name": name, "value": 294.0 + j} for j, name in enumerate([
                        "T_wall_12_A", "T_wall_12_B",
                        "T_wall_23_A", "T_wall_23_B",
                        "T_wall_34_A", "T_wall_34_B",
                        "T_wall_41_A", "T_wall_41_B"
                    ])],
                ],
            },
            "t_sample": 10,
            "update_inputs_on_callback": False,
            "save_results": True,
            "outputs": [
                {"name": "T_out_1", "value": 293.15, "alias": "T1"},
                {"name": "T_out_2", "value": 340.15, "alias": "T2"},
                {"name": "T_out_3", "value": 340.15, "alias": "T3"},
                {"name": "T_out_4", "value": 340.15, "alias": "T4"},
            ],
            "inputs": [
                {"name": "mDot_1", "value": 0.02, "alias": "mDot_1"},
                {"name": "mDot_2", "value": 0.02, "alias": "mDot_2"},
                {"name": "mDot_3", "value": 0.02, "alias": "mDot_3"},
                {"name": "mDot_4", "value": 0.02, "alias": "mDot_4"},
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

    fig, ax = plt.subplots(5, 1, sharex=True)
    for i in range(4):
        plot_mpc(series=mpc_results["variable"][f"T{i+1}"] - 273.15, ax=ax[i], plot_actual_values=True, plot_predictions=True)
        ax[i].axhline(ub[i] - 273.15, color="grey", linestyle="--", label=f"upper boundary {i+1}")
        ax[i].set_ylabel(f"$T_{{room{i+1}}}$ / °C")
        ax[i].legend()
    for i in range(4):
        plot_mpc(series=mpc_results["variable"][f"mDot_{i+1}"], ax=ax[4], plot_actual_values=True, plot_predictions=True)
    ax[4].set_ylabel("$\\dot{m}$ / kg/s")
    ax[4].set_xlabel("simulation time / s")
    ax[4].set_ylim([0, 0.06])
    ax[4].set_xlim([0, until])
    ax[4].legend()
    plt.show()

if __name__ == "__main__":
    run_example(with_plots=True, with_dashboard=True, until=7200, log_level=logging.INFO)
