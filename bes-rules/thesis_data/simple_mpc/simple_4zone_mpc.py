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

ub = [295.15, 296.15, 297.15, 298.15]

class MyCasadiModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        *[CasadiInput(name=f"mDot_{i+1}", value=0.02, unit="m³/s") for i in range(4)],
        *[CasadiInput(name=f"load_{i+1}", value=100 + 50*i, unit="W") for i in range(4)],
        *[CasadiInput(name=f"T_in_{i+1}", value=290.15, unit="K") for i in range(4)],
        *[CasadiInput(name=f"T_upper_{i+1}", value=ub[i], unit="K") for i in range(4)],
    ]
    states: List[CasadiState] = [
        *[CasadiState(name=f"T{i+1}", value=293.15 + i, unit="K") for i in range(4)],
        *[CasadiState(name=f"T_slack_{i+1}", value=0, unit="K") for i in range(4)],
    ]
    parameters: List[CasadiParameter] = [
        CasadiParameter(name="cp", value=1000, unit="J/kg*K"),
        CasadiParameter(name="C", value=100000, unit="J/K"),
        CasadiParameter(name="s_T", value=1, unit="-"),
        CasadiParameter(name="r_mDot", value=1, unit="-"),
        CasadiParameter(name="k_wall", value=0.1, unit="W/K"),
    ]
    outputs: List[CasadiOutput] = [
        *[CasadiOutput(name=f"T_out_{i+1}", unit="K") for i in range(4)],
    ]

class MyCasadiModel(CasadiModel):
    config: MyCasadiModelConfig

    def setup_system(self):
        inputs = {inp.name: inp for inp in self.config.inputs}
        states = {st.name: st for st in self.config.states}
        outputs = {out.name: out for out in self.config.outputs}

        T = [states[f"T{i+1}"] for i in range(4)]
        T_in = [inputs[f"T_in_{i+1}"] for i in range(4)]
        mDot = [inputs[f"mDot_{i+1}"] for i in range(4)]
        load = [inputs[f"load_{i+1}"] for i in range(4)]
        slack = [states[f"T_slack_{i+1}"] for i in range(4)]
        T_upper = [inputs[f"T_upper_{i+1}"] for i in range(4)]
        T_out = [outputs[f"T_out_{i+1}"] for i in range(4)]

        connections = {
            0: [1, 3],
            1: [0, 2],
            2: [1, 3],
            3: [2, 0],
        }

        for i in range(4):
            heat_exchange = sum([self.k_wall * (T[j] - T[i]) for j in connections[i]])
            T[i].ode = self.cp * mDot[i] / self.C * (T_in[i] - T[i]) + load[i] / self.C + heat_exchange / self.C
            T_out[i].alg = T[i]

        self.constraints = [(0, T[i] + slack[i], T_upper[i]) for i in range(4)]

        objective = sum([
            self.r_mDot * mDot[i] + self.s_T * slack[i]**2 for i in range(4)
        ])
        return objective

# Der Rest des Codes bleibt unverändert (AGENT_MPC, AGENT_SIM, run_example, plot)


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
                *[{"name": f"T_in_{i+1}", "value": 290.15} for i in range(4)],
                *[{"name": f"load_{i+1}", "value": 100 + 50*i} for i in range(4)],
                *[{"name": f"T_upper_{i+1}", "value": ub[i]} for i in range(4)],
            ],
            "controls": [
                *[{"name": f"mDot_{i+1}", "value": 0.02, "ub": 0.05, "lb": 0} for i in range(4)],
            ],
            "outputs": [
                *[{"name": f"T_out_{i+1}"} for i in range(4)],
            ],
            "states": [
                *[{
                    "name": f"T{i+1}", "value": 293.15 + i, "ub": 303.15, "lb": 288.15,
                    "alias": f"T{i+1}", "source": "SimAgent"
                } for i in range(4)],
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
                    *[{"name": f"T{i+1}", "value": 293.15 + i} for i in range(4)],
                ],
            },
            "t_sample": 10,
            "update_inputs_on_callback": False,
            "save_results": True,
            "outputs": [
                *[{"name": f"T_out_{i+1}", "value": 293.15 + i, "alias": f"T{i+1}"} for i in range(4)],
            ],
            "inputs": [
                *[{"name": f"mDot_{i+1}", "value": 0.02, "alias": f"mDot_{i+1}"} for i in range(4)],
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
