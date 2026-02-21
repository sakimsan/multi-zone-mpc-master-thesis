"""
Plots for the control parameters
"""
import logging
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
from ebcpy import TimeSeriesData

from bes_rules.configs import StudyConfig
from bes_rules.plotting import utils

logger = logging.getLogger(__name__)


def plot_time_constants_heat_map(
        dfs: Dict[str, dict],
        save_path: Path
):
    for input_name, data in dfs.items():
        fig, axes = plt.subplots(1, len(data["max"]) + 1, sharey=True)
        df = data["time_constant"]
        df = df.sort_index()
        df = df.sort_index(axis=1)
        sns.heatmap(df, cmap="rocket", ax=axes[0], cbar=True)
        axes[0].set_title("Time constants in h")
        for ax, df, name in zip(axes[1:], data["max"].values(), data["max"].keys()):
            df = df.sort_index()
            df = df.sort_index(axis=1)
            sns.heatmap(df, cmap="rocket", ax=ax, cbar=True)
            ax.set_title(name[-10:])
            ax.set_ylabel(None)
            ax.set_xlabel(df.columns.name)
        axes[0].set_ylabel(df.index.name)
        fig.tight_layout()
        fig.savefig(save_path.joinpath(f"time_constants_{input_name}.png"))

    plt.show()


def get_time_constants_dfs(
        study_config: StudyConfig,
        start_time_search: float,
        x_variable: str = "TOda_start",
        y_variable: str = "timeInt",
):
    dfs = {}
    _, input_configs = utils.get_all_results_from_config(study_config=study_config)
    for input_config in input_configs:
        sim_results_path = study_config.study_path.joinpath(
            "DesignOptimizationResults",
            input_config.get_name())
        df = pd.DataFrame()
        df.index.name = x_variable
        df.columns.name = y_variable
        save_max_of_variables = ["outputs.hydraulic.gen.QEleHea_flow.value",
                                 "hydraulic.control.PIDCtrl.TSet"]
        units = [(1000, 0), (1, 273.15)]
        df_variables_max = {}
        for var in save_max_of_variables:
            df_variables_max[var] = df.copy()

        for file in os.listdir(sim_results_path):
            if not file.endswith(".mat"):
                continue

            TZoneSet_name = "building.useProBus.TZoneSet[1]"
            variable = "building.buiMeaBus.TZoneMea[1]"
            variable_names = list(df_variables_max.keys()) + [variable, TZoneSet_name, x_variable, y_variable]

            tsd = TimeSeriesData(sim_results_path.joinpath(file), variable_names=variable_names).to_df()
            critical_value = tsd.iloc[-1][TZoneSet_name]
            mask_mea = tsd.loc[start_time_search:, variable] > critical_value

            def _get_first_true(_mask):
                if not np.any(_mask):
                    return np.inf
                return _mask.idxmax()

            end_time = _get_first_true(mask_mea)
            start_time = start_time_search
            time_constant = (end_time - start_time) / 3600

            x = round(tsd.iloc[-1][x_variable] - 273.15, 0)
            y = tsd.iloc[-1][y_variable]
            for var, _unit in zip(df_variables_max.keys(), units):
                df_variables_max[var].loc[x, y] = tsd.loc[start_time:end_time, var].max() / _unit[0] - _unit[1]
            df.loc[x, y] = time_constant
            plt.figure()
            tsd = tsd.loc[start_time - 3600:start_time + 24 * 3600]
            plt.plot(tsd.index / 3600, tsd.loc[:, TZoneSet_name] - 273.15, color="blue")
            plt.plot(tsd.index / 3600, tsd.loc[:, variable] - 273.15, color="r")
            #plt.axvline(start_time / 3600, color="black", linestyle="--")
            plt.axvline(end_time / 3600, color="black", linestyle="--")
            plt.suptitle(f"{time_constant=} | {x_variable}={x} | {y_variable}={y}")
            plt.savefig(sim_results_path.joinpath(f"{file}.png"))
            plt.close("all")
        dfs[input_config.get_name()] = {"time_constant": df, "max": df_variables_max}
    return dfs


if __name__ == '__main__':
    plt.rcParams.update(
        {"figure.figsize": [13 / 2.54 * 2.6, 15 / 2.54 * 3 / 3],
         "font.size": 12,
         "figure.dpi": 250,
         "font.family": "Arial",
         "backend": "TkAgg"
         }
    )

    CONFIG = StudyConfig.from_json(r"D:\fwu\07_Results\ControlEstimationRadiator\study_config.json")
    DFS = get_time_constants_dfs(study_config=CONFIG, start_time_search=181*3600)
    plot_time_constants_heat_map(dfs=DFS, save_path=CONFIG.study_path)
