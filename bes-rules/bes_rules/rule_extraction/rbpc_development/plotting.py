import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

from bes_rules.rule_extraction.rbpc_development import utils
from bes_rules.plotting import EBCColors
from bes_rules.plotting.utils import get_figure_size


def _get_index_total_seconds_in_hours(index: pd.DatetimeIndex):
    return ((index.day - index[0].day) * 86400 + index.minute * 60 + index.hour * 3600 + index.second) / 3600


def plot_pv_day(day, df, tsd, save_path, year_shift_milp: int = 0):
    first_day_in_year = datetime.datetime(2023, 1, 1)
    pv_milp = df.loc[
              first_day_in_year + datetime.timedelta(days=day + year_shift_milp * 365):
              first_day_in_year + datetime.timedelta(seconds=(day + 1 + year_shift_milp * 365) * 86400 - 1)
              ]
    pv_dym = tsd.loc[
             first_day_in_year + datetime.timedelta(days=day):
             first_day_in_year + datetime.timedelta(seconds=(day + 1) * 86400 - 1)
             ]
    plt.figure()
    plt.plot(_get_index_total_seconds_in_hours(pv_dym.index), pv_dym, color="red")
    plt.plot(_get_index_total_seconds_in_hours(pv_milp.index), pv_milp, color="blue", linestyle="--")
    plt.xticks(np.arange(0, 25, 4))
    plt.savefig(save_path.joinpath(f"compare_milp_to_dym_pv_{day}.png"))
    plt.close("all")


def plot_daily_results(save_path: Path, successive_days: int = 1):
    df = utils.load_results(save_path)
    df = df.astype(float)
    os.makedirs(save_path.joinpath("daily_plots"), exist_ok=True)
    start_time = df.index[0]
    end_time = df.index[-1]
    n_days = (end_time - start_time).days
    def _get_date_name(d):
        return f"{d.day}. {d.month_name()}"

    for day in range(0, int(n_days) + 1, successive_days):
        df_day = df.loc[
                 start_time + datetime.timedelta(days=day):
                 start_time + datetime.timedelta(seconds=(day + successive_days) * 86400 - 1)
                 ].copy()
        fig, ax = plt.subplots(3, 1, figsize=get_figure_size(n_columns=successive_days*0.8+0.4, height_factor=2))
        _x_values = _get_index_total_seconds_in_hours(df_day.index)
        ax[0].plot(_x_values, (df_day["Q_HP"] + df_day["Q_HR"]) / 1000, label="$\dot{Q}_\mathrm{HP+EH}$", color=EBCColors.dark_red)
        ax[0].plot(_x_values, df_day["Q_Penalty"] / 1000, label="$\dot{Q}_\mathrm{Pen}$", color=EBCColors.dark_grey)
        ax[0].plot(_x_values, df_day["Q_Hou"] / 1000, label="$\dot{Q}_\mathrm{Dem}$", color=EBCColors.blue)
        ax[0].plot(_x_values, df_day["Q_Sto_Loss"] / 1000, label="$\dot{Q}_\mathrm{Sto,Los}$", color=EBCColors.grey)
        ax[0].plot(_x_values, df_day["QStoDischarge"] / 1000, label="$\dot{Q}_\mathrm{Sto,Dis}$", color=EBCColors.red)
        ax[0].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
        ax[0].set_ylabel("$\dot{Q}$ in kW")

        ax[1].plot(_x_values, df_day["TBufSet"] - 273.15, label="$T_\mathrm{Sto}$", color=EBCColors.light_red)
        ax[1].plot(_x_values, df_day["THeaCur"] - 273.15, label="$T_\mathrm{Sto,Min}$", color=EBCColors.dark_grey)
        ax[1].plot(_x_values, df_day["TBufSet"] - 273.15, label="$T_\mathrm{EH}$", color=EBCColors.dark_red)
        ax[1].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
        ax[1].set_ylabel("$T$ in °C")

        ax[2].plot(_x_values, df_day["P_PV"] / 1000, label="P_PV", color=EBCColors.blue)
        ax[2].plot(_x_values, (df_day["P_EL_HP"] + df_day["P_EL_HR"]) / 1000, label="$P_\mathrm{el,HP+EH}$", color=EBCColors.dark_red)
        ax[2].plot(_x_values, df_day["P_EL_Dem"] / 1000, label="$P_\mathrm{el,HA}$", color=EBCColors.grey)
        ax[2].plot(_x_values, (df_day["P_EL_Dem"] + df_day["P_EL_HP"] + df_day["P_EL_HR"]) / 1000, label="$P_\mathrm{el,Dem}$", linestyle="dashed", color=EBCColors.red)
        ax[2].set_ylabel("$P_\mathrm{el}$ in kW")
        ax[2].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
        for _ax in ax:
            _ax.set_xticks(np.arange(0, 24 * successive_days + 1, 4))

        fig.suptitle(f"{_get_date_name(df_day.index[0])} - {_get_date_name(df_day.index[-1])}")
        fig.align_ylabels()
        fig.tight_layout()
        fig.savefig(save_path.joinpath("daily_plots", f"Day_{day}-{day + successive_days}.png"))
        plt.close("all")


def plot_clustered_days(save_path: Path, z: np.ndarray, n_clusters: int, save_path_plots, with_filter: bool):
    df = utils.load_results(save_path=save_path)
    if with_filter:
        from bes_rules.rule_extraction.rbpc_development.clustering import cluster_filter
        df = cluster_filter(df)
    min_dT = round((df["TBufSet"] - df["THeaCur"]).min())
    max_dT = round((df["TBufSet"] - df["THeaCur"]).max())
    df_days = utils.split_df_into_days(df)
    assert len(df_days) == z.shape[0]
    fig, ax = plt.subplots(n_clusters, 1, figsize=get_figure_size(n_columns=1, height_factor=n_clusters / 2.5))
    if isinstance(ax, plt.Axes):
        ax = [ax]
    cluster_idx_to_plot_idx = {}
    n_ax_cluster_already_plotted = 0
    plot_last = {}
    for day_idx, df_day in enumerate(df_days):
        cluster_idx = np.where(z[:, day_idx] == 1)[0][0]
        if cluster_idx in cluster_idx_to_plot_idx:
            ax_idx = cluster_idx_to_plot_idx[cluster_idx]
        else:
            ax_idx = n_ax_cluster_already_plotted
            cluster_idx_to_plot_idx[cluster_idx] = ax_idx
            n_ax_cluster_already_plotted += 1
        _x_values = _get_index_total_seconds_in_hours(df_day.index)

        if z[day_idx, day_idx] == 1:
            plot_last[ax_idx] = [_x_values, df_day["TBufSet"] - df_day["THeaCur"]]
            continue
        else:
            ax[ax_idx].plot(
                _x_values,
                df_day["TBufSet"] - df_day["THeaCur"],
                color=EBCColors.light_grey)
    for ax_idx, plot_last_data in plot_last.items():
        ax[ax_idx].plot(*plot_last_data, color="black")
    for _ax in ax:
        _ax.set_xlim(0, 24)
        _ax.set_xticks(np.arange(0, 25, 4))
        _ax.set_yticks(np.arange(0, max_dT + 4, 10))
        _ax.set_ylabel("$\Delta T_\mathrm{OH}$ in K")
    ax[-1].set_xlabel("Time in h")
    fig.tight_layout()
    fig.savefig(save_path_plots.joinpath(f"Clustered_days_{n_clusters}.png"))
    plt.close("all")


def plot_clustered_days_other(save_path: Path, z: np.ndarray, n_clusters: int, save_path_plots, column):
    df = utils.load_results(save_path=save_path)
    min_ = round((df[column]).min())
    max_ = round((df[column]).max())
    df_days = utils.split_df_into_days(df)
    assert len(df_days) == z.shape[0]
    fig, ax = plt.subplots(n_clusters, 1, figsize=get_figure_size(n_columns=1, height_factor=n_clusters / 2.5))
    if isinstance(ax, plt.Axes):
        ax = [ax]
    cluster_idx_to_plot_idx = {}
    n_ax_cluster_already_plotted = 0
    plot_last = {}
    for day_idx, df_day in enumerate(df_days):
        cluster_idx = np.where(z[:, day_idx] == 1)[0][0]
        if cluster_idx in cluster_idx_to_plot_idx:
            ax_idx = cluster_idx_to_plot_idx[cluster_idx]
        else:
            ax_idx = n_ax_cluster_already_plotted
            cluster_idx_to_plot_idx[cluster_idx] = ax_idx
            n_ax_cluster_already_plotted += 1
        _x_values = _get_index_total_seconds_in_hours(df_day.index)

        if z[day_idx, day_idx] == 1:
            plot_last[ax_idx] = [_x_values, df_day[column]]
            continue
        else:
            ax[ax_idx].plot(
                _x_values,
                df_day[column],
                color=EBCColors.light_grey)
    for ax_idx, plot_last_data in plot_last.items():
        ax[ax_idx].plot(*plot_last_data, color="black")
    for _ax in ax:
        _ax.set_xlim(0, 24)
        _ax.set_xticks(np.arange(0, 25, 4))
        step = round((max_-min_)/4) if column != "yValSet" else 1
        _ax.set_yticks(np.arange(min_, max_, step))
        _ax.set_ylabel(column)
    ax[-1].set_xlabel("Time in h")
    fig.tight_layout()
    fig.savefig(save_path_plots.joinpath(f"Clustered_days_{n_clusters}_{column}.png"))
    plt.close("all")


def plot_convergence_of_clustering(save_path, cluster_results: dict, n_days: int = None):
    x_data = list(cluster_results.keys())
    y_data = [res["obj"] for res in cluster_results.values()]
    if n_days is not None:
        y_data = y_data[:n_days]
        x_data = x_data[:n_days]
    y_data_gradient = [y_data[idx-1] - y for idx, y in enumerate(y_data) if idx > 0]
    fig, ax = plt.subplots(figsize=get_figure_size(n_columns=1, height_factor=0.8))
    #ax.plot(x_data, y_data, marker="^", color="blue", label="$MSE$")
    gradient_label = "$\\Delta MSE$"
    ax.plot(x_data[1:], y_data_gradient, marker="^", color="red", label=gradient_label)
    ax.set_ylabel(gradient_label)
    #ax.set_ylabel("$MSE$")
    ax.set_xlabel("Number of clusters")

    #ax.set_xticks(np.arange(1, n_days) + 1)
    #ax.legend()
    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"Cluster_convergence_{n_days}.png"))


def plot_spread_of_clusters(save_path, cluster_results: dict, n_days: int):
    all_n_day = []
    cluster_dTs = []
    for n_day in range(2, n_days + 1):
        cluster_temperature_difference_sum = utils.calculate_sum_of_overheating(cluster_results[n_day])
        cluster_names = [dT for dT in np.unique(cluster_temperature_difference_sum)]
        cluster_dTs.extend(cluster_names)
        all_n_day.extend(n_day * np.ones(len(cluster_names)))
    fig, ax = plt.subplots(figsize=get_figure_size(n_columns=1, height_factor=0.8))
    ax.scatter(all_n_day, cluster_dTs, marker="s", color="red")
    ax.set_xticks(range(2, n_days + 1))
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("$\\Delta T_\mathrm{OH,Sum}$ in K")
    fig.tight_layout()
    fig.savefig(save_path.joinpath("Cluster_spread.png"))
