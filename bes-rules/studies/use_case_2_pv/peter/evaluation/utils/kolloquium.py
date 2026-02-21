import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates

from agentlib_mpc.utils.plotting.basic import EBCColors

from utils.plotting import save_plot, set_plot_settings, get_labels_control
from utils.plot_dictionary import plot_dict

yellow = (250 / 255, 222 / 255, 0 / 255)


def plot_pv_day(df: pd.DataFrame, save_path: str):
    start = df.index[0] + pd.Timedelta(days=75)
    end = start + pd.Timedelta(days=1)
    _df = df.loc[start:end, :]

    save_path = f"{save_path}/kolloquium"

    set_plot_settings()

    fig, ax = plt.subplots(figsize=(3.6, 2))

    ax.fill(_df["P_pv"], color=yellow)

    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(dates.HourLocator(interval=12))
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_yticks([])
    ax.set_ylabel("PV-Leistung")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_plot(fig=fig, save_path=save_path, save_name="pv_day")



T_to_C = 273.15


def optimization_results(df: pd.DataFrame, save_path: str):
    set_plot_settings()

    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(6.1, 5.8), layout="constrained")

    ax[0].plot(df["T_Air"]-T_to_C, label=plot_dict["T_Air"], color=EBCColors.ebc_palette_sort_2[0])
    ax[0].set_ylabel(plot_dict["T_Air"] + r" in \textdegree C")

    ax[1].plot(df["TBufSet"]-T_to_C, label=plot_dict["TBufSet"], color=EBCColors.ebc_palette_sort_2[1])
    ax[1].set_ylabel(plot_dict["TBufSet"] + r" in \textdegree C")

    ax[2].plot(df["valve_actual"]*100, label="", color=EBCColors.ebc_palette_sort_2[2])
    ax[2].set_ylabel(f"Ventilöffnung")
    ax[2].set_ylim(0, 1.1*df["valve_actual"].max()*100)
    ax[2].yaxis.set_major_formatter(ticker.PercentFormatter())

    ax[3].plot(df["P_el_hp"]/1000, label="WP", color=EBCColors.ebc_palette_sort_2[5])
    ax[3].plot(df["P_pv"]/1000, label="PV", color=EBCColors.ebc_palette_sort_2[3])
    ax[3].set_ylabel(r"$P_\mathrm{el}$ in kW")
    ax[3].set_ylim(0, 1.1*df["P_pv"].max()/1000)
    ax[3].legend()

    ax[4].plot(df["Tamb"]-T_to_C, label=plot_dict["Tamb"], color=EBCColors.ebc_palette_sort_2[4])
    ax[4].set_ylabel(plot_dict["Tamb"] + r" in \textdegree C")
    ax[4].set_xlabel("Datum")

    ax[4].set_xlim(df.index[0], df.index[-1])
    ax[4].xaxis.set_major_locator(dates.DayLocator(interval=1))
    ax[4].xaxis.set_major_formatter(dates.DateFormatter("%d.%m"))
    # ax[4].xaxis.set_minor_locator(dates.HourLocator(interval=12))
    # ax[4].xaxis.set_minor_formatter(dates.DateFormatter("%H:%M"))

    for _ax in ax:
        _ax.grid()

    fig.align_ylabels(ax)

    save_plot(fig=fig, save_path=save_path, save_name="optimization_results")


    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6.1, 3), layout="constrained")
    ax[0].plot(df["Qdot_Air"]/1000, label=plot_dict["Qdot_Air"], color=EBCColors.ebc_palette_sort_2[0])
    ax[0].plot(df["QTraCon_flow"]/1000, label=plot_dict["QTraCon_flow"], color=EBCColors.ebc_palette_sort_2[1])
    ax[0].plot(df["Qdot_Air_int"]/1000, label=plot_dict["Qdot_Air_int"], color=EBCColors.ebc_palette_sort_2[2])
    ax[0].plot(df["Qdot_Air_ext"]/1000, label=plot_dict["Qdot_Air_ext"], color=EBCColors.ebc_palette_sort_2[3])

    ax[0].set_ylabel(r"Q in kW")
    ax[0].grid()
    ax[0].legend(loc="upper left")


    ax[1].plot(df["T_Air"]-T_to_C, label="Raumluft", color=EBCColors.ebc_palette_sort_2[0])
    ax[1].plot(df["T_IntWall_sur"]-T_to_C, label="Int. Wand, Oberfläche", color=EBCColors.ebc_palette_sort_2[1])
    ax[1].plot(df["T_IntWall"]-T_to_C, label="Int. Wand", color=EBCColors.ebc_palette_sort_2[2])

    ax[1].set_ylabel(r"T in °C")
    ax[1].set_xlabel("Datum")

    ax[1].set_xlim(df.index[0], df.index[-1])
    ax[1].xaxis.set_major_locator(dates.DayLocator(interval=1))
    ax[1].xaxis.set_major_formatter(dates.DateFormatter("%d.%m"))
    ax[1].grid()
    ax[1].legend()
    fig.align_ylabels(ax)

    save_plot(fig=fig, save_path=save_path, save_name="heat_flows")