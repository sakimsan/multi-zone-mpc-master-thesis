import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agentlib_mpc.utils.plotting.basic import EBCColors

from utils.plotting import save_plot, set_plot_settings, get_labels_control
from utils.plot_dictionary import plot_dict
import matplotlib.ticker as ticker


def get_dt(df: pd.DataFrame, freq="h"):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")
    match freq:
        case "h":
            conv = 60 * 60
        case "min":
            conv = 60
        case "s":
            conv = 1
        case _:
            raise ValueError("freq must be 'h', 'min' or 's'")
    df["dt"] = df.index.to_series().diff().dt.total_seconds() / conv
    return df


def monthly(save_path: str, df: pd.DataFrame, features: callable, with_plots: bool = False):
    months = {
        "Apr": 4,
        "Mai": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Okt": 10,
        "Nov": 11,
        "Dez": 12,
        "Jan": 1,
        "Feb": 2,
        "Mär": 3,
    }

    columns = features(df).keys()
    df_months = pd.DataFrame(index=list(months.keys()), columns=columns)
    for month, i in months.items():
        _df = df[df.index.year == 2015]
        _df = _df[_df.index.month == i]
        df_months.loc[month] = features(_df)

    if with_plots:
        set_plot_settings()
        for col in columns:
            fig, ax = plt.subplots(layout="constrained")
            df_months[col].plot(kind="bar", ax=ax)

            ax.set_ylabel(plot_dict[col] if col in plot_dict else col)

            save_plot(fig=fig, save_path=save_path, save_name=col)

    return df, df_months


def compare_monthly_bar(monthly_dfs: list, save_path: str, save_name: str):
    df = pd.concat(monthly_dfs, axis=1)

    set_plot_settings()
    for prop in df.columns.levels[1]:
        plt.xticks(rotation=0)
        fig, ax = plt.subplots(figsize=(24/2.51, 7/2.51), layout="constrained")
        ax = df.loc[:, (slice(None), [prop])].plot(kind="bar", ax=ax)
        handles, previous_labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        if prop in ["SCR", "SSD"]:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.grid(which="major", axis="y")
        ax.set_ylabel(plot_dict[prop] if prop in plot_dict else prop)
        controls = df.columns.levels[0]
        fig.legend(handles=handles, labels=get_labels_control(previous_labels, controls), loc="outside right upper")
        plt.xticks(rotation=0)
        save_plot(fig=fig, save_path=save_path, save_name=prop)

    if len(monthly_dfs) > 1:
        controls = df.columns.levels[0]
        props = df.columns.levels[1]
        for control in controls:
            for prop in props:
                for contr in controls:
                    if contr != control:
                        df[f"{control}_base", f"{contr}_{prop}_abs"] = df[contr, prop] - df[control, prop]
                        df[f"{control}_base", f"{contr}_{prop}_rel"] = df[contr, prop] / df[control, prop].replace(to_replace=0, value=np.nan)

    df = df.sort_index(axis=1)
    os.makedirs(save_path, exist_ok=True)
    df.to_excel(f"{save_path}/{save_name}.xlsx")


def compare_time_series(save_path: str, dfs: list, properties: list):
    df = pd.concat(dfs, axis=1).interpolate(method="index")
    os.makedirs(save_path, exist_ok=True)

    controls = df.columns.levels[0]

    set_plot_settings()
    begin = df.index[0]
    b = begin
    dt = pd.Timedelta(days=7)
    end = pd.Timestamp(year=2015, month=4, day=1)  # df.index[-1]
    while b < end:
        e = b + dt
        _df = df.loc[b:e]

        for prop in properties:
            fig, ax = plt.subplots(figsize=(30/2.51, 10/2.51), layout="constrained")
            # for control in df.columns.levels[0]:
            for control in controls:
                if prop == "T_Air":
                    ax.plot(_df[control, prop]-273.15, label=control, linewidth=3)
                elif not (control == "rbpc_zcbe" and prop == "THeaCur"):
                    ax.plot(_df[control, prop], label=control, linewidth=3)
            handles, previous_labels = ax.get_legend_handles_labels()

            if prop == "T_Air":
                ax.hlines(y=20, xmin=_df.index[0], xmax=_df.index[-1], color=EBCColors.dark_grey)
                ax.hlines(y=24, xmin=_df.index[0], xmax=_df.index[-1], color=EBCColors.dark_grey)

            ax.set_xlim(_df.index[0], _df.index[-1])
            y_min = min(_df.loc[:, (slice(None), [prop])].min())
            y_max = max(_df.loc[:, (slice(None), [prop])].max())
            if prop == "T_Air":
                y_min = y_min - 273.15
                y_max = y_max - 273.15
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel(plot_dict[prop] if prop in plot_dict else prop)
            ax.grid(which="major", axis="x")
            fig.legend(labels=get_labels_control(previous_labels, controls), loc="outside right upper")

            save_plot(fig=fig, save_path=f"{save_path}/{prop}", save_name=f"{str(b.dayofyear)}-{str(e.dayofyear - 1)}")

        b = e
