import pandas as pd
import matplotlib.pyplot as plt

from agentlib_mpc.utils.plotting.basic import EBCColors

from utils.timely import get_dt
from utils.plotting import save_plot, set_plot_settings


T_Air_lb_control = 273.15 + 20  # control to 20°C
T_Air_lb_dis = T_Air_lb_control - 2      # discomfort interval 2K


def get_comfort(df: pd.DataFrame):
    if "dT_lb_dis" not in df.columns:
        mask_T_Air_lb = df["T_Air"] < T_Air_lb_dis
        df["dT_lb_dis"] = 0
        df.loc[mask_T_Air_lb, "dT_lb_dis"] = df.loc[mask_T_Air_lb, "T_Air"] - T_Air_lb_dis

    if "dt" not in df.columns:
        df = get_dt(df, "h")

    if "dT_lb_dis_integral" not in df.columns:
        df["dT_lb_dis_integral"] = df["dT_lb_dis"] * df["dt"]

    cumulated_dT_lb_dis = df["dT_lb_dis_integral"].sum()

    return cumulated_dT_lb_dis, df


def comfort(save_path: str, df: pd.DataFrame):
    cumulated_dT_lb_dis, df = get_comfort(df)

    set_plot_settings()

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, layout="constrained")
    ax[0].plot(df.loc[:, "T_Air"]-273.15, label="T_Air")
    ax[0].hlines(y=T_Air_lb_control-273.15, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.light_grey, label="T_Air_lb_control")
    ax[0].hlines(y=T_Air_lb_dis-273.15, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey, label="T_Air_lb_dis")
    ax[1].plot(df.loc[:, "dT_lb_dis"], linewidth=1, color=EBCColors.blue, label="lb")
    ax[1].hlines(y=0, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey)

    # format plot
    for _ax in ax:
        _ax.set_xlim(df.index[0], df.index[-1])

    save_plot(fig=fig, save_path=save_path, save_name="comfort")

    return df

