import matplotlib.pyplot as plt
import pandas as pd

from agentlib_mpc.utils.plotting.basic import EBCColors
from agentlib_mpc.utils.plotting.mpc import interpolate_colors

from utils.files import load_sim_mpc, load_pre_mpc
from utils.timely import monthly, get_dt
from utils.plotting import set_plot_settings, save_plot


def prediction_quality(load_path: str, save_path: str, columns: list, begin: pd.Timestamp, end: pd.Timestamp):
    df_sim = load_sim_mpc(f"{load_path}/sim_agent.csv")
    df_pre = load_pre_mpc(f"{load_path}/mpc_agent.csv")

    set_plot_settings()
    fig, ax = plt.subplots(layout="constrained")

    number_of_predictions = (begin-end)/pd.Timedelta(minutes=30)
    for column in columns:
        i = -1
        for _df_pre in df_pre:
            if _df_pre.index[0] < begin or _df_pre.index[0] > end:
                continue
            else:
                i = i + 1
                progress = i / number_of_predictions
                prediction_color = interpolate_colors(
                    progress=progress,
                    colors=[EBCColors.red, EBCColors.grey],
                )
                ax.plot(_df_pre.loc[begin:end, column], linewidth=0.5, color=prediction_color)
        ax.plot(df_sim.loc[begin:end, column], color=EBCColors.dark_grey, label="Sim")

        # format plot
        ax.set_xlim(begin, end)
        ax.grid()

        save_plot(fig=fig, save_path=save_path, save_name=column)


def get_lb_viol(df: pd.DataFrame):
    if "T_Air" not in df.columns:
        raise ValueError("T_Air must be in columns")
    lb = 273.15 + 20
    df["lb_viol"] = 0.0
    viol_mask = df["T_Air"] < lb
    df.loc[viol_mask, "lb_viol"] = df["T_Air"] - lb
    return df


def get_lb_viol_tol(df: pd.DataFrame):
    if "T_Air" not in df.columns:
        raise ValueError("T_Air must be in columns")
    lb = 273.15 + 20 - 2
    df["lb_viol_tol"] = 0.0
    viol_mask = df["T_Air"] < lb
    df.loc[viol_mask, "lb_viol_tol"] = df["T_Air"] - lb
    return df


def get_lb_viol_int(df: pd.DataFrame):
    if "lb_viol" not in df.columns:
        df = get_lb_viol(df)
    if "dt" not in df.columns:
        df = get_dt(df)
    df["lb_viol_int"] = df["lb_viol"] * df["dt"]
    return df


def get_lb_viol_tol_int(df: pd.DataFrame):
    if "lb_viol_tol" not in df.columns:
        df = get_lb_viol_tol(df)
    if "dt" not in df.columns:
        df = get_dt(df)
    df["lb_viol_tol_int"] = df["lb_viol_tol"] * df["dt"]
    return df


def get_quality_features(df: pd.DataFrame):
    if "lb_viol_int" not in df.columns:
        df = get_lb_viol_int(df)
    if "lb_viol_tol_int" not in df.columns:
        df = get_lb_viol_tol_int(df)
    try:
        days = (df.index[-1] - df.index[0]).days
    except IndexError:
        days = 1
    features = {
        "lb_viol_int": df["lb_viol_int"].sum()/days,
        "lb_viol_tol_int": df["lb_viol_tol_int"].sum()/days,
    }
    return features


def quality(save_path: str, df: pd.DataFrame):
    df = get_lb_viol(df)
    df = get_lb_viol_tol(df)
    df, df_months = monthly(save_path=save_path, df=df, features=get_quality_features)
    return df, df_months

