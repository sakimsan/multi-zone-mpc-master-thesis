import ast
import json
import locale
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from agentlib_mpc.utils.plotting.basic import EBCColors
from agentlib_mpc.utils.plotting.interactive import show_dashboard
from agentlib_mpc.utils.plotting.mpc import interpolate_colors
from cycler import cycler
from matplotlib import dates

from bes_rules.configs.plotting import EBCColors

locale.setlocale(locale.LC_ALL, "de_DE")
latex_textwidth = 6.1035  # inches
latex_texthight = 8.74187  # inches
pgf = False

start_date = pd.Timestamp(year=2015, month=1, day=1)


def load_mpc(path, prediction_series: bool):
    float_rows = {0: str}
    for i in range(1, 49):
        float_rows[i] = float
    df = pd.read_csv(path, dtype=float_rows, header=[0, 1]).rename(columns={"Unnamed: 0_level_1": "time"})
    df.drop(axis=1, level=0, columns=["upper", "lower", "parameter"], inplace=True)
    df.columns = df.columns.droplevel(level=0)
    df["time"] = df["time"].apply(ast.literal_eval)
    df["time_sim"] = df["time"].apply(lambda x: x[0])
    df["time_sim"] = start_date + pd.to_timedelta(df["time_sim"], unit="s")
    df["time_pre"] = df["time"].apply(lambda x: x[1])
    df["time_pre"] = pd.to_timedelta(df["time_pre"], unit="s")
    df.drop("time", axis=1, inplace=True)
    df.set_index(["time_sim", "time_pre"], inplace=True)

    if prediction_series:
        data = []
        idx = pd.IndexSlice
        beg_pred = df.index.get_level_values("time_pre").unique()[0]
        end_pred = df.index.get_level_values("time_pre").unique()[-1]
        sim_times = df.index.get_level_values("time_sim").unique()
        for sim_time in sim_times:
            prediction = df.loc[idx[sim_time, beg_pred:end_pred], :].copy()
            prediction.set_index(prediction.index.droplevel(level="time_sim"), inplace=True)
            prediction.set_index(prediction.index + sim_time, inplace=True)
            prediction.interpolate(method="index", axis="index", inplace=True)
            data.append(prediction)

    else:
        index_first = df.index.to_list()

        idx = pd.IndexSlice
        data = df.loc[idx[:, index_first[0][1]], :]
        data.set_index(data.index.droplevel("time_pre"), inplace=True)

        # some rows just got predictions, so we fill them with the first predicted value
        # Todo: add prediction time to index
        index_pred = df.loc[idx[index_first[0][0], :], :].index.to_list()
        for label in data.columns.to_list():
            i = 0
            while data[label].isna().all():
                helper = df.loc[idx[:, index_pred[i][1]], label]
                if helper.notna().any():
                    helper.index = helper.index.droplevel(level="time_pre")
                    data.loc[:, label] = helper
                i = i + 1
        data = data.loc[data.index[0] + pd.Timedelta(days=9):, :]

    return data


def load_mpc_stats(path):
    columns = ["iter_count", "success"]
    df = pd.read_csv(path, index_col=0, header=0).loc[:, columns]
    df.index = start_date + pd.to_timedelta(df.index, unit="s")
    return df.loc[df.index[0] + pd.Timedelta(days=9):, :]


def load_sim(path):
    with open("mapping_besmod.json") as mapping:
        column_mapping = json.load(mapping)
    column_mapping["name"] = "time"

    skip_rows = [0, 2]
    df = pd.read_csv(path, skiprows=skip_rows, header=0, dtype=float).rename(columns=column_mapping)
    df["time"] = start_date + pd.to_timedelta(df["time"], unit="s")
    df.set_index("time", inplace=True)

    for label in df.columns.to_list():
        if label.startswith("QTra_flow") or label.startswith("mdot"):
            df[label] = -df[label]

    return df.loc[df.index[0] + pd.Timedelta(days=1):, :]


def load_agent(path, prediction_series=True):
    if "stats" in path:
        return load_mpc_stats(path)
    if "mpc" in path:
        return load_mpc(path, prediction_series=prediction_series)
    if "sim" in path:
        return load_sim(path)
    raise ValueError("Could not match path with agent.")


def plot_intervals(df: pd.DataFrame, savepath: str, columns: list):
    set_plot_settings()
    colors = EBCColors.ebc_palette_sort_2
    timedelta = [df.index[-1] - df.index[0], pd.Timedelta(days=4)]
    for dt in timedelta:
        begin = df.index[0]
        b = begin
        end = df.index[-1]
        while b < end:
            e = b + dt
            _df = df.loc[b:e, :]
            for col in columns:
                # splited plot
                fig, ax = plt.subplots(nrows=len(col), ncols=1, sharex=True, figsize=(latex_textwidth, latex_texthight),
                                       layout="constrained")
                if isinstance(ax, plt.Axes):
                    ax = [ax]

                for i, _ax in enumerate(ax):
                    _ax.plot(_df.loc[:, col[i]], color=colors[i], label=col[i])
                    format_plot(df=_df, fig=fig, ax=_ax, column=col[i])
                    if all(col[0][0] == st[0] for st in col) and not any("Q" == st[0] for st in col):
                        y_min = _df.loc[:, col].min().min()
                        y_max = _df.loc[:, col].max().max()
                        _ax.set_ylim(y_min, y_max)

                save_plot(fig=fig, savepath=f"{savepath}/{str(dt.days).zfill(2)}/{str(b.day_of_year).zfill(3)}",
                          savename=f"{''.join(col)}_split")

                # one plot
                fig, ax = plt.subplots(layout="constrained")
                for c in col:
                    ax.plot(_df.loc[:, c], linewidth=1, label=c)
                format_plot(df=_df, fig=fig, ax=ax, column="")
                save_plot(fig=fig, savepath=f"{savepath}/{str(dt.days).zfill(2)}/{str(b.day_of_year).zfill(3)}",
                          savename=f"{''.join(col)}_one")

            b = e


def check_comfort(df: pd.DataFrame, path):
    set_plot_settings()
    with open(f"{path}/configs/mpc.json") as config:
        config = json.load(config)
    for param in config["modules"][1]["parameters"]:
        if param["name"] == "T_Air_ub":
            T_Air_ub = float(param["value"])
        elif param["name"] == "T_Air_lb":
            T_Air_lb = float(param["value"])

    df.loc[:, "T_Air_ub_check"] = 0
    df.loc[:, "T_Air_lb_check"] = 0

    mask_T_Air_ub = df["T_Air"] > T_Air_ub
    mask_T_Air_lb = df["T_Air"] < T_Air_lb

    df.loc[mask_T_Air_ub, "T_Air_ub_check"] = df.loc[mask_T_Air_ub, "T_Air"] - T_Air_ub
    df.loc[mask_T_Air_lb, "T_Air_lb_check"] = df.loc[mask_T_Air_lb, "T_Air"] - T_Air_lb

    savepath = f"{path}/evaluation/comfort"

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, layout="constrained")
    ax[0].plot(df.loc[:, "T_Air"], label="T_Air")
    ax[0].hlines(y=T_Air_ub, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey, label="T_Air_ub")
    ax[1].plot(df.loc[:, "T_Air_ub_check"], linewidth=1, color=EBCColors.dark_red, label="ub")
    for _ax in ax:
        format_plot(df=df, fig=fig, ax=_ax, column="")
    save_plot(fig=fig, savepath=savepath, savename="check_comfort_ub")

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, layout="constrained")
    ax[0].plot(df.loc[:, "T_Air"], label="T_Air")
    ax[0].hlines(y=T_Air_lb, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey, label="T_Air_lb")
    ax[1].plot(df.loc[:, "T_Air_lb_check"], linewidth=1, color=EBCColors.blue, label="lb")
    ax[1].hlines(y=0, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey)
    for _ax in ax:
        format_plot(df=df, fig=fig, ax=_ax, column="")
    save_plot(fig=fig, savepath=savepath, savename="check_comfort_lb")

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, layout="constrained")
    ax[0].plot(df.loc[:, "T_Air"], label="T_Air")
    ax[0].hlines(y=T_Air_ub, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey, label="T_Air_ub")
    ax[0].hlines(y=T_Air_lb, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey, label="T_Air_lb")
    ax[1].plot(df.loc[:, "T_Air_ub_check"], linewidth=1, color=EBCColors.dark_red, label="ub")
    ax[1].hlines(y=0, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey)
    ax[2].plot(df.loc[:, "T_Air_lb_check"], linewidth=1, color=EBCColors.blue, label="lb")
    ax[2].hlines(y=0, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey)
    for _ax in ax:
        format_plot(df=df, fig=fig, ax=_ax, column="")
    save_plot(fig=fig, savepath=savepath, savename="check_comfort_sep")

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, layout="constrained")
    ax[0].plot(df.loc[:, "T_Air"], label="T_Air")
    ax[0].hlines(y=T_Air_ub, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey, label="T_Air_ub")
    ax[0].hlines(y=T_Air_lb, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey, label="T_Air_lb")
    ax[1].plot(df.loc[:, "T_Air_ub_check"], linewidth=1, color=EBCColors.dark_red, label="ub")
    ax[1].plot(df.loc[:, "T_Air_lb_check"], linewidth=1, color=EBCColors.blue, label="lb")
    ax[1].hlines(y=0, xmin=df.index[0], xmax=df.index[-1], color=EBCColors.dark_grey)
    for _ax in ax:
        format_plot(df=df, fig=fig, ax=_ax, column="")
    save_plot(fig=fig, savepath=savepath, savename="check_comfort")

    os.makedirs(savepath, exist_ok=True)
    f = open(f"{savepath}/comfort.txt", "w")
    f.write(f"upper_max:\t\t{df.loc[:, 'T_Air_ub_check'].max()}\n")
    f.write("\n\n")
    f.write(f"lower_max:\t\t{df.loc[:, 'T_Air_lb_check'].min()}\n")
    f.write("\n\n")
    f.close()
    print(f"{savepath}/comfort.txt")

    (df.loc[mask_T_Air_ub, :].index.to_series().diff().dt.total_seconds() / (60 * 60)).to_csv(
        f"{savepath}/comfort_ub.csv")
    (df.loc[mask_T_Air_lb, :].index.to_series().diff().dt.total_seconds() / (60 * 60)).to_csv(
        f"{savepath}/comfort_lb.csv")


def get_dt(df: pd.DataFrame):
    df["dt"] = df.index.to_series().diff().dt.total_seconds() / (60 * 60)  # in h
    df.loc[df.index[0], "dt"] = df.loc[df.index[1], "dt"]
    df.loc[df.index[-1], "dt"] = 0
    return df


def analyse_electric_energy(df: pd.DataFrame, path):
    savepath = f"{path}/evaluation"
    if not "dt" in df.columns:
        df = get_dt(df)

    # electric energy
    df["W_el_feed_from_grid"] = df["P_el_feed_from_grid"] * df["dt"]
    W_el_feed_from_grid = df["W_el_feed_from_grid"].sum()
    df["W_el_feed_into_grid"] = df["P_el_feed_into_grid"] * df["dt"]
    W_el_feed_into_grid = df["W_el_feed_into_grid"].sum()

    df["P_el_cons"] = df["P_el_feed_from_grid"] - df["P_el_feed_into_grid"] + df["P_pv"]
    df["W_el_cons"] = df["P_el_cons"] * df["dt"]
    W_el_cons = df["W_el_cons"].sum()

    df["P_el_cons_dev_light"] = df["P_el_cons"] - df["P_el_hp"] - df["P_el_EleHea"]

    df["W_pv"] = df["P_pv"] * df["dt"]
    W_pv = df["W_pv"].sum()

    df["P_pv_feed"] = df["P_el_feed_into_grid"]
    df["W_pv_feed"] = df["P_pv_feed"] * df["dt"]
    W_pv_feed = df["W_pv_feed"].sum()

    df["P_pv_cons"] = df["P_pv"] - df["P_pv_feed"]
    df["W_pv_cons"] = df["P_pv_cons"] * df["dt"]
    W_pv_cons = df["W_pv_cons"].sum()
    SC = W_pv_cons / W_pv
    SSD = W_pv_cons / W_el_cons

    df["P_el_cons_wo_pv"] = df["P_el_cons"] - df["P_pv_cons"]

    os.makedirs(savepath, exist_ok=True)
    f = open(savepath + "/electric_energy.txt", "w")
    f.write(f"W_el_feed_from_grid:\t\t{W_el_feed_from_grid}\n")
    f.write(f"W_el_feed_into_grid:\t\t{W_el_feed_into_grid}\n")
    f.write(f"W_el_cons:\t\t{W_el_cons}\n")
    f.write(f"W_pv:\t\t{W_pv}\n")
    f.write(f"W_pv_cons:\t\t{W_pv_cons}\n")
    f.write(f"W_pv_feed:\t\t{W_pv_feed}\n")
    f.write(f"SC:\t\t{SC}\n")
    f.write(f"SSD:\t\t{SSD}\n")
    f.close()
    print(savepath + "/electric_energy.txt")

    return df


def get_model_parameters(path):
    from bes_rules.utils.modelica_parser import parse_modelica_record
    model_parameters_mo = parse_modelica_record(f"{path}/configs/Hamburg_BJ1994.mo")
    return model_parameters_mo


def analyse_thermal_energy(df: pd.DataFrame, path):
    savepath = f"{path}/evaluation"
    if not "dt" in df.columns:
        df = get_dt(df)

    mp_mo = get_model_parameters(path)

    VAir = mp_mo['VAir']
    air_rho = 1.25
    air_cp = 1000

    AInttot = sum(mp_mo['AInt']) if isinstance(mp_mo['AInt'], list) else mp_mo['AInt']
    AExttot = sum(mp_mo['AExt']) if isinstance(mp_mo['AExt'], list) else mp_mo['AExt']
    ARooftot = sum(mp_mo['ARoof']) if isinstance(mp_mo['ARoof'], list) else mp_mo['ARoof']
    AFloortot = sum(mp_mo['AFloor']) if isinstance(mp_mo['AFloor'], list) else mp_mo['AFloor']
    AWintot = sum(mp_mo['AWin']) if isinstance(mp_mo['AWin'], list) else mp_mo['AWin']

    hConInt = mp_mo['hConInt']
    hConExt = mp_mo['hConExt']
    hConRoof = mp_mo['hConRoof']
    hConFloor = mp_mo['hConFloor']
    hConWin = mp_mo['hConWin']

    k_int_air = hConInt * AInttot
    k_ext_air = hConExt * AExttot
    k_roof_air = hConRoof * ARooftot
    k_floor_air = hConFloor * AFloortot
    k_win_air = hConWin * AWintot

    # heatflows
    df["Qdot_Air_vent"] = df["ventRate_airExc"] * VAir * air_rho * air_cp * (df["Tamb"] - df["T_Air"]) * (1 / 3600)

    df["dT_Air_dt"] = (1 / (VAir * air_rho * air_cp)) * df["Qdot_Air"]

    df["Qdot_sol_rad"] = df["Q_RadSol_or_1"] + df["Q_RadSol_or_2"] + df["Q_RadSol_or_3"] + df["Q_RadSol_or_4"]

    return df


def plot_sim(df: pd.DataFrame, path):
    set_plot_settings()

    savepath = f"{path}/evaluation"
    columns = [
        ["T_Air", "TBufSet", "yValSet"],
        ["QTra_flow", "TBufSet", "yValSet"],
        ["T_Air", "TBufSet", "P_pv", "Tamb"],
        ["T_Air", "QTra_flow", "P_el_hp", "P_pv"],
        ["T_Air", "QTra_flow", "P_el_hp", "P_pv", "Tamb"],
        ["T_Air", "Tamb", "QTra_flow", "P_el_hp", "P_pv"],
        ["T_Air", "Tamb", "Qdot_sol_rad", "QTra_flow", "P_el_hp", "P_pv"],
        ["TBufSet", "THeaCur"],
        ["TBufSet", "TBuf", "TTraSup"],
        ["T_Air", "TBufSet", "yValSet", "P_pv"],
        ["T_Air", "P_pv"],
        ["T_Air", "QTra_flow"],
        ["T_Air", "QTra_flow"],
        ["T_Air", "T_IntWall_sur", "T_ExtWall_sur"],
        ["T_Air", "T_IntWall", "T_IntWall_sur"],
        ["T_Air", "T_ExtWall", "T_ExtWall_sur", "T_ExtWall_pre"],
        ["T_Air", "T_IntWall", "T_ExtWall", "T_Roof", "T_Floor"],
        ["QTra_flow", "Qdot_sol_rad"],
        ["Qdot_Air", "QTra_flow", "Qdot_Air_int", "Qdot_Air_ext"],
        ["Qdot_Air_int", "Qdot_IntWall"],
        ["Qdot_Air", "QTraCon_flow", "Qdot_Air_int", "Qdot_Air_ext"],
        ["Qdot_Air", "QTraCon_flow", "Qdot_Air_int", "Qdot_Air_ext", "Qdot_Air_vent"],
        ["ventRate_airExc"],
        ["Qdot_Air_int"],
        ["Qdot_Air"],
        ["dT_Air_dt"],
        ["P_el_feed_from_grid", "P_el_feed_into_grid"],
        ["P_el_cons", "P_el_cons_dev_light", "P_el_hp", "P_el_EleHea"],
        ["P_el_cons", "P_pv_cons", "P_el_cons_wo_pv"],
        ["P_el_cons", "P_pv_cons"],
        ["P_el_cons", "P_pv"],
        ["P_pv", "P_pv_cons", "P_pv_feed"],
        ["P_pv", "P_el_hp", "QTra_flow"],
        ["P_pv", "P_el_hp"],
    ]
    plot_intervals(df=df, savepath=savepath, columns=columns)


def evaluate(path):
    df = load_agent(path=f"{path}/sim_agent.csv")
    df = analyse_electric_energy(df, path)
    df = analyse_thermal_energy(df, path)
    check_comfort(df, path)
    plot_sim(df, path)


def get_figsize():
    plot_witdh = latex_textwidth
    plot_higth = latex_texthight / 3 - 1  # latex_texthight/2 - 1
    return plot_witdh, plot_higth


def set_plot_settings():
    plt.rcParams["figure.figsize"] = get_figsize()
    plt.rcParams["figure.constrained_layout.use"] = True

    plt.rcParams["axes.prop_cycle"] = cycler(color=EBCColors.ebc_palette_sort_2)

    plt.rcParams["axes.formatter.use_locale"] = True

    plt.rcParams["figure.dpi"] = 450

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11

    plt.rcParams["text.usetex"] = True
    plt.rcParams["pgf.preamble"] = "\n".join([
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath}"
    ])
    plt.rcParams["pgf.rcfonts"] = False
    plt.rcParams["pgf.texsystem"] = "xelatex"

    plt.rcParams["date.autoformatter.month"] = "%B"
    plt.rcParams["date.autoformatter.day"] = "%d.%m"
    plt.rcParams["date.autoformatter.hour"] = "%H:%M"


def latex_formater(text):
    if "_" in text:
        text = text.replace("_", " ")
    return text


def format_plot(df: pd.DataFrame, fig: plt.Figure, ax: plt.Axes, column):
    ylabel = column
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()

    if len(df.index) > 1:
        ax.set_xlim(df.index[0], df.index[-1])

    ticks = False
    if ticks:
        dt_total = df.index[-1] - df.index[0]
        dt_total_days = dt_total.days
        dt_total_seconds = dt_total.total_seconds()
        if dt_total_days <= 1.1:
            minor_loc = dates.HourLocator(interval=6)
            major_loc = dates.DayLocator(interval=1)
            minor_for = dates.DateFormatter("%H:%M")
            major_for = dates.DateFormatter("%d.%m")
        else:
            if dt_total_days <= 32.1:
                k = min(dt_total_days, 4)
                n = k
                minor_for = dates.DateFormatter("%d.%m")
                major_for = dates.DateFormatter("%d.%m")
            else:
                n = 6
                k = 1
                minor_for = dates.DateFormatter("%d.%m")
                major_for = dates.DateFormatter("%d.%m.%Y")

            dt_minor = round(dt_total_seconds / n)
            dt_major = round(dt_total_seconds / k)
            minor_loc = dates.SecondLocator(interval=dt_minor)
            major_loc = dates.SecondLocator(interval=dt_major)

        ax.xaxis.set_minor_locator(minor_loc)
        ax.xaxis.set_minor_formatter(minor_for)
        ax.xaxis.set_major_locator(major_loc)
        ax.xaxis.set_major_formatter(major_for)


def save_plot(fig, savepath, savename):
    if len(savename) > 100:
        savename = savename[:48] + savename[-6:]
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(fname=f"{savepath}/{savename}.png", format="png")
    if pgf:
        fig.savefig(fname=f"{savepath}/{savename}.pgf", backend="pgf")
    print(f"{savepath}/{savename}.png")
    plt.close("all")


def plot_sim(path):
    set_plot_settings()
    df = load_agent(path=f"{path}/sim_agent.csv")

    savepath = f"{path}/plots_sim"
    col_to_plot = df.columns.to_list()
    for c in col_to_plot:
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(df.loc[:, c], color=EBCColors.dark_grey, label="Sim")

        format_plot(df=df, fig=fig, ax=ax, column=c)
        save_plot(fig=fig, savepath=savepath, savename=c)


def plot_mpc(path):
    set_plot_settings()
    df = load_agent(path=f"{path}/mpc_agent.csv")
    df.pop(0)
    df_mod = load_agent(path=f"{path}/mpc_agent.csv", prediction_series=False)

    savepath = f"{path}/plots_mpc"
    col_to_plot = df[0].columns.to_list()
    for c in col_to_plot:
        fig, ax = plt.subplots(layout="constrained")
        number_of_predictions = len(df)
        if not isinstance(c, list):
            for i, _df in enumerate(df):
                progress = i / number_of_predictions
                prediction_color = interpolate_colors(
                    progress=progress,
                    colors=[EBCColors.red, EBCColors.grey],
                )
                ax.plot(_df.loc[:, c], linewidth=0.5, color=prediction_color)
        ax.plot(df_mod.loc[:, c], color=EBCColors.dark_red, label="MPC")

        format_plot(df=df_mod, fig=fig, ax=ax, column=c)
        save_plot(fig=fig, savepath=savepath, savename=c)


def plot_mpc_stats(path):
    set_plot_settings()
    df = load_agent(f"{path}/stats_mpc_agent.csv")

    savepath = f"{path}/stats"
    col_to_plot = df.columns.to_list()
    for c in col_to_plot:
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(df.loc[:, c], color=EBCColors.dark_red, label="MPC")

        format_plot(df=df, fig=fig, ax=ax, column=c)
        save_plot(fig=fig, savepath=savepath, savename=c)


def plot_comparison(path):
    set_plot_settings()
    df_mpc = load_agent(path=f"{path}/mpc_agent.csv")
    df_mpc.pop(0)
    df_sim = load_agent(path=f"{path}/sim_agent.csv")

    col_mpc = df_mpc[0].columns.to_list()
    col_sim = df_sim.columns.to_list()
    col_comb = list(set(col_mpc) & set(col_sim))

    # plot
    savepath = f"{path}/plots_comparison"
    for c in col_comb:
        fig, ax = plt.subplots(layout="constrained")

        number_of_predictions = len(df_mpc)
        for i, _df in enumerate(df_mpc):
            progress = i / number_of_predictions
            prediction_color = interpolate_colors(
                progress=progress,
                colors=[EBCColors.red, EBCColors.grey],
            )
            ax.plot(_df.loc[:, c], linewidth=0.5, color=prediction_color)
        ax.plot(df_sim.loc[:, c], color=EBCColors.dark_grey, label="Sim")

        format_plot(df=df_sim, fig=fig, ax=ax, column=c)
        save_plot(fig=fig, savepath=savepath, savename=c)


def plot_results(save_path: Path):
    plot_mpc(path=save_path)
    plot_comparison(path=save_path)
    plot_sim(path=save_path)
    plot_mpc_stats(path=save_path)
    evaluate(path=save_path)


def load_and_show_dashboard(save_path: Path):
    mpc_stats = pd.read_csv(save_path.joinpath("stats_mpc_agent.csv"))
    mpc_results = pd.read_hdf(save_path.joinpath("results_mpc_module.csv"), key="mpc")
    show_dashboard(data=mpc_results, stats=mpc_stats, scale="days")
