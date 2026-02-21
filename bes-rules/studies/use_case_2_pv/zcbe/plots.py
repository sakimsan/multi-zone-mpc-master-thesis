import logging

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FONT_SIZE = 11

sns.set_context("paper", rc={"font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE})

from bes_rules.plotting.ecos2023 import pv_influence
from bes_rules.plotting.design_space import plot_scatter_for_x_over_multiple_y

NEW_RC_PARAMS = {
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "figure.dpi": 250,
    # "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ['Segoe UI Symbol', 'simHei', 'Arial', 'sans-serif']
    # "backend": "TkAgg",
}



def plot_metric_for_control(study: str):
    base_pars = dict(
        base_path=Path(r"D:\zcbe"),
        save_path=Path(r"D:\zcbe\plots_control"),
        study_name=f"BES_{study}",
        show=False,
        cmap="rocket",
        y_variable="parameterStudy.TBiv",
        x_variable="parameterStudy.VPerQFlow",
    )
    pv_influence.plot_control_heat_map(
        save_name=f"5_metrics_{study}",
        z_variables={
            "self_sufficiency_degree": False,
            "self_consumption_rate": False,
            # "outputs.hydraulic.gen.eleHea.totOnTim": True,
            "SCOP_Sys": False,
            "outputs.electrical.dis.PEleLoa.integral": True,
            "outputs.electrical.dis.PEleGen.integral": False,
            "outputs.hydraulic.gen.heaPum.numSwi": True,
        },
        constant_optimization_variables={
            "parameterStudy.f_design": 1,
        },
        **base_pars
    )


def plot_influence_control():
    custom_plot_config = {"variables": {
        "parameterStudy.VPerQFlow": {
            "factor": 1 / 5.5,
            "offset": 0,
            "unit": "l",
            "label": "$V$"
        }}, "rcParams": NEW_RC_PARAMS}

    base_pars = dict(
        base_path=Path(r"D:\zcbe"),
        z_variable="costs_total",
        minimize=True,
        show=False,
        cmap="rocket",
        y_variable="parameterStudy.TBiv",
        x_variable="parameterStudy.VPerQFlow",
        custom_plot_config=custom_plot_config
    )
    pv_influence.plot_control_influence(
        save_name=f"4_MPC_vs_NoCtrl",
        study_name_ctrl_1=f"BES_No_RBC",
        study_name_ctrl_2=f"BES_MPC_NPL_OE",
        alias_1="NoRBC",
        alias_2="MPC",
        f_design=1,
        **base_pars
    )
    pv_influence.plot_control_influence(
        save_name=f"4_PRBC_vs_NoCtrl",
        study_name_ctrl_1=f"BES_No_RBC",
        study_name_ctrl_2=f"BES_RBPC",
        alias_1="NoRBC",
        alias_2="PRBC",
        f_design=1,
        **base_pars
    )
    pv_influence.plot_control_influence(
        save_name=f"4_PRBC_vs_MPC",
        study_name_ctrl_1=f"BES_MPC_NPL_OE",
        study_name_ctrl_2=f"BES_RBPC",
        alias_1="MPC",
        alias_2="PRBC",
        f_design=1,
        **base_pars
    )
    pv_influence.plot_control_influence(
        save_name=f"4_PRBC_vs_RBC",
        study_name_ctrl_1=f"BES_RBC",
        study_name_ctrl_2=f"BES_RBPC",
        alias_1="RBC",
        alias_2="PRBC",
        f_design=1,
        **base_pars
    )
    pv_influence.plot_control_influence(
        save_name=f"4_RBC_vs_No_RBC",
        study_name_ctrl_1=f"BES_No_RBC",
        study_name_ctrl_2=f"BES_RBC",
        alias_1="NoRBC",
        alias_2="RBC",
        f_design=1,
        **base_pars
    )
    pv_influence.plot_control_influence(
        save_name=f"4_MPC_vs_NPL",
        study_name_ctrl_1=f"BES_MPC_NPL_OE",
        study_name_ctrl_2=f"BES_MPC",
        alias_1="NPL",
        alias_2="MPC",
        f_design=1,
        **base_pars
    )


def _get_config(study):
    from bes_rules.configs import StudyConfig
    base_path = Path(r"D:\zcbe").joinpath(f"BES_{study}")
    return StudyConfig.from_json(base_path.joinpath("study_config.json"))


def plot_detailed_results(study: str):
    config = _get_config(study)
    plot_scatter_for_x_over_multiple_y(
        study_config=config,
        save_path=base_path.joinpath("plots_detailed"),
        x_variable="parameterStudy.TBiv",
        y_variables=[
            "costs_total",
            "outputs.building.dTComHea[1]",
            "outputs.hydraulic.gen.PEleEleHea.integral",
            "outputs.hydraulic.gen.PEleHeaPum.integral",
            "outputs.building.eneBal[1].traGain.integral",
            "outputs.DHW.Q_flow.integral",
        ]
    )


def compare_supervisory_control_performance(study_1="MPC", study_2="RBPC"):
    import pandas as pd
    import matplotlib.pyplot as plt
    storage_small_index = 0
    storage_big_index = 2
    input_name = "TRY2015_536322100078_Jahr_B1994_retrofit_SingleDwelling_M_South"
    res_name = "DesignOptimizationResults"
    base_path = Path(r"D:\zcbe")
    config_1 = base_path.joinpath(f"BES_{study_1}", res_name, input_name)
    config_2 = base_path.joinpath(f"BES_{study_2}", res_name, input_name)
    set_name = "TBufSet"
    mea_name = "outputs.hydraulic.disCtrl.TStoBufTopMea"
    df_1_small = pd.read_excel(config_1.joinpath(f"Design_{storage_small_index}.xlsx"))
    df_2_small = pd.read_excel(config_2.joinpath(f"Design_{storage_small_index}.xlsx"))
    df_1_big = pd.read_excel(config_1.joinpath(f"Design_{storage_big_index}.xlsx"))
    df_2_big = pd.read_excel(config_2.joinpath(f"Design_{storage_big_index}.xlsx"))
    TOda_name = "outputs.weather.TDryBul"

    def setup_plots(s1, s2, ylabel_format, x_label):
        from bes_rules.plotting.utils import get_figure_size
        f, a = plt.subplots(2, 2, sharex=True, sharey=True,
                            figsize=get_figure_size(n_columns=2, height_factor=2))
        a[0, 0].set_title(s1)
        a[0, 1].set_title(s2)
        a[0, 0].set_ylabel(ylabel_format.replace(",}", ",small}"))
        a[1, 0].set_ylabel(ylabel_format.replace(",}", ",big}"))
        a[1, 0].set_xlabel(x_label)
        a[1, 1].set_xlabel(x_label)
        return f, a

    def legend_and_save_plot(f, a, legend: bool, save_path: Path):
        if legend:
            a[0, 0].legend(ncol=2)
        f.savefig(save_path)

    fig, ax = setup_plots(study_1, study_2, ylabel_format="$T_\mathrm{Sto,}$ in °C", x_label="Time")
    ax[0, 0].plot(df_1_small[set_name] - 273.15, label="set", color="red")
    ax[0, 0].plot(df_1_small[mea_name] - 273.15, label="mea", color="blue", linestyle="--")
    ax[0, 1].plot(df_2_small[set_name] - 273.15, label="set", color="red")
    ax[0, 1].plot(df_2_small[mea_name] - 273.15, label="mea", color="blue", linestyle="--")
    ax[1, 0].plot(df_1_big[set_name] - 273.15, label="set", color="red")
    ax[1, 0].plot(df_1_big[mea_name] - 273.15, label="mea", color="blue", linestyle="--")
    ax[1, 1].plot(df_2_big[set_name] - 273.15, label="set", color="red")
    ax[1, 1].plot(df_2_big[mea_name] - 273.15, label="mea", color="blue", linestyle="--")
    legend_and_save_plot(fig, ax, legend=True, save_path=base_path.joinpath("BigSmall_set_mea.png"))

    dT_1_small = df_1_small[set_name] - df_1_small[mea_name]
    dT_2_small = df_2_small[set_name] - df_2_small[mea_name]
    dT_1_big = df_1_big[set_name] - df_1_big[mea_name]
    dT_2_big = df_2_big[set_name] - df_2_big[mea_name]

    # Difference
    fig, ax = setup_plots(study_1, study_2,
                          ylabel_format="$\Delta T_\mathrm{Sto,}$ in K",
                          x_label="Time")
    ax[0, 0].plot(dT_1_small, color="red")
    ax[0, 1].plot(dT_2_small, color="red")
    ax[1, 0].plot(dT_1_big, color="red")
    ax[1, 1].plot(dT_2_big, color="red")
    legend_and_save_plot(fig, ax, legend=False, save_path=base_path.joinpath("BigSmall_dT_time.png"))

    fig, ax = setup_plots(study_1, study_2,
                          ylabel_format="$\Delta T_\mathrm{Sto,}$ in K",
                          x_label="$T_\mathrm{Oda}$ in °C")
    ax[0, 0].scatter(df_1_small[TOda_name] - 273.15, dT_1_small, color="red", s=1)
    ax[0, 1].scatter(df_2_small[TOda_name] - 273.15, dT_2_small, color="red", s=1)
    ax[1, 0].scatter(df_1_big[TOda_name] - 273.15, dT_1_big, color="red", s=1)
    ax[1, 1].scatter(df_2_big[TOda_name] - 273.15, dT_2_big, color="red", s=1)
    legend_and_save_plot(fig, ax, legend=False, save_path=base_path.joinpath("BigSmall_dT_scatter.png"))


def plot_detailed_results(study: str):
    from bes_rules.configs import StudyConfig
    from bes_rules.plotting.design_space import plot_scatter_for_x_over_multiple_y
    base_path = Path(r"D:\zcbe").joinpath(f"BES_{study}")
    CONFIG = StudyConfig.from_json(base_path.joinpath("study_config.json"))
    plot_scatter_for_x_over_multiple_y(
        study_config=CONFIG,
        save_path=base_path.joinpath("plots_detailed"),
        x_variable="parameterStudy.TBiv",
        y_variables=[
            "costs_total",
            "outputs.building.dTComHea[1]",
            "outputs.hydraulic.gen.PEleEleHea.integral",
            "outputs.hydraulic.gen.PEleHeaPum.integral",
            "outputs.building.eneBal[1].traGain.integral",
            "outputs.DHW.Q_flow.integral",
        ]
    )


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    plt.rcParams.update(NEW_RC_PARAMS)

    # compare_supervisory_control_performance()
    plot_influence_control()
    # plot_metric_for_control(study="MPC")
    # plot_metric_for_control(study="No_RBC")
    # plot_metric_for_control(study="RBC")
    # plot_metric_for_control(study="RBPC")
    # plot_detailed_results(study="RBC")
    # plot_detailed_results(study="No_RBC")
    # plot_detailed_results(study="RBC")
    # plot_detailed_results("MPC")
