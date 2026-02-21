import itertools

import pandas as pd

import logging
from bes_rules.configs import StudyConfig, PlotConfig

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", rc={"font.size": 11,"axes.titlesize":11,"axes.labelsize":11})

from bes_rules.plotting.ecos2023 import pv_influence
from bes_rules.plotting import utils


def plot_3_appendix_controls():
    pv_influence.plot_control_heat_map(
        base_path=Path(r"D:\fwu-jre"),
        study_name="BESCtrlOptTest",
        z_variables=["costs_operating", "SCOP_Sys", "self_sufficiency_degree", "self_consumption_rate",
                     #"hp_coverage_rate",
                     #"outputs.control.PVHys_KPI.numSwi",
                     #"outputs.hydraulic.gen.eleHea.numSwi",
                     #"outputs.control.DHWOverheat_KPI.numSwi",
                     #"outputs.control.BufOverheat_KPI.numSwi",
                     ],
        minimize=[True, False, False, False],
        show=True,
        x_variable="parameterStudy.BufOverheatdT",
        y_variable="parameterStudy.DHWOverheatTemp",
        cmap="rocket",
        save_name="plots_controls"
    )


def plot_3_2_design_and_control():

    base_pars = dict(
        base_path=Path(r"D:\fwu-jre"),
        study_name="BESPVDesOptAllSingleSide",
        show=True,
        cmap="rocket",
        y_variable="parameterStudy.TBiv",
        x_variable="parameterStudy.VPerQFlow",
    )
    pv_influence.plot_control_heat_map(
        save_name=f"3_2_design",
        z_variables=["self_sufficiency_degree", "self_consumption_rate", "outputs.hydraulic.gen.eleHea.totOnTim", "SCOP_Sys",],
        minimize=[False, False, True, False],
        constant_optimization_variables={
            "parameterStudy.f_design": 1,
        },
        **base_pars
    )
    pv_influence.plot_control_heat_map(
        save_name=f"3_2_design_shareOfPEle",
        z_variables=["costs_total", "self_sufficiency_degree", "self_consumption_rate", "SCOP_Sys"],
        minimize=[True, False, False, False],
        constant_optimization_variables={
            "parameterStudy.f_design": 1,
        },
        argopt_plot_variable="parameterStudy.ShareOfPEleNominal",
        **base_pars
    )


def plot_3_1_influence_pv(side="Single"):
    base_pars = dict(
        base_path=Path(r"D:\fwu-jre"),
        study_name_ctrl=f"BESPVDesOptAll{side}Side",
        study_name_no_ctrl="BESNoCtrl",
        z_variable="costs_total",
        minimize=True,
        show=True,
        cmap="rocket",
        y_variable="parameterStudy.TBiv",
        x_variable="parameterStudy.VPerQFlow",
    )
    pv_influence.plot_pv_influence(
        save_name=f"3_1_total_pv_100_{side}",
        f_design=1,
        **base_pars
    )
    #pv_influence.plot_pv_influence(
    #    save_name=f"3_1_total_pv_50_{side}",
    #    f_design=0.5,
    #    **base_pars
    #)


def plot_mpp_over_thres(specific_volume_factor):
    base_path = Path(r"D:\fwu-jre")
    study_two_side = "BESPVDesOptAllTwoSide"
    study_single_side = "BESPVDesOptAllSingleSide"
    x_variable = "PElePVMPP"
    y_variable = "PElePVMPP"
    plot_config = PlotConfig.load_default()

    config_single_side = StudyConfig.from_json(base_path.joinpath(study_single_side, "study_config.json"))
    config_two_side = StudyConfig.from_json(base_path.joinpath(study_two_side, "study_config.json"))

    dfs_sin, input_variations = utils.get_all_results_from_config(config_single_side)
    dfs_two, _ = utils.get_all_results_from_config(config_two_side)
    dfs = []
    for df_sin, df_two, input_config in zip(dfs_sin, dfs_two, input_variations):
        df = merge_single_and_two_side(df_sin=df_sin, df_two=df_two)
        df.loc[:, "weather"] = input_config.weather.get_name()
        dfs.append(df)

    df = pd.concat(dfs)
    df.index = range(len(df))
    df = plot_config.scale_df(df=df)
    opt_var = "parameterStudy.ShareOfPEleNominal"

    var_names = [
        "parameterStudy.TBiv",
        "parameterStudy.VPerQFlow",
        "PV_Direction",
        "weather",
        "parameterStudy.f_design"
    ]
    vars_to_extract = [
        opt_var,
        "PElePVMPP",
        "outputs.control.PEle_nominal",
        "parameterStudy.VPerQFlow",
        "hydraulic.distribution.parStoBuf.V"
    ]
    metric = "costs_total"
    optima = get_optima_subset(df=df, var_names=var_names, metric=metric, vars_to_extract=vars_to_extract)

    df_plot = pd.DataFrame(columns=["thres", "f", "V_per_mpp", "V_per_Q"])

    for idx, opt in enumerate(optima):
        _f = _get_f(opt)
        thr = _get_thr(opt)
        vols_per_mpp = _get_v_per_mpp(opt)
        if len(opt) == 4:
            continue   # No influence at all
        else:
            df_plot.loc[idx, "f"] = _f.iloc[thr.argmax()]
            df_plot.loc[idx, "V_per_mpp"] = vols_per_mpp.iloc[thr.argmax()]
            df_plot.loc[idx, "V_per_Q"] = opt.iloc[thr.argmax()]["parameterStudy.VPerQFlow"]
            df_plot.loc[idx, "thres"] = thr.max()

    plt.figure()
    plt.gca().set_xlabel("$f_\mathrm{MPP,HP}$")
    plt.gca().set_ylabel("$f_\mathrm{HP,PV}$")
    plt.scatter(df_plot["f"], df_plot["thres"], c=df_plot[specific_volume_factor], cmap="rocket")
    if specific_volume_factor == "V_per_mpp":
        plt.gcf().suptitle("$V_\mathrm{P_\mathrm{el,PV,MPP}}$")
    else:
        plt.gcf().suptitle("$V_\mathrm{\dot{Q}_\mathrm{Bui}}$")
    plt.colorbar()

    plt.gcf().tight_layout()
    plt.gcf().savefig(base_path.joinpath("3_3_mpp_over_threshold.pdf"))
    plt.show()


def merge_single_and_two_side(df_sin, df_two):
    df_sin.loc[:, "PV_Direction"] = "south"
    df_sin.loc[:, "PElePVMPP"] = df_sin.loc[:, "outputs.electrical.gen.PElePVMPP"]

    df_two.loc[:, "PV_Direction"] = "south-north"
    df_two.loc[:, "PElePVMPP"] = df_two.loc[:, "outputs.electrical.gen.PElePVMPP"] * 2
    df = pd.concat([df_sin, df_two])
    df.index = range(len(df))
    return df


def get_optima_subset(df: pd.DataFrame, var_names: list, metric: str, vars_to_extract: list):
    optima = []
    uniques = [df.loc[:, var].unique() for var in var_names]
    for var_unique in itertools.product(*uniques):
        only_thre = df.copy()
        for var, var_val in zip(var_names, var_unique):
            only_thre = only_thre.loc[only_thre.loc[:, var] == var_val]
        optimum = only_thre.loc[only_thre.loc[:, metric] == only_thre.loc[:, metric].min(), vars_to_extract]
        if len(optimum) > 0:
            optima.append(optimum)
    return optima


def _get_f(row):
    return 1000 * row.loc[:, "PElePVMPP"] / row.loc[:, "outputs.control.PEle_nominal"]


def _get_thr(row):
    return row.loc[:, "parameterStudy.ShareOfPEleNominal"]


def _get_v_per_mpp(row):
    return row.loc[:, "hydraulic.distribution.parStoBuf.V"] * 1000 / row.loc[:, "PElePVMPP"]


if __name__ == '__main__':
    plt.rcParams.update(
        {"figure.figsize": [17 / 2.54, 24 / 2.54 / 3],
         "font.size": 11,
         "figure.dpi": 250,
         "font.family": "Arial"
         }
    )

    logging.basicConfig(level="INFO")
    plot_3_1_influence_pv(side="Single")

    #plot_mpp_over_thres(specific_volume_factor="V_per_Q")
    plot_3_2_design_and_control()
