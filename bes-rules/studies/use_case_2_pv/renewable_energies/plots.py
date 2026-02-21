import logging

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from bes_rules.input_analysis.plotting import _get_data_for_boundary_type
from bes_rules.input_analysis.pv import load_and_filter_results
from bes_rules.plotting import EBCColors
from bes_rules.plotting.boxplots import plot_box

from bes_rules.plotting.ecos2023 import pv_influence
from bes_rules import RESULTS_FOLDER

BASE_PATH = Path(r"R:\_Dissertationen\fwu\03_Paper\RE")
BASE_PATH = RESULTS_FOLDER.joinpath("ZCBE_Journal")

save_name = {"NoElec": "NoInfluence", "": "Influence"}


def plot_3_2_design_and_control():
    for elec in ["NoElec", ""]:
        print("Case ", elec)
        base_pars = dict(
            base_path=BASE_PATH,
            save_path=BASE_PATH.joinpath("plots_control"),
            study_name=f"BESCtrl{elec}",
            show=False,
            cmap="rocket",
            y_variable="parameterStudy.TBiv",
            x_variable="parameterStudy.VPerQFlow",
        )
        pv_influence.plot_control_heat_map(
            save_name=f"f=opt_{save_name[elec]}",
            z_variables={
                "self_sufficiency_degree": False,
                "self_consumption_rate": False,
                #"outputs.hydraulic.gen.eleHea.totOnTim": True,
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
        pv_influence.plot_control_heat_map(
            save_name=f"fHPPVOpt_for_TBiv_over_V_{save_name[elec]}",
            z_variables={
                "costs_total": True,
                "self_sufficiency_degree": False,
                "self_consumption_rate": False,
                "SCOP_Sys": False,
                "outputs.electrical.dis.PEleLoa.integral": True,
                "outputs.electrical.dis.PEleGen.integral": False
            },
            constant_optimization_variables={
                "parameterStudy.f_design": 1,
            },
            argopt_plot_variable="parameterStudy.ShareOfPEleNominal",
            **base_pars
        )


def plot_3_2_design_and_control_share():
    for elec in ["NoElec", ""]:
        print("Case ", elec)
        base_pars = dict(
            base_path=BASE_PATH,
            save_path=BASE_PATH.joinpath("plots_storage"),
            study_name=f"BESCtrl{elec}",
            show=False,
            cmap="rocket",
            z_variables={
                "self_sufficiency_degree": False,
                "self_consumption_rate": False,
                # "outputs.hydraulic.gen.eleHea.totOnTim": True,
                "SCOP_Sys": False,
                "outputs.electrical.dis.PEleLoa.integral": True,
                "outputs.electrical.dis.PEleGen.integral": False,
                "outputs.hydraulic.gen.heaPum.numSwi": True,
            },
            y_variable="parameterStudy.TBiv",
            x_variable="parameterStudy.ShareOfPEleNominal",
        )
        import numpy as np
        pv_influence.plot_control_heat_map(
            save_name=f"TBiv_over_fHPPV_at_VOpt_{save_name[elec]}",
            constant_optimization_variables={
                "parameterStudy.f_design": 1
            },
            **base_pars
        )
        continue
        for sto in np.linspace(5, 150, 8):
            sto = int(sto)
            pv_influence.plot_control_heat_map(
                save_name=f"V={sto}_{elec}_share",
                constant_optimization_variables={
                    "parameterStudy.f_design": 1,
                    "parameterStudy.VPerQFlow": sto
                },
                **base_pars
            )


def plot_3_1_influence_pv():
    base_pars = dict(
        base_path=BASE_PATH,
        z_variable="costs_total",
        minimize=True,
        show=False,
        cmap="rocket",
        y_variable="parameterStudy.TBiv",
        x_variable="parameterStudy.VPerQFlow",
    )
    for elec in ["NoElec", ""]:
        print("Case ", elec)
        pv_influence.plot_pv_influence(
            save_name=save_name[elec],
            study_name_ctrl=f"BESCtrl{elec}",
            study_name_no_ctrl=f"BESNoCtrl{elec}",
            f_design=1,
            **base_pars
        )


def plot_all_results():
    pv_influence.create_table_with_all_optima(
        base_path=BASE_PATH,
        study_names=["BESCtrl", "BESNoCtrl", "BESNoCtrl"],
        study_alias=["Control", "No Control", "No PV"],
        optimization_variables=[
            "parameterStudy.TBiv",
            "parameterStudy.VPerQFlow",
            "parameterStudy.ShareOfPEleNominal"
        ],
        objective_functions={
            "costs_total": "min",
            "self_sufficiency_degree": "max",
            # "self_consumption_rate": "max",
            "SCOP_Sys": "max"
        }
    )


def change_rc_params_for_paper():
    sns.set_context("paper", rc={"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})
    plt.rcParams.update(
        {
            "font.size": 11,
            "figure.dpi": 250,
            #"text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ['Segoe UI Symbol', 'simHei', 'Arial', 'sans-serif'],
            # "backend": "TkAgg",
        }
    )


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    change_rc_params_for_paper()
    plot_3_1_influence_pv()
    plot_3_2_design_and_control_share()
    plot_3_2_design_and_control()
    plot_all_results()


def plot_for_paper(save_path: Path, years_to_skip: list):
    metric = "self_sufficiency_degree"
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15.5 / 2.6 * 1.2, 15.5 / 2.6 / 2 * 2))

    for interval, axes in zip(["1H", "1D"], ax):
        df_results = load_and_filter_results(save_path, interval, years_to_skip=years_to_skip)
        _type_data = {}
        for _type_name in [
            "case",
            "building",
        ]:
            _type_data.update(
                _get_data_for_boundary_type(
                    df_results=df_results,
                    metric=metric,
                    boundary_type=_type_name,
                    years_to_skip=years_to_skip
                )
            )

        df = pd.DataFrame(_type_data)
        palette = [EBCColors.light_grey] * 6 + [EBCColors.dark_grey] * 24
        plot_box(df=df, orient='v', axes=axes, color_palette=palette)

        renames = {
            "case": "(a) PV Installation",
            "building": "(b) Building Envelope",
            "weather": "Weather",
            "self_sufficiency_degree": "$SSD$ in %"
        }
        axes.set_ylabel(renames.get(metric, metric))
        axes.set_ylim([0, 100])
        interval_s = "Hour" if interval == "1H" else "Day"
        axes.set_title(f"Sampling Interval: {interval_s}")
    ax[1].tick_params(axis="x", rotation=90)
    import matplotlib.patches as mpatches

    bui_patch = mpatches.Patch(color=EBCColors.dark_grey, label='Building Envelope')
    pv_patch = mpatches.Patch(color=EBCColors.light_grey, label='PV Installation')
    ax[0].legend(handles=[pv_patch, bui_patch], ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"{metric}_building_and_pv.png"))
    fig.savefig(save_path.joinpath(f"{metric}_building_and_pv.pdf"))
    plt.close("all")
