"""
Plots for the design space
"""
import os.path
import logging
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from bes_rules.configs import PlotConfig, StudyConfig
from bes_rules.objectives.annuity import Annuity, AnnuityMapping
from bes_rules.plotting import utils
from bes_rules.simulation_based_optimization import SurrogateBuilder
from bes_rules.utils.pareto import get_pareto_efficient_points_for_df

logger = logging.getLogger(__name__)


def plot_scatter_for_x_over_multiple_y(
        study_config: StudyConfig,
        x_variable: str,
        y_variables: List[str],
        plot_config: PlotConfig = None,
        save_path: Path = None,
        show=False,
):
    dfs, input_configs = utils.get_all_results_from_config(study_config=study_config)
    plot_config = utils.load_plot_config(plot_config=plot_config)

    for df, input_config in zip(dfs, input_configs):
        df = plot_config.scale_df(df)
        fig, axes = utils.create_plots(
            plot_config=plot_config,
            x_variables=[x_variable],
            y_variables=y_variables
        )
        for _y_variable, _ax in zip(y_variables, axes[:, 0]):
            _ax.scatter(df.loc[:, x_variable], df.loc[:, _y_variable])

        axes[0, 0].legend(bbox_to_anchor=(0, 1), loc="lower left")
        utils.save(
            fig=fig, axes=axes,
            save_path=save_path.joinpath(input_config.get_name()),
            show=show, with_legend=False, file_endings=["png"]
        )


def plot_pv_scatter(
        df: pd.DataFrame,
        plot_config: PlotConfig,
        x_variable: str,
        y_variables: Union[List[str], str],
        save_path: Path = None,
        show=False,
):
    if isinstance(y_variables, str):
        y_variables = [y_variables]

    pv_area = {
        0: {"label": "Kein PV", "color": "red", "marker": "o"},
        0.5: {"label": "Halb PV", "color": "blue", "marker": "^"},
        1: {"label": "Voll PV", "color": "green", "marker": "s"}
    }

    fig, axes = utils.create_plots(
        plot_config=plot_config,
        x_variables=[x_variable],
        y_variables=y_variables
    )
    for pv_factor, pv_plot_options in pv_area.items():
        mask = df.loc[:, "parameterStudy.f_design"] == pv_factor

        for _y_variable, _ax in zip(y_variables, axes[:, 0]):
            _ax.scatter(df.loc[mask, x_variable], df.loc[mask, _y_variable],
                        label=pv_plot_options["label"],
                        color=pv_plot_options["color"],
                        marker=pv_plot_options["marker"])

    axes[0, 0].legend(bbox_to_anchor=(0, 1), loc="lower left", ncol=3)
    utils.save(fig=fig, axes=axes, save_path=save_path, show=show, with_legend=False, file_endings=["png"])


def plot_biv_rule(config: StudyConfig):
    plot_config = utils.load_plot_config()
    dfs, input_configs = utils.get_all_results_from_config(study_config=config)

    for df, input_config in zip(dfs, input_configs):
        plot_pv_scatter(
            df=df,
            plot_config=plot_config,
            save_path=config.base_path.joinpath(input_config.get_name() + "_biv_rule"),
            x_variable="parameterStudy.TBiv",
            y_variables=["costs_invest", "costs_operating", "costs_total", "emissions"],
            show=True,
        )


def plot_biv_dependencies(config: StudyConfig, k_el_array: list, plot_obj_space=True, polynomial=True):
    plt.rcParams.update(
        {"figure.figsize": [13 / 2.54 * 1.25, 15 / 2.54 * 2 / 3],
         "font.size": 16,
         "figure.dpi": 250,
         "font.family": "Arial"
         }
    )
    colors = ["red", "blue", "black", "green", "gray"]
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    idx_col = 0
    input_config = config.inputs.get_permutations()[0]
    study_name = input_config.get_name()
    log_path = SurrogateBuilder.create_and_get_log_path(
        base_path=config.study_path, study_name=study_name
    )
    plot_config = PlotConfig.load_default()
    obj_map = AnnuityMapping()
    x_variable = "parameterStudy.TBiv"
    #x_variable = obj_map.heat_pump_size
    vars = list(obj_map.dict().values())
    if x_variable not in vars:
        vars.append(x_variable)
    y_variable = "costs_total"
    # X-Axis
    t_biv_opt = {}
    for k_el in k_el_array:
        df = SurrogateBuilder.load_design_optimization_log(file_path=log_path)
        mask = df.loc[:, "parameterStudy.VPerQFlow"] == 12
        df = df.loc[mask, vars]
        real_simulated_values = df[obj_map.heat_pump_size].values
        df.index = real_simulated_values
        _min = df.loc[:, obj_map.heat_pump_size].min()
        _max = df.loc[:, obj_map.heat_pump_size].max()
        new_index = np.linspace(_min, _max, 200)[::-1]
        df_new = pd.DataFrame(index=new_index, columns=obj_map.dict().values())
        df = pd.concat([df, df_new])
        df = df.sort_index().astype(float)
        df = df.interpolate().dropna()
        df = Annuity(k_el=k_el/100).calc(df)
        df.loc[:, y_variable] = ((df.loc[:, y_variable] - df.loc[:, y_variable].min())
                                 / df.loc[:, y_variable].min() * 100)
        df = plot_config.scale_df(df=df)
        arg_min = df.loc[:, y_variable].argmin()
        t_biv_opt[k_el] = df.iloc[arg_min][x_variable]
        if plot_obj_space:
            ax.plot(
                df.loc[:, x_variable],
                df.loc[:, y_variable],
                color=colors[idx_col],
            )
            ax.scatter(
                df.loc[real_simulated_values, x_variable],
                df.loc[real_simulated_values, y_variable],
                color=colors[idx_col],
                marker="s",
                label="$c_\mathrm{el} = %s$ ct/kWh" % int(k_el)
            )
        idx_col += 1
    savedir = Path(r"D:\00_temp")
    os.makedirs(savedir, exist_ok=True)
    if plot_obj_space:
        ax.set_xlabel(plot_config.get_label_and_unit(x_variable))
        ax.set_ylabel("$\Delta K_\mathrm{Tot}$ in %")
        ax.set_ylim([-1, 15])
        ax.set_yticks([0, 5, 10, 15])
        ax.legend(bbox_to_anchor=(0, 1, 1, 1), loc="lower left", ncol=2)
        ax.grid()
        fig.tight_layout()
        plt.savefig(savedir.joinpath(f"relative_deviation_motivation_{len(k_el_array)}.svg"))
    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel(plot_config.get_label_and_unit(x_variable))
    ax.set_xlabel("$c_\mathrm{el}$ in ct/kWh")
    x_ = np.array(list(t_biv_opt.keys()))
    if x_variable == "parameterStudy.TBiv":
        offset = 273.15
    else:
        offset = 0
    from bes_rules.rule_extraction.innovization import PowerLawRegressor, PolynomialRegressor
    y_ = offset + np.array(list(t_biv_opt.values()))
    ax.scatter(x_, y_ - offset, color="blue", label="$x_\mathrm{opt}$")
    if polynomial:
        pars_poly = PolynomialRegressor.get_parameters(x=x_, y=y_, n_order=13)
        print(pars_poly)
        x_detailed = np.linspace(x_.min(), x_.max(), 1000)
        y_reg_poly = PolynomialRegressor.eval(
            x=x_detailed,
            parameters=pars_poly
        )
        ax.plot(x_detailed, y_reg_poly - offset, color="black", label="Regel")
    else:
        ret = PowerLawRegressor.get_parameters(x=x_, y=y_)
        y_regression = PowerLawRegressor.eval(x=x_, parameters=ret)
        print(f"{ret[1]}*c_el ^{ret[0]}")
        ax.plot(x_, y_regression - offset, color="black", label="Regel")

    ax.legend(bbox_to_anchor=(0, 1, 1, 1), loc="lower left", ncol=2)
    ax.grid()
    fig.tight_layout()
    plt.savefig(savedir.joinpath(f"rule_explanation_{len(k_el_array)}.svg"))
    plt.show()


def plot_pareto_front_grid(config: StudyConfig, objectives: list):
    plot_config = PlotConfig.load_default()
    assert len(objectives) == 2, "Only 2D front supported for plotting"
    opt_var = "parameterStudy.TBiv"
    for input_config in config.inputs.get_permutations():
        study_name = input_config.get_name()
        log_path = SurrogateBuilder.create_and_get_log_path(
            base_path=config.study_path, study_name=input_config.get_name()
        )
        if not os.path.exists(log_path):
            logger.error("Can not plot %s, no xlsx result file!", study_name)
            continue
        df = SurrogateBuilder.load_design_optimization_log(file_path=log_path)
        df = plot_config.scale_df(df=df)
        df_pareto_efficient = get_pareto_efficient_points_for_df(df=df.copy(), objectives=objectives)
        fig, ax = plt.subplots(1, 2)
        y_obj, x_obj = objectives
        ax[0].set_xlabel(plot_config.get_label_and_unit(x_obj))
        ax[0].set_ylabel(plot_config.get_label_and_unit(y_obj))

        ax[0].scatter(df_pareto_efficient.loc[:, x_obj], df_pareto_efficient.loc[:, y_obj],
                   color="red", marker="s")
        ax[0].set_ylabel(plot_config.get_label_and_unit(y_obj))
        ax[1].set_xlabel(plot_config.get_label_and_unit(opt_var))
        ax[1].scatter(df_pareto_efficient.loc[:, opt_var],
                      df_pareto_efficient.loc[:, y_obj],
                      color="red", marker="s")
        fig.tight_layout()
        plt.savefig(f"netz.svg")
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    CONFIG = StudyConfig.from_json(r"D:\00_temp\02_storage_VDI\EVUControl\study_config.json")
    plot_scatter_for_x_over_multiple_y(
        study_config=CONFIG, save_path=Path(r"D:\00_temp"), x_variable="parameterStudy.VPerQFlow",
        y_variables=[
            "costs_total",
            "outputs.building.dTComHea[1]",
            "outputs.hydraulic.gen.heaPum.numSwi"
        ]
    )

    #plot_biv_dependencies(config=CONFIG, k_el_array=np.linspace(18, 40, 2))
    #plot_biv_dependencies(config=CONFIG, k_el_array=np.linspace(5, 60, 30), plot_obj_space=False)
