import os
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from bes_rules.configs import PlotConfig, StudyConfig
from bes_rules.objectives.annuity import Annuity
from bes_rules.plotting import utils
from bes_rules.plotting.utils import get_figure_size

INPUT_CONFIG_RENAMES = {
        "TRY2045_474856110632_Somm_B1950_adv_retrofit_SingleDwelling_M_SouthNorth": "max-1",
        "TRY2045_474856110632_Somm_B1950_adv_retrofit_SingleDwelling_M_EastWest": "max-2",
        "TRY2015_536322100078_Jahr_B1994_retrofit_SingleDwelling_M_South": "mean"
    }


def calc_costs_and_get_variables(df, plot_config):
    df = Annuity().calc(df)
    #df = Annuity(i_the_0=StochasticParameter(value=0), i_the_a=StochasticParameter(value=0)).calc(df)

    df = plot_config.scale_df(df=df)
    if df.columns.nlevels == 2:
        df = df.droplevel(0, axis=1)
    df.loc[:, "parameterStudy.TBiv"] = df.loc[:, "parameterStudy.TBiv"].apply(_round_or_int)
    df.loc[:, "parameterStudy.VPerQFlow"] = df.loc[:, "parameterStudy.VPerQFlow"].apply(_to_int)

    return df


def plot_pv_influence(
        base_path: Path,
        study_name_no_ctrl: str,
        study_name_ctrl: str,
        x_variable: str,
        y_variable: str,
        z_variable: str,
        minimize: bool,
        save_name: str,
        show=False,
        cmap=None,
        f_design=1,
):
    plot_config = PlotConfig.load_default(update_rc=False)

    config_no_ctrl = StudyConfig.from_json(base_path.joinpath(study_name_no_ctrl, "study_config.json"))
    config_ctrl = StudyConfig.from_json(base_path.joinpath(study_name_ctrl, "study_config.json"))

    fig_all, axes_all = [], []
    # No PV:
    kwargs_for_map = dict(
        x_variable=x_variable,
        y_variable=y_variable,
        z_variables=[z_variable],
        minimize=[minimize],
        argopt_plot_variable=None
    )
    kwargs_for_plot = dict(
        x_variable=x_variable,
        plot_config=plot_config,
        cmap=cmap,
        z_variable=z_variable,
        minimize=minimize,
        argopt_plot_variable=None,
    )

    dfs, input_configs = utils.get_all_results_from_config(
        study_config=config_ctrl
    )
    dfs_no_ctrl, input_configs_no_pv = utils.get_all_results_from_config(
        study_config=config_no_ctrl
    )
    for df_pv_ctrl, df_no_ctrl, input_config, input_config_no_pv in zip(dfs, dfs_no_ctrl, input_configs, input_configs_no_pv):
        # First Plot: No PV vs. PV with no control
        df_pv_ctrl = calc_costs_and_get_variables(df_pv_ctrl, plot_config)
        df_no_ctrl = calc_costs_and_get_variables(df_no_ctrl, plot_config)
        df_no_pv = df_no_ctrl.loc[df_no_ctrl.loc[:, "parameterStudy.f_design"] == 0]
        df_pv_no_ctrl = df_no_ctrl.loc[df_no_ctrl.loc[:, "parameterStudy.f_design"] == f_design]

        plot_comparison_of_two_cases_as_heatmap(
            df_1=df_no_pv, df_2=df_pv_no_ctrl,
            kwargs_for_map=kwargs_for_map,
            kwargs_for_plot=kwargs_for_plot, name_1="No PV",
            name_2="No Control", y_variable=y_variable, plot_config=plot_config,
            save_path=str(base_path.joinpath(f"pv_{save_name}_{INPUT_CONFIG_RENAMES[input_config.get_name(with_user=False)]}"))
        )
        plot_comparison_of_two_cases_as_heatmap(
            df_1=df_pv_no_ctrl, df_2=df_pv_ctrl,
            kwargs_for_map=kwargs_for_map,
            kwargs_for_plot=kwargs_for_plot, name_1="No Control",
            name_2="Control", y_variable=y_variable, plot_config=plot_config,
            save_path=str(base_path.joinpath(f"control_{save_name}_{INPUT_CONFIG_RENAMES[input_config.get_name(with_user=False)]}"))
        )
    if show:
        plt.show()


def plot_control_influence(
        base_path: Path,
        study_name_ctrl_1: str,
        study_name_ctrl_2: str,
        alias_1: str,
        alias_2: str,
        x_variable: str,
        y_variable: str,
        z_variable: str,
        minimize: bool,
        save_name: str,
        show=False,
        cmap=None,
        f_design=1,
        custom_plot_config: dict = None
):
    plot_config = PlotConfig.load_default(update_rc=False)
    if custom_plot_config:
        plot_config.update_config(custom_plot_config)
    config_first_ctrl = StudyConfig.from_json(base_path.joinpath(study_name_ctrl_1, "study_config.json"))
    config_second_ctrl = StudyConfig.from_json(base_path.joinpath(study_name_ctrl_2, "study_config.json"))

    fig_all, axes_all = [], []
    # No PV:
    kwargs_for_map = dict(
        x_variable=x_variable,
        y_variable=y_variable,
        z_variables=[z_variable],
        minimize=[minimize],
        argopt_plot_variable=None
    )
    kwargs_for_plot = dict(
        x_variable=x_variable,
        plot_config=plot_config,
        cmap=cmap,
        z_variable=z_variable,
        minimize=minimize,
        argopt_plot_variable=None,
    )

    dfs_1, input_configs_1 = utils.get_all_results_from_config(
        study_config=config_first_ctrl
    )
    dfs_2, input_configs_2 = utils.get_all_results_from_config(
        study_config=config_second_ctrl
    )
    for df_1, df_2, input_config_1, input_config_2 in zip(dfs_1, dfs_2, input_configs_1, input_configs_2):
        assert input_config_1.get_name() == input_config_2.get_name(), "Input configs don't match!"
        # First Plot: No PV vs. PV with no control
        df_1 = calc_costs_and_get_variables(df_1, plot_config)
        df_2 = calc_costs_and_get_variables(df_2, plot_config)
        df_1 = df_1.loc[df_1.loc[:, "parameterStudy.f_design"] == f_design]
        df_2 = df_2.loc[df_2.loc[:, "parameterStudy.f_design"] == f_design]

        plot_comparison_of_two_cases_as_heatmap(
            df_1=df_1, df_2=df_2,
            kwargs_for_map=kwargs_for_map,
            kwargs_for_plot=kwargs_for_plot,
            name_1=alias_1,
            name_2=alias_2,
            y_variable=y_variable, plot_config=plot_config,
            save_path=str(base_path.joinpath(f"pv_{save_name}_{INPUT_CONFIG_RENAMES[input_config_1.get_name(with_user=False)]}"))
        )
    if show:
        plt.show()



def plot_comparison_of_two_cases_as_heatmap(
        df_1, df_2, kwargs_for_map,
        kwargs_for_plot, name_1: str,
        y_variable, plot_config, save_path, name_2,
        three_color_bars=True
):
    if three_color_bars:
        width_ratios = [1, 1, 1]
    else:
        width_ratios = [1, 1.2, 1.2]
    fig, axes = plt.subplots(1, 3, sharey=False,
                             width_ratios=width_ratios, figsize=get_figure_size(n_columns=1.5))
    df_map_1 = extract_df_map(df=df_1, **kwargs_for_map)[0]
    df_map_2 = extract_df_map(df=df_2, **kwargs_for_map)[0]
    if three_color_bars:
        vmin = None
        vmax = None
    else:
        vmin = min(df_map_1.min().min(), df_map_2.min().min())
        vmax = max(df_map_1.max().max(), df_map_2.max().max())
    title_1 = plot_config.get_label_and_unit(kwargs_for_plot["z_variable"], linebreak=True).replace("Tot", f"Tot,{name_1}")
    title_2 = plot_config.get_label_and_unit(kwargs_for_plot["z_variable"], linebreak=True).replace("Tot", f"Tot,{name_2}")
    _plot_single_ax(df=df_map_1, ax=axes[0], plot_cbar=three_color_bars, vmin=vmin, vmax=vmax, title=title_1, **kwargs_for_plot)
    _plot_single_ax(df=df_map_2, ax=axes[1], plot_cbar=True, vmin=vmin, vmax=vmax, title=title_2, **kwargs_for_plot)

    change = (df_map_2 - df_map_1) / df_map_1 * 100
    change_min = change.min().min()
    change_max = change.max().max()
    print(f"Change {name_2} to {name_1}", change.min().min(), change.max().max())

    title = "$\Delta C_\mathrm{" + name_1 + "," + name_2 + "}$\nin \\%"
    _plot_single_ax(
        df=change, ax=axes[2], plot_cbar=True,
        vmin=change_min, vmax=change_max, plot_hatch=False,
        title=title, **kwargs_for_plot)
    axes[0].set_ylabel(plot_config.get_label_and_unit(y_variable))

    cases = [f"(1)", f"(2)", "(3)"]
    #cases = [f"(1) {name_1}", f"(2) {name_2}", "(3) Change"]
    for case, ax in zip(cases, axes):
        ax.set_title(f"{case} {ax.title.get_text()}")
        ax.set_ylabel(plot_config.get_label_and_unit(y_variable))
        #ax.set_title(f"{case}:\n {ax.title.get_text()}")

    #fig.tight_layout(w_pad=0.6)
    fig.tight_layout()
    fig.savefig(save_path + ".png")
    fig.savefig(save_path + ".pdf")
    plt.close("all")

def plot_control_heat_map(
        base_path: Path,
        study_name: str,
        x_variable: str,
        y_variable: str,
        z_variables: Dict[str, bool],
        save_name: str,
        show=False,
        cmap=None,
        constant_optimization_variables: dict = None,
        argopt_plot_variable: str = None,
        save_path: Path = None
):
    minimize = list(z_variables.values())
    z_variables = list(z_variables.keys())
    if save_path is None:
        save_path = base_path
    os.makedirs(save_path, exist_ok=True)
    if constant_optimization_variables is None:
        constant_optimization_variables = {}

    plot_config = PlotConfig.load_default(update_rc=False)

    config = StudyConfig.from_json(base_path.joinpath(study_name, "study_config.json"))
    dfs, inputs_configs = utils.get_all_results_from_config(config)
    for df, input_config in zip(dfs, inputs_configs):
        if df is None:
            continue
        df = calc_costs_and_get_variables(df=df, plot_config=plot_config)
        for var_name, var_value in constant_optimization_variables.items():
            df = df.loc[df.loc[:, var_name] == var_value]

        width_ratios = [1] * len(z_variables)
        if argopt_plot_variable:
            width_ratios[-1] = 1.2

        fig, axes = plt.subplots(1, len(z_variables), sharey=False, width_ratios=width_ratios,
                                 figsize=get_figure_size(2.5 if len(z_variables) > 4 else 1.4))

        df_maps = extract_df_map(
            df=df,
            x_variable=x_variable,
            y_variable=y_variable,
            z_variables=z_variables,
            minimize=minimize,
            argopt_plot_variable=argopt_plot_variable)
        i = 0
        for _df_map, z_variable, _minimize, ax in zip(df_maps, z_variables, minimize, axes):
            _plot_single_ax(df=_df_map,
                            z_variable=z_variable,
                            minimize=_minimize,
                            ax=ax,
                            plot_config=plot_config,
                            vmin=None,
                            vmax=None,
                            plot_cbar=(argopt_plot_variable is None) or (i == len(z_variables) - 1),
                            x_variable=x_variable, cmap=cmap,
                            argopt_plot_variable=argopt_plot_variable)
            i += 1
        for ax in axes:
            ax.set_ylabel(plot_config.get_label_and_unit(y_variable))

        fig.tight_layout()
        file_name = str(save_path.joinpath(f"{save_name}_{INPUT_CONFIG_RENAMES[input_config.get_name(with_user=False)]}"))
        fig.savefig(file_name + ".png")
        fig.savefig(file_name + ".pdf")

    if show:
        plt.show()
    plt.close("all")

def _plot_single_ax(df, z_variable, minimize, ax, plot_config, x_variable, cmap, argopt_plot_variable,
                    plot_cbar, vmin, vmax, title=None, discrete_colors=False, plot_hatch=True):
    if argopt_plot_variable is None and minimize:
        reversed = True
    elif argopt_plot_variable is None and not minimize:
        reversed = False
    else:
        vmin = 0.1
        vmax = 1
        reversed = False
    cmap = sns.color_palette(f"{cmap}{'_r' if reversed else ''}", as_cmap=argopt_plot_variable is None)
    sns.heatmap(df,
                cmap=cmap, ax=ax,
                vmin=vmin, vmax=vmax,
                cbar=plot_cbar, cbar_kws=dict(format='%.1f'))
    if minimize:
        zm = np.ma.masked_greater(df.values, df.min().min())
    else:
        zm = np.ma.masked_less(df.values, df.max().max())

    x = np.arange(len(df.columns) + 1)
    y = np.arange(len(df.index) + 1)
    if argopt_plot_variable is None and plot_hatch:
        ax.pcolor(x, y, zm, hatch="xxxxxx", alpha=0.)  # , edgecolor="black" if _minimize else "white")
    if title is None:
        if argopt_plot_variable is None:
            title = plot_config.get_label_and_unit(z_variable, linebreak=True)
        else:
            title = plot_config.get_label(z_variable)
    ax.set_title(title)
    if len(df.columns) <= 4:
        ticks = [df.columns[0], df.columns[-1]]
        ax.set_xticks([0.5, len(df.columns)-0.5])
    else:
        ticks = [df.columns[0], df.columns[int(len(df.columns) / 2)-1], df.columns[-1]]
        ax.set_xticks([0.5, int(len(df.columns)/2) - 0.5, len(df.columns)-0.5])
    ax.set_xticklabels(ticks)
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylabel(None)
    ax.set_xlabel(plot_config.get_label_and_unit(x_variable, linebreak=True))
    return ax


def extract_df_map(df, x_variable, y_variable, z_variables, minimize, argopt_plot_variable):
    df_maps = [pd.DataFrame() for _ in z_variables]
    func = {
        True: np.min,
        False: np.max
    }
    for y_val in df.loc[:, y_variable].unique():
        for x_val in df.loc[:, x_variable].unique():
            _mask = (y_val == df.loc[:, y_variable]) & (x_val == df.loc[:, x_variable])
            for _i_z, z_variable in enumerate(z_variables):
                minmax_func = func[minimize[_i_z]]
                if argopt_plot_variable is not None:
                    _d = df.loc[_mask]
                    all_opts = _d.loc[_d.loc[:, z_variable] == minmax_func(_d.loc[:, z_variable]), argopt_plot_variable]
                    df_maps[_i_z].loc[y_val, x_val] = all_opts.max()
                else:
                    z_vars = df.loc[_mask, z_variable]
                    if len(z_vars) == 0:
                        raise Exception
                    else:
                        df_maps[_i_z].loc[y_val, x_val] = minmax_func(z_vars)
    return [df.sort_index(axis=1).sort_index(axis=0) for df in df_maps]


def _round_or_int(val):
    if round(val, 1) == int(val):
        return int(val)
    return round(val, 1)


def _to_int(val):
    return int(val)


def create_table_with_all_optima(
        study_names: List[str],
        study_alias: List[str],
        base_path: Path,
        optimization_variables: List[str],
        objective_functions: dict
):
    plot_config = PlotConfig.load_default(update_rc=False)
    df_table = pd.DataFrame(dtype=object)

    idx = 0
    for with_elec in ["NoElec", ""]:
        for study_name, _study_alias in zip(study_names, study_alias):
            config = StudyConfig.from_json(base_path.joinpath(study_name + with_elec, "study_config.json"))
            dfs, inputs_configs = utils.get_all_results_from_config(config)
            for df, input_config in zip(dfs, inputs_configs):
                df = Annuity().calc(df)
                df = plot_config.scale_df(df=df)
                _objective_functions = objective_functions.copy()
                _optimization_variables = optimization_variables.copy()
                if _study_alias in ["Control", "No Control"]:
                    constant_optimization_variables = {"parameterStudy.f_design": 1}
                    _objective_functions = objective_functions.copy()
                    if _study_alias == "No Control":
                        _optimization_variables.remove("parameterStudy.ShareOfPEleNominal")
                else:
                    constant_optimization_variables = {"parameterStudy.f_design": 0}
                    _objective_functions.pop("self_sufficiency_degree")
                    _objective_functions.pop("self_consumption_rate", None)
                    _optimization_variables.remove("parameterStudy.ShareOfPEleNominal")

                for var_name, var_value in constant_optimization_variables.items():
                    df = df.loc[df.loc[:, var_name] == var_value]
                results = _get_optimal_values(
                    df=df,
                    optimization_variables=_optimization_variables,
                    objective_functions=_objective_functions
                )
                for result in results:
                    results_for_table = dict(
                        study_alias=_study_alias,
                        inputs_name=INPUT_CONFIG_RENAMES[input_config.get_name(with_user=False)],
                        user_electricity="No" if with_elec == "NoElec" else "Yes",
                        **result
                    )
                    df_table.loc[idx, results_for_table.keys()] = results_for_table.values()
                    idx += 1
    df_table.set_index("study_alias").to_excel(base_path.joinpath("AllResults.xlsx"))


def _get_optimal_values(
        df,
        optimization_variables: List[str],
        objective_functions: dict):
    func = {
        "min": np.min,
        "max": np.max
    }
    results = []
    renames = {
        "costs_total": "$C_\mathrm{tot}$",
        "self_sufficiency_degree": "$SSD$",
        "SCOP_Sys": "$SCOP_\mathrm{Sys}$"
    }
    names = optimization_variables + list(objective_functions.keys())
    for _objective_name, _objective_function in objective_functions.items():
        _objective_function = func[_objective_function]
        if np.any(df.loc[:, _objective_name].isna()):
            raise ValueError("Contains NaNs")
            #results.append({name: "NaN" for name in names})
            #continue
        optimal_mask = df.loc[:, _objective_name] == _objective_function(df.loc[:, _objective_name])
        if np.count_nonzero(optimal_mask) == 1:
            results.append({
                "objective": renames[_objective_name],
                **_get_dict_from_row(row=df.loc[optimal_mask], names=names)
            })
        else:
            raise ValueError("Multiple optimal values")
    return results


def _get_dict_from_row(row, names):
    return {name: round(value, 1) for name, value in zip(names, row[names].values[0])}
