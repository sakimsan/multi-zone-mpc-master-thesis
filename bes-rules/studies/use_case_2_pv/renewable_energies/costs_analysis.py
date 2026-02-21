import itertools
import os
import shutil
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from bes_rules import configs, STARTUP_BESMOD_MOS, BESRULES_PACKAGE_MO
from bes_rules.plotting import utils
from bes_rules.objectives import get_all_objectives, Annuity
from bes_rules.objectives.annuity import AnnuityMapping
from bes_rules.objectives.stochastic_parameter import StochasticParameter
from bes_rules.simulation_based_optimization import SurrogateBuilder
from bes_rules.configs.plotting import PlotConfig

PLOT_CONFIG = PlotConfig.load_default()

import plots   # Trigger rcParams
from bes_rules.plotting.utils import get_figure_size

INPUT_CONFIG_RENAMES = {
        "TRY2045_474856110632_Somm_B1950_adv_retrofit_SingleDwelling_M_SouthNorth": "max-1",
        "TRY2045_474856110632_Somm_B1950_adv_retrofit_SingleDwelling_M_EastWest": "max-2",
        "TRY2015_536322100078_Jahr_B1994_retrofit_SingleDwelling_M_South": "mean"
    }


def recalculate_all_objectives(study_name: str, base_path: Path):
    config = configs.StudyConfig.from_json(base_path.joinpath(study_name, "study_config.json"))
    dfs, input_configs = utils.get_all_results_from_config(config)
    for df, input_config in zip(dfs, input_configs):
        P_mp0 = 285  # From record
        A_mod = 1.000 * 1.670  # From record
        use_two_side = "useTwoRoo=true" in input_config.modifiers.modifiers[0]
        numGenUnits = 2 if use_two_side else 1
        f_design = df.loc[:, "parameterStudy.f_design"]
        ARooSid = input_config.building.building_parameters.roof_area / 2
        numMod = [f_design * ARooSid / A_mod for _ in range(numGenUnits)]
        P_MPP = np.sum([numMod[i] * P_mp0 for i in range(numGenUnits)], axis=0)
        if "PElePVMPP" not in df.columns:
            logging.warning("Recalculating maximal power point, include 'PElePVMPP' in the next study!")
            df.loc[:, "PElePVMPP"] = P_MPP
        else:
            if not np.all(np.isclose(P_MPP, df.loc[:, "PElePVMPP"])):
                raise ValueError("Results do not match")
        for obj in get_all_objectives():
            df = obj.calc(df=df)
        study_name = input_config.get_name()
        df_path = SurrogateBuilder.create_and_get_log_path(
            base_path=config.study_path, study_name=study_name, create=False
        )
        df_path_reference = df_path.parent.joinpath(df_path.stem + "_old.xlsx")
        shutil.copy(df_path, df_path_reference)
        SurrogateBuilder.save_design_optimization_log(file_path=df_path, df=df)


def optimize_price_assumptions(study_name: str, base_path: Path, objective: str):
    config = configs.StudyConfig.from_json(
        base_path.joinpath(study_name, "study_config.json")
    )
    dfs, input_configs = utils.get_all_results_from_config(config)
    TBiv_name = "parameterStudy.TBiv"
    V_name = "parameterStudy.VPerQFlow"
    control_name = "parameterStudy.ShareOfPEleNominal"
    variable_names = [TBiv_name, V_name, control_name]
    k_el = np.arange(0.1, 0.5, 0.05).round(2)
    feed_in_ratios = np.array([1, 3, 5, 7, 9, 11, 13, 15])
    all_results = {}
    for df, input_config in zip(dfs, input_configs):
        ref_df = pd.DataFrame(index=k_el, columns=feed_in_ratios)
        df_correct_unit = PLOT_CONFIG.scale_df(df)
        variables_at_optimum = {var: ref_df.copy() for var in variable_names}
        storage_savings = {round(storage_vol, 2): ref_df.copy() for storage_vol in df.loc[:, V_name].unique()}
        for _k_el, _feed_in_ratio in itertools.product(k_el, feed_in_ratios):
            ann = Annuity(
                k_el=StochasticParameter(value=_k_el),
                k_el_feed_in=StochasticParameter(value=_k_el/_feed_in_ratio)
            )
            df = ann.calc(df)

            # Difference to no storage:
            TBiv_name = "parameterStudy.TBiv"
            V_name = "parameterStudy.VPerQFlow"
            control_name = "parameterStudy.ShareOfPEleNominal"
            cols = [TBiv_name, V_name, control_name, "invest_tes"] + list(ann.mapping.model_dump().values())
            # df = df.loc[df.loc[:, control_name] == 1]
            df_deltas = []
            for TBiv in df.loc[:, TBiv_name].unique():
                df_biv = df.loc[df.loc[:, TBiv_name] == TBiv, cols].copy()
                df_idx_vmin = df_biv.loc[df_biv.loc[:, V_name].min() == df_biv.loc[:, V_name]]
                df_deltas.append(df_biv - pd.concat([df_idx_vmin] * 8).values)
            df_deltas = pd.concat(df_deltas)
            df.loc[:, "delta_W_el_feed_in"] = df_deltas.loc[:, ann.mapping.electric_energy_feed_in].values / 3600000
            df.loc[:, "delta_W_el_demand"] = df_deltas.loc[:, ann.mapping.electric_energy_demand].values / 3600000
            df.loc[:, "delta_I_storage"] = df_deltas.loc[:, "invest_tes"].values

            df.loc[:, "possible_savings_if_storage_costs_nothing"] = (
                (
                        - df.loc[:, "delta_W_el_demand"] +
                        df.loc[:, "delta_W_el_feed_in"] / _feed_in_ratio
                ) * _k_el
            )

            df.loc[:, "savings_from_larger_storage"] = (
                (
                        - df.loc[:, "delta_W_el_demand"] +
                        df.loc[:, "delta_W_el_feed_in"] / _feed_in_ratio
                ) * _k_el - df.loc[:, "delta_I_storage"]
            )

            cost_optima_idx = df.loc[:, objective].argmin()
            for variable_name, df_var in variables_at_optimum.items():
                df_var.loc[_k_el, _feed_in_ratio] = df_correct_unit.iloc[cost_optima_idx][variable_name]
            for storage_vol in df.loc[:, V_name].unique():
                mask_optimal_TBiv = df.loc[:, TBiv_name] == df.iloc[cost_optima_idx][TBiv_name]
                mask_optimal_control = df.loc[:, control_name] == df.iloc[cost_optima_idx][control_name]
                mask_storage = df.loc[:, V_name] == storage_vol
                highest_saving = df.loc[
                    mask_optimal_TBiv & mask_storage & mask_optimal_control,
                    "possible_savings_if_storage_costs_nothing"
                ]
                if len(highest_saving) > 1:
                    raise ValueError("Multiple maximal savings exist. Should not happen")
                storage_savings[round(storage_vol, 2)].loc[_k_el, _feed_in_ratio] = highest_saving.values[0]

        all_results[INPUT_CONFIG_RENAMES[input_config.get_name(with_user=False)]] = {
            "vars_at_optimum": variables_at_optimum,
            "storage_savings": storage_savings
        }

    return all_results


def plot_price_assumption_optima_heatmap(all_results: dict, save_path: Path, objective: str):
    for input_name, variables_at_optimum in all_results.items():
        variables_at_optimum = variables_at_optimum["vars_at_optimum"]
        fig, axes = plt.subplots(1, len(variables_at_optimum), sharey=True, figsize=get_figure_size(1.5, height_factor=0.7))
        for ax, variable_name, df_var in zip(axes, variables_at_optimum.keys(), variables_at_optimum.values()):
            sns.heatmap(df_var.astype(float), cmap="rocket", ax=ax, cbar=True)
            ax.set_title(PLOT_CONFIG.get_label_and_unit(variable_name))
            ax.set_xlabel("$f_\mathrm{Fee,Dem}$\nin -")
            ax.tick_params(axis="y", rotation=0)
            ax.tick_params(axis="x", rotation=0)
            ticks = [df_var.columns[0], df_var.columns[int(len(df_var.columns) / 2) - 1], df_var.columns[-1]]
            ax.set_xticks([0.5, int(len(df_var.columns) / 2) - 0.5, len(df_var.columns) - 0.5])
            ax.set_xticklabels(ticks)
        axes[0].set_ylabel("$c_\mathrm{el,Dem}$ in ct/kWh")

        fig.tight_layout()
        fig.savefig(save_path.joinpath("heatmap_all_" + input_name + f"_{objective}.pdf"))
        fig.savefig(save_path.joinpath("heatmap_all_" + input_name + f"_{objective}.png"))
        plt.close("all")


def plot_price_assumption_TBiv_and_storages(all_results: dict, save_path: Path, objective: str, with_optimal_biv: bool):
    TBiv_name = "parameterStudy.TBiv"
    V_name = "parameterStudy.VPerQFlow"
    label = PLOT_CONFIG.get_label_and_unit(V_name)
    storage_sizes = [25.71, 150]
    for input_name, variables_at_optimum in all_results.items():
        fig, axes = plt.subplots(
            1, len(storage_sizes) + int(with_optimal_biv),
            sharey=False, figsize=get_figure_size(1 + int(with_optimal_biv) * 0.5, height_factor=1)
        )
        if with_optimal_biv:
            df_var_TBiv = variables_at_optimum["vars_at_optimum"][TBiv_name]
            sns.heatmap(df_var_TBiv.astype(float), cmap="rocket", ax=axes[0], cbar=True)
            axes[0].set_title("(1) $T_\mathrm{Biv,Opt}$\nin °C")
        for idx, storage_vol in enumerate(storage_sizes):
            i = idx + int(with_optimal_biv)
            df_var = variables_at_optimum["storage_savings"][storage_vol]
            sns.heatmap(df_var.astype(float), cmap="rocket", ax=axes[i], cbar=True)
            # v_\mathrm{\dot{Q}_\mathrm{Bui}}=
            axes[i].set_title(
                "(%s) $\Delta C_\mathrm{Op}(%s \,\mathrm{l/kW})$\nin €/a" % (
                    i + 1, int(round(storage_vol, 0))
                )
            )
        for ax in axes:
            ax.set_ylabel("$c_\mathrm{el,Dem}$ in ct/kWh")
            ax.set_xlabel("$f_\mathrm{Fee,Dem}$\nin -")
            ax.tick_params(axis="y", rotation=0)
            ax.tick_params(axis="x", rotation=0)
            ticks = [df_var.columns[0], df_var.columns[int(len(df_var.columns) / 2) - 1], df_var.columns[-1]]
            ax.set_xticks([0.5, int(len(df_var.columns) / 2) - 0.5, len(df_var.columns) - 0.5])
            ax.set_xticklabels(ticks)
        fig.tight_layout()
        biv = "" if with_optimal_biv else "_noBiv"
        save_name = f"prices_{input_name}_Influence_{objective}{biv}"
        fig.savefig(save_path.joinpath(f"{save_name}.png"))
        fig.savefig(save_path.joinpath(f"{save_name}.pdf"))
        plt.close("all")



if __name__ == '__main__':
    #run_design_optimization(get_config(
    #    study_name="BESCtrl",
    #    model_name="BESRules.PVAndHPDesignAndControlOptimization.PythonAPICtrlOpt",
    #    test=False,
    #))
    from plots import BASE_PATH, change_rc_params_for_paper

    change_rc_params_for_paper()

    #recalculate_all_objectives(study_name="BESCtrl", base_path=BASE_PATH)
    #recalculate_all_objectives(study_name="BESNoCtrl", base_path=BASE_PATH)
    os.makedirs(BASE_PATH.joinpath("cost_analysis"), exist_ok=True)
    for OBJECTIVE in ["costs_total", "costs_operating"]:
        RES = optimize_price_assumptions(study_name="BESCtrl", base_path=BASE_PATH, objective=OBJECTIVE)
        #plot_price_assumption_optima_heatmap(RES, save_path=BASE_PATH.joinpath("cost_analysis"), objective=OBJECTIVE)
        plot_price_assumption_TBiv_and_storages(RES, save_path=BASE_PATH.joinpath("cost_analysis"), objective=OBJECTIVE,
                                                with_optimal_biv=True)
