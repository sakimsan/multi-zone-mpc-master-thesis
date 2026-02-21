import logging
import os
from pathlib import Path
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import pandas as pd

from bes_rules.plotting.utils import PlotConfig
from bes_rules.rule_extraction.regression.regressors import Regressor
from bes_rules.plotting.utils import get_figure_size

logger = logging.getLogger(__name__)


def plot_optimality_gap(
        optimal_design_regressions: Union[Dict[str, np.ndarray], np.ndarray],
        objective_values: List[pd.Series],
        objective_name: str,
        feature_values: np.ndarray,
        design_variable: str,
        feature_names: List[str],
        save_path: Path,
        plot_config: PlotConfig = None
):
    if len(feature_names) > 1:
        print("Not yet supported")
        return
    feature_name = feature_names[0]
    if plot_config is None:
        plot_config = PlotConfig.load_default()
    fig, ax = plt.subplots(
        1, 1,
        figsize=get_figure_size(n_columns=1, quadratic=True)
    )
    heatmap = pd.DataFrame()
    feature_order = feature_values[0].argsort()

    for feature_idx in feature_order:
        objective_values_idx = objective_values[feature_idx]
        feature_value = plot_config.scale(feature_name, feature_values[0, feature_idx])
        x_name = objective_values_idx.index.name
        y_objective = objective_values_idx.values
        y_min = np.nanmin(y_objective)
        y_objective_percent_deviation = (y_objective - y_min) / y_min * 100
        x_values_scaled = plot_config.scale(x_name, objective_values_idx.index)
        if feature_value in heatmap.index:
            heatmap.loc[feature_value, x_values_scaled] = np.nanmax(
                [
                    y_objective_percent_deviation,
                    heatmap.loc[feature_value, x_values_scaled]
                ], axis=0
            )
        else:
            heatmap.loc[feature_value, x_values_scaled] = y_objective_percent_deviation
    heatmap = heatmap.sort_index(axis=1)
    n_design_samples = len(heatmap.columns)
    new_index = list(set(
        list(heatmap.index) +
        list(np.linspace(heatmap.index.min(), heatmap.index.max(), n_design_samples)))
    )
    heatmap = heatmap.reindex(new_index).sort_index()
    x = heatmap.index.values
    y = heatmap.columns.values
    X, Y = np.meshgrid(x, y)
    Z = heatmap.values.T  # Transpose to match meshgrid orientation
    # Plot heatmap
    # cmap = "magma_r" as another good option
    pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap="coolwarm",
                        norm=colors.PowerNorm(gamma=0.15, vmin=0, vmax=np.nanmax(Z)))
    cbar = plt.colorbar(pcm)
    if not isinstance(optimal_design_regressions, dict):
        optimal_design_regressions = {None: optimal_design_regressions}
        with_legend = False
    else:
        with_legend = False
    for rule_name, regression in optimal_design_regressions.items():
        ax.plot(
            plot_config.scale(feature_name, feature_values[0, feature_order]),
            plot_config.scale(design_variable, regression[feature_order]),
            label=f"Rule: {with_legend}"
        )

    ax.set_title(plot_config.get_label(objective_name))
    cbar.set_label('Optimalitätsgap in %', rotation=270, labelpad=15)
    cbar.set_ticks([0, 1, 5] + list(np.arange(10, np.nanmax(Z), 20)))
    ax.set_ylabel(plot_config.get_label_and_unit(design_variable))
    ax.set_xlabel(plot_config.get_label_and_unit(feature_name))
    if with_legend:
        ax.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
    fig.tight_layout()
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path)
    plt.close("all")


def plot_single_regression(
        regressor: Regressor,
        parameters: np.ndarray,
        optimal_design_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str],
        design_rule_string: str,
        design_variable: str,
        save_path: Path,
        plot_config: PlotConfig,
        objective_name: str,
        deviation_per_optimal_design: np.ndarray
):
    fig, ax = plt.subplots(
        1, 1,
        figsize=get_figure_size(
            n_columns=1.5,
            quadratic=True
        ))
    max_deviation = np.nanmax(deviation_per_optimal_design)

    color_args = dict(
        norm=colors.PowerNorm(gamma=0.15, vmin=0, vmax=max(10, max_deviation)),
        cmap='coolwarm', s=5
    )

    if len(feature_names) == 1:
        feature_name = feature_names[0]
        idx = 0
        ax.set_xlabel(plot_config.get_label_and_unit(feature_name))
        features_sorted = feature_values[:, feature_values[idx].argsort()]
        optimal_design_regressions = regressor.eval(
            x=features_sorted,
            parameters=parameters
        )
        ax.plot(
            plot_config.scale(feature_name, features_sorted[idx]),
            plot_config.scale(design_variable, optimal_design_regressions),
            color="red",
            label="Regression"
        )
        mappable = ax.scatter(
            plot_config.scale(feature_name, feature_values[idx]),
            plot_config.scale(design_variable, optimal_design_values),
            label="Optima", **color_args, c=deviation_per_optimal_design
        )
    else:
        ax.set_xlabel(plot_config.get_label_and_unit(design_variable + "_rule"))
        sort_idx = optimal_design_values.argsort()
        optimal_design_regressions = regressor.eval(
            x=feature_values[:, sort_idx],
            parameters=parameters
        )
        mappable = ax.scatter(
            plot_config.scale(design_variable, optimal_design_regressions),
            plot_config.scale(design_variable, optimal_design_values[sort_idx]),
            label="Regression", **color_args, c=deviation_per_optimal_design[sort_idx]
        )
        ax.plot(
            plot_config.scale(design_variable, optimal_design_values[sort_idx]),
            plot_config.scale(design_variable, optimal_design_values[sort_idx]),
            color="black",
            label="Idealer Fit"
        )
    cbar = fig.colorbar(mappable)
    cbar.set_label('Optimalitätsgaps in %', rotation=270, labelpad=15)
    if max_deviation <= 10:
        cbar.set_ticks([0, 1, 5, 10])
    else:
        cbar.set_ticks([0, 1, 5] + list(np.arange(10, max_deviation, 20)))

    ax.legend()
    ax.set_ylabel(plot_config.get_label_and_unit(design_variable + "_optimum"))
    fig.suptitle(design_rule_string)
    fig.tight_layout()
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_convergence(all_deviations: dict, save_path: Path, objective_name: str, plot_config: PlotConfig):
    objective_label = plot_config.get_label(objective_name).replace("$", "")
    deviations_to_plot = {
        "mean": "$\Delta \\bar{%s}$" % objective_label + " in %",
        "max": "$\max (\Delta %s)$" % objective_label + " in %",
    }
    fig, axes = plt.subplots(len(deviations_to_plot), 1, sharex=True, figsize=get_figure_size(n_columns=1))
    for ax, metric, ylabel in zip(axes, deviations_to_plot.keys(), deviations_to_plot.values()):
        experiments = list(all_deviations[metric].keys())
        values = list(all_deviations[metric].values())
        full = values[-1]
        ax.plot(experiments[:-1], values[:-1], color="red", marker="s", label="OVP-Iteration")
        ax.set_ylabel(ylabel)
        ax.axhline(full, label="Vollfaktoriell")
        #ax.set_xticks(list(values.keys()))
        #ax.set_xticklabels([str(k) for k in list(values.keys())[:-1]] + ["full"], rotation=90)
    axes[0].legend(loc="upper right", ncol=1)
    axes[-1].set_xlabel("Anzahl Versuche")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(save_path)
