import json
import pathlib

import numpy as np
from pydantic import BaseModel, ConfigDict

from ebcpy import DymolaAPI, TimeSeriesData

from bes_rules import BESRULES_PACKAGE_MO, STARTUP_BESMOD_MOS
from bes_rules.configs.plotting import PlotConfig
from bes_rules.plotting import utils

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

CUSTOM_PLOT_CONFIG = {
    "heaPum.refCyc.sigBus.icefacHPMea": {"label": "iceFac", "quantity": "Percent"},
    "TConOutMea": {"label": "$T_\mathrm{VL,WP}$", "quantity": "Temperature"},
    "heaPum.refCyc.sigBus.TConInMea": {"label": "$T_\mathrm{RL,WP}$", "quantity": "Temperature"},
    "heaPum.refCyc.sigBus.TEvaInMea": {"label": "$T_\mathrm{Auß}$", "quantity": "Temperature"},
    "heaPum.refCyc.sigBus.TConOutMea": {"label": r"TConOut", "quantity": "Temperature"},
    "heaPum.refCyc.sigBus.yMea": {"label": "$n_\mathrm{Ver}$", "quantity": "Percent"},
    "PEleCom": {"label": "$P_\mathrm{el,WP}$", "quantity": "Power"},
    "heaPum.refCyc.sigBus.PEleMea": {"label": "$P_\mathrm{el,WP}$", "quantity": "Power"},
    "heaPum.COP": {"label": "$COP$", "quantity": "One"},
    "COPMea": {"label": "$COP$", "quantity": "One"},
    "heaPum.QCon_flow": {"label": "QCon", "quantity": "Power"},
    "QConMea": {"label": "QCon", "quantity": "Power"},
    "heaPum.refCyc.sigBus.dTSupHeaSet": {"label": "$\Delta T_\mathrm{ÜH}$", "quantity": "TemperatureDifference"},
    "heaPum.refCyc.sigBus.mConMea_flow": {"label": "$\dot{m}_\mathrm{Kon}$", "quantity": "MassFlowRate"}
}


class ValidationCase(BaseModel):
    model_name: str
    name: str
    result_path: str = None
    model_config = ConfigDict(protected_namespaces=())


def add_measurement_uncertainty_T(TMea):
    if np.any(TMea) > 100:
        raise ValueError("Function expects degC")
    # Table 3.2
    a_gt = 0.1 * (0.3 + 0.05 * TMea)
    a_kt1 = 0.3
    a_kt2 = 0.005
    u_T = (a_gt ** 2 / 3 + a_kt1 ** 2 / 3 + a_kt2 ** 2 / 3) ** 0.5
    return u_T


def add_measurement_uncertainty_PEle(PEle):
    u_PEle = PEle * 0.0128
    return u_PEle


def flatten_nested_list(l: list):
    flat_l = []
    for entry in l:
        if isinstance(entry, list):
            flat_l.extend(entry)
        else:
            flat_l.append(entry)
    return list(set(flat_l))


def simulate(validation_cases: list, sim_setup: dict, save_path: pathlib.Path):
    dym_api = DymolaAPI(
        model_name=None,
        mos_script_pre=STARTUP_BESMOD_MOS,
        packages=[BESRULES_PACKAGE_MO],
        show_window=True,
        debug=True,
        equidistant_output=False,
        n_cpu=1,
        working_directory=save_path.joinpath("00_DymolaWorkdir")
    )
    dym_api.set_sim_setup(sim_setup)
    result_names = [validation_case.name for validation_case in validation_cases]
    results = dym_api.simulate(
        return_option="savepath",
        model_names=[validation_case.model_name for validation_case in validation_cases],
        result_file_name=result_names
    )
    if len(result_names) == 1:
        results = [results]
    results = dict(zip(result_names, results))
    for case in validation_cases:
        case.result_path = results[case.name]
    with open(save_path.joinpath("validation_case_results.json"), "w") as file:
        json.dump([validation_case.model_dump() for validation_case in validation_cases], file)
    return validation_cases


def plot_results_plotly(
        validation_cases: list, variables_to_plot: list,
        save_path: pathlib.Path,
        custom_plot_config: dict):
    plot_config = PlotConfig.load_default()
    plot_config.update_config({"variables": {**CUSTOM_PLOT_CONFIG, **custom_plot_config}})
    all_variable_names = flatten_nested_list(variables_to_plot)

    for case in validation_cases:
        result_path = case.result_path
        if result_path is None:
            result_path = save_path.joinpath("00_DymolaWorkdir", f"{case.name}.mat")
        df = TimeSeriesData(result_path, variable_names=all_variable_names).to_df()
        df = plot_config.scale_df(df)
        df.index /= 60  # Convert to minutes

        fig = make_subplots(rows=len(variables_to_plot), cols=1, shared_xaxes=True, vertical_spacing=0.05)
        for i, variable_to_plot in enumerate(variables_to_plot, start=1):
            if isinstance(variable_to_plot, str):
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[variable_to_plot], name=variable_to_plot,
                               mode='lines', line=dict(color='blue')),
                    row=i, col=1
                )
                fig.update_yaxes(title_text=plot_config.get_label_and_unit(variable_to_plot), row=i, col=1)
            else:
                mea_name, sim_name = variable_to_plot
                k = 1.65
                if mea_name == "PEleCom":
                    u = add_measurement_uncertainty_PEle(df[mea_name])
                elif mea_name == "TConOutMea":
                    u = add_measurement_uncertainty_T(df[mea_name])
                else:
                    u = None
                if u is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[mea_name].values + k * u,
                            name=f"{mea_name} uncertainty",
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ),
                        row=i, col=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[mea_name].values - k * u,
                            name=f"{mea_name} uncertainty",
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',  # fill area between traces
                            fillcolor='rgba(255,0,0,0.2)',  # red with 0.2 opacity
                            showlegend=False
                        ),
                        row=i, col=1
                    )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[mea_name].values,
                        name=f"{mea_name} (mea)",
                        mode='lines',
                        line=dict(color='red', dash='dash')
                    ),
                    row=i, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[sim_name], name=f"{sim_name} (sim)",
                               mode='lines', line=dict(color='blue')),
                    row=i, col=1
                )
                fig.update_yaxes(title_text=plot_config.get_label_and_unit(mea_name), row=i, col=1)

        fig.update_layout(
            height=400 * len(variables_to_plot),
            title_text=f"Results for {case.name}",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(title_text="Time in min", row=len(variables_to_plot), col=1)

        # Save as interactive HTML
        fig.write_html(save_path.joinpath(f"{case.name}_interactive.html"))


def _plot_error(ax: plt.Axes, x_data, y_data):
    ax.hist2d(x_data, y_data,
              bins=50,
              cmap='viridis',
              norm=plt.cm.colors.LogNorm())  # Log scale for better visualization


def plot_error_hist(all_results: dict, y_errors: list, x_variables: list, save_path: pathlib.Path, plot_config):
    for case, data in all_results.items():
        logger.info(f"Creating plot for %s", case)
        fig, axes = plt.subplots(
            len(y_errors), len(x_variables),
            figsize=utils.get_figure_size(2, len(y_errors)),
            sharex='col', sharey='row'
        )
        for i_x, x_variable in enumerate(x_variables):
            for i_y, y_error in enumerate(y_errors):
                logger.info("Plotting %s and %s with %s data points", x_variable, y_error[0], len(data[x_variable]))
                ax = axes[i_y, i_x]
                _plot_error(
                    ax=ax,
                    x_data=data[x_variable],
                    y_data=data[y_error[0]] - data[y_error[1]]
                )
                axes[i_y, 0].set_ylabel("$\Delta$" + plot_config.get_label_and_unit(y_error[0]))
            axes[-1, i_x].set_xlabel(plot_config.get_label_and_unit(x_variable))
        #fig.suptitle(case)
        fig.tight_layout()
        fig.savefig(save_path.with_stem(f"{case}_{save_path.stem}"))
    plt.show()


def plot_error_over_error(all_results: dict, y_errors: list, save_path: pathlib.Path, plot_config):
    for case, data in all_results.items():
        logger.info(f"Creating plot for %s", case)
        fig, axes = plt.subplots(
            len(y_errors) - 1, 1,
            figsize=utils.get_figure_size(2, len(y_errors)),
            sharex=False, sharey=False
        )
        if isinstance(axes, plt.Axes):
            axes = [axes]
        for i_y, y_error in enumerate(y_errors):
            if i_y == 0:
                continue
            ax = axes[i_y - 1]
            _plot_error(
                ax=ax,
                x_data=data[y_errors[0][0]] - data[y_errors[0][1]],
                y_data=data[y_error[0]] - data[y_error[1]]
            )
            ax.axhline(0, color="black", linestyle="--")
            ax.axvline(0, color="black", linestyle="--")
            ax.set_ylabel("$\Delta$" + plot_config.get_label_and_unit(y_error[0]))
        axes[-1].set_xlabel(plot_config.get_label_and_unit(y_errors[0][0]))
        #fig.suptitle(case)
        fig.tight_layout()
        fig.savefig(save_path.with_stem(f"{case}_{save_path.stem}"))
    plt.show()
