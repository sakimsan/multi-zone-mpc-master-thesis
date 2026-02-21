import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from vclibpy.components.compressors import Compressor
from vclibpy.media import RefProp
import custom_compressor
from bes_rules import DATA_PATH, RESULTS_FOLDER
from bes_rules.plotting import utils, EBCColors
import pandas as pd

import os
import matplotlib.pyplot as plt

T_con_outs = np.array([
    35,
    45,
    55,
    65,
    70
])
VARIABLES = ["eta_glob", "lambda_h", "eta_is", "eta_mech"]


def plot_compressor_3d(T_con_out: float, compressor: Compressor, save_path: pathlib.Path, variables: list = None):
    # Create data points
    T_eva_in = np.arange(-15, 26, 5) + 273.15
    n = np.linspace(0.2, 1, 20)
    T_eva_in, n = np.meshgrid(T_eva_in, n)
    # Calculate Z values
    if variables is None:
        variables = VARIABLES

    calc_compressor_vector = np.vectorize(custom_compressor.calc_compressor)
    Z = calc_compressor_vector(T_con_out=T_con_out, T_eva_in=T_eva_in, n=n, compressor=compressor)
    # Create the 3D plot
    if len(variables) <= 3:
        shape = (1, 3)
    else:
        shape = (2, int(len(variables) / 2))
    fig, axes = plt.subplots(*shape, subplot_kw=dict(projection='3d'), squeeze=False)
    for variable, ax in zip(variables, axes.flatten()):
        get_value = np.vectorize(
            lambda x: float(x.get(variable).value) if variable in x.get_variable_names() else np.NaN
        )
        surf = ax.plot_surface(T_eva_in - 273.15, n, get_value(Z), cmap='viridis', edgecolor='none')
        # Add labels and title
        ax.set_xlabel('T_eva_in in °C')
        ax.set_ylabel('n in -')
        ax.set_zlabel(variable)
        # Add colorbar
        # ax.colorbar(surf)
    name = f'T_con_out={int(T_con_out - 273.15)}'
    fig.suptitle(f'{name} °C')
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(save_path.joinpath(f"{name}_3d.png"))
    plt.close("all")


def plot_compressor_2d(T_con_out: float, compressor: Compressor, save_path: pathlib.Path, variables: list = None,
                       n_max: float = 120):
    plot_config = utils.load_plot_config(language="de")
    # Create data points
    T_eva_in = np.arange(-10, 16, 5) + 273.15
    n = np.linspace(0.2 * 120, n_max, 20) / 120
    T_eva_in, n = np.meshgrid(T_eva_in, n)
    if variables is None:
        variables = VARIABLES

    # Calculate Z values
    calc_compressor_vector = np.vectorize(custom_compressor.calc_compressor)
    Z = calc_compressor_vector(T_con_out=T_con_out, T_eva_in=T_eva_in, n=n, compressor=compressor)
    # Create the 3D plot
    if len(variables) <= 3:
        shape = (1, 3)
    else:
        shape = (2, int(len(variables) / 2))

    fig, axes = plt.subplots(*shape, sharex=True, figsize=utils.get_figure_size(2, 2))
    # For each unique T_eva_in value, create a line
    for variable, ax in zip(variables, axes.flatten()):
        get_value = np.vectorize(
            lambda x: plot_config.scale(variable, float(x.get(variable).value))
            if variable in x.get_variable_names() else np.NaN
        )
        for i in range(T_eva_in.shape[1]):
            temp = T_eva_in[0, i] - 273.15  # Get temperature value
            ax.plot(
                plot_config.scale("n_compressor", n[:, 0]),
                get_value(Z[:, i]),
                label=f'{temp:.1f}°C',
                marker='o'
            )
        ax.set_ylabel(plot_config.get_label_and_unit(variable))
        ax.grid(True)
    for ax in axes[-1, :]:
        ax.set_xlabel(plot_config.get_label_and_unit("n_compressor"))
    axes[0, 1].legend(bbox_to_anchor=(1, 1), ncol=1, loc="upper left")
    T_con_out_degC = int(T_con_out - 273.15)
    fig.suptitle("$T_\mathrm{VL}=%s$ °C" % T_con_out_degC)
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(save_path.joinpath(f"T_con_out={T_con_out_degC}_2d.png"))
    plt.close("all")


def plot_compressors_2d(T_con_out: float, compressors: dict, save_path: pathlib.Path, variables: list = None,
                        n_max: float = 120):
    plot_config = utils.load_plot_config(language="de")
    # Create data points
    T_eva_in = np.arange(-10, 16, 10) + 273.15
    n = np.arange(0.2, n_max / 120 + 0.01, 0.1)
    T_eva_in, n = np.meshgrid(T_eva_in, n)
    if variables is None:
        variables = VARIABLES

    # Calculate Z values
    calc_compressor_vector = np.vectorize(custom_compressor.calc_compressor)
    for compressor in compressors:
        compressors[compressor]["data"] = calc_compressor_vector(
            T_con_out=T_con_out, T_eva_in=T_eva_in, n=n,
            compressor=compressors[compressor]["model"]
        )
    # Create the 3D plot
    if len(variables) <= 3:
        shape = (1, 3)
    else:
        shape = (2, int(len(variables) / 2))

    colors = [EBCColors.blue, EBCColors.grey, EBCColors.red]

    fig, axes = plt.subplots(*shape, sharex=True, figsize=utils.get_figure_size(2, 2))
    # For each unique T_eva_in value, create a line
    for variable, ax in zip(variables, axes.flatten()):
        get_value = np.vectorize(
            lambda x: plot_config.scale(variable, float(x.get(variable).value))
            if variable in x.get_variable_names() else np.NaN
        )
        for i in range(T_eva_in.shape[1]):
            temp = T_eva_in[0, i] - 273.15  # Get temperature value
            for compressor, items in compressors.items():
                ax.plot(
                    plot_config.scale("n_compressor", n[:, 0]),
                    get_value(items["data"][:, i]),
                    color=colors[i],
                    label=f'{compressor}: {temp:.1f}°C',
                    **items["plot_kwargs"]
                )
        ax.set_ylabel(plot_config.get_label_and_unit(variable))
        ax.grid(True)
    for ax in axes[-1, :]:
        ax.set_xlabel(plot_config.get_label_and_unit("n_compressor"))
    axes[0, 1].legend(bbox_to_anchor=(1, 1), ncol=1, loc="upper left")
    T_con_out_degC = int(T_con_out - 273.15)
    fig.suptitle("$T_\mathrm{VL}=%s$ °C" % T_con_out_degC)
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(save_path.joinpath(f"T_con_out={T_con_out_degC}_2d.png"))
    plt.close("all")


def plot_compressors_all_T_con_out(compressors, save_path, T_con_out_max: float = 70, n_max: float = 120):
    for T_con_out in T_con_outs[T_con_outs <= T_con_out_max]:
        plot_compressors_2d(
            T_con_out=T_con_out + 273.15, compressors=compressors,
            save_path=save_path, n_max=n_max
        )


def plot_compressor_all_T_con_out(compressor, save_path, T_con_out_max: float = 70, n_max: float = 120):
    for T_con_out in T_con_outs[T_con_outs <= T_con_out_max]:
        # plot_compressor_3d(
        #    T_con_out=T_con_out + 273.15, compressor=compressor,
        #    save_path=save_path
        # )
        plot_compressor_2d(
            T_con_out=T_con_out + 273.15, compressor=compressor,
            save_path=save_path, n_max=n_max
        )


def plot_2d_c10(compressor: Compressor, variables: list, save_path: pathlib.Path):
    markers = ["o", "D", "v", "s"]
    colors = ["orange", "blue", "green", "gray"]
    # Create data points
    T_eva_in = np.array([-3.0, 2.0, 7.0, 12.0]) + 273.15
    n = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]) / 120
    T_eva_in_mesh, n_mesh = np.meshgrid(T_eva_in, n)
    calc_compressor_vector = np.vectorize(custom_compressor.calc_compressor)
    # Create the 3D plot
    if len(variables) <= 3:
        shape = (1, 3)
    else:
        shape = (2, int(len(variables) / 2))
    # limits = {
    #    35: [0.62, 0.72],
    #    50: [0.61, 0.7],
    #    65: [0.54, 0.65]
    # }
    for T_con_out in [35, 50, 65]:
        # limit = limits[T_con_out]
        T_con_out += 273.15
        # Calculate Z values
        Z = calc_compressor_vector(T_con_out=T_con_out, T_eva_in=T_eva_in_mesh, n=n_mesh, compressor=compressor)
        fig, axes = plt.subplots(*shape, squeeze=False)
        for variable, ax in zip(variables, axes.flatten()):
            get_value = np.vectorize(
                lambda x: float(x.get(variable).value) if variable in x.get_variable_names() else np.NaN)
            for i, T_eva in enumerate(T_eva_in):
                ax.plot(
                    n * 120, get_value(Z[:, i]),
                    label=f"{int(T_eva - 273.15)} °C",
                    marker=markers[i], color=colors[i]
                )
                # Add labels and title
                ax.set_ylabel(f"{variable} in %")
                ax.set_xlabel('n in Hz')
            # ax.set_ylim(limit)
        axes[0, 0].legend(bbox_to_anchor=(0, 1), ncol=2, loc="lower left")
        name = f'T_con_out={T_con_out - 273.15}'
        fig.suptitle(f'{name} °C')
        os.makedirs(save_path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path.joinpath(f"{name}.png"))
        plt.close("all")


def plot_2d_login(save_path, compressor: Compressor):
    variables = ["eta_glob", "lambda_h"]
    path = DATA_PATH.joinpath("map_generation", "login_efficiencies")
    dfs = [pd.read_excel(path.joinpath(variable + ".xlsx"), header=[0, 1, 2]) for variable in variables]
    T_eva_in = {
        -3: ("o", "orange"),
        2: ("D", "blue"),
        7: ("v", "green"),
        12: ("s", "gray")
    }
    shape = (2, 2)
    for T_con_out in [35, 50, 65]:
        fig, axes = plt.subplots(*shape, squeeze=False)
        for T_eva, args in T_eva_in.items():
            marker, color = args
            eta_mech = []
            eta_is = None
            n_eta = None
            for i, variable in enumerate(variables):
                df = dfs[i]
                ax = axes.flatten()[i]
                n = df.loc[:, (T_con_out, T_eva, "n")].values
                is_na = np.isnan(n)
                n = n[~is_na]
                eta = df.loc[~is_na, (T_con_out, T_eva, variable)].values
                ax.plot(
                    n,
                    eta,
                    label=f"{T_eva} °C",
                    marker=marker, color=color
                )
                # Add labels and title
                ax.set_ylabel(f"{variable} in -")
                ax.set_xlabel('n in Hz')
                if variable == "eta_glob":
                    eta_mech = np.array([
                        custom_compressor.calc_compressor(
                            T_con_out=T_con_out + 273.15,
                            T_eva_in=T_eva + 273.15,
                            n=_n / 120,
                            compressor=compressor).get("eta_mech").value
                        for _n in n
                    ])
                    eta_is = eta / eta_mech
                    n_eta = n
            for i, eta, variable in zip([2, 3], [eta_is, eta_mech], ["eta_is", "eta_mech"]):
                ax = axes.flatten()[i]
                ax.plot(
                    n_eta,
                    eta,
                    label=f"{T_eva} °C",
                    marker=marker, color=color
                )
                # Add labels and title
                ax.set_ylabel(f"{variable} in -")
                ax.set_xlabel('n in Hz')

        axes[0, 0].legend(bbox_to_anchor=(0, 1), ncol=2, loc="lower left")
        name = f'T_con_out={T_con_out}'
        fig.suptitle(f'{name} °C')
        os.makedirs(save_path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path.joinpath(f"{name}.png"))
        plt.close("all")


def plot_all_vitocal_compressor_options(med_prop: RefProp):
    results_folder = RESULTS_FOLDER.joinpath("vitocal")
    cases = custom_compressor.get_eta_mech_cases()
    for eta_mech_name in cases:
        # C10 Data
        for regression in [True]:
            # no regression makes more sense as direct data leads to wrong eta interpolations.
            for corrected in [False]:
                # Corrected = True remove low mass flow rates but this is a random assumption
                CORR_STR = "_corr" if corrected else ""
                REG_STR = "_reg" if regression else ""
                compressor = custom_compressor.get_vitocal_compressor(
                    med_prop=med_prop, eta_mech_name=eta_mech_name,
                    c10_name=f"10C_WHP07600{CORR_STR}", regression=regression
                )
                # plot_2d_c10(
                #    variables=VARIABLES,
                #    compressor=compressor,
                #    save_path=results_folder.joinpath("plots_c10_2d", eta_mech_name + CORR_STR + REG_STR)
                # )
                plot_compressor_all_T_con_out(
                    compressor=compressor,
                    save_path=results_folder.joinpath("plots_c10", eta_mech_name + CORR_STR + REG_STR),
                    n_max=100
                )
        # Login data
        compressor = custom_compressor.get_login_compressor(
            med_prop=med_prop, eta_mech_name=eta_mech_name
        )
        # plot_2d_login(
        #    compressor=compressor,
        #    save_path=results_folder.joinpath("plots_login_2d", eta_mech_name)
        # )
        plot_compressor_all_T_con_out(
            compressor=compressor,
            save_path=results_folder.joinpath("plots_login", eta_mech_name),
            n_max=100
        )


def compare_login_to_c10(med_prop: RefProp):
    results_folder = RESULTS_FOLDER.joinpath("vitocal")
    cases = custom_compressor.get_eta_mech_cases()
    for eta_mech_name in cases:
        regression = True
        corrected = False
        CORR_STR = "_corr" if corrected else ""
        REG_STR = "_reg" if regression else ""
        compressor_c10 = custom_compressor.get_vitocal_compressor(
            med_prop=med_prop, eta_mech_name=eta_mech_name,
            c10_name=f"10C_WHP07600{CORR_STR}", regression=regression
        )
        compressor_login = custom_compressor.get_login_compressor(
            med_prop=med_prop, eta_mech_name=eta_mech_name
        )
        compressors = {
            "Messung": {"model": compressor_login, "plot_kwargs": {"marker": "s", "linestyle": "--"}},
            "EN 12900": {"model": compressor_c10, "plot_kwargs": {"marker": "o", "linestyle": "-"}}
        }
        plot_compressors_all_T_con_out(
            compressors=compressors,
            save_path=results_folder.joinpath("plots_compare_c10_login", eta_mech_name + CORR_STR + REG_STR),
            n_max=100
        )


def plot_optihorst_compressor(ref_prop: RefProp):
    for config_name in [
        "EN_MEN412_EN412",
        "EN_MEN412_Linear"
    ]:
        compressor = custom_compressor.get_optihorst_compressor(ref_prop, config_name=config_name)

        plot_compressor_all_T_con_out(
            compressor=compressor,
            save_path=RESULTS_FOLDER.joinpath("plots_optihorst", config_name),
            T_con_out_max=60
        )


if __name__ == '__main__':
    from bes_rules import REF_PROP_PATH

    REF_PROP = RefProp(
        fluid_name="Propane",
        ref_prop_path=REF_PROP_PATH.as_posix()
    )
    # plot_optihorst_compressor(REF_PROP)
    plot_all_vitocal_compressor_options(REF_PROP)
    compare_login_to_c10(REF_PROP)
