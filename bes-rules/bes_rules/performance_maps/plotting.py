import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

from bes_rules.performance_maps.utils import read_data_sheet, apply_constraints
from bes_rules.plotting import EBCColors
from bes_rules import DATA_PATH

Q_con_name = "Q_con_outer"
Q_con_nom_name = f"{Q_con_name}_nom"


def load_result(optimizer_path: Path, parameter_combination: int, frosting: bool, T_con: float):
    datasheet_df = read_data_sheet(
        file_path=DATA_PATH.joinpath("map_generation", "Vitocal_13kW_Datenblatt.xlsx")
    )
    if T_con in datasheet_df['T_con_out'].values:
        datasheet_df = datasheet_df[datasheet_df['T_con_out'] == T_con]
    else:
        warnings.warn(f"Der Wert {T_con=} ist nicht im DataFrame enthalten.")

    path = optimizer_path.joinpath(f"ParameterCombination_{parameter_combination}")
    deviation_path = path.joinpath(
        f"Deviations_{parameter_combination}.xlsx"
    )
    deviation_df = pd.read_excel(deviation_path)
    if frosting:
        vclib_path = path.joinpath("map_with_frosting.csv")
    else:
        vclib_path = path.joinpath("Standard_Propane.csv")

    vclib_df = pd.read_csv(vclib_path, sep=';')

    vclib_df = apply_constraints(vclib_df)

    vclib_df[Q_con_name] /= 1000  # Convert Q_con from W to kW
    vclib_df['T_con_out'] -= 273.15  # Convert T_con_out from K to °C
    vclib_df['T_eva_in'] -= 273.15  # Convert T_eva_in from K to °C

    vclib_df['T_con_out'] = vclib_df['T_con_out'].round()  # Round the values to the nearest integer
    vclib_df['T_eva_in'] = vclib_df['T_eva_in'].round().astype('Int64')  # um Probleme im späteren code zu vermeiden
    vclib_df = vclib_df[vclib_df['T_con_out'] == T_con]
    deviation_df = deviation_df[deviation_df['T_con_out'] == T_con]

    vclib_cop_df = get_vclib_map_for_best_fitting_compressor_speed(
        vclib_df=vclib_df, deviation_df=deviation_df
    )

    return vclib_cop_df, vclib_df, datasheet_df


def get_vclib_map_for_best_fitting_compressor_speed(
        vclib_df: pd.DataFrame,
        deviation_df: pd.DataFrame
):
    vclib_cop_df = pd.DataFrame()
    for idx, T_eva in enumerate(vclib_df['T_eva_in'].unique()):
        vclib_temp = vclib_df[vclib_df['T_eva_in'] == T_eva]
        column_name = f"N_{T_eva}"
        N = deviation_df[column_name].iloc[0]
        if pd.isna(N):
            COP = np.nan
            Q_con = np.nan
        else:
            idx = vclib_temp[vclib_temp['n'] == N].index[0]
            COP = vclib_temp.loc[idx, "COP"]
            Q_con = vclib_temp.loc[idx, Q_con_name]
        vclib_cop_df.loc[idx, "T_eva_in"] = T_eva
        vclib_cop_df.loc[idx, "N"] = N
        vclib_cop_df.loc[idx, "COP_nom"] = COP
        vclib_cop_df.loc[idx, Q_con_nom_name] = Q_con
        vclib_cop_df.loc[idx, "COP_min"] = vclib_temp["COP"].max()
        vclib_cop_df.loc[idx, "COP_max"] = vclib_temp["COP"].min()

    return vclib_cop_df


def optimization_plot(
        optimizer_path: Path,
        parameter_combination: int,
        T_con: float = None,
        frosting: bool = False,
        save_name: str = None
):
    vclib_cop_df, vclib_df, datasheet_df = load_result(
        optimizer_path=optimizer_path,
        parameter_combination=parameter_combination,
        frosting=frosting,
        T_con=T_con
    )

    # initiate Plot
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(6.1, 5))
    # First Plot for Q_nom
    add_lines_and_shading(ax1, datasheet_df, vclib_df, vclib_cop_df)

    # Second subplot for COP
    ax2.plot(
        datasheet_df['T_eva_in'], datasheet_df["COP"],
        label='$COP_{\mathrm{Nenn}}$ Datenblatt',
        color=EBCColors.red)
    ax2.plot(
        vclib_cop_df['T_eva_in'], vclib_cop_df['COP_nom'],
        label='$COP_{\mathrm{Nenn}}$ VCLibPy',
        color=EBCColors.blue)
    ax2.set_xlabel('Außentemperatur in °C')
    ax2.set_ylabel('$COP_{{\mathrm{{Nenn}}}}$ in /')
    # ax2.set_ylabel('$COP$ in /')
    ax2.set_xlim([-21, 31])  # Set the range for T_eva_in
    # ax2.set_ylim([0, 8])  # Set the range for COP
    # ax2.set_yticks(np.arange(0, 9, 2))
    ax2.legend(loc='upper left')
    ax2.grid(True)

    fig.tight_layout()
    if save_name is None:
        save_name = str(parameter_combination)

    plt.savefig(optimizer_path.joinpath(f"datasheet_comparison_{save_name}_{T_con=}.png"))
    plt.close("all")


def add_lines_and_shading(ax, datasheet_df, vclib_df, vclib_cop_df):
    max_Q_con_per_T_eva_in = get_max_Q_con_per_T_eva_in(vclib_df)
    min_Q_con_per_T_eva_in = get_min_Q_con_per_T_eva_in(vclib_df)

    ax.plot(
        max_Q_con_per_T_eva_in['T_eva_in'],
        max_Q_con_per_T_eva_in[Q_con_name],
        label='$\dot{Q}_\mathrm{max}$ VCLibPy',
        color=EBCColors.blue,
        linestyle="-."
    )
    ax.plot(
        datasheet_df['T_eva_in'], datasheet_df['Q_nom'],
        label=r'$\dot{Q}_{\mathrm{Nenn}}$ Datenblatt',
        color=EBCColors.red)
    ax.plot(
        vclib_cop_df['T_eva_in'], vclib_cop_df[Q_con_nom_name],
        label=r'$\dot{Q}_{\mathrm{Nenn}}$ VCLibPy',
        color=EBCColors.blue)
    ax.plot(
        min_Q_con_per_T_eva_in['T_eva_in'],
        min_Q_con_per_T_eva_in[Q_con_name],
        label='$\dot{Q}_\mathrm{min}$ VCLibPy',
        color=EBCColors.blue,
        linestyle="--"
    )
    ax.fill_between(
        datasheet_df['T_eva_in'], datasheet_df['Q_min'], datasheet_df['Q_max'],
        color=EBCColors.light_grey,
        alpha=0.5)

    ax.set_ylabel('Wärmeleistung in kW')
    ax.set_xlim([-21, 31])  # Set the range for T_eva_in
    ax.set_ylim([0, 25])  # Set the range for Q_nom
    ax.set_xticklabels([])  # This line removes the x-axis tick labels
    ax.legend(loc='upper left')
    ax.grid(True)


def get_max_Q_con_per_T_eva_in(vclib_df):
    return vclib_df.groupby('T_eva_in')[Q_con_name].max().reset_index()


def get_min_Q_con_per_T_eva_in(vclib_df):
    return vclib_df.groupby('T_eva_in')[Q_con_name].min().reset_index()


def plot_COP_scatter(
        optimizer_path: Path,
        parameter_combination: int,
        T_con: float = None,
        frosting: bool = False,
        relative_COP: bool = False
):
    vclib_cop_df, vclib_df, datasheet_df = load_result(
        optimizer_path=optimizer_path,
        parameter_combination=parameter_combination,
        frosting=frosting,
        T_con=T_con
    )
    # Merging the dataframes on 'T_eva_in'
    merged_df = pd.merge(vclib_df, vclib_cop_df, on='T_eva_in')

    # Calculating the relative COP difference
    merged_df['Relative_COP_Diff'] = (
            (merged_df["COP"] - merged_df['COP_min']) /
            (merged_df['COP_max'] - merged_df['COP_min']) * 100
    )

    # initiate Plot
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(6.1, 6))

    # First Plot for Q_nom
    add_lines_and_shading(ax1, datasheet_df, vclib_df, vclib_cop_df)

    if relative_COP:
        cop_name = "Relative_COP_Diff"
    else:
        cop_name = "COP"

    for T_eva_in in merged_df['T_eva_in'].unique():
        mask_eva_vclib_df = vclib_cop_df["T_eva_in"] == T_eva_in
        mask_eva = merged_df['T_eva_in'] == T_eva_in
        sc = ax1.scatter(
            merged_df.loc[mask_eva, 'T_eva_in'],
            merged_df.loc[mask_eva, Q_con_name],
            c=merged_df.loc[mask_eva, cop_name] / vclib_cop_df.loc[mask_eva_vclib_df, "COP_nom"].values[0],
            cmap='inferno',
            s=20
        )

    # Second subplot for COP
    ax2.plot(datasheet_df['T_eva_in'], datasheet_df["COP"],
             label='$COP_{\mathrm{Nenn}}$ Datenblatt',
             color=EBCColors.dark_red)
    ax2.plot(vclib_cop_df['T_eva_in'], vclib_cop_df["COP_nom"],
             label='$COP_{\mathrm{Nenn}}$ VCLibPy',
             color=EBCColors.blue)
    ax2.set_xlabel('Außentemperatur in °C')
    ax2.set_ylabel('$COP$ in /')
    ax2.set_xlim([-21, 31])  # Set the range for T_eva_in
    ax2.legend(loc='upper left')
    ax2.grid(True)

    # colorbar
    cbar = fig.colorbar(sc, ax=ax1)
    if relative_COP:
        cbar.set_label('Relative $COP$ Abweichung  (%)')
    else:
        cbar.set_label('$COP$ in /')
    # cbar.ax.set_position([0.095, 0.06, 0.875, 0.05])
    # [left, bottom, width, height]
    fig.tight_layout()
    ax1_pos = ax1.get_position()
    ax2_pos = ax2.get_position()
    ax2.set_position([ax2_pos.x0, ax2_pos.y0, ax1_pos.width, ax2_pos.height])

    plt.savefig(optimizer_path.joinpath(f"cop_scatter_{parameter_combination}_{T_con=}.png"))
    plt.close("all")


def plot_COP_over_n(
        optimizer_path: Path,
        parameter_combination: int,
        T_con: float = None,
        frosting: bool = False,
):
    vclib_cop_df, vclib_df, datasheet_df = load_result(
        optimizer_path=optimizer_path,
        parameter_combination=parameter_combination,
        frosting=frosting,
        T_con=T_con
    )

    # Merging the dataframes on 'T_eva_in'
    merged_df = pd.merge(vclib_df, vclib_cop_df, on='T_eva_in')

    # Calculating the relative COP difference
    merged_df['Relative_COP_Diff'] = (
            (merged_df["COP"] - merged_df['COP_min']) /
            (merged_df['COP_max'] - merged_df['COP_min']) * 100
    )

    color_dict = {
        -20: EBCColors.dark_red,
        -15: EBCColors.red,
        -10: EBCColors.light_red,
        -7: EBCColors.dark_grey,
        2: EBCColors.grey,
        7: EBCColors.light_grey,
        10: EBCColors.blue,
        20: EBCColors.light_blue,
        30: EBCColors.green,
    }

    # Plotting COP over Drehzahl with different colored lines for each Außentemperatur
    fig, ax = plt.subplots(figsize=(6.1, 4))
    merged_df['n'] *= 100

    for T_eva_in, group in merged_df.groupby('T_eva_in'):
        ax.plot(
            group['n'], group['COP'],
            label='$T_\mathrm{U}$ = %s °C' % T_eva_in,
            color=color_dict.get(T_eva_in, EBCColors.light_grey)
        )

    ax.set_xlabel('Relative Drehzahl in %')
    ax.set_ylabel('$COP$ in /')
    ax.grid(True)
    # Reverse the order of the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right')
    plt.savefig(optimizer_path.joinpath(f"cop_over_n_{parameter_combination}_{T_con=}.png"))
    plt.close("all")


if __name__ == '__main__':
    from bes_rules import RESULTS_FOLDER

    PATH = RESULTS_FOLDER.joinpath("vitocal", "map_ihx_new")

    T_cons = [35, 45, 55, 65, 70]
    FROSTING = True
    ITERATE = 0
    for T_CON in T_cons:
        #optimization_plot(
        #    T_con=T_CON,
        #    parameter_combination=ITERATE,
        #    optimizer_path=PATH,
        #    frosting=FROSTING
        #)
        plot_COP_scatter(
            T_con=T_CON,
            parameter_combination=ITERATE,
            optimizer_path=PATH,
            frosting=FROSTING
        )
        plot_COP_over_n(
            T_con=T_CON,
            parameter_combination=ITERATE,
            optimizer_path=PATH,
            frosting=FROSTING,
        )
