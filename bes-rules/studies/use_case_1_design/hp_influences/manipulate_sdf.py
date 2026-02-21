import itertools
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sdf
from bes_rules.plotting import EBCColors, get_figure_size


def remove_compressor_speed_influence_on_COP(
        filepath: pathlib.Path,
        save_path: pathlib.Path
):
    """
    Remove the influence of the compressor speed on the
    COP and electrical power consumption.

    Args:
        filepath (pathlib.Path): sdf file
        save_path (pathlib.Path): Directory where to store the sdf file.
    """
    variables_to_remove_influence = ["COP", "COP_outer"]
    calc_P_el = {"COP": 0, "Q_con": 0}
    dataset = sdf.load(str(filepath))
    for flowsheet in dataset.groups:
        for fluid in flowsheet.groups:
            for data in fluid.datasets:
                if data.name in variables_to_remove_influence:
                    data.data = cast_max_to_n(data.data)
                    if data.name == "COP_outer":
                        calc_P_el["COP"] = data.data
                if data.name == "Q_con_outer":
                    calc_P_el["Q_con"] = data.data
            for data in fluid.datasets:
                if data.name == "P_el":
                    data.data = calc_P_el["Q_con"] / calc_P_el["COP"]

    sdf.save(save_path, group=dataset)


def cast_max_to_n(array_3d):
    """
    For each combination of T_con and T_eva, find the maximum along n
    and cast it back to all n indices.

    Parameters:
    -----------
    array_3d : numpy.ndarray
        Array with shape (n, T_con, T_eva)

    Returns:
    --------
    numpy.ndarray
        Array with same shape where max values are repeated along n axis
    """
    # Find max along n axis (axis=0)
    max_values = np.max(array_3d, axis=0)  # Shape: (T_con, T_eva)

    # Broadcast max values back to original shape
    # Creates a view where max_values is repeated for each n
    return np.broadcast_to(max_values, array_3d.shape)


def create_2d_table_from_sdf(filepath):
    COP = None
    QCon = None
    T_eva_in = None
    T_con_in = None
    dT = None
    n = None
    T2 = None
    # Get values
    dataset = sdf.load(str(filepath))
    for flowsheet in dataset.groups:
        for fluid in flowsheet.groups:
            for data in fluid.datasets:
                if data.name == "COP_outer":
                    COP = data.data
                if data.name == "Q_con_outer":
                    QCon = data.data
                    n = data.scales[0].data
                    T_con_in = data.scales[1].data
                    T_eva_in = data.scales[2].data
                    dT = data.scales[3].data
                if data.name == "T_2":
                    T2 = data.data

    COP_nom = pd.DataFrame(columns=T_eva_in.data)
    COP_max = pd.DataFrame(columns=T_eva_in.data)
    COP_min = pd.DataFrame(columns=T_eva_in.data)
    QCon_nom = pd.DataFrame(columns=T_eva_in.data)
    QCon_max = pd.DataFrame(columns=T_eva_in.data)
    QCon_min = pd.DataFrame(columns=T_eva_in.data)
    T2_max = pd.DataFrame(columns=T_eva_in.data)
    T2_min = pd.DataFrame(columns=T_eva_in.data)
    for i_con, _T_con_in in enumerate(T_con_in):
        for i_eva, _T_eva_in in enumerate(T_eva_in):
            dT_nominal = get_dT_nominal(_T_con_in)
            print(dT_nominal)
            n_nominal = get_n_at_TOda(_T_eva_in - 273.15)
            _T_con_out = _T_con_in
            mask_dT_scale = dT == dT_nominal
            COP_nom.loc[_T_con_out, _T_eva_in] = np.interp(n_nominal, n, COP[:, i_con, i_eva, mask_dT_scale].flatten())
            QCon_nom.loc[_T_con_out, _T_eva_in] = np.interp(n_nominal, n, QCon[:, i_con, i_eva, mask_dT_scale].flatten())
            QCon_max.loc[_T_con_out, _T_eva_in] = QCon[-1, i_con, i_eva, mask_dT_scale].max()
            QCon_min.loc[_T_con_out, _T_eva_in] = QCon[0, i_con, i_eva, mask_dT_scale].min()
            COP_max.loc[_T_con_out, _T_eva_in] = COP[-1, i_con, i_eva, mask_dT_scale].max()
            COP_min.loc[_T_con_out, _T_eva_in] = COP[0, i_con, i_eva, mask_dT_scale].min()
            T2_max.loc[_T_con_out, _T_eva_in] = T2[-1, i_con, i_eva, mask_dT_scale].min()
            T2_min.loc[_T_con_out, _T_eva_in] = T2[0, i_con, i_eva, mask_dT_scale].min()

    # Export to Modelica
    PEle = QCon_max / COP_nom
    print(df_to_modelica_table(PEle, "tabPEle"))
    print(df_to_modelica_table(QCon_max, "tabQCon_flow"))
    from bes_rules import DATA_PATH
    QCon_max.to_excel(DATA_PATH.joinpath("OptiHorstQConMax.xlsx"), sheet_name="QConMaxFromVCLibPy")
    COP_nom.to_excel(DATA_PATH.joinpath("OptiHorstCOP.xlsx"), sheet_name="COPNomFromVCLibPy")

    for T_con in QCon_nom.index:
        fig, ax = plt.subplots(3, 1, sharex=True)
        x = QCon_nom.columns - 273.15
        ax[0].plot(x, QCon_nom.loc[T_con] / 1000, label="Nominal", color=EBCColors.blue)
        ax[0].plot(x, QCon_max.loc[T_con] / 1000, label="Maximum", color=EBCColors.red, linestyle="--")
        ax[0].plot(x, QCon_min.loc[T_con] / 1000, label="Minimum", color=EBCColors.dark_grey, linestyle="--")
        ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=3)
        ax[0].set_ylabel("$\dot{Q}$ in kW")
        ax[1].plot(x, COP_nom.loc[T_con], label="Nominal", color=EBCColors.blue)
        ax[1].plot(x, COP_max.loc[T_con], label="Maximum", color=EBCColors.red, linestyle="--")
        ax[1].plot(x, COP_min.loc[T_con], label="Minimum", color=EBCColors.dark_grey, linestyle="--")
        ax[1].set_ylabel("$COP$ in -")
        ax[2].plot(x, T2_max.loc[T_con] - 273.15, label="Maximum", color=EBCColors.red, linestyle="--")
        ax[2].plot(x, T2_min.loc[T_con] - 273.15, label="Minimum", color=EBCColors.red, linestyle="--")
        ax[2].axhline(140, label="Envelope", color="black")
        ax[2].set_ylabel("$T_\mathrm{Max}$ in °C")
        ax[-1].set_xlabel("$T_\mathrm{Auß}$ in °C")
        fig.suptitle("$T_\mathrm{VL}=$" + f"{T_con-273.15} °C")#
        fig.tight_layout()
    plt.show()


def plot_Q_and_COP_over_n(filepath):
    COP = None
    QCon = None
    T_eva_in = None
    T_con_in = None
    dT = None
    n = None
    T2 = None
    # Get values
    dataset = sdf.load(str(filepath))
    for flowsheet in dataset.groups:
        for fluid in flowsheet.groups:
            for data in fluid.datasets:
                if data.name == "COP_outer":
                    COP = data.data
                if data.name == "Q_con_outer":
                    QCon = data.data
                    n = data.scales[0].data
                    T_con_in = data.scales[1].data
                    T_eva_in = data.scales[2].data
                    dT = data.scales[3].data
                if data.name == "T_2":
                    T2 = data.data

    QConRel = {_n: [] for _n in n}
    COPRel = {_n: [] for _n in n}

    for i_con, _T_con_in in enumerate(T_con_in):
        for i_eva, _T_eva_in in enumerate(T_eva_in):
            dT_nominal = get_dT_nominal(_T_con_in)
            n_nominal = get_n_at_TOda(_T_eva_in - 273.15)
            _T_con_out = _T_con_in
            mask_dT_scale = dT == dT_nominal
            COP_nom = np.interp(n_nominal, n, COP[:, i_con, i_eva, mask_dT_scale].flatten())
            QConMax = QCon[-1, i_con, i_eva, mask_dT_scale].max()
            for idx_n, _n in enumerate(n):
                QConRel[_n].append(QCon[idx_n, i_con, i_eva, mask_dT_scale] / QConMax)
                COPRel[_n].append(COP[idx_n, i_con, i_eva, mask_dT_scale] / COP_nom)

    QConRelMax = {_n: np.max(QConRel[_n]) * 100 for _n in n}
    QConRelMean = {_n: np.mean(QConRel[_n]) * 100 for _n in n}
    QConRelMin = {_n: np.min(QConRel[_n]) * 100 for _n in n}
    COPRelMax = {_n: np.max(COPRel[_n]) * 100 for _n in n}
    COPRelMean = {_n: np.mean(COPRel[_n]) * 100 for _n in n}
    COPRelMin = {_n: np.min(COPRel[_n]) * 100 for _n in n}

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=get_figure_size(2))
    ax[0].plot(n, QConRelMean.values(), label="4D Mittelwert", color=EBCColors.blue)
    ax[0].fill_between(n, QConRelMin.values(), QConRelMax.values(), label="4D Bereich", color=EBCColors.blue, alpha=0.2, linewidth=0.0)
    ax[0].plot(n, n * 100, label="2D", color=EBCColors.red)

    ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=1)
    ax[0].set_ylabel("$\dot{Q}/\dot{Q}_\mathrm{Max}$ in %")
    ax[1].plot(n, COPRelMean.values(), label="4D Mittelwert", color=EBCColors.blue)
    ax[1].fill_between(n, COPRelMin.values(), COPRelMax.values(), label="4D Bereich", color=EBCColors.blue, alpha=0.2, linewidth=0.0)
    ax[1].plot(n, np.ones(len(n)) * 100, label="2D Annahme", color=EBCColors.red)
    ax[1].set_ylabel("$COP/COP_\mathrm{Nom}$ in %")
    ax[0].set_xlabel("$n$ in -")
    ax[1].set_xlabel("$n$ in -")
    fig.tight_layout()
    from bes_rules import LATEX_FIGURES_FOLDER
    fig.savefig(LATEX_FIGURES_FOLDER.joinpath("Appendix", "4_wp", "relative_values_optihorst.png"))
    plt.show()


def df_to_modelica_table(df, name: str):
    # Header erstellen
    header = f"{name}=["

    # Erste Zeile (0 und Spaltennamen)
    first_row = "\n  0"
    for col in df.columns:
        first_row += f", {float(col)}"
    first_row += ";"
    # Datenzeilen
    data_rows = ""
    for idx, row in df.iterrows():
        data_row = f"\n  {float(idx)}"
        for val in row:
            data_row += f", {val}"
        data_rows += data_row + ";"

    # Tabelle zusammenbauen
    table = header + first_row + data_rows[:-1] + "]"

    return table


def get_dT_nominal(T_con_in: float):
    if T_con_in >= 273.15 + 55:
        return 10
    if T_con_in >= 273.15 + 45:
        return 8
    return 5


def get_n_at_TOda(TOda):
    points = {
        -7: 1,
        2: 0.5,
        7: 0.4
    }
    return np.interp(
        TOda,
        list(points.keys()),
        list(points.values())
    )


def plot_part_load_over_TOda():
    TOda = np.arange(-15, 20, 0.5)
    n = get_n_at_TOda(TOda)
    fig, ax = plt.subplots(1, 1)
    ax.plot(TOda, n)
    plt.show()


if __name__ == '__main__':
    # remove_compressor_speed_influence_on_COP(
    #    filepath=pathlib.Path(r"E:\02_Paper\01_vclibpy\Results\MEN_MEN_ENTests.sdf"),
    #    save_path=pathlib.Path(r"E:\02_Paper\01_vclibpy\Results\MEN_MEN_ENTests_COP_const_over_n.sdf")
    # )
    #plot_part_load_over_TOda()
    SDF_PATH = pathlib.Path(r"E:\02_Paper\01_vclibpy\Results_4d_dT\EN_MEN412_Linear\OptiHorst_R410A.sdf")
    SDF_PATH = pathlib.Path(r"D:\02_Paper\01_vclibpy\Results\EN_MEN412_Linear\OptiHorst_R410A.sdf")
    #create_2d_table_from_sdf(filepath=SDF_PATH)
    plot_Q_and_COP_over_n(filepath=SDF_PATH)
