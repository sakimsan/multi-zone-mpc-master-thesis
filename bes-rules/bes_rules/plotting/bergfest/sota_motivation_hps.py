"""
Used for "Figure 7.7: Characteristic performancemaps of HP according to manufacturer’s data sheet"
Author: pme-fwu
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def plot_sota_motivation_hps():

    # Change savepath:
    savepath = os.path.dirname(__file__)
    plt.rcParams.update(
        {"figure.figsize": [13 / 2.54 * 0.75, 15 / 2.54 * 2 / 3],
         "font.size": 16,
         "figure.dpi": 250,
         "font.family": "Arial"
         }
    )
    assert os.path.isdir(savepath), "Given directory is not a valid savepath"

    # Load data:
    csv_path_ashp = os.path.join(os.path.dirname(__file__), "Motivation_Diss_Huchtemann_ASHP.csv")
    csv_path_gshp = os.path.join(os.path.dirname(__file__), "Motivation_Diss_Huchtemann_GSHP.csv")

    data_ashp = pd.read_csv(csv_path_ashp, sep=";")
    data_gshp = pd.read_csv(csv_path_gshp, sep=";")

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("$\mathrm{SCOP}_\mathrm{Norm}$ [-]")
    ax.set_ylabel("$\mathrm{SCOP}_\mathrm{Mess}$ [-]")

    ax.plot([0, 10], [0, 10], color='k')
    ax.scatter(data_ashp["SCOPCalc"], data_ashp["SCOPMeas"], marker="+", label="Luft", color="red", s=200)
    #ax.scatter(data_gshp["SCOPCalc"], data_gshp["SCOPMeas"], marker=".", label="Sole", color="blue", s=200)

    ax.set_xlim([1, 4])
    ax.set_xticks([2, 3, 4])
    ax.set_ylim([1, 4])
    ax.set_yticks([1, 2, 3, 4])
    #ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(savepath, "sota-motivation_hps.svg"))
    plt.show()


if __name__ == "__main__":
    plot_sota_motivation_hps()
