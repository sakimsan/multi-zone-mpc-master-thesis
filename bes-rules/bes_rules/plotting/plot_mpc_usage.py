import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def plot():
    plt.rcParams.update(
        {
         "figure.figsize": [6.24, 9],
         "font.size": 14,
         "figure.dpi": 250,
         "figure.autolayout": True
         }
    )
    path = pathlib.Path(r"N:\Forschung\EBC1005_EON_IHC_Testbench_GES\Data\Reglerentwicklung\Results\BES_IHC\DesignOptimizationResults\TRY2015_523845130645_Jahr_WSchV1984_SingleDwelling_NoDHW_0K-Per-IntGai_elePro")
    df = pd.read_excel(path.joinpath("Design_2023.xlsx"), index_col=0)
    fig, ax = plt.subplots(3, 1, sharex=True)
    df.index /= 3600
    df = df.loc[7800:8100]
    df.index -= 7800
    ax[0].plot(df.index, df.loc[:, "c_grid"] * 1000, color="blue")
    ax[1].plot(df.index, df.loc[:, "TBufSet"] - 273.15, color="blue", label="set")
    ax[1].plot(df.index, df.loc[:, "hydraulic.control.sigBusDistr.TStoBufTopMea"] - 273.15, color="red", linestyle="--", label="mea")
    ax[1].legend(loc="upper right")
    ax[0].set_ylabel("$c_\mathrm{el}$ in €/kWh")
    ax[1].set_ylabel("$T_\mathrm{flow}$ in °C")
    ax[2].set_ylabel("$T_\mathrm{Oda}$ in °C")
    ax[2].plot(df.index, df.loc[:, "hydraulic.control.weaBus.TDryBul"] - 273.15, color="blue")
    ax[2].set_xlabel("Time in h")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(path.joinpath("example_case.png"))
    plt.show()


if __name__ == '__main__':
    plot()