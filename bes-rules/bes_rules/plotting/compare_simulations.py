from pathlib import Path

import matplotlib.pyplot as plt

from bes_rules.configs import StudyConfig, PlotConfig

from ebcpy import TimeSeriesData


def compare_simulations(path: Path, idx_outlier: int, idx_reference: int):
    tsd_ref = TimeSeriesData(path.joinpath(f"iterate_{idx_reference}.hdf")).to_df()
    tsd_out = TimeSeriesData(path.joinpath(f"iterate_{idx_outlier}.hdf")).to_df()
    plot_config = PlotConfig.load_default()
    tsd_out = plot_config.scale_df(tsd_out)
    tsd_ref = plot_config.scale_df(tsd_ref)
    variables = [
        "outputs.hydraulic.gen.PEleHR.integral",
        "outputs.hydraulic.gen.PEleHR.value",
        "outputs.building.TZone[1]"
    ]
    fig, axes = plt.subplots(len(variables), 1, sharex=True)

    for ax, variable in zip(axes, variables):
        for color, label, tsd in zip(["blue", "red"], ["reference", "outlier"], [tsd_ref, tsd_out]):
            ax.plot(
                tsd.index, tsd.loc[:, variable],
                label=label, color=color,
                linestyle="--" if label == "outlier" else "-"
            )
        ax.set_ylabel(plot_config.get_label_and_unit(variable))
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    CONFIG = StudyConfig.from_json(r"R:\_Dissertationen\fwu\06_Diss\03_Ergebnisse\Test180Cases\study_config.json")
    compare_simulations(
        CONFIG.study_path.joinpath("DesignOptimizationResults", "TRY2015_485809134724_Jahr_NoRetrofit1983_SingleDwelling_M_0K-Per-IntGai"),
        idx_outlier=2, idx_reference=4
    )
