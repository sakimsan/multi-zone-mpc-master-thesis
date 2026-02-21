from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from ebcpy import TimeSeriesData

from bes_rules.plotting.utils import get_figure_size
from bes_rules import DATA_PATH
from bes_rules.configs.plotting import PlotConfig


def plot_storage_motivation(save_path: Union[Path, str]):
    save_path = Path(save_path)
    plot_config = PlotConfig.load_default()
    file = DATA_PATH.joinpath("storage_motivation.mat")
    df = TimeSeriesData(file).to_df()
    df.index /= 3600
    periods = {
        "good_control": [120, 128],
        "dhw_tapping": [140, 148],
        "cycling": [1014, 1022],
        "evu_block": [128, 136]
    }
    var = "building.buiMeaBus.TZoneMea[1]"
    for name, times in periods.items():
        fig, ax = plt.subplots(1, 1, figsize=get_figure_size(n_columns=1.5))
        start, end = times
        sub_df = df.loc[start:end]
        ax.plot(sub_df.index - sub_df.index[0], sub_df[var] - 273.15)
        ax.set_ylabel(plot_config.get_label_and_unit(var))
        ax.set_xlabel("Time in h")
        ax.set_ylim([18, 21])
        fig.tight_layout()
        fig.savefig(save_path.joinpath(f"storage_motivation_{name}.png"))


if __name__ == '__main__':
    plot_storage_motivation(save_path=r"D:\00_temp")
