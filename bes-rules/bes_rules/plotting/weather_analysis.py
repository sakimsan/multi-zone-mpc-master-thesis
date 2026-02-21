import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from bes_rules.configs import WeatherConfig, PlotConfig
from bes_rules.plotting import EBCColors
from bes_rules.boundary_conditions.weather import get_all_weather_configs


def analyze_weather_data(weather_config: WeatherConfig, save_path: Path):
    df = weather_config.get_hourly_weather_data()
    t_oda_start = int(weather_config.TOda_nominal - 273.15)
    t_odas = np.arange(t_oda_start, t_oda_start + 6)
    data_dict = {}
    for t_oda in t_odas:
        mask_smaller_t_oda = df["t"] < t_oda
        n_consecutive_trues = np.cumsum(mask_smaller_t_oda)
        n_consecutive_trues = n_consecutive_trues[n_consecutive_trues > 0]
        if np.any(n_consecutive_trues > 0):
            data_dict[t_oda] = n_consecutive_trues
    df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data_dict.items()]))
    try:
        hours_below_toda_nominal_plus_one = df[int(weather_config.TOda_nominal - 273.15) + 1].max()
    except KeyError:
        hours_below_toda_nominal_plus_one = 0

    fig, ax = plt.subplots(1, 1)
    sns.violinplot(data=df, ax=ax, orient="v", color=EBCColors.light_grey)

    for i, col in enumerate(df.columns, 0):
        ax.scatter([i], [df[col].max()], marker='d', facecolor='black')
    ax.set_title("$T_\mathrm{Oda,Nom}=$" + f"{round(weather_config.TOda_nominal - 273.15, 2)} °C")
    ax.set_xlabel("$T_\mathrm{Oda}$ in °C")
    ax.set_ylabel("Consecutive hours below $T_\mathrm{Oda}$ in h")
    fig.tight_layout()
    try:
        fig.savefig(save_path.joinpath(weather_config.get_name(pretty_print=True) + ".png"))
    except Exception as err:
        print(err)
    return hours_below_toda_nominal_plus_one


def plot_all_weathers(save_path: Path):
    os.makedirs(save_path, exist_ok=True)
    all_results = {}
    for weather_config in get_all_weather_configs():
        all_results[weather_config.get_name(pretty_print=True)] = [analyze_weather_data(
            weather_config, save_path=save_path
        )]
    pd.DataFrame(all_results).to_excel(save_path.joinpath("weather_critical.xlsx"))


if __name__ == '__main__':
    plot_all_weathers(Path(r"E:\00_temp\weather_analysis"))
