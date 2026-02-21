import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bes_rules.rule_extraction.clustering import clustering_medoid
from bes_rules.configs.inputs import WeatherConfig
from bes_rules.boundary_conditions.weather import get_weather_configs_by_names
from bes_rules.plotting import EBCColors


def cluster_weather_year(weather_config: WeatherConfig):
    df = weather_config.get_hourly_weather_data()
    # Only heating season
    n_per_day = 24
    start_day_heating = 274
    end_day_heating = 119  # 120 normally, but should be only full weeks
    df = pd.concat([df.iloc[:end_day_heating * n_per_day], df.iloc[start_day_heating * n_per_day:]])
    # Trick clustering to believe seven days are just one day
    n_weeks = int(len(df) / 7 / 24)
    number_of_clusters = 1
    inputs = np.array([
        df.loc[:, "t"],
        df.loc[:, "B"],
        df.loc[:, "D"],
    ])
    weights = [0.9, 0.03, 0.07]
    (inputs, nc, z, inputsTransformed, obj) = clustering_medoid.cluster(
        inputs,
        number_of_clusters,
        n_days=n_weeks,
        norm=2,
        mip_gap=0,
        weights=weights
    )

    fig, ax = plt.subplots(1, 1)
    plot_last = []
    week_clustered = None
    for n_week in range(n_weeks):
        _x_values = range(24 * 7)
        _y_values = df.iloc[int(n_week * 24 * 7):int((n_week + 1) * 24 * 7)]["t"]
        if z[n_week, n_week] == 1:
            week_clustered = n_week
            plot_last = [_x_values, _y_values]
            continue
        ax.plot(_x_values, _y_values, color=EBCColors.light_grey)

    ax.plot(*plot_last, color="black")
    if week_clustered < end_day_heating / 7:
        start_day = week_clustered * 7
    else:
        start_day = start_day_heating + (week_clustered * 7 - 119)
    ax.set_title(f"Start day of clustered week: {start_day}")
    ax.set_xlabel("Hour in Week")
    ax.set_ylabel("$T_\mathrm{Oda}$ in °C")
    plt.show()
    return inputs


if __name__ == '__main__':
    cluster_weather_year(get_weather_configs_by_names(region_names=["Potsdam"])[0])
