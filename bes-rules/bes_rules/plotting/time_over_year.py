import pathlib
from typing import List

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bes_rules.configs import PlotConfig


def plot_days_in_year_over_hours_in_day(
        df: pd.DataFrame,
        variables: List[str],
        plot_config: PlotConfig,
        save_path: pathlib.Path = None
):
    df = df.loc[:86400 * 365 - 1]
    df.to_datetime_index(origin=datetime.datetime(2023, 1, 1))
    df = df.to_df()
    df = df[~df.index.duplicated(keep='first')]

    df = df.loc[:, variables]
    df = plot_config.scale_df(df=df)

    # Extract day of the year and minute of the day from the datetime index
    df['day_of_year'] = df.index.dayofyear
    df['hour_of_day'] = df.index.hour + df.index.minute / 60 + 1
    for variable in variables:
        # Pivot the DataFrame to create a matrix for the heatmap
        heatmap_data = df.pivot(index='day_of_year', columns='hour_of_day', values=variable)

        # Create the heatmap
        plt.figure()  # figsize=(15, 8))
        sns.heatmap(heatmap_data, cmap='rocket_r', cbar_kws={'label': plot_config.get_label_and_unit(variable)})

        # Customize the plot
        plt.title(plot_config.get_label_and_unit(variable))
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Year')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path.joinpath(f"time_over_day_and_year_{variable.split('.')[-1]}.png"))
    # Show the plot
    plt.show()
