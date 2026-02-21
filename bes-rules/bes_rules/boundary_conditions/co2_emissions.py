from datetime import datetime

import json
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _load_data(path: pathlib.Path):
    with open(path, "r") as file:
        data = json.load(file)
    df = pd.DataFrame(data["data"])
    df["data"] = df["data"].apply(lambda x: x["value"])
    df["epoch"] = df["epoch"].apply(datetime.fromtimestamp)
    df = df.rename(columns={"epoch": "time", "data": "co2"}).set_index("time")
    #df = df.dropna(axis=1)
    np.count_nonzero(df.isna())
    df = df.sort_index()
    return df


def plot_co2_values(
        path: pathlib.Path,
        number_of_days=-1
):
    df = _load_data(path=path)
    #df = df.loc[(df.index[-1] - pd.DateOffset(days=number_of_days)):]
    plt.plot(df)
    plt.ylabel("CO2-Emissionen in g/kWh")
    plt.xlabel("Zeit")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    from bes_rules import DATA_PATH
    PATH = DATA_PATH.joinpath("co2", "co2.json")
    plot_co2_values(PATH)
