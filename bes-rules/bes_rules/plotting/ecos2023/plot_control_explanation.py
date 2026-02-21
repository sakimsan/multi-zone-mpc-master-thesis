from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from ebcpy import TimeSeriesData



def plot_control_explanation(hdf_path: Path, single_day: int = 91):
    tsd = TimeSeriesData(hdf_path)
    name_p_ele_gen = "outputs.electrical.gen.PElePV.value"
    days_in_year = range(365)
    max_per_day = []
    for day in days_in_year:
        max_per_day.append(tsd.loc[day*86400:(day + 1)*86400, name_p_ele_gen].max() / 1000)

    
    fig, ax = plt.subplots(1, 2, sharey=True)
    
    ax[1].set_xlabel("Time in h")
    ax[0].set_xlabel("Time in d")
    
    ax[0].set_ylabel("$P_\mathrm{el}$ in kW")
    daytime_start_sec = (single_day + 5/24)* 86400
    daytime_end_sec = (single_day + 18/24)*86400
    
    ax[1].plot(
        (tsd.loc[daytime_start_sec:daytime_end_sec].index - single_day * 86400)/ 3600, 
        tsd.loc[daytime_start_sec:daytime_end_sec, name_p_ele_gen] / 1000,
        color="red"
    )
    ax[0].scatter(
        list(days_in_year),
        max_per_day,
        color="red"
    )
    fig.tight_layout()
    plt.savefig("D:\control_explanation.svg")
    
if __name__=="__main__":
    plt.rcParams.update(
        {"figure.figsize": [15 / 2.54, 24 / 2.54 / 2],
         "font.size": 16,
         "figure.dpi": 250,
         "font.family": "Arial"
         }
    )
    plot_control_explanation(hdf_path=r"D:\fwu-jre\BESPVDesOptAllTEST\DesignOptimizationResults\BESPVDesOptAll_weather_TRY2015_535312085881_Wint_building_Case_1_SingleDwelling_dhw_profile_M_user_Standard\iterate_1.hdf")