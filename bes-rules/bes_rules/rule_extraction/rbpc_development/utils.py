import datetime
import logging
import os
import pathlib
import json
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ebcpy.preprocessing import convert_datetime_index_to_float_index
from ebcpy import TimeSeriesData

from bes_rules.utils.function_fit import create_linear_regression, plot_surface_of_function_fit
from bes_rules.utils.heat_pumps import load_vitocal250_COPs

logger = logging.getLogger(__name__)


def load_results(save_path: pathlib.Path):
    config_path = pathlib.Path(__file__).parents[4].joinpath(
        "student_theses/peter/mapping_besmod.json"
    )
    with open(config_path) as mapping:
        column_mapping = json.load(mapping)
    column_mapping["name"] = "time"

    start_day = pd.Timestamp(year=2015, month=1, day=1)

    skip_rows = [0, 2]
    df = pd.read_csv(save_path / "sim_agent.csv", skiprows=skip_rows, header=0, dtype=float).rename(
        columns=column_mapping)
    df = df.rename(columns={"P_pv": "P_PV", "Tamb": "T_outdoor_air"})
    df["time"] = start_day + pd.to_timedelta(df["time"], unit="s")
    df.set_index("time", inplace=True)
    # select timeinterval
    dt = pd.Timedelta(days=2)
    df = df.loc[
         pd.Timestamp(year=2015, month=3, day=1) - dt:pd.Timestamp(year=2015, month=3, day=31, hour=23, minute=59) - dt,
         :]
    df.index = df.index + dt
    pv_path = pathlib.Path(__file__).parents[4].joinpath(
        "student_theses/peter/pv.hdf"
    )
    df_sol = TimeSeriesData(pv_path).to_df().loc[:, "outputs.electrical.gen.PElePV.value"]
    df_sol.index = start_day + pd.to_timedelta(df_sol.index, unit="seconds")
    df = pd.merge(left=df, right=df_sol, left_index=True, right_index=True, how="left")
    df.rename(columns={"outputs.electrical.gen.PElePV.value": "P_el_pv"}, inplace=True)
    df = df.interpolate(method="time")

    return df


def _get_start_day(save_path: pathlib.Path):
    return int(save_path.name.split("_")[0].replace("start=", ""))


def merge_start_and_end_of_year(
        save_path_start_of_year: pathlib,
        save_path_end_of_year: pathlib,

):
    df_end = load_results(save_path_end_of_year)
    df_end = convert_datetime_index_to_float_index(df_end)
    df_start = load_results(save_path_start_of_year)
    df_start = convert_datetime_index_to_float_index(df_start)
    df_start.index += 365 * 86400
    df_end.index += _get_start_day(save_path_end_of_year) * 86400
    df = pd.concat([df_end, df_start])
    new_save_path = save_path_end_of_year.parent.joinpath(
        save_path_end_of_year.name.split("_")[0] + "_" +
        save_path_start_of_year.name.split("_")[1], "results.xlsx"
    )
    os.makedirs(new_save_path.parent, exist_ok=True)
    df.to_excel(new_save_path)
    print(new_save_path)


def split_df_into_days(df: pd.DataFrame) -> List[pd.DataFrame]:
    start_time = df.index[0]
    end_time = df.index[-1]
    n_days = (end_time - start_time).days
    dfs = []
    for day in range(int(n_days) + 1):
        df_day = df.loc[
                 start_time + datetime.timedelta(days=day):
                 start_time + datetime.timedelta(seconds=(day + 1) * 86400 - 1)
                 ].copy()
        if not df_day.empty:
            dfs.append(df_day)
    return dfs


def check_results_correctness(df):
    greater_zeros = ["Q_HP", "P_EL_HP", "P_EL_HR", "P_EL_Dem", "P_PV", "COP_HP", "Q_HR", "Q_Hou", "Q_Penalty"]
    for col in greater_zeros:
        if df[col].min() < -1e-5:
            logger.error(f"{col} must be greater zero but is {df[col].min()}")
    diff_elec = df["P_EL_HR"] + df["P_EL_HP"] + df["P_EL_Dem"] - df["P_PV"] - df["P_EL"]
    if not np.all(np.isclose(diff_elec, 0, atol=1.e-5, rtol=1.e-3)):
        logger.warning(f"Electricity balance must be zero but is at max {diff_elec.abs().max()} Wh")
        plt.figure()
        plt.plot(diff_elec)
        plt.show()

    energy_balance_time_no_charge = df["Q_HP"] + df["Q_HR"] + df["Q_Penalty"] - df["Q_Hou"] - df["Q_Sto_Loss"]
    energy_balance_time = energy_balance_time_no_charge - df["QStoDischarge"]
    if not np.all(np.isclose(
            energy_balance_time.dropna(),
            0, atol=1.e-3, rtol=1.e-2
    )):
        logger.warning(f"Electricity balance must be zero but is at max {energy_balance_time.dropna().abs().max()} Wh")
        plt.figure()
        plt.plot(energy_balance_time)
        plt.show()

    diff_storage = (df.iloc[-1]["Q_Sto_Energy"] - df.iloc[0]["Q_Sto_Energy"]) / 3600  # [Jh/s] -> [Wh]
    df_integral = df.sum() * 0.25  # [W * h] -> [Wh]

    diff_energy = (
            df_integral["Q_HP"] +
            df_integral["Q_HR"] +
            df_integral["Q_Penalty"] -
            df_integral["Q_Hou"] -
            df_integral["Q_Sto_Loss"] -
            diff_storage -
            energy_balance_time_no_charge.iloc[0] * 0.25
    # First time step is "free" heating as the storage temperature is fixed
    )
    if not np.isclose(diff_energy, 0, atol=1.e-3, rtol=1.e-2):
        logger.warning(f"Energy balance must be zero but is {diff_energy} Wh")


def create_heat_pump_regression():
    df_cop = load_vitocal250_COPs("cop_extrapolation")
    df_Q = load_vitocal250_COPs("QConMax")
    cop = df_cop.to_numpy().flatten()
    Q = df_Q.to_numpy().flatten()
    TCon = np.concatenate([df_cop.columns.to_numpy() for _ in df_cop.index])
    TAir = np.array([[idx for _ in df_cop.columns] for idx in df_cop.index]).flatten()
    variables = {
        "TAir": TAir,
        "TCon": TCon,
        "TAir*TCon": TAir * TCon,
        "TAir**2": TAir * TAir,
        "TAir**2*TCon": TAir * TAir * TCon,
        "TAir**3": TAir * TAir * TAir
    }
    create_linear_regression(variables=variables, y=cop, y_name="COP")
    create_linear_regression(variables=variables, y=Q, y_name="Q_HP_max")
    # For PEle, part load needs to be considered:
    part_loads = [0.25, 0.5, 0.75, 1]
    cop_part_load = np.concatenate([cop for _ in part_loads])
    TCon_part_load = np.concatenate([TCon for _ in part_loads])
    TAir_part_load = np.concatenate([TAir for _ in part_loads])
    Q_part_load = np.concatenate([Q * part_load for part_load in part_loads])

    variables = {
        "TAir": TAir_part_load,
        "TCon": TCon_part_load,
        "Q": Q_part_load,
        "COP": cop_part_load,
        "TAir*TCon": TAir_part_load * TCon_part_load,
        "TAir*Q": Q_part_load * TAir_part_load,
        "TAir*COP": cop_part_load * TAir_part_load,
        "TAir**2": TAir_part_load * TAir_part_load,
        "TAir**2*TCon": TAir_part_load * TAir_part_load * TCon_part_load,
        "TAir**2*Q": TAir_part_load * TAir_part_load * Q_part_load,
        "TAir**2*COP": cop_part_load * TAir_part_load * Q_part_load,
        "TAir**3": TAir_part_load * TAir_part_load * TAir_part_load,
    }
    create_linear_regression(variables=variables, y=Q_part_load / cop_part_load, y_name="PEleHP")


def calculate_sum_of_overheating(clustering_result_day):
    return [
        np.sum(values).round(0)
        for values in clustering_result_day["results"]["Clusterseries"].values
    ]


def get_cluster_time_series_map(clustering_result_day):
    daily_sums = calculate_sum_of_overheating(clustering_result_day)
    time_series = clustering_result_day["results"]["Clusterseries"].values
    return {daily_sum: time_series_day for daily_sum, time_series_day in zip(daily_sums, time_series)}


def get_features(df: pd.DataFrame, idx_day, n_days: int):
    _start_day = df.index[0] + datetime.timedelta(days=idx_day)
    _end_day = _start_day + datetime.timedelta(days=1)
    if idx_day + 1 < n_days:
        shift_suntime = datetime.timedelta(hours=6)
    else:
        shift_suntime = datetime.timedelta(hours=0)
    feature_data_per_day = {
        "PElePVMax": np.max(df.loc[_start_day:_end_day, "P_PV"].values),
        "TAirMea": np.mean(df.loc[_start_day + shift_suntime:_end_day + shift_suntime, "T_outdoor_air"].values),
    }
    return feature_data_per_day


def cop(T_Vl, TAir):
    return (
            965.902231300392 +
            -9.533246340114223 * TAir +
            -1.029123761692837 * T_Vl +
            0.008767379505684366 * TAir * T_Vl +
            0.029481697344765898 * TAir ** 2 +
            -1.899325852627404e-05 * TAir ** 2 * T_Vl +
            -2.6599338912753304e-05 * TAir ** 3
    )


def QCon(T_Vl, TAir):
    return (
            3189231.4110641507 +
            -32220.980830563 * TAir +
            -1685.3455766577608 * T_Vl +
            10.644665614582552 * TAir * T_Vl +
            109.27161006239132 * TAir ** 2 +
            -0.016922003609194197 * TAir ** 2 * T_Vl +
            -0.12370531548537612 * TAir ** 3
    )


if __name__ == "__main__":
    #create_heat_pump_regression()
    plot_surface_of_function_fit(variable="QCon", version="old", regression=QCon)
    plot_surface_of_function_fit(variable="QCon", version="new", regression=QCon)
    plot_surface_of_function_fit(variable="COP", version="old", regression=cop)
    plot_surface_of_function_fit(variable="COP", version="new", regression=cop)
    # merge_start_and_end_of_year(
    #     save_path_end_of_year=pathlib.Path(r"D:/fwu/02_Paper/zcbe/open_loop/start=274_stop=365"),
    #     save_path_start_of_year=pathlib.Path(r"D:/fwu/02_Paper/zcbe/open_loop/start=0_stop=120")
    # )
