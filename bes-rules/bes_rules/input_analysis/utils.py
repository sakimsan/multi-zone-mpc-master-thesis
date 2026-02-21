import datetime
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from ebcpy.preprocessing import convert_index_to_datetime_index


from bes_rules.input_analysis.heat_pump_system import estimate_heat_pump_system_demands
from bes_rules.input_analysis.pv import estimate_pv_generation, radTil
from bes_rules.input_analysis import dhw
from bes_rules import DATA_PATH
from bes_rules.configs.inputs import InputConfig


def get_time_series_data_for_one_case(weather_config, case_name, building_config,
                                      interval: str, save_path: Path, day_of_year: int, n_days: int):
    save_path_json = save_path.joinpath("pv_simulations.json")
    save_path_pickle = save_path.joinpath(interval, f"simulation_data_buildings_interval={interval}.pickle")
    with open(save_path_json, "r") as file:
        results = json.load(file)
    df_path = results[weather_config.get_name() + "____" + case_name]
    with open(save_path_pickle, "rb") as file:
        simulated_heat_demand = pickle.load(file)
    polynoms = create_cop_regression_curves()
    TRoom = 273.15 + 20
    df_sum, df_max, df_toda_mean, P_household, Q_demand_DHW = _load_and_convert_df_to_interval(
        df_path, interval,
        weather_config
    )

    cop_polynom_bui = polynoms[weather_config.get_name(location_name=True)]["bui"]
    cop_polynom_dhw = polynoms[weather_config.get_name(location_name=True)]["dhw"]
    cop_bui = cop_polynom_bui(df_toda_mean)
    cop_dhw = cop_polynom_dhw(df_toda_mean)
    P_hp_bui, P_hp_dhw, Q_demand_building, Q_demand_building_est = estimate_heat_pump_system_demands(
        TOda_mean=df_toda_mean,
        simulated_heat_demand=simulated_heat_demand,
        Q_demand_DHW=Q_demand_DHW,
        P_household=P_household,
        building=building_config,
        weather_config=weather_config,
        cop_bui=cop_bui, cop_dhw=cop_dhw, TRoom=TRoom,
    )
    P_pv = estimate_pv_generation(
        radTil_sum=df_sum[radTil].values,
        building=building_config
    )

    def _to_df(ndarray, _day_of_year, _n_days=2, interval="1H"):
        if interval == "1H":
            ndarray = ndarray[_day_of_year * 24:(_day_of_year + _n_days) * 24]
            series = pd.Series(ndarray, index=np.arange(0, _n_days * 24))
            series_shift = series.shift()
            series_shift.index -= 0.01  # Avoid random swap of values with same index in sort_index()
            return pd.concat([series, series_shift]).sort_index()
        ndarray = ndarray[_day_of_year * 1:(_day_of_year + _n_days) * 1] / 24
        return pd.Series([ndarray[0], ndarray[0], ndarray[1], ndarray[1]], index=[0, 24, 24, 48])

    P_demand_sum = P_hp_bui + P_hp_dhw + P_household

    return (
        _to_df(P_pv, day_of_year, n_days, interval),
        _to_df(P_demand_sum, day_of_year, n_days, interval),
        _to_df(Q_demand_building, day_of_year, n_days, interval),
        _to_df(Q_demand_DHW.values, day_of_year, n_days, interval),
    )


def load_household_profile(
        number_of_occupants: int,
        interval: str,
        init_period: float,
        first_day_of_year: datetime.datetime,
        last_day_of_year: datetime.datetime
):
    elec_profile = DATA_PATH.joinpath("electricity_profiles", f"{number_of_occupants}_occupants.txt")
    P_household = pd.read_csv(elec_profile, skiprows=[0, 1], index_col=[0], sep="\t")
    if np.any(P_household.index.duplicated()):
        raise IndexError("Duplicated index!")
    P_household = periodic_append_of_init_period(
        series=P_household,
        init_period=init_period,
        first_day_of_year=first_day_of_year,
        last_day_of_year=last_day_of_year
    )
    P_household = P_household.resample(interval).sum()
    # to kWh, interval is fixed in files
    P_household *= _get_factor_to_watt_hours(simulation_interval=900)
    P_household = P_household.values[:, 0]
    return P_household


def periodic_append_of_init_period(
        series: pd.Series,
        init_period: float,
        first_day_of_year: datetime.datetime,
        last_day_of_year: datetime.datetime
):
    # Modelica simulates periodically. Append some higher number than init_period
    # and later trim:
    series_periodic = series.loc[:init_period * 2]
    series_periodic.index += series.index[-1] + (series.index[1] - series.index[0])  # First entry is 0
    # Trim first day
    series = series.loc[init_period:]
    series = pd.concat([series, series_periodic])
    series.index -= series.index[0]
    series = convert_index_to_datetime_index(series, origin=first_day_of_year)
    # Trim full year
    series = series.loc[:last_day_of_year]
    return series


def _get_factor_to_watt_hours(simulation_interval: float):
    """
    If the simulation interval is exactly one hour, no conversion is needed.
    If the simulation interval is, e.g. 900 s, building the sum with resample.sum
    will yield four (3600 / 900) values in each hour. Thus, the sum has to be
    divided by four to get Wh.
    Or, as the values are in W, summing them gets you W * simulation_interval s.
    To get W*h, multiply by "h" and divide by 3600
    -> W * simulation_interval s * h / 3600 s
    -> W * h * (simulation_interval s / 3600 s)
    """
    return simulation_interval / 3600


def get_sampled_demands(
        df_building: pd.DataFrame,
        interval: str,
        input_config: InputConfig,
        simulation_interval: float,
        dhw_daily_per_person: float = 25
):
    df_toda_mean = df_building.loc[:, "outputs.weather.TDryBul"].resample(interval).mean()

    Q_demand_building = df_building.loc[:, "outputs.electrical.tra.PHea[1].value"]  # in W
    if np.any(Q_demand_building.index.duplicated()):
        raise IndexError("Duplicated index!")
    # in W * simulation_interval / interval
    Q_demand_building = Q_demand_building.resample(interval).sum()
    Q_demand_building *= _get_factor_to_watt_hours(simulation_interval=simulation_interval)
    if np.any(Q_demand_building < 0):
        raise ValueError(f"Given heat demand is negative in {input_config.get_name()}")

    Q_demand_DHW = dhw.create_mean_hourly_dhw_profile(
        number_of_occupants=input_config.building.number_of_occupants,
        hourly_index=df_building.index,
        dhw_daily_per_person=dhw_daily_per_person
    )
    Q_demand_DHW = Q_demand_DHW.resample(interval).sum().loc[:, "Q_demand_DHW"]

    P_household = load_household_profile(
        number_of_occupants=input_config.building.number_of_occupants,
        interval=interval,
        init_period=86400 * 2,
        first_day_of_year=df_toda_mean.index[0],
        last_day_of_year=df_toda_mean.index[-1]
    )
    # All powers are in W * h / interval, so e.g. Wh/h or Wh/d etc.
    return (
        Q_demand_building,
        Q_demand_DHW,
        df_toda_mean,
        P_household,
    )
