import numpy as np
import pandas as pd

from bes_rules.input_analysis.heat_pump_system import PartialHeatPump
from typing import Type


class OptihorstDataBA:
    TOda = np.array([-15, -10, -7, 2, 7, 12])
    PEle = np.array([5.547, 3.897, 2.907, 1.510, 1.484, 1.433])
    QNom = np.array([8.787, 8.017, 6.631, 4.773, 5.394, 6.730])
    QMax = np.array([7.13, 8.1, 8.684, 10.334, 11.174, 11.944])
    COP = QNom / PEle


def simulate(self, parameters, log_df):
    round_bin_Tj = False
    use_DIN_hours = False
    df_hourly_weather_data = self.input_config.weather.get_hourly_weather_data()
    temperature_hours = []
    clustered_weather_data = {"Temperature": [], "Hours": []}
    for item in df_hourly_weather_data.loc[:, "DryBulbTemp"]:
        if round_bin_Tj:
            temperature_hours.append(int(round(item, 0)))
        else:
            temperature_hours.append(item)
    count = 0
    for item in temperature_hours:
        if item <= 15:
            count += 1

    temperature_hours[2879:6551 + 2] = []
    # TODO: Use real clustering method
    if round_bin_Tj:
        for Tj in range(-30, 16):
            hours = 0
            for item in temperature_hours:
                if Tj == item:
                    hours += 1
            clustered_weather_data["Temperature"].append(Tj)
            if not use_DIN_hours:
                clustered_weather_data["Hours"].append(hours)
            else:
                clustered_weather_data["Hours"] = [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    25, 23, 24, 27, 68, 91, 89, 165, 173, 240, 280, 320, 357, 356,
                    303, 330, 326, 348, 335, 315, 215, 169, 151, 105,
                    74
                ]  # BIN-Hours for medium climate in DIN 14825
    else:
        # np.arange macht keine Schritte von 0.1 in floats, deshalb werden integer verwendet
        for Tj in range(-300, 160, 1):
            hours = 0
            for item in temperature_hours:
                if Tj == item * 10:
                    hours += 1
            clustered_weather_data["Temperature"].append(Tj / 10)
            clustered_weather_data["Hours"].append(hours)


def calc_COP(TOda, TSup, heat_pump: Type[PartialHeatPump], TOda_nominal: float):
    COP = heat_pump.COP(TCon=TSup, TAir=TOda)
    QHeaPumNom = heat_pump.QConNom(TCon=TSup, TAir=TOda)

    part_load = calc_part_load(TOda=TOda, TOda_nominal=TOda_nominal)

    CR = part_load * QDem_flow_nominal / QHeaPumNom
    one_series = pd.Series(1, index=TOda.index)
    CR = np.minimum(CR, one_series)

    Cd = 0.9  # According to EN 14825 if not known
    COP_corrected = COP * CR / (Cd * CR + (1 - Cd))  # equation (23)
    return COP_corrected


def calc_part_load(TOda, TOda_nominal: float, THeaThr: float = 273.15 + 16):
    # EN 14825 uses 16 °C as default. "pl(Tj)" in 14825, equation (23)
    return (TOda - THeaThr) / (TOda_nominal - THeaThr)
