import logging

import numpy as np
import pandas as pd
from typing import Type
from bes_rules.input_analysis.heat_pump_system import PartialHeatPump, VitoCal250

logger = logging.getLogger(__name__)


def get_SCOP(
        heat_pump: Type[PartialHeatPump],
        THeaThr: float,
        TOda_nominal: float,
        TSup_nominal: float,
        dTCon_nominal: float,
        dTCon_measured: float,
        eps: float,
        operation: str,
        hybrid: bool,
        TCutOff: float = None,
        alpha_dhw: float = 0.18,
        inverter: bool = True,
):
    # Assumption: Frosting is included in data
    factor_points = {
        "F1": (-7, 35),
        "F2": (2, 35),
        "F3": (7, 35)
    }

    if TCutOff is not None and operation != "parallel":
        TSup_nominal, dTCon_nominal = correct_alternative_operation(
            TCutOff=TCutOff,
            TOda_nominal=TOda_nominal,
            TSup_nominal=TSup_nominal,
            dTCon_nominal=dTCon_nominal,
        )
        logger.debug("Corrected values for TCutOff=%s: TSup_nominal=%s, dTCon_nominal=%s",
                     TCutOff, TSup_nominal, dTCon_nominal)
    factors = get_correction_factor(
        THeaThr=THeaThr,
        TOda_nominal=TOda_nominal,
        TSup_nominal=TSup_nominal,
        inverter=inverter
    )
    COP = {
        factor: heat_pump.COP(TCon=273.15 + point[1], TAir=273.15 + point[0]) for factor, point in factor_points.items()
    }
    # uncomment to test against JAZ-Rechner from BWP. Matches exactly.
    #c = [2.70033389, 3.70027248, 4.898958971]
    #COP = {
    #    factor: c[i] for i, factor in enumerate(factor_points)
    #}
    f_dT = get_temperature_spread_correction(dTCon_nominal=dTCon_nominal, dTCon_measured=dTCon_measured)
    SCOP_H = f_dT / sum(factors[factor] / COP[factor] for factor in factor_points)
    logger.debug(",".join(f"COP_{f}={_cop}" for f, _cop in COP.items()))
    logger.debug("f_dT=%s,%s", f_dT, ",".join(f"{f}={v}" for f, v in factors.items()))

    f_dT_W = get_temperature_spread_correction(dTCon_nominal=5, dTCon_measured=dTCon_measured)
    f_1 = 1  # TDHW_nominal = 50 °C Assumption
    f_2 = 0.716  # HE in storage
    SCOP_W = COP["F3"] * f_dT_W * f_1 * f_2
    alpha_hp = get_alpha(eps=eps, operation=operation)
    assert 0 <= alpha_dhw <= 1, "alpha_dhw must be smaller than 1"
    SCOP_WPA = 1 / (
            (1 - alpha_dhw) * alpha_hp / SCOP_H +
            alpha_dhw * alpha_hp / SCOP_W +
            1 - alpha_hp
    )
    logger.debug("alpha_hp=%s", alpha_hp)
    logger.debug("SCOP_H=%s", SCOP_H)
    logger.debug("SCOP_W=%s", SCOP_W)
    logger.debug("SCOP_WPA=%s", SCOP_WPA)
    return SCOP_WPA, SCOP_H, SCOP_W, alpha_hp


def get_alpha(eps, operation: str):
    assert 0 <= eps <= 1, "eps is expected relative"
    eps *= 100
    alphas = {
        "eps": [100, 90, 80, 70, 60, 50, 40, 30, 20],
        "alternative": [1.00, 0.99, 0.97, 0.94, 0.91, 0.86, 0.76, 0.57, 0.33],
        "parallel": [1.00, 1.00, 1.00, 0.99, 0.98, 0.97, 0.93, 0.87, 0.72],
    }
    df = pd.DataFrame(alphas).set_index("eps")
    if eps not in df.index:
        df.loc[eps] = np.NAN
        df = df.sort_index()
        df = df.interpolate(method="index")
        df = df.ffill().bfill()
    return df.loc[eps, operation]


def get_correction_factor(
        THeaThr: float,
        TOda_nominal: float,
        TSup_nominal: float,
        inverter: bool = True,

):
    THeaThr_degC = int(THeaThr - 273.15)
    if THeaThr_degC == 15:
        if not inverter:
            # Tabelle 11, fixed speed 15 °C
            data = {
                'TOda_nominal': [-8, -8, -8, -10, -10, -10, -12, -12, -12, -14, -14, -14, -16, -16, -16],
                'TOda': [-7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7],
                'Factor': ['F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3'],
                30: [0.008, 0.468, 0.370, 0.020, 0.450, 0.418, 0.036, 0.514, 0.346, 0.090, 0.537, 0.278, 0.140, 0.538,
                     0.232],
                35: [0.008, 0.496, 0.388, 0.022, 0.476, 0.437, 0.039, 0.542, 0.361, 0.096, 0.566, 0.290, 0.150, 0.566,
                     0.241],
                40: [0.009, 0.527, 0.407, 0.024, 0.504, 0.458, 0.042, 0.573, 0.377, 0.104, 0.598, 0.303, 0.161, 0.596,
                     0.252],
                45: [0.010, 0.562, 0.428, 0.026, 0.536, 0.480, 0.046, 0.609, 0.395, 0.112, 0.634, 0.317, 0.175, 0.630,
                     0.263],
                50: [0.011, 0.602, 0.451, 0.029, 0.573, 0.505, 0.050, 0.649, 0.415, 0.123, 0.674, 0.332, 0.191, 0.669,
                     0.275],
                55: [0.012, 0.648, 0.477, 0.032, 0.616, 0.532, 0.055, 0.694, 0.436, 0.135, 0.720, 0.349, 0.210, 0.712,
                     0.289],
                60: [0.013, 0.695, 0.503, 0.035, 0.658, 0.560, 0.061, 0.740, 0.458, 0.147, 0.766, 0.366, 0.229, 0.756,
                     0.302]
            }
        else:
            # Tabelle 12, inverter 15 °C
            data = {
                'TOda_nominal': [-8, -8, -8, -10, -10, -10, -12, -12, -12, -14, -14, -14, -16, -16, -16],
                'TOda': [-7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7],
                'Factor': ['F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3'],
                30: [0.008, 0.458, 0.353, 0.020, 0.439, 0.398, 0.036, 0.500, 0.328, 0.089, 0.521, 0.264, 0.138, 0.520,
                     0.219],
                35: [0.008, 0.483, 0.368, 0.022, 0.462, 0.413, 0.038, 0.525, 0.340, 0.095, 0.547, 0.272, 0.147, 0.544,
                     0.226],
                40: [0.009, 0.512, 0.383, 0.024, 0.488, 0.430, 0.041, 0.553, 0.352, 0.102, 0.575, 0.282, 0.158, 0.571,
                     0.233],
                45: [0.010, 0.544, 0.400, 0.026, 0.517, 0.447, 0.045, 0.584, 0.366, 0.110, 0.606, 0.292, 0.171, 0.600,
                     0.241],
                50: [0.011, 0.581, 0.419, 0.028, 0.550, 0.466, 0.049, 0.619, 0.380, 0.119, 0.640, 0.303, 0.185, 0.632,
                     0.249],
                55: [0.012, 0.623, 0.439, 0.031, 0.587, 0.487, 0.054, 0.658, 0.396, 0.131, 0.679, 0.315, 0.203, 0.668,
                     0.258],
                60: [0.013, 0.665, 0.460, 0.035, 0.625, 0.508, 0.059, 0.698, 0.412, 0.142, 0.718, 0.326, 0.221, 0.703,
                     0.267]
            }
    elif THeaThr_degC == 12:
        if not inverter:
            # Tabelle 13, fixed-spped 12 °C
            data = {
                'TOda_nominal': [-8, -8, -8, -10, -10, -10, -12, -12, -12, -14, -14, -14, -16, -16, -16],
                'TOda': [-7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7],
                'Factor': ['F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3'],
                30: [0.010, 0.553, 0.292, 0.027, 0.537, 0.344, 0.046, 0.595, 0.272, 0.109, 0.601, 0.211, 0.167, 0.589,
                     0.169],
                35: [0.011, 0.586, 0.307, 0.029, 0.567, 0.360, 0.049, 0.628, 0.284, 0.117, 0.633, 0.220, 0.179, 0.619,
                     0.176],
                40: [0.012, 0.622, 0.323, 0.031, 0.601, 0.377, 0.053, 0.665, 0.298, 0.125, 0.669, 0.230, 0.193, 0.653,
                     0.183],
                45: [0.013, 0.664, 0.341, 0.034, 0.640, 0.397, 0.058, 0.706, 0.313, 0.136, 0.709, 0.241, 0.209, 0.690,
                     0.192],
                50: [0.014, 0.711, 0.360, 0.037, 0.684, 0.418, 0.063, 0.752, 0.329, 0.148, 0.755, 0.253, 0.228, 0.732,
                     0.201],
                55: [0.016, 0.766, 0.383, 0.042, 0.735, 0.442, 0.070, 0.806, 0.347, 0.163, 0.806, 0.266, 0.251, 0.780,
                     0.212],
                60: [0.017, 0.822, 0.405, 0.046, 0.786, 0.465, 0.077, 0.859, 0.365, 0.178, 0.858, 0.280, 0.274, 0.828,
                     0.222]
            }
        else:
            # Tabelle 14, inverter 12 °C
            data = {
                'TOda_nominal': [-8, -8, -8, -10, -10, -10, -12, -12, -12, -14, -14, -14, -16, -16, -16],
                'TOda': [-7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7],
                'Factor': ['F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3'],
                30: [0.010, 0.541, 0.280, 0.026, 0.523, 0.329, 0.045, 0.579, 0.259, 0.107, 0.583, 0.200, 0.164, 0.570,
                     0.160],
                35: [0.011, 0.571, 0.292, 0.028, 0.551, 0.342, 0.048, 0.608, 0.269, 0.115, 0.612, 0.207, 0.175, 0.596,
                     0.165],
                40: [0.012, 0.605, 0.305, 0.031, 0.582, 0.356, 0.052, 0.641, 0.280, 0.123, 0.643, 0.215, 0.188, 0.625,
                     0.171],
                45: [0.013, 0.643, 0.320, 0.034, 0.617, 0.371, 0.057, 0.677, 0.291, 0.133, 0.678, 0.223, 0.203, 0.657,
                     0.177],
                50: [0.014, 0.687, 0.336, 0.037, 0.657, 0.388, 0.062, 0.718, 0.303, 0.144, 0.717, 0.232, 0.221, 0.692,
                     0.183],
                55: [0.015, 0.736, 0.354, 0.041, 0.702, 0.406, 0.068, 0.764, 0.317, 0.158, 0.761, 0.242, 0.242, 0.731,
                     0.191],
                60: [0.017, 0.786, 0.372, 0.045, 0.747, 0.425, 0.075, 0.810, 0.330, 0.172, 0.805, 0.251, 0.264, 0.771,
                     0.198]
            }
    elif THeaThr_degC == 10:
        if not inverter:
            # Tabelle 15, fixed-speed 10 °C
            data = {
                'TOda_nominal': [-8, -8, -8, -10, -10, -10, -12, -12, -12, -14, -14, -14, -16, -16, -16],
                'TOda': [-7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7],
                'Factor': ['F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3'],
                30: [0.012, 0.625, 0.228, 0.033, 0.613, 0.272, 0.055, 0.658, 0.210, 0.126, 0.648, 0.155, 0.191, 0.622,
                     0.119],
                35: [0.013, 0.662, 0.240, 0.036, 0.648, 0.285, 0.059, 0.694, 0.219, 0.135, 0.682, 0.162, 0.205, 0.654,
                     0.124],
                40: [0.014, 0.703, 0.253, 0.039, 0.688, 0.299, 0.064, 0.735, 0.230, 0.145, 0.721, 0.170, 0.220, 0.690,
                     0.130],
                45: [0.016, 0.751, 0.267, 0.042, 0.732, 0.314, 0.069, 0.781, 0.242, 0.158, 0.765, 0.178, 0.239, 0.730,
                     0.136],
                50: [0.017, 0.805, 0.283, 0.047, 0.783, 0.332, 0.076, 0.832, 0.254, 0.172, 0.814, 0.187, 0.261, 0.775,
                     0.143],
                55: [0.019, 0.867, 0.301, 0.052, 0.841, 0.351, 0.084, 0.892, 0.269, 0.189, 0.870, 0.197, 0.287, 0.825,
                     0.150],
                60: [0.021, 0.930, 0.319, 0.057, 0.899, 0.370, 0.092, 0.951, 0.283, 0.206, 0.926, 0.207, 0.314, 0.876,
                     0.158]
            }
        else:
            # TAbelle 16, inverter 10 °C
            data = {
                'TOda_nominal': [-8, -8, -8, -10, -10, -10, -12, -12, -12, -14, -14, -14, -16, -16, -16],
                'TOda': [-7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7, -7, 2, 7],
                'Factor': ['F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3', 'F1', 'F2', 'F3'],
                30: [0.012, 0.611, 0.219, 0.033, 0.598, 0.260, 0.054, 0.640, 0.200, 0.124, 0.629, 0.148, 0.188, 0.602,
                     0.113],
                35: [0.013, 0.645, 0.229, 0.035, 0.630, 0.271, 0.058, 0.673, 0.208, 0.133, 0.659, 0.153, 0.201, 0.630,
                     0.117],
                40: [0.014, 0.684, 0.240, 0.038, 0.666, 0.283, 0.063, 0.709, 0.217, 0.143, 0.694, 0.159, 0.216, 0.661,
                     0.121],
                45: [0.015, 0.727, 0.252, 0.042, 0.706, 0.295, 0.068, 0.749, 0.226, 0.154, 0.731, 0.165, 0.233, 0.694,
                     0.126],
                50: [0.017, 0.777, 0.265, 0.046, 0.751, 0.309, 0.074, 0.795, 0.236, 0.167, 0.774, 0.172, 0.253, 0.732,
                     0.131],
                55: [0.019, 0.834, 0.280, 0.051, 0.803, 0.324, 0.082, 0.846, 0.246, 0.183, 0.821, 0.179, 0.278, 0.774,
                     0.136],
                60: [0.021, 0.890, 0.295, 0.056, 0.855, 0.340, 0.089, 0.897, 0.257, 0.199, 0.869, 0.187, 0.302, 0.816,
                     0.141]
            }
    else:
        raise ValueError(f"Only 10, 12 and 15 °C supported as THeaThr, you passed {THeaThr_degC=}")
    factors = {}
    df = pd.DataFrame(data)
    TSup_nominal_degC = TSup_nominal - 273.15
    TOda_nominal_degC = TOda_nominal - 273.15
    for factor in ["F1", "F2", "F3"]:
        df_sub = df.loc[df.loc[:, "Factor"] == factor]
        df_sub = df_sub.drop(["TOda", "Factor"], axis=1).set_index("TOda_nominal")
        df_sub = _interpolate_df(df_sub, TOda_nominal_degC, TSup_nominal_degC)
        factors[factor] = df_sub.loc[TOda_nominal_degC, TSup_nominal_degC]
    return factors


def get_temperature_spread_correction(
        dTCon_nominal: float,
        dTCon_measured: float
):
    data = {
        'dTCon_nominal': [3, 4, 5, 6, 7, 8, 9, 10],
        3: [1.000, 1.010, 1.020, 1.031, 1.041, 1.051, 1.061, 1.072],
        4: [0.990, 1.000, 1.010, 1.020, 1.031, 1.041, 1.051, 1.061],
        5: [0.980, 0.990, 1.000, 1.010, 1.020, 1.031, 1.041, 1.051],
        6: [0.969, 0.980, 0.990, 1.000, 1.010, 1.020, 1.031, 1.041],
        7: [0.959, 0.969, 0.980, 0.990, 1.000, 1.010, 1.020, 1.031],
        8: [0.949, 0.959, 0.969, 0.980, 0.990, 1.000, 1.010, 1.020],
        9: [0.939, 0.949, 0.959, 0.969, 0.980, 0.990, 1.000, 1.010],
        10: [0.928, 0.939, 0.949, 0.959, 0.969, 0.980, 0.990, 1.000]
    }
    df = pd.DataFrame(data).set_index("dTCon_nominal")
    df = _interpolate_df(df, dTCon_nominal, dTCon_measured)
    return df.loc[dTCon_nominal, dTCon_measured]


def correct_alternative_operation(
        TCutOff: float,
        TOda_nominal: float,
        TSup_nominal: float,
        dTCon_nominal: float
):
    if TCutOff <= TOda_nominal:
        return TSup_nominal, dTCon_nominal

    # Tabelle 29
    data_dT = {
        'TOda_nominal': [-8, -8, -8, -10, -10, -10, -12, -12, -12, -14, -14, -14, -16, -16, -16],
        'TSup_nominal': [70, 55, 35, 70, 55, 35, 70, 55, 35, 70, 55, 35, 70, 55, 35],
        -10: [None, None, None, 15.0, 10.0, 7.0, 13.9, 9.3, 6.5, 12.9, 8.6, 6.0, 12.1, 8.1, 5.6],
        -5: [13.0, 8.7, 6.1, 12.0, 8.0, 5.6, 11.1, 7.4, 5.2, 10.3, 6.9, 4.8, 9.7, 6.5, 4.5],
        -2: [11.1, 7.4, 5.2, 10.2, 6.8, 4.8, 9.4, 6.3, 4.4, 8.8, 5.9, 4.1, 8.2, 5.5, 3.8],
        0: [9.8, 6.5, 4.6, 9.0, 6.0, 4.2, 8.3, 5.6, 3.9, 7.8, 5.2, 3.6, 7.3, 4.8, 3.4],
        2: [8.5, 5.7, 4.0, 7.8, 5.2, 3.6, 7.2, 4.8, 3.4, 6.7, 4.5, 3.1, 6.3, 4.2, 3.0]
    }
    data_sup = {
        'TOda_nominal': [-8, -8, -8, -10, -10, -10, -12, -12, -12, -14, -14, -14, -16, -16, -16],
        'TSup_nominal': [70, 55, 35, 70, 55, 35, 70, 55, 35, 70, 55, 35, 70, 55, 35],
        -10: [None, None, None, 70, 55, 35, 67, 53, 34, 65, 51, 33, 63, 50, 33],
        -5: [65, 52, 34, 63, 50, 33, 60, 48, 32, 58, 47, 31, 57, 46, 31],
        -2: [60, 48, 32, 58, 47, 31, 56, 45, 31, 54, 44, 30, 53, 43, 30],
        0: [57, 46, 31, 55, 44, 30, 53, 43, 30, 51, 42, 29, 50, 41, 29],
        2: [53, 43, 30, 51, 42, 29, 50, 41, 29, 48, 40, 28, 47, 39, 28]
    }
    return_values = []
    for data in [data_sup, data_dT]:
        df = pd.DataFrame(data)
        TOda_nominal_degC = TOda_nominal - 273.15
        TSup_nominal_degC = TSup_nominal - 273.15
        TCutOff_degC = TCutOff - 273.15
        if TOda_nominal_degC in df.loc[:, "TOda_nominal"].values:
            df_sub = df.loc[df.loc[:, "TOda_nominal"] == TOda_nominal_degC]
            df_sub = df_sub.drop(["TOda_nominal"], axis=1).set_index("TSup_nominal")
            value = _interpolate_df(df=df_sub, idx=TSup_nominal_degC, col=TCutOff_degC).loc[
                TSup_nominal_degC, TCutOff_degC]
        elif TSup_nominal_degC in df.loc[:, "TOda_nominal"].values:
            df_sub = df.loc[df.loc[:, "TSup_nominal"] == TSup_nominal_degC]
            df_sub = df_sub.drop(["TSup_nominal"], axis=1).set_index("TOda_nominal")
            value = _interpolate_df(df=df_sub, idx=TOda_nominal_degC, col=TCutOff_degC).loc[
                TOda_nominal_degC, TCutOff_degC]
        else:
            value = interpolate_temperature(
                TOda_nominal=TOda_nominal_degC,
                TSup_nominal=TSup_nominal_degC,
                TCutOff=TCutOff_degC,
                df=df
            )
        return_values.append(value)
    supply, delta = return_values
    return supply + 273.15, delta


def interpolate_temperature(TOda_nominal, TSup_nominal, TCutOff, df):
    # Find nearest values in available data
    TOda_values = sorted(list(set(df['TOda_nominal'])))
    TSup_values = sorted(list(set(df['TSup_nominal'])))
    TCutOff_values = [-10, -5, -2, 0, 2]

    # Find bracketing values
    TOda_low = max([x for x in TOda_values if x <= TOda_nominal])
    TOda_high = min([x for x in TOda_values if x >= TOda_nominal])

    TSup_low = max([x for x in TSup_values if x <= TSup_nominal])
    TSup_high = min([x for x in TSup_values if x >= TSup_nominal])

    TCutOff_low = max([x for x in TCutOff_values if x <= TCutOff])
    TCutOff_high = min([x for x in TCutOff_values if x >= TCutOff])

    # Get all corner values for tri-linear interpolation
    results = []
    for TOda in [TOda_low, TOda_high]:
        for TSup in [TSup_low, TSup_high]:
            for TC in [TCutOff_low, TCutOff_high]:
                col = TC

                mask = (df['TOda_nominal'] == TOda) & (df['TSup_nominal'] == TSup)
                row = df[mask]

                if not row.empty and not pd.isna(row[col].values[0]):
                    results.append({
                        'TOda': TOda,
                        'TSup': TSup,
                        'TCutOff': TC,
                        'value': row[col].values[0],
                    })

    # Perform tri-linear interpolation
    def interpolate_point(x, x1, x2, y1, y2):
        if x1 == x2:
            return y1
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

    # First interpolate TCutOff for each TOda/TSup combination
    TCutOff_interp = []
    for TOda in set(r['TOda'] for r in results):
        for TSup in set(r['TSup'] for r in results):
            points = [r for r in results if r['TOda'] == TOda and r['TSup'] == TSup]
            if len(points) >= 2:
                value = interpolate_point(TCutOff, points[0]['TCutOff'], points[1]['TCutOff'],
                                          points[0]['value'], points[1]['value'])
                TCutOff_interp.append({
                    'TOda': TOda,
                    'TSup': TSup,
                    'value': value
                })

    # Then interpolate TSup
    TSup_interp = []
    for TOda in set(r['TOda'] for r in TCutOff_interp):
        points = [r for r in TCutOff_interp if r['TOda'] == TOda]
        if len(points) >= 2:
            value = interpolate_point(TSup_nominal, points[0]['TSup'], points[1]['TSup'],
                                      points[0]['value'], points[1]['value'])
            TSup_interp.append({
                'TOda': TOda,
                'value': value
            })

    # Finally interpolate TOda
    if len(TSup_interp) >= 2:
        final_value = interpolate_point(TOda_nominal, TSup_interp[0]['TOda'], TSup_interp[1]['TOda'],
                                        TSup_interp[0]['value'], TSup_interp[1]['value'])

        return final_value

    raise ValueError("Unable to interpolate with given values")


def _interpolate_df(df, idx, col):
    if idx not in df.index:
        df.loc[idx] = np.NAN
        df = df.sort_index()
        df = df.interpolate(method="index")
        df = df.ffill().bfill()
    if col not in df.columns:
        df.loc[:, col] = np.NAN
        df = df.sort_index(axis=1)
        df = df.interpolate(method="index", axis=1)
        df = df.ffill(axis=1).bfill(axis=1)
    return df


if __name__ == '__main__':

    class ExampleHeatPump(PartialHeatPump):
        @staticmethod
        def COP(TCon, TAir):
            if TAir == 273.15 - 7:
                return 2.9
            elif TAir == 273.15 + 2:
                return 3.6
            elif TAir == 273.15 + 7:
                return 4.3
            raise ValueError


    logging.basicConfig(level="DEBUG")
    default_args = dict(
        heat_pump=ExampleHeatPump,
        TOda_nominal=273.15 - 12,
        TSup_nominal=273.15 + 55,
        THeaThr=273.15 + 15,
        dTCon_nominal=10,
        eps=0.7,
        TCutOff=273.15 - 5,
        dTCon_measured=5,
        inverter=True
    )
    # 8.5.1
    get_SCOP(
        **default_args,
        alpha_dhw=0,
        hybrid=True,
        operation="parallel"
    )
    # 8.5.1
    get_SCOP(
        **default_args,
        alpha_dhw=0,
        hybrid=False,
        operation="parallel"
    )
    # 8.5.2
    get_SCOP(
        **default_args,
        alpha_dhw=0,
        hybrid=True,
        operation="alternative"
    )
    # 8.5.3
    get_SCOP(
        **default_args,
        alpha_dhw=0,
        hybrid=False,
        operation="alternative"
    )
    # 8.5.1
    get_SCOP(
        heat_pump=VitoCal250,
        TOda_nominal=273.15 - 8,
        TSup_nominal=273.15 + 60,
        THeaThr=273.15 + 15,
        dTCon_nominal=10,
        eps=0.7,
        TCutOff=273.15 - 5,
        dTCon_measured=5,
        inverter=True,
        alpha_dhw=0.18,
        hybrid=False,
        operation="parallel"
    )
