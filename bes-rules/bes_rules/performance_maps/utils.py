import pathlib

import pandas as pd
import numpy as np

from bes_rules import DATA_PATH


def read_data_sheet(
        file_path: pathlib.Path,
        evaporator_values_to_ignore: list = None,
        condenser_values_to_ignore: list = None,
        use_condenser_inlet: bool = False,
        use_evaporator_inlet: bool = True,
):
    eva_name = "T_eva_in" if use_evaporator_inlet else "T_eva_out"
    con_name = "T_con_in" if use_condenser_inlet else "T_con_out"
    if evaporator_values_to_ignore is None:
        evaporator_values_to_ignore = []
    if condenser_values_to_ignore is None:
        condenser_values_to_ignore = []
    df = pd.read_excel(file_path)
    df = df.transpose()
    df.columns = df.iloc[0]  # Set the first row as the column names
    df = df[1:]  # Remove the first row
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
    df = df[~df[con_name].isin(condenser_values_to_ignore)]
    df = df[~df[eva_name].isin(evaporator_values_to_ignore)]
    return df


def get_points_outside_envelope(vclibpy_df: pd.DataFrame):
    """
    Check if points are outside compressor envelope.
    Returns a boolean mask where True indicates points outside the envelope.
    """
    T_sh = vclibpy_df["dT_eva_superheating"]
    T_sh = np.ones(len(T_sh)) * 4  # Assumption extended envelope
    p_eva = vclibpy_df["p_eva"]
    p_con = vclibpy_df["p_con"]

    # Define points, calculated using ref-prop based on Temperatures
    p_at_70 = 25.867581413555435
    # p5 = (-30, 55)
    p5 = (1.6783190985798868, 19.07169434577533)
    # p6 = (-30, 60)
    p6 = (1.6783190985798868, 21.167498136979944)
    # p7 = (-20, 70)
    p7 = (2.4451792337933593, p_at_70)
    # p8 = (-5, 70)
    p8 = (4.060364346093652, p_at_70)

    # Initialize mask for points outside envelope
    outside_mask = np.zeros_like(p_eva, dtype=bool)

    # Linear interpolation for T_sh > 5
    high_sh_mask = T_sh > 5
    if np.any(high_sh_mask):
        # Calculate slope and intercept for p5-p8 line
        slope_high = (p8[1] - p5[1]) / (p8[0] - p5[0])
        intercept_high = p5[1] - slope_high * p5[0]

        # Points above line are outside
        line_high = slope_high * p_eva[high_sh_mask] + intercept_high
        outside_mask[high_sh_mask] |= (p_con[high_sh_mask] > line_high)

    # Linear interpolation for T_sh <= 5
    low_sh_mask = ~high_sh_mask
    if np.any(low_sh_mask):
        # Calculate slope and intercept for p6-p7 line
        slope_low = (p7[1] - p6[1]) / (p7[0] - p6[0])
        intercept_low = p6[1] - slope_low * p6[0]

        # Points above line are outside
        line_low = slope_low * p_eva[low_sh_mask] + intercept_low
        outside_mask[low_sh_mask] |= (p_con[low_sh_mask] > line_low)

    # General condition: p_con < p_at_70
    outside_mask |= (p_con > p_at_70)

    return outside_mask


def apply_constraints(vclibpy_df: pd.DataFrame):
    # TODO: Operational Envelope translation to T_2_max
    vars_to_set_nan = [
        "Q_con_outer",
        "Q_con",
        "Q_eva",
        "Q_eva_outer",
        "COP",
        "P_el"
    ]
    #return vclibpy_df
    mask_oil = vclibpy_df["T_2"] > 273.15 + 130
    mask_max_P_el = vclibpy_df["P_el"] > 5400
    mask_ope_env = get_points_outside_envelope(vclibpy_df)
    #mask_speed = minimal_maximal_compressor_speed(vclibpy_df)
    or_mask = mask_ope_env | mask_max_P_el | mask_oil
    #or_mask = mask_max_P_el | mask_oil
    for var in vars_to_set_nan:
        vclibpy_df.loc[or_mask, var] = np.nan
    return vclibpy_df


def minimal_maximal_compressor_speed(vclibpy_df: pd.DataFrame):
    n_violated = np.array([False] * len(vclibpy_df))
    datasheet_df = read_data_sheet(
        DATA_PATH.joinpath("map_generation", "Vitocal_13kW_Datenblatt.xlsx")
    )
    for T_con in vclibpy_df.loc[:, "T_con_out"].unique():
        mask_T_con = vclibpy_df.loc[:, "T_con_out"] == T_con
        for T_eva_in in vclibpy_df.loc[:, "T_eva_in"].unique():
            mask_T_eva = vclibpy_df.loc[:, "T_eva_in"] == T_eva_in
            if T_con > 100:
                T_con -= 273.15
            if T_eva_in > 100:
                T_eva_in -= 273.15
            datasheet_entry = datasheet_df.loc[(
                    (datasheet_df.loc[:, "T_con_out"] == round(T_con, 0)) &
                    (datasheet_df.loc[:, "T_eva_in"] == round(T_eva_in, 0))
            )]
            n_min = datasheet_entry["n_min"].values[0]
            n_max = datasheet_entry["n_max"].values[0]
            n_violated = n_violated | (
                    (
                            (vclibpy_df.loc[:, "n"] < n_min) |
                            (vclibpy_df.loc[:, "n"] > n_max)
                    ) & mask_T_con & mask_T_eva
            )
    return n_violated
