import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from bes_rules import DATA_PATH
from bes_rules.performance_maps import custom_automation
from bes_rules.performance_maps.utils import apply_constraints
from sklearn.metrics import mean_squared_error


logger = logging.getLogger(__name__)


def calc_deviation_to_data_sheet(
        datasheet_df: pd.DataFrame,
        vclibpy_df: pd.DataFrame,
        save_path: Path,
        weighting_T_con: dict = None,
        weight_cop=0.5,
        weight_q_max=0.25,
        weight_q_nom=0.15,
        weight_q_min=0.1,
        frosting=False,
        K_r: float = 0.2,
        use_condenser_inlet: bool = False,
        use_evaporator_inlet: bool = True
):
    """
    This function evaluates the error between a manufacturer's datasheet "datasheet_df" and a calculated heat pump map
    from VCLibPy "vclibpy_df" and returns a weighted mean squared error (MSE).

    Parameters:
    - vclibpy_df (pd.DataFrame): DataFrame containing the performance data of the parameter combination from VCLibPy.
    - save_path (Path): Path where the VCLibPy inputs have been stored.
    - weighting_T_con (dict): If None, equal weighting is applied to all condenser temperatures.
    - weight_cop (float): Weighting factor for the COP.
    - weight_q_max (float): Weighting factor for the maximum heat output.
    - weight_q_nom (float): Weighting factor for the nominal heat output.
    - weight_q_min (float): Weighting factor for the minimum heat output.
    - frosting (bool): If True, frosting correction is applied to the data.
    - K_r (float): Correction factor (0.2 or 1), see MA Julius Klupp or BA Marius Haas.
    - use_condenser_inlet (bool): If True, use condenser inlet temperature for calculations.
    - use_evaporator_inlet (bool): If True, use evaporator inlet temperature for calculations.

    Returns:
    - float: The weighted mean squared error (MSE) of the input parameter combination.
    """
    eva_name = "T_eva_in" if use_evaporator_inlet else "T_eva_out"
    con_name = "T_con_in" if use_condenser_inlet else "T_con_out"
    Q_con_name = "Q_con_outer"

    vclibpy_df[con_name] -= 273.15  # Convert T_con_in from K to °C
    vclibpy_df[eva_name] -= 273.15  # Convert T_con_in from K to °C
    vclibpy_df[con_name] = vclibpy_df[con_name].round()  # Round the values to the nearest integer
    vclibpy_df[eva_name] = vclibpy_df[eva_name].round()  # to avoid problems in later code
    vclibpy_df[Q_con_name] /= 1000  # Convert Q_con from W to kW
    vclibpy_df["P_el"] /= 1000  # Convert Q_con from W to kW

    # Set operational envelope
    vclibpy_df = apply_constraints(vclibpy_df)

    if frosting:
        with open(DATA_PATH.joinpath("map_generation", f"Zhu_frosting_values.json"), "r") as file:
            frosting_df = pd.DataFrame(json.load(file)[str(K_r)])

        # Set IceFac = 1 for values which are not in frosting map
        for T_eva in vclibpy_df[eva_name].unique():
            if T_eva not in frosting_df.loc[:, eva_name].values:
                frosting_df.loc[len(frosting_df), [eva_name, "IceFac"]] = [T_eva, 1]

        # Recalculate values
        vclibpy_df = vclibpy_df.merge(frosting_df, left_on=eva_name, right_on=eva_name)
        vclibpy_df[Q_con_name] = (
                vclibpy_df["P_el"] +
                (vclibpy_df[Q_con_name] - vclibpy_df["P_el"]) * vclibpy_df["IceFac"]
        )
        vclibpy_df["COP"] = vclibpy_df[Q_con_name] / vclibpy_df["P_el"]
        vclibpy_df_copy_si = vclibpy_df.copy()
        vclibpy_df_copy_si[con_name] += 273.15
        vclibpy_df_copy_si[eva_name] += 273.15
        vclibpy_df_copy_si[Q_con_name] *= 1000
        vclibpy_df_copy_si["P_el"] *= 1000
        csv_path = save_path.parent.joinpath("map_with_frosting.csv")
        vclibpy_df_copy_si.to_csv(csv_path, sep=";")
        custom_automation.create_sdf_from_csv(
            csv_path=csv_path,
            use_condenser_inlet=use_condenser_inlet,
            flowsheet="Standard",
            fluid="Propane"
        )

    # Check if the sum of weighting factors equals 1
    if sum([weight_cop, weight_q_max, weight_q_nom, weight_q_min]) != 1:
        logger.warning(
            "Sum of weightings is not 1."
            f"Current values: {weight_cop=}, {weight_q_max=}, {weight_q_nom=}, {weight_q_min=}"
        )

    # Initialize the DataFrame to store the final results
    columns_eva_n = [con_name, eva_name, "N", "COP_MSE", "Q_nom_MSE", "COP_Q_nom_MSE"]
    deviation_df = pd.DataFrame()
    # Loop for each condenser temperature
    T_con_values = datasheet_df[con_name].unique()
    if weighting_T_con is None:
        weighting_T_con = {T_con: 1 / len(T_con_values) for T_con in T_con_values}

    for idx_con, T_con in enumerate(T_con_values):
        T_con_weight = weighting_T_con[T_con]

        # Filter data for the current condenser temperature
        group_T_con_vclib = vclibpy_df[vclibpy_df[con_name] == T_con]
        group_T_con_datasheet = datasheet_df[datasheet_df[con_name] == T_con]

        # Group the data by eva_name and get the maximum Q_con_name for each group
        max_Q_con_per_T_eva = group_T_con_vclib.groupby(eva_name)[Q_con_name].max().reset_index()
        min_Q_con_per_T_eva = group_T_con_vclib.groupby(eva_name)[Q_con_name].min().reset_index()

        # Calculate the relative weight for the MSE calculation
        COP_relative = weight_cop / (weight_cop + weight_q_nom)
        Q_nom_relative = weight_q_nom / (weight_cop + weight_q_nom)

        # Loop for each evaporator temperature
        # Assignment of evaporator temperature values for the loop
        T_eva_values = group_T_con_vclib[eva_name].unique()
        df_con = pd.DataFrame(columns=columns_eva_n)
        for idx_eva, T_eva in enumerate(T_eva_values):
            group_T_eva_vclib = group_T_con_vclib[group_T_con_vclib[eva_name] == T_eva]
            datasheet_con_eva = group_T_con_datasheet[group_T_con_datasheet[eva_name] == T_eva]
            if len(datasheet_con_eva) > 1:
                raise IndexError(f"Multiple candidates in datasheet for {T_con=} and {T_eva=}")
            datasheet_con_eva = datasheet_con_eva.iloc[0].to_dict()

            # Loop for each rotational speed
            # Assignment of rotational speeds for the loop
            n_values = group_T_eva_vclib["n"].unique()
            df_con_eva_n = pd.DataFrame(columns=columns_eva_n)
            for idx_n, N in enumerate(n_values):
                vclib_eva_con_n = group_T_eva_vclib[group_T_eva_vclib["n"] == N]
                if len(vclib_eva_con_n) > 1:
                    raise IndexError(f"Multiple candidates in results for {T_eva=}, {T_con=}, {N=}")
                vclib_eva_con_n = vclib_eva_con_n.iloc[0].to_dict()
                if np.isnan(vclib_eva_con_n["COP"]) or np.isnan(datasheet_con_eva["COP"]):
                    COP_MSE = np.nan
                else:
                    # Actual MSE calculation for each individual value pair
                    COP_MSE = mean_squared_error(
                        [vclib_eva_con_n["COP"]],
                        [datasheet_con_eva["COP"]]
                    )

                if np.isnan(vclib_eva_con_n[Q_con_name]) or np.isnan(datasheet_con_eva["Q_nom"]):
                    Q_nom_MSE = np.nan
                else:
                    # Actual MSE calculation for each individual value pair
                    Q_nom_MSE = mean_squared_error(
                        [vclib_eva_con_n[Q_con_name]],
                        [datasheet_con_eva["Q_nom"]]
                    )
                # Include relative weighting
                COP_Q_nom_MSE = COP_relative * COP_MSE + Q_nom_relative * Q_nom_MSE
                df_con_eva_n.loc[idx_n, con_name] = T_con
                df_con_eva_n.loc[idx_n, eva_name] = T_eva
                df_con_eva_n.loc[idx_n, "N"] = N
                df_con_eva_n.loc[idx_n, "COP_MSE"] = COP_MSE
                df_con_eva_n.loc[idx_n, "Q_nom_MSE"] = Q_nom_MSE
                df_con_eva_n.loc[idx_n, "COP_Q_nom_MSE"] = COP_Q_nom_MSE

            # Drop compressor speeds which yield NaN
            df_con_eva_n_no_nans = df_con_eva_n.loc[~df_con_eva_n.loc[:, "COP_MSE"].isna()]
            if len(df_con_eva_n_no_nans) == 0:
                if not np.isnan(datasheet_con_eva["COP"]):
                    logger.warning(
                        f"Result is NaN for all compressor speeds at {T_con=}, {T_eva=}, "
                        f"but datasheet not. Setting result for this point to NaN!"
                    )
                min_df = df_con_eva_n.iloc[0].to_dict()  # Use the first row since all values are NaN
                min_df["N"] = np.nan  # set the rotational speed to NaN so that the point is not displayed later
            else:
                min_df = df_con_eva_n_no_nans.loc[df_con_eva_n_no_nans["COP_Q_nom_MSE"].idxmin()].copy()
            for col in columns_eva_n:
                df_con.loc[idx_eva, col] = min_df[col]

        # Handling NAN values in the datasheet or VCLibPy calculation
        for Q_name, Q_con_per_T_eva in zip(
            ["Q_max", "Q_min"],
            [max_Q_con_per_T_eva, min_Q_con_per_T_eva],
        ):
            Q_con_vclib = Q_con_per_T_eva[Q_con_name]
            Q_con_datasheet = group_T_con_datasheet[Q_name]
            vclib_nan = np.isnan(Q_con_vclib).values
            datasheet_nan = np.isnan(Q_con_datasheet).values
            both_nan = datasheet_nan & vclib_nan
            either_nan = datasheet_nan | vclib_nan
            if np.isnan(Q_con_datasheet[~both_nan]).any():
                logger.warning(
                    f"Datasheet contains nan for {Q_name} at {T_con=}, "
                    "but vclib result is not None.")
            if np.isnan(Q_con_vclib[~both_nan]).any():
                logger.warning(
                    f"VCLib results contain nan for Q_max at {T_con=}, "
                    "but datasheet is not None.")
            deviation_df.loc[idx_con, Q_name + "_MSE"] = mean_squared_error(
                Q_con_vclib[~either_nan],
                Q_con_datasheet[~either_nan]
            )

        max_COP_Q_nom_mse = df_con["COP_Q_nom_MSE"].mean()
        # Saving the running results
        deviation_df.loc[idx_con, con_name] = T_con
        deviation_df.loc[idx_con, "T_con_weighting"] = T_con_weight
        deviation_df.loc[idx_con, "MAX_COP_Q_nom_MSE"] = max_COP_Q_nom_mse

        # Add rotational speeds to the result DataFrame for different evaporator temperatures
        for n, T_eva in zip(df_con["N"], df_con[eva_name]):
            deviation_df.loc[idx_con, f"N_{int(T_eva)}"] = n

    deviation_df.to_excel(save_path)

    # Final weighting
    COP_Q_nom_deviation = np.dot(deviation_df["T_con_weighting"], deviation_df["MAX_COP_Q_nom_MSE"])
    Q_max_deviation = np.dot(deviation_df["T_con_weighting"], deviation_df["Q_max_MSE"])
    Q_min_deviation = np.dot(deviation_df["T_con_weighting"], deviation_df["Q_min_MSE"])

    return (
            COP_Q_nom_deviation * (weight_cop + weight_q_nom) +
            Q_max_deviation * weight_q_max +
            Q_min_deviation * weight_q_min
    )
