"""
Module with functions to validate vclibpy map dynamically.

Visuals:
- 220613_1000_13_1415: good fit, but inlet and outlet temperatures are swapped
- 220614_0920_15_1430: good fit, but inlet and outlet temperatures are swapped
- 220623_1500_24_1248: good fit, but inlet and outlet temperatures are swapped
- 220822_1100_1615_test: All good git, EN_MEN412 best
- 220823_1045_1900_test: Overall good fit, except TSup > 60 °C and n_comp=0.3
- 220824_1000_1900_test: All good fit, MEN slightly worse at low n_comp=0.3
- 220825_1000_1100_test: MEN good, rest bad fit at high TSup and low TOda
- 221013_1030_13_1800: on/off, ok fit
- 221012_1120_12_1800: on/off, good git
- 221018_1130_19_0830: on/off, ok fit but bad PEle with low dTCon
- 221020_1030_21_0730: on/off, ok fit, same day as above
- 221103_1310_04_1130: long operation, T good, P ok fit
- 221109_1110_10_0900: on/off, really low TSup and low TOda -> bad fit but irrelevant. However, the map does not extrapolate and hold constant
- 221110_1150_11_1030: long operation, first good, then bad fit (frosting)? --> Frosting from MA Florian Will
- 221114_1015_15_0830: long operation and on/off, EN best fit, overall ok
- 221116_1100_17_1030: long operation, first good, then bad fit (higher TOda -> too high COP)--> Frosting from MA Florian Will
- 221117_1610_18_1545: long operation, first good, then bad fit (higher TOda -> too high COP)--> Frosting from MA Florian Will
- 221122_1836_23_1720: On/Off+long, perfect fit
- 221123_1730_24_1750: on/off, bad fit, high TOda
- 221125_1410_26_1400: only short, ok fit
- 221126_1520_27_1410: On/Off+long, perfect fit
- 221128_1015_29_1000: low TSup, not so good fit
- 221129_1300_30_1200: low TSup and three on/off cycles -> hold constant therefor bad fit
- 221202_1050_03_1000: On/Off, bad fit, TOda high+fluctuates
- 221203_1220_04_1115: Bad fit at low TSup
- 221205_1100_06_1000: Long operation, perfect fit
- 221213_0915_13_1550: Steps, good fit, nice visuals for TSup, EN model best fit
- 221214_1715_15_1700: on/off low TSup, as above
- 230118_0700_18_1400: all perfect except MEN, long operation
"""
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ebcpy import TimeSeriesData
from ebcpy.preprocessing import convert_datetime_index_to_float_index
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt

from bes_rules import RESULTS_FOLDER
from bes_rules.plotting import utils
from bes_rules.utils import validation

logger = logging.getLogger(__name__)

SAVE_PATH = RESULTS_FOLDER.joinpath("VCLibPyValidation")

FAULTY_EXPERIMENTS = [
    "221110_1150_11_1030",
    "221116_1100_17_1030",
    "221117_1610_18_1545",
    "221128_1015_29_1000",
    "221129_1300_30_1200",
    "221202_1050_03_1000",
    "221123_1730_24_1750",
    "optihorst_2025-01-21"
]

MAP_STATES_OLD_DATA = {
    "T_eva_out": {"name": "T_eva_out", "factor": 1, "offset": 273.15, "unit": "K"},
    "T1": {"name": "T_1", "factor": 1, "offset": 273.15, "unit": "K"},
    "T2": {"name": "T_2", "factor": 1, "offset": 273.15, "unit": "K"},
    "T2_dash": {"name": "T_2_ihe", "factor": 1, "offset": 273.15, "unit": "K"},
    "T3": {"name": "T_3", "factor": 1, "offset": 273.15, "unit": "K"},
    "T4_dash": {"name": "T_4", "factor": 1, "offset": 273.15, "unit": "K"},
    "T1_dash": {"name": "T_1_ihe", "factor": 1, "offset": 273.15, "unit": "K"},
    "m_flow_ref": {"name": "m_flow_ref", "factor": 1, "offset": 0, "unit": "kg/s"},
    "p1": {"name": "p_1", "factor": 1e5, "offset": 0, "unit": "Pa"},
    "p2": {"name": "p_2", "factor": 1e5, "offset": 0, "unit": "Pa"},
    "p2_dash": {"name": "p_2_ihe", "factor": 1e5, "offset": 0, "unit": "Pa"},
    # "p3": {"name": "p_2", "factor": 1e5, "offset": 0, "unit": "Pa"},
    # "p4_dash": {"name": "p_1_ihe", "factor": 1e5, "offset": 0, "unit": "Pa"},
    "p1_dash": {"name": "p_1_ihe", "factor": 1e5, "offset": 0, "unit": "Pa"}
}

MAP_STATES_NEW_DATA = {
    # "detHP/StateMachine.p_CompIn": {"name": "T_eva_out", "factor": 1, "offset": 273.15, "unit": "K"},
    "detHP/StateMachine.T_CompIn": {"name": "T_1", "factor": 1, "offset": 273.15, "unit": "K"},
    "detHP/StateMachine.T_CompOut": {"name": "T_2", "factor": 1, "offset": 273.15, "unit": "K"},
    "detHP/StateMachine.T_CondIn": {"name": "T_2_ihe", "factor": 1, "offset": 273.15, "unit": "K"},
    "detHP/StateMachine.T_CondOut": {"name": "T_3", "factor": 1, "offset": 273.15, "unit": "K"},
    # "detHP/StateMachine.T_EvapIn": {"name": "T_4", "factor": 1, "offset": 273.15, "unit": "K"},
    "detHP/StateMachine.T_EvapOut": {"name": "T_1_ihe", "factor": 1, "offset": 273.15, "unit": "K"},
    "detHP/StateMachine.m_CompOut": {"name": "m_flow_ref", "factor": 1, "offset": 0, "unit": "kg/s"},
    "detHP/StateMachine.p_CompIn": {"name": "p_1", "factor": 1e5, "offset": 0, "unit": "Pa"},
    "detHP/StateMachine.p_CompOut": {"name": "p_2", "factor": 1e5, "offset": 0, "unit": "Pa"},
    "detHP/StateMachine.p_CondIn": {"name": "p_2_ihe", "factor": 1e5, "offset": 0, "unit": "Pa"},
    # "detHP/StateMachine.p_CompIn": {"name": "p_2", "factor": 1e5, "offset": 0, "unit": "Pa"},
    # "detHP/StateMachine.p_CompIn": {"name": "p_1", "factor": 1e5, "offset": 0, "unit": "Pa"},
    "detHP/StateMachine.p_EvapOut": {"name": "p_1_ihe", "factor": 1e5, "offset": 0, "unit": "Pa"},
    # "lambda_h_calc": {"name": "lambda_h", "factor": 1, "offset": 0, "unit": "%"}
}


def add_lambda_h_to_data(df):
    from vclibpy.media import CoolProp
    med_prop = CoolProp(fluid_name="R410A")

    m_flow = "detHP/StateMachine.m_CompOut"
    n = "detHP/StateMachine.rps_Comp"
    p1 = "detHP/StateMachine.p_CompIn"
    T1 = "detHP/StateMachine.T_CompIn"

    d = []
    for idx, row in df.iterrows():
        try:
            d.append(med_prop.calc_state("PT", row[p1] * 1e5, row[T1] + 273.15).d)
        except Exception as err:
            print(err)
            d.append(np.nan)

    df.loc[:, "d1"] = d
    V_h = 42.3e-6  # panasonic C-SDP205H02B
    mask_nan = np.isnan(df.loc[:, "d1"])
    df.loc[:, "lambda_h_calc"] = 0
    df.loc[~mask_nan, "lambda_h_calc"] = df.loc[~mask_nan, m_flow].values / (
            V_h * df.loc[~mask_nan, n] * df.loc[~mask_nan, "d1"]
    )
    df.loc[df.loc[:, "lambda_h_calc"] < 0, "lambda_h_calc"] = 0
    return df


def run_validation(tsd_path, recalculate: bool = False, five_d: bool = True, update_plot: bool = True):
    validation_cases = get_validation_cases(five_d)
    order_in_txt = [
        "TConOutMea",
        "TConInMea",
        "mCon_flow",
        "PEleMeaFan",
        "PEleMeaInv",
        "TEvaInMea",
        "ySet",
    ]
    columns = {
        "T_con_out": {"name": "TConOutMea", "factor": 1, "offset": 273.15},
        "T_con_in": {"name": "TConInMea", "factor": 1, "offset": 273.15},
        "V_flow_water": {"name": "mCon_flow", "factor": 0.997 / 3600, "offset": 0},
        "P_el_fan": {"name": "PEleMeaFan", "factor": 1000, "offset": 0},
        "P_el": {"name": "PEleMeaInv", "factor": 1000, "offset": 0},
        "T_eva_in": {"name": "TEvaInMea", "factor": 1, "offset": 273.15},
        "n_comp": {"name": "ySet", "factor": 1 / 110, "offset": 0}
    }
    columns_hdf = {
        "detHP/IndoorUnit.T_FlowOutlet_IDU": {"name": "TConOutMea", "factor": 1, "offset": 273.15},
        "detHP/IndoorUnit.T_CondIn_IDU": {"name": "TConInMea", "factor": 1, "offset": 273.15},
        "detHP/IndoorUnit.massflow_IDU": {"name": "mCon_flow", "factor": 1 / 3600, "offset": 0},
        "detHP/StateMachine.Pel_Fan": {"name": "PEleMeaFan", "factor": 1000, "offset": 0},
        "detHP/StateMachine.Pel_Inv": {"name": "PEleMeaInv", "factor": 1000, "offset": 0},
        "detHP/StateMachine.T_Air_In": {"name": "TEvaInMea", "factor": 1, "offset": 273.15},
        "detHP/StateMachine.rps_Comp": {"name": "ySet", "factor": 1 / 110, "offset": 0}
    }
    variables_to_plot = [
        ["TConOutMea", "heaPum.refCyc.sigBus.TConOutMea"],
        ["PEleCom", "heaPum.refCyc.sigBus.PEleMea"],
        "heaPum.refCyc.sigBus.yMea",
        "heaPum.refCyc.sigBus.TConInMea",
        "heaPum.refCyc.sigBus.TEvaInMea",
        "heaPum.refCyc.sigBus.mConMea_flow"
    ]
    if five_d:
        columns["dTSupHeaSet"] = {"name": "dTSupHeaSet", "factor": 1, "offset": 0}
        columns_hdf["detHP/StateMachine.T_Superheat_mes"] = {"name": "dTSupHeaSet", "factor": 1, "offset": 0}
        order_in_txt.append("dTSupHeaSet")
        variables_to_plot.append("heaPum.refCyc.sigBus.dTSupHeaSet")

    for file in os.listdir(tsd_path):
        file_path = tsd_path.joinpath(file)
        if not file_path.is_dir() and file_path.suffix == "":
            os.rename(file_path, str(file_path) + ".csv")
            file_path = Path(str(file_path) + ".csv")
        if file_path.suffix not in [".csv", ".hdf"]:
            continue
        if file_path.stem in FAULTY_EXPERIMENTS:
            continue
        save_path_case = SAVE_PATH.joinpath(file_path.stem)
        if os.path.exists(save_path_case) and not recalculate and not update_plot:
            continue
        if file_path.suffix == ".csv":
            mapped_states = MAP_STATES_OLD_DATA
        else:
            mapped_states = MAP_STATES_NEW_DATA

        start_index = 8 if five_d else 7
        mapped_states_to_plot = [
            [f"tabMea.y[{start_index + i}]", f"heaPum.refCyc.sigBus.extSta[{i}]"]
            for i in range(1, len(mapped_states) + 1)
        ]

        if not os.path.exists(save_path_case) or recalculate:
            tsd = TimeSeriesData(file_path).to_df()
            os.makedirs(save_path_case, exist_ok=True)
            clean_df = pd.DataFrame(index=tsd.index)
            if file_path.suffix == ".csv":
                if file_path.stem in [
                    "220613_1000_13_1415",
                    "220614_0920_15_1430",
                    "220623_1500_24_1248",
                ]:
                    columns_to_use = columns.copy()
                    columns_to_use["T_con_in"] = columns["T_con_out"]
                    columns_to_use["T_con_out"] = columns["T_con_in"]
                else:
                    columns_to_use = columns
                tsd.loc[:, "dTSupHeaSet"] = tsd.loc[:, "T1"] - tsd.loc[:, "T_e"]
            else:
                columns_to_use = columns_hdf
                tsd = add_lambda_h_to_data(tsd)

            map_states = [state["name"] for state in mapped_states.values()]
            map_units = [state["unit"] for state in mapped_states.values()]
            i = 0
            for col, data in {**columns_to_use, **mapped_states}.items():
                clean_df.loc[:, data["name"]] = (tsd.loc[:, col] + data["offset"]) * data["factor"]
                i += 1
            clean_df = clean_df.reindex(order_in_txt + map_states, axis=1)
            clean_df = convert_datetime_index_to_float_index(df=clean_df)
            txt_path = save_path_case.joinpath("vclibpy_tsd.txt")
            clean_df = clean_df.interpolate(method="index").ffill().bfill()
            convert_tsd_to_modelica_txt(
                tsd=clean_df,
                table_name="ValidationData",
                save_path_file=txt_path
            )
            states_modifier = 'nExtSta=%s, dataUnitExt={"%s"}, datasetExt={"%s"}' % (
                len(map_states),
                '", "'.join(map_units),
                '", "'.join(map_states)
            )
            for validation_case in validation_cases:
                validation_case.model_name = validation_case.model_name.split("(")[0]
                validation_case.model_name += f'(tabMea(fileName="{txt_path.as_posix()}"), {states_modifier})'

            validation_cases = validation.simulate(
                validation_cases,
                sim_setup=dict(stop_time=clean_df.index[-1], output_interval=2),
                save_path=save_path_case
            )
        else:
            with open(save_path_case.joinpath("validation_case_results.json"), "r") as file:
                validation_cases = json.load(file)
            validation_cases = [
                validation.ValidationCase(**validation_case)
                for validation_case in validation_cases
            ]
        mapped_states_to_plot = []
        validation.plot_results_plotly(
            validation_cases=validation_cases,
            variables_to_plot=variables_to_plot + mapped_states_to_plot,
            custom_plot_config=get_mapped_states_config(mapped_states, five_d=five_d),
            save_path=save_path_case
        )


def get_mapped_states_config(mapped_states: dict, five_d):
    quantities = {
        "Pa": "Pressure", "K": "Temperature", "kg/s": "MassFlowRate", "%": "Percent"
    }
    start_index = 8 if five_d else 7

    return {
        **{
            f"heaPum.refCyc.sigBus.extSta[{i + 1}]":
                {"label": state["name"],
                 "quantity": quantities.get(state["unit"])}
            for i, state in enumerate(mapped_states.values())
        },
        **{
            f"tabMea.y[{start_index + i + 1}]": {"label": state["name"],
                                                 "quantity": quantities.get(state["unit"])}
            for i, state in enumerate(mapped_states.values())
        }
    }


def get_validation_cases(five_d):
    if five_d:
        return [
            validation.ValidationCase(model_name="BESRules.Validation.VCLibPy5D.EN_MEN412_EN412",
                                      name="EN_MEN412_EN412"),
            # validation.ValidationCase(model_name="BESRules.Validation.VCLibPy5D.EN_MEN412_Linear",
            #                          name="EN_MEN412_Linear"),
            # validation.ValidationCase(model_name="BESRules.Validation.VCLibPy5D.EN_MEN412_Reg", name="EN_MEN412_Reg"),
        ]
    return [
        # validation.ValidationCase(model_name="BESRules.Validation.VCLibPy.MEN", name="MEN"),
        # validation.ValidationCase(model_name="BESRules.Validation.VCLibPy.EN_MEN412_EN412", name="EN_MEN412_EN412"),
        validation.ValidationCase(model_name="BESRules.Validation.VCLibPy.EN_MEN412_EN412", name="EN_MEN412_EN412"),
        # validation.ValidationCase(model_name="BESRules.Validation.VCLibPy.OptiHorstNew", name="OptiHorstNew"),
    ]


def plot_errors_over_inputs(which_data, five_d, with_mapped_states: bool = True, with_cop_and_qcon: bool = True):
    use_all_which_data = which_data == "all"
    new_data = which_data == "new"
    if use_all_which_data:
        with_mapped_states = False
    x_variables = [
        "heaPum.refCyc.sigBus.yMea",
        "heaPum.refCyc.sigBus.TConInMea",
        "heaPum.refCyc.sigBus.TEvaInMea",
        "heaPum.refCyc.sigBus.mConMea_flow"
    ]
    if five_d:
        x_variables.append("heaPum.refCyc.sigBus.dTSupHeaSet")
    y_errors = [
        ["TConOutMea", "heaPum.refCyc.sigBus.TConOutMea"],
        ["PEleCom", "heaPum.refCyc.sigBus.PEleMea"],
    ]
    if with_cop_and_qcon:
        y_errors.append(["COPMea", "heaPum.COP"])
        y_errors.append(["QConMea", "heaPum.QCon_flow"])
    if with_mapped_states:
        if new_data:
            mapped_states = MAP_STATES_NEW_DATA
        else:
            mapped_states = MAP_STATES_OLD_DATA
        start_index = 8 if five_d else 7
        y_errors = y_errors + [
            [f"tabMea.y[{start_index + i}]", f"heaPum.refCyc.sigBus.extSta[{i}]"]
            for i in range(1, len(mapped_states) + 1)
        ]
        plot_variables_config = {**validation.CUSTOM_PLOT_CONFIG, **get_mapped_states_config(mapped_states, five_d)}
    else:
        plot_variables_config = validation.CUSTOM_PLOT_CONFIG

    plot_config = utils.load_plot_config()
    plot_config.update_config({"variables": plot_variables_config})
    variables = validation.flatten_nested_list(y_errors) + x_variables + ["PEleFan"]
    all_results = {}
    all_times = {}
    df_stats = pd.DataFrame()
    for path in os.listdir(SAVE_PATH):
        res_path = SAVE_PATH.joinpath(path, "validation_case_results.json")
        if not os.path.exists(res_path):
            continue
        if path in FAULTY_EXPERIMENTS:
            continue
        if not use_all_which_data:
            if new_data and not path.startswith("optihorst") or not new_data and path.startswith("optihorst"):
                continue
        with open(res_path, "r") as file:
            validation_cases = json.load(file)
        logger.info(f"Reading path %s", path)

        for validation_case in validation_cases:
            validation_case = validation.ValidationCase(**validation_case)
            variables_to_load = variables.copy()
            if with_cop_and_qcon:
                variables_to_load.remove("COPMea")
                variables_to_load.remove("QConMea")
            df = TimeSeriesData(validation_case.result_path,
                                variable_names=variables_to_load).to_df()
            dt_total = df.index[-1] - df.index[0]
            # df = convert_index_to_datetime_index(df)
            # df = df.resample("1min").mean()
            # df = convert_datetime_index_to_float_index(df)
            df = plot_config.scale_df(df)
            df.index /= 60  # Convert to minutes
            # Filter first 3 min of init in simulation
            df = df.loc[3:]
            # Filter startup-processes:
            df = filter_startup_transition(df=df, startup_delay=1)
            # df = filter_cooldown_transition(df=df, startup_delay=1)
            # Filter defrost
            if np.any(df.loc[:, "PEleFan"] > 0):  # Sometimes PEleFan was not logged
                df = df.loc[df.loc[:, "PEleFan"] > 1]  # If fan is off, hp is either off or in defrost
            df = df.loc[df.loc[:, "heaPum.refCyc.sigBus.mConMea_flow"] > 0.01]  # If pump is off

            if df.empty:
                logger.warning("No data left after filtering for case %s", path)
                continue
            # Calc COPMea:
            df.loc[:, "QConMea"] = (
                    (df.loc[:, "TConOutMea"] - df.loc[:, "heaPum.refCyc.sigBus.TConInMea"]) *
                    4.184 *
                    df.loc[:, "heaPum.refCyc.sigBus.mConMea_flow"]
            )
            df.loc[:, "COPMea"] = df.loc[:, "QConMea"] / df.loc[:, "PEleCom"]

            from ebcpy.utils.statistics_analyzer import StatisticsAnalyzer
            for y_error in y_errors:
                df_stats.loc[path, f"{validation_case.name}_{y_error[0]}_RMSE"] = StatisticsAnalyzer.calc_rmse(
                    meas=df.loc[:, y_error[0]],
                    sim=df.loc[:, y_error[1]]
                )
                df_stats.loc[path, f"{validation_case.name}_{y_error[0]}_ME"] = np.mean(
                    df.loc[:, y_error[0]] - df.loc[:, y_error[1]]
                )
            if validation_case.name in all_results:
                all_results[validation_case.name] = {
                    variable: np.hstack([all_results[validation_case.name][variable], df.loc[:, variable].values])
                    for variable in variables
                }
                all_times[validation_case.name] += dt_total
            else:
                all_results[validation_case.name] = {variable: df.loc[:, variable].values
                                                     for variable in variables}
                all_times[validation_case.name] = dt_total
    print("Total operation times:", ", ".join([f"{case}: {time / 2 / 3600} h" for case, time in all_times.items()]))
    df_stats.to_excel(SAVE_PATH.joinpath(f"RMSE_stats_{which_data}.xlsx"))
    validation.plot_error_hist(
        all_results=all_results,
        x_variables=x_variables,
        y_errors=y_errors,
        save_path=SAVE_PATH.joinpath(f"{which_data}.png"),
        plot_config=plot_config
    )
    validation.plot_error_over_error(
        all_results=all_results,
        y_errors=y_errors,
        save_path=SAVE_PATH.joinpath(f"error_over_error{which_data}.png"),
        plot_config=plot_config
    )


def filter_startup_transition(df, startup_delay: float = 60):
    var = "heaPum.refCyc.sigBus.yMea"

    # Create mask for valid data points
    mask = df[var] > 5  # Start with non-zero points. All below 5 % speed is off
    if startup_delay == 0:
        return df[mask]

    # Find where control variable becomes non-zero
    transitions = (df[var] > 0) & (df[var].shift(1) == 0)
    transition_times = df.loc[transitions].index

    # Exclude startup_delay seconds after each transition
    for t_start in transition_times:
        mask = mask & ~((df.index >= t_start) &
                        (df.index <= t_start + startup_delay))

    return df[mask]


def filter_cooldown_transition(df, startup_delay: float = 60):
    var = "heaPum.refCyc.sigBus.yMea"

    # Create mask for valid data points
    mask = df[var] > 0

    # Find where control variable becomes non-zero
    transitions = (df[var] == 0) & (df[var].shift(1) > 0)
    transition_times = df.loc[transitions].index

    # Exclude startup_delay seconds after each transition
    for t_start in transition_times:
        mask = mask & ~((df.index <= t_start) &
                        (df.index >= t_start - startup_delay))

    return df[mask]


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    PATHS = [
        Path(r"D:\02_Paper\vclibpy\OptiHorst\stationäres_Kennfeld"),
        Path(r"D:\02_Paper\vclibpy\OptiHorst\genutzt"),
        # Path(r"D:\02_Paper\vclibpy\OptiHorst\keine Punkte"),
        Path(r"D:\02_Paper\vclibpy\OptiHorst\neu_2025"),
        # Path(r"D:\02_Paper\vclibpy\OptiHorst\Prozessumkehr")
    ]
    # for PATH in PATHS:
    #    run_validation(tsd_path=PATH, five_d=True, update_plot=True)
    plot_errors_over_inputs(which_data="all", five_d=True, with_mapped_states=False, with_cop_and_qcon=False)
    # plot_errors_over_inputs(which_data="new", five_d=True, with_mapped_states=True)
    # plot_errors_over_inputs(which_data="old", five_d=True, with_mapped_states=True)
