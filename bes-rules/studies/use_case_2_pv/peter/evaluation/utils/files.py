import ast
import json
from pathlib import Path

import pandas as pd
from ebcpy import TimeSeriesData

start_date = pd.Timestamp(year=2015, month=1, day=1)


def get_besmod_mapping():
    config_path = Path(__file__).parents[4].joinpath(
        "student_theses/peter/mapping_besmod.json"
    )
    with open(config_path) as mapping:
        besmod_mapping = json.load(mapping)
    return besmod_mapping


def _apply_own_conventions(df):
    for label in df.columns.to_list():
        if label.startswith("QTra_flow") or label.startswith("mdot"):
            df[label] = -df[label]
    return df


def load_pre_mpc(save_path: str):
    float_rows = {0: str}
    for i in range(1, 49):
        float_rows[i] = float
    df = pd.read_csv(save_path, dtype=float_rows, header=[0, 1]).rename(columns={"Unnamed: 0_level_1": "time"})
    df.drop(axis=1, level=0, columns=["upper", "lower", "parameter"], inplace=True)
    df.columns = df.columns.droplevel(level=0)
    df["time"] = df["time"].apply(ast.literal_eval)
    df["time_sim"] = df["time"].apply(lambda x: x[0])
    df["time_sim"] = start_date + pd.to_timedelta(df["time_sim"], unit="s")
    df["time_pre"] = df["time"].apply(lambda x: x[1])
    df["time_pre"] = pd.to_timedelta(df["time_pre"], unit="s")
    df.drop("time", axis=1, inplace=True)
    df.set_index(["time_sim", "time_pre"], inplace=True)

    dfs = []
    idx = pd.IndexSlice
    beg_pred = df.index.get_level_values("time_pre").unique()[0]
    end_pred = df.index.get_level_values("time_pre").unique()[-1]
    sim_times = df.index.get_level_values("time_sim").unique()
    for sim_time in sim_times:
        prediction = df.loc[idx[sim_time, beg_pred:end_pred], :].copy()
        prediction.set_index(prediction.index.droplevel(level="time_sim"), inplace=True)
        prediction.set_index(prediction.index + sim_time, inplace=True)
        prediction.interpolate(method="index", axis="index", inplace=True)
        dfs.append(prediction)

    return dfs


def load_sim_mpc(save_path: str):
    column_mapping = get_besmod_mapping()
    column_mapping["name"] = "time"

    df = pd.read_csv(save_path, skiprows=[0, 2], header=0, dtype=float).rename(columns=column_mapping)
    df["time"] = start_date + pd.to_timedelta(df["time"], unit="s")
    df.set_index("time", inplace=True)

    df = _apply_own_conventions(df)

    return df


def load_sim_besrules(path):
    column_mapping = get_besmod_mapping()

    df = pd.read_excel(path, index_col=0, header=0).rename(columns=column_mapping)
    df.index = pd.Timestamp(year=2015, month=1, day=1) + pd.to_timedelta(df.index, unit="s")

    df = _apply_own_conventions(df)

    return df


def load_sim_mat(path):
    column_mapping = get_besmod_mapping()

    columns = list(column_mapping.keys())
    not_in_mat = ["hydraulic.transfer.outBusTra.TRet[1]", "building.thermalZone[1].ROM.convRoof.Q_flow", "building.thermalZone[1].ROM.convFloor.Q_flow", "building.thermalZone[1].ROM.convExtWall.Q_flow", "hydraulic.transfer.outBusTra.TSup[1]", "building.thermalZone[1].ROM.convIntWall.Q_flow", "building.thermalZone[1].ROM.convWin.Q_flow"]
    columns = [col for col in columns if col not in not_in_mat]
    df = TimeSeriesData(path, variable_names=columns)
    df.to_datetime_index(origin=pd.Timestamp(year=2015, month=1, day=1))
    df = df.to_df().rename(columns=column_mapping)

    return df

'''
def load_sim(path, control):
    print("Load file: ", path)
    match control:
        case "mpc_wo_UB":
            df = load_sim_mpc(path)
        case "mpc_w_UB":
            df = load_sim_mpc(path)
        case "rbpc_wo_UB":
            df = load_sim_besrules(path)
        case "rbc":
            df = load_sim_mat(path)
        case "noSGReady":
            df = load_sim_mat(path)
        case "rbpc_zcbe":
            df = load_sim_besrules(path).rename(columns={"outputs.building.TZone[1]": "T_Air"})
        case _:
            raise ValueError(f"Control type not found: {control}")
    return df.loc[start_date + pd.Timedelta(days=2):start_date + pd.Timedelta(days=364), :]
'''