import numpy as np
import pandas as pd
from ebcpy.preprocessing import convert_index_to_datetime_index
import datetime
from bes_rules.simulation_based_optimization.milp.milp_model import heating_curve


def convert_time_series_inputs_to_df(time_series_inputs: pd.DataFrame, time_step: float):
    T_outdoor_air = time_series_inputs.loc[:, "T_Air"]
    x_time_step = np.arange(0.0, len(T_outdoor_air), time_step)
    x_hourly = np.arange(0.0, len(T_outdoor_air), 1.0)
    T_outdoor_air = np.interp(x_time_step, x_hourly, T_outdoor_air)
    df = pd.DataFrame({
        "P_PV": time_series_inputs.loc[:, "P_PV"],
        "T_outdoor_air": T_outdoor_air
    })
    df.index *= 3600 / 4
    first_day_in_year = datetime.datetime(2023, 1, 1)

    return convert_index_to_datetime_index(df, origin=first_day_in_year)


def run_rbpc(simulation_settings, time_series_inputs, solver_kwargs, **kwargs):
    from bes_rules.rule_extraction.rbpc_development import utils
    day_in_year = simulation_settings['start_time'] / 24
    n_days = simulation_settings['overall_stop_time'] / 24
    df = convert_time_series_inputs_to_df(time_series_inputs, time_step=simulation_settings["time_step"])
    x = utils.get_features(df=df, idx_day=day_in_year, n_days=n_days)
    clf = solver_kwargs['clf']
    cluster_map = solver_kwargs["cluster_map"]
    y = clf.predict([list(x.values())])[0]  # Make to list and get single return as predict expect 2D data
    dTBufSet = cluster_map[y]
    # if rbpc does not want to overheat use local valve control
    start_day = df.index[0] + datetime.timedelta(days=day_in_year)
    end_day = start_day + datetime.timedelta(hours=simulation_settings['control_horizon'])
    T_outdoor_air_control_horizon = df.loc[start_day:end_day, "T_outdoor_air"].values
    THeaCur = heating_curve(T_outdoor_air_control_horizon)
    return {
        "actExtDHWCtrl": [False] * len(dTBufSet),
        "actExtBufCtrl": [True] * len(dTBufSet),
        "TBufSet": THeaCur + dTBufSet
    }
