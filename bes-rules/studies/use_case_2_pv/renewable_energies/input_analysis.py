import logging
from pathlib import Path

import pandas as pd

from bes_rules.input_analysis import input_analysis, heat_pump_system, pv, plotting
from bes_rules.input_analysis.pv import load_and_filter_results
from bes_rules.utils.functions import argmean, argmedian

years_to_skip = ["1859", "1918", "1948"]


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    from studies.use_case_2_pv.renewable_energies import plots
    plots.change_rc_params_for_paper()
    SAVE_PATH = plots.BASE_PATH.joinpath("00_PV_analysis")
    # Create files relevant for analysis:
    # input_analysis.extract_heat_demand_and_convert_to_interval(dict(key="Test",
    #    df_path=SAVE_PATH.joinpath("SimulationResults",
    #                               "TRY2045_506745079707_Wint" + "____" + "2015_standard" + ".mat"),
    #    interval="1H"))
    # cop.create_cop_regression_curves(save_path=SAVE_PATH)
    # pv.run_pv_simulations(save_path=SAVE_PATH)
    # plotting.plot_weather_cases_only(SAVE_PATH, interval=INTERVAL)
    # run_building_simulations(save_path=SAVE_PATH)
    plotting.plot_three_cases(save_path=SAVE_PATH, day_of_year=31, study_path=plots.BASE_PATH.joinpath("BESCtrl"))

    for INTERVAL in [
        "1H",
        # "12H", "15H", "18H", "21H",
        # "24H",
        "1D"
    ]:
        pass
        # input_analysis.load_and_convert_buildings(SAVE_PATH, interval=INTERVAL)
        # input_analysis.analyze_buildings_and_weathers(SAVE_PATH, interval=INTERVAL, use_mp=True)
        # plotting.plot_cluster(save_path=SAVE_PATH, interval=INTERVAL, years_to_skip=years_to_skip)
        # input_analysis.get_five_max_mean_and_median(save_path=SAVE_PATH, interval=INTERVAL, years_to_skip=years_to_skip)
    # plotting.plot_for_paper(save_path=SAVE_PATH)


def get_five_max_mean_and_median(save_path: Path, interval: str, years_to_skip: list, metric: str = "self_sufficiency_degree"):
    df_results = load_and_filter_results(save_path, interval, years_to_skip=years_to_skip)
    df_results = df_results.sort_values(metric, ascending=False)
    idx_max = df_results.iloc[:5]
    idx_max.loc[:, "type"] = "max"
    _iloc_mean = argmean(df_results.loc[:, metric])
    idx_mean = df_results.iloc[[_iloc_mean + i for i in range(-2, 3)]]
    idx_mean.loc[:, "type"] = "mean"
    _iloc_median = argmedian(df_results.loc[:, metric])
    idx_median = df_results.iloc[[_iloc_median + i for i in range(-2, 3)]]
    idx_median.loc[:, "type"] = "median"
    pd.concat([idx_max, idx_mean, idx_median]).to_excel(
        save_path.joinpath(interval, "possible_cases_for_optimization.xlsx"))
