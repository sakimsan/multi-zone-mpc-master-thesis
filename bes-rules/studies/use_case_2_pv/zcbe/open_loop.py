import logging
import os
import pathlib

import matplotlib.pyplot as plt

from bes_rules.rule_extraction.rbpc_development import clustering
from bes_rules.simulation_based_optimization.milp.external_control import run_external_control
from bes_rules.simulation_based_optimization.milp.milp_model import run_milp_model


def run_easy_mpc(start_day: int, end_day: int, save_path: pathlib.Path):
    start_time = start_day * 24  # start time in hours
    time_step = 0.25  # step size in hours
    stop_time = end_day * 24  # end time in hours
    VStoBuf_small = 0.0276
    sInsBuf_small = 0.051
    UABuf_small = 17.5 / 35
    VStoBuf_big = 0.8291662102778516
    sInsBuf_big = 0.1775908064792074
    UABuf_big = 1.4847686356997805
    time_series_inputs = load_time_series(time_step)
    model_parameters = dict(
        VStoBuf=VStoBuf_small,
        sInsBuf=sInsBuf_small,
        UABuf=UABuf_small,
        VStoDHW=0.123417,
        sInsDHW=0.07781021589429941,
        UADHW=0.9062959182573429,
        eta_hr=0.97,
        m_flow_hp=0.19002700205345954,
        QEleHea_flow_nominal=175.944992065429,
        scalingFactor=0.7242177836517408,
    )
    os.makedirs(save_path, exist_ok=True)
    run_external_control(
        external_control=run_milp_model,
        model_parameters=model_parameters,
        time_series_inputs=time_series_inputs,
        start_time=start_time * 3600,
        stop_time=stop_time * 3600,
        output_interval=time_step * 3600,
        fmu=None,
        closed_loop=False,
        save_path=save_path.joinpath("Results.xlsx"),
        with_dhw=False,
        control_horizon=4,
        minimal_part_load_heat_pump=0,
        init_period=0,
        get_df=True
    )


if __name__ == '__main__':
    logging.basicConfig(level="INFO")

    plt.rcParams.update(
        {
            "font.size": 11,
            "figure.dpi": 250,
            # "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ['Segoe UI Symbol', 'simHei', 'Arial', 'sans-serif']
            # "backend": "TkAgg",
        }
    )

    START_DAY = 274
    END_DAY = 120
    SAVE_PATH = pathlib.Path(r"D:\zcbe\open_loop").joinpath(f"start={START_DAY}_stop={END_DAY}")
    #run_easy_mpc(start_day=START_DAY, end_day=END_DAY, save_path=SAVE_PATH)
    #plot_daily_results(save_path=SAVE_PATH, successive_days=3)
    #compare_pv(save_path=SAVE_PATH)
    #clustering.perform_clustering(save_path=SAVE_PATH)
    #decision_tree.create_decision_tree_for_multiple_settings(save_path=SAVE_PATH, n_days_total=5)
    clustering.plot_results(save_path=SAVE_PATH)
    #decision_tree.plot_train_test_scores(save_path=SAVE_PATH)
