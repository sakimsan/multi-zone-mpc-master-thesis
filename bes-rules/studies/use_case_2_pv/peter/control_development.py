import logging
import os
import pathlib
import pickle
import datetime
import numpy as np
from ebcpy import TimeSeriesData

#from bes_rules.rule_extraction.rbpc_development.external_control import run_external_control
from bes_rules.rule_extraction.rbpc_development.plotting import plot_daily_results, plot_pv_day
#from bes_rules.rule_extraction.rbpc_development.milp_model import run_milp_model
from bes_rules.rule_extraction.rbpc_development import clustering
from bes_rules.rule_extraction.rbpc_development import decision_tree
from bes_rules.rule_extraction.rbpc_development import utils
import matplotlib.pyplot as plt

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


REFERENCE_PV_FILE = pathlib.Path(r"D:/fwu/02_Paper/zcbe").joinpath(
    "BES_No_RBC",
    "DesignOptimizationResults",
    "TRY2015_536322100078_Jahr_B1994_retrofit_SingleDwelling_M_South",
    "iterate_0.mat"
)


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    path = r"D:/06_Results/ba_peter/RBPC/BJ1994_hc_no_T_Air_ub_march"
    SAVE_PATH = pathlib.Path(path)
    # plot_daily_results(save_path=SAVE_PATH, successive_days=3)
    # compare_pv(save_path=SAVE_PATH)
    clustering.perform_clustering(save_path=SAVE_PATH)
    clustering.plot_results(save_path=SAVE_PATH)
    decision_tree.create_decision_tree_for_multiple_settings(save_path=SAVE_PATH, n_days_total=3)
    decision_tree.plot_train_test_scores(save_path=SAVE_PATH)
