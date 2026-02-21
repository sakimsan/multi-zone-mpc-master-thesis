import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bes_rules import RESULTS_FOLDER, PC_SPECIFIC_SETTINGS
from bes_rules import configs
from bes_rules.input_variations import run_input_variations
from bes_rules.simulation_based_optimization import AgentLibMPC
from studies_ssr.sfh_mpc_hom_monovalent_spawn import MPC_UTILS_PATH, base_design_optimization
from studies_ssr.sfh_mpc_hom_monovalent_spawn.buf_influences import compare_buf_sizes
from agentlib_mpc.utils.analysis import load_sim, load_mpc_stats
from pathlib import Path

def plot_sim_stats(study_name: str, sim_result_name: str):
    sim_results_path = RESULTS_FOLDER.joinpath(
        "SFH_MPCRom_monovalent_spawn",
        study_name,
        "DesignOptimizationResults",
        "TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai_"
    )


    # Variante A: Raw-String (r"...")
    sim_results_path = Path(
        r"D:\fwu-ssr\res\Studies_not_coupled\studies_parallel\DesignOptimizationResults\TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai2h_Arbeitswoche_")

    all_variables = [
        'yValSet',
        'TBufSet'
    ]

    mpc_results = load_sim(sim_results_path.joinpath(sim_result_name))

    zones = ["livingroom", "kitchen", "hobby", "wcstorage", "corridor", "bedroom", "children", "corridor2", "bath", "children2", "attic", "one", "two"]
    #zones = ["corridor"]
    base_vars = ["yValSet", "T_Air", "QTra_flow", "Q_RadSol_or", "TAir_lb_slack", "TAir_ub_slack", "TSetOneZone"]
    #base_vars = ["yValSet", "T_Air", "Q_RadSol_or", "TSetOneZone"]

    global_vars = ["TBufSet", "yValSet", "T_Air", "P_el_hp", "TTraSup_slack", "THeaPumSup_slack", "QHeaPum_flow_slack", "PEleHeaPum_slack", "TTraSup"]
    #global_vars = []

    wanted_vars = global_vars + [f"{var}_{zone}" for zone in zones for var in base_vars]


    filtered_results = mpc_results.loc[:, [
                                            col for col in mpc_results.columns
                                            if col[1] in wanted_vars
                                          ]]

    #filtered_results = mpc_results.loc[:, [col for col in mpc_results.columns if str(col[1]).__contains__("livingroom")]]

    load_mpc_stats(data=filtered_results, scale="hours")


if __name__ == "__main__":
    study_name = "mpc_design_CISBAT2025_2hgrid_48hpred_3600s"
    sim_result_name = "Design_0_sim_agent.csv"
    plot_sim_stats(study_name=study_name, sim_result_name=sim_result_name)
