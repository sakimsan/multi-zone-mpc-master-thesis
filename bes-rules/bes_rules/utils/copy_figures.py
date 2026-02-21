import os
import shutil
import socket
from pathlib import Path

from bes_rules import LATEX_FIGURES_FOLDER


LATEX_FIGURES_FOLDER_EBC_SIM = Path(r"R:\_Dissertationen\fwu\06_Diss\04_Schreiben\Figures")

from studies.use_case_1_design.no_dhw import PATH_INPUT_ANALYSIS as PATH_INPUT_ANALYSIS_NO_DHW
from studies.use_case_1_design.dhw import PATH_INPUT_ANALYSIS as PATH_INPUT_ANALYSIS_DHW


def copy_figures_to_ebc_sim_folder():
    rules_static = PATH_INPUT_ANALYSIS_NO_DHW.joinpath("manual_innovization", "TEASER", "parameterStudy.TBiv", "costs_total")
    static_oed = PATH_INPUT_ANALYSIS_NO_DHW.joinpath("OED", "verification")
    latex_figures = {
        "4_wp/static/GTZ_Q_T.png": rules_static.joinpath("Q_demand_total_GTZ_Ti_HT_THyd_nominal", "all_all_PowerLaw.png"),
        "4_wp/static/GTZ_Q.png": rules_static.joinpath("Q_demand_total_GTZ_Ti_HT", "all_all_PowerLaw.png"),
        "4_wp/static/GTZ.png": rules_static.joinpath("GTZ_Ti_HT", "all_all_PowerLaw.png"),
        "4_wp/static/feature_correlations.png": PATH_INPUT_ANALYSIS_DHW.joinpath("OED", "features_correlation_pearson.png"),
        "4_wp/static/OED_convergence.png": static_oed.joinpath("parameterStudy.TBiv_costs_total_Q_demand_total_GTZ_Ti_HT_THyd_nominal_all_all_PowerLaw.png"),
        "4_wp/static/feature_values.png": static_oed.joinpath("iter_3", "OED_results_0.png")
    }
    for goal_path, local_path in latex_figures.items():
        path_on_ebc_sim = LATEX_FIGURES_FOLDER_EBC_SIM.joinpath(goal_path)
        os.makedirs(path_on_ebc_sim.parent, exist_ok=True)
        print("Copying file", goal_path)
        shutil.copy(local_path, path_on_ebc_sim)


def copy_figures_to_latex_folder():
    shutil.copytree(LATEX_FIGURES_FOLDER_EBC_SIM, LATEX_FIGURES_FOLDER, dirs_exist_ok=True)


if __name__ == '__main__':
    if socket.gethostname() == 'Laptop-EBC221':
        copy_figures_to_latex_folder()
    else:
        copy_figures_to_ebc_sim_folder()
