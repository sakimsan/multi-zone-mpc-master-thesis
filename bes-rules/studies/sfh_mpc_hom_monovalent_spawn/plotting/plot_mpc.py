from bes_rules import RESULTS_FOLDER
from collections import defaultdict
from pathlib import Path

def plot_mpc_stats(study_name: str, mpc_result_name: str):
    from agentlib_mpc.utils.analysis import load_mpc
    from agentlib_mpc.utils.plotting.interactive import show_dashboard
    mpc_results_path = RESULTS_FOLDER.joinpath(
        "Problem_Studies",
        "mpc_design_Aufheizverhalten",
        "DesignOptimizationResults",
        "TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai_"
    )
    #mpc_results_path = Path(r"C:\Users\tbc-ssr\Desktop")
    mpc_results = load_mpc(mpc_results_path.joinpath(mpc_result_name))


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

    #filtered_results = mpc_results.loc[:, [col for col in mpc_results.columns if str(col[1]).__contains__("GroundFloor")]]

    show_dashboard(data=filtered_results, scale="hours")


if __name__ == "__main__":
    study_name = "mpc_design_Nachtabsenkung_TARP_Max"
    mpc_result_name = "Design_0_mpc_agent.csv"
    plot_mpc_stats(study_name=study_name, mpc_result_name=mpc_result_name)