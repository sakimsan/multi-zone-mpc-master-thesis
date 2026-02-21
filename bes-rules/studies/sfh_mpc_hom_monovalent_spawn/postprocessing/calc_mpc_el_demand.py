from bes_rules import RESULTS_FOLDER
from agentlib_mpc.utils.analysis import load_sim


def print_electricity_demand(study_name: str, sim_result_name: str, start_time=0, stop_time=0):
    sim_results_path = RESULTS_FOLDER.joinpath(
        "SFH_MPCRom_monovalent_spawn",
        study_name,
        "DesignOptimizationResults",
        "TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai_"
    )

    df = load_sim(sim_results_path.joinpath(sim_result_name))
    heat_pump_el_demand = df["outputs.hydraulic.gen.PEleHeaPum.integral"]
    heat_pump_el_demand_start = heat_pump_el_demand.loc[start_time]
    heat_pump_el_demand_end = heat_pump_el_demand.loc[stop_time]
    heat_pump_el_demand_res = heat_pump_el_demand_end - heat_pump_el_demand_start
    heat_pump_el_demand_res_kWh = heat_pump_el_demand_res / 3600 / 1000

    print(f"Electricity demand of the heat pump: {heat_pump_el_demand_res_kWh} kWh")

if __name__ == "__main__":
    study_name = "mpc_design_CISBAT2025_6hgrid_48hpred_3600s"
    sim_result_name = "Design_2_sim_agent.csv"
    print_electricity_demand(study_name=study_name, sim_result_name=sim_result_name, start_time=24*24*3600, stop_time=31*24*3600-3600)