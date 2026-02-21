import pandas as pd

from utils.files import load_sim
from utils.mpc import optimization_results
from utils.quality import quality, prediction_quality
from utils.comfort import comfort
from utils.costs import costs
from utils.electric import electrical
from utils.timely import compare_monthly_bar, compare_time_series
from utils.kolloquium import plot_pv_day


def evaluate_one(control: str, load_path: str, save_path: str, with_opt_results: bool, with_koll: bool, categories: list):
    save_path = f"{save_path}/{control}"

    df = load_sim(load_path, control)

    if with_opt_results and "mpc" in control:
        begin = pd.Timestamp(year=2015, month=1, day=1) + pd.Timedelta(days=75)
        end = begin + pd.Timedelta(days=4)
        optimization_results(df=df.loc[begin:end, :], save_path=save_path)
        prediction_quality(load_path=load_path.replace("/sim_agent.csv", ""),
                           save_path=save_path,
                           columns=["T_Air", "P_el_hp"],
                           begin=begin,
                           end=end)

    if with_koll and "mpc" in control:
        plot_pv_day(df=df, save_path=save_path)

    evaluations = {}
    if "comfort" in categories:
        df = comfort(save_path=save_path, df=df)
    if "quality" in categories:
        df, df_months_quality = quality(save_path=save_path, df=df)
        df_months_quality.columns = pd.MultiIndex.from_product([[control], df_months_quality.columns])
        evaluations["quality"] = df_months_quality
    if "costs" in categories:
        df, df_months_costs = costs(save_path=save_path, df=df)
        df_months_costs.columns = pd.MultiIndex.from_product([[control], df_months_costs.columns])
        evaluations["costs"] = df_months_costs
    if "electric" in categories:
        df, df_months_electric = electrical(save_path=save_path, df=df)
        df_months_electric.columns = pd.MultiIndex.from_product([[control], df_months_electric.columns])
        evaluations["electric"] = df_months_electric

    df.to_excel(f"{save_path}/{control}.xlsx")
    df.columns = pd.MultiIndex.from_product([[control], df.columns])
    evaluations["time_series"] = df
    return evaluations


def evaluate(controls: dict, save_path: str, with_opt_results: bool, with_koll: bool, categories: list, compare_properties: list = None):
    dfs = []
    monthly_quality = []
    monthly_costs = []
    monthly_electric = []
    for control in controls.keys():
        load_path = controls[control]
        evaluations = evaluate_one(control=control, load_path=load_path, with_opt_results=with_opt_results, with_koll=with_koll, save_path=save_path, categories=categories)
        dfs.append(evaluations["time_series"])

        if "quality" in evaluations:
            monthly_quality.append(evaluations["quality"])
        if "costs" in evaluations:
            monthly_costs.append(evaluations["costs"])
        if "electric" in evaluations:
            monthly_electric.append(evaluations["electric"])

    save_path = f"{save_path}/comparison"
    if "quality" in categories:
        compare_monthly_bar(monthly_dfs=monthly_quality, save_path=save_path, save_name="quality")
    if "costs" in categories:
        compare_monthly_bar(monthly_dfs=monthly_costs, save_path=save_path, save_name="costs")
    if "electric" in categories:
        compare_monthly_bar(monthly_dfs=monthly_electric, save_path=save_path, save_name="electric")
    if compare_properties is not None:
        compare_time_series(dfs=dfs, save_path=save_path, properties=compare_properties)


def main():
    from pathlib import Path
    base_path = Path(r"D:/06_Results/ba_peter")
    save_path = base_path.joinpath("evaluation", "results")
    controls = {
        "mpc_wo_UB": base_path.joinpath("MPC/no_T_ub/sim_agent.csv"),
        "mpc_w_UB": base_path.joinpath("MPC/standard/sim_agent.csv"),
        "rbpc_wo_UB": base_path.joinpath("RBPC/BJ1994_hc_no_T_Air_ub_march/sim_rbpc.xlsx"),
        "rbpc_zcbe": base_path.joinpath("RBPC_ZCBE/Design_0.xlsx"),
        "noSGReady": base_path.joinpath("RBC_noSGReady/iterate_0.mat"),
        "rbc": base_path.joinpath("RBC/iterate_0.mat"),
    }
    with_opt_results = True
    with_koll = False
    categories = [
        "comfort",
        "quality",
        "costs",
        "electric"
    ]
    compare_properties = ["T_Air"]
    evaluate(controls=controls, save_path=save_path, categories=categories, compare_properties=compare_properties, with_opt_results=with_opt_results, with_koll=with_koll)


if __name__ == "__main__":
    main()
