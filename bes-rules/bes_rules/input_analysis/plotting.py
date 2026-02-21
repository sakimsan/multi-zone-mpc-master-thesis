import json
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from bes_rules.boundary_conditions.building import get_all_tabula_sfh_buildings, create_buildings
from bes_rules.boundary_conditions.weather import get_all_weather_configs

from bes_rules.utils.functions import argmean
from bes_rules.input_analysis.utils import get_time_series_data_for_one_case
from bes_rules.input_analysis.pv import load_and_filter_results, radTil, radTil_value
from bes_rules.plotting.boxplots import plot_box


def plot_cluster(save_path: Path, interval: str, years_to_skip: list):
    df_results = load_and_filter_results(save_path, interval, years_to_skip=years_to_skip)
    cols = df_results.columns
    df_for_simulation = pd.DataFrame(columns=cols)
    for metric in [
        "own_consumption",
        "self_sufficiency_degree",
        "pv_generation",
        "WEle_demand",
        "W_pv_grid_feed_in",
        "solar_share_dhw",
        "solar_share_bui",
        "solar_share_house"
    ]:
        df_m = df_results.loc[:, metric]
        try:
            df_for_simulation.loc[f"{metric}_max", cols] = df_results.loc[df_m == df_m.max(), cols].values
        except ValueError:
            df_for_simulation.loc[f"{metric}_max", cols] = df_results.loc[df_m == df_m.max(), cols].values[0, :]
        try:
            df_for_simulation.loc[f"{metric}_min", cols] = df_results.loc[df_m == df_m.min(), cols].values
        except ValueError:
            df_for_simulation.loc[f"{metric}_min", cols] = df_results.loc[df_m == df_m.min(), cols].values[0, :]
        df_for_simulation.loc[f"{metric}_mean", cols] = df_results.iloc[argmean(df_m)][cols].values
        for _type_name in [
            "building",
            "weather",
            "case"
        ]:
            _type_data = _get_data_for_boundary_type(
                df_results=df_results, metric=metric,
                boundary_type=_type_name, years_to_skip=years_to_skip)
            fig, axes = plt.subplots(figsize=(15.5 / 2.6 / 2, 15.5 / 2.6 / 2 * 1.2))
            df = pd.DataFrame(_type_data)
            plot_box(df=df, orient='h', axes=axes)

            renames = {
                "case": "(a) PV Installation", "building": "(b) Building Envelope", "weather": "Weather",
                "self_sufficiency_degree": "$SSD$ in %"
            }
            axes.set_title(renames[_type_name])
            axes.set_xlabel(renames.get(metric, metric))
            if metric in ["own_consumption", "self_sufficiency_degree"]:
                axes.set_xlim([0, 100])
            fig.tight_layout()
            fig.savefig(save_path.joinpath(interval, f"{metric}_{_type_name}.png"))
            fig.savefig(save_path.joinpath(interval, f"{metric}_{_type_name}.pdf"))
            plt.close("all")


def _get_data_for_boundary_type(
        df_results: pd.DataFrame,
        boundary_type: str,
        years_to_skip: list,
        metric: str):
    _type_data = {}
    for _type_value in df_results.loc[:, boundary_type].unique():
        if _type_value in ["WestEast", "NorthSouth"] and boundary_type == "case":
            raise ValueError("Filtering did not work")
        if boundary_type == "building" and _type_value.split("_")[0] in years_to_skip:
            raise ValueError("Filtering did not work")
        _value = df_results.loc[df_results.loc[:, boundary_type] == _type_value, metric]
        if _type_value == "EastWest":
            _type_data["East-West"] = _value
        elif _type_value == "SouthNorth":
            _type_data["South-North"] = _value
        elif boundary_type == "building":
            _type_data[_type_value.replace("-standard", "-no-retrofit")] = _value
        else:
            _type_data[_type_value] = _value
    return _type_data


def plot_weather_cases_only(save_path: Path, interval: str = "D"):
    save_path_json = save_path.joinpath("pv_simulations.json")
    with open(save_path_json, "r") as file:
        pv_results = json.load(file)
    for res_tuple, df_path in pv_results.items():
        weather_name, case_name = res_tuple.split("____")
        df_sum, df_max, df_toda_mean = _load_and_convert_df_to_interval(df_path, interval)
        _plot_single_weather_case(
            case_name=case_name,
            df_rad_max=df_max[radTil_value].values,
            df_rad_sum=df_sum[radTil].values,
            df_t_mean=df_toda_mean,
            save_path=save_path.joinpath(weather_name),
        )


def _plot_single_weather_case(case_name: str, df_rad_sum, df_rad_max, df_t_mean, save_path: Path):
    fig, ax = plt.subplots(1, 1)
    ax_twin = ax.twinx()
    ax.scatter(df_t_mean - 273.15,
               df_rad_sum,
               color="red", label="$H_\mathrm{PV,sum,d}$")
    ax_twin.scatter(df_t_mean - 273.15, df_rad_max,
                    color="blue", label="$H_\mathrm{PV,max,d}$")
    ax.set_xlabel("$\hat{T}_\mathrm{Oda}$ in °C")
    ax.set_ylabel("$H_\mathrm{PV,sum,d}$ in kWh/m2")
    ax.legend()
    ax_twin.legend()
    ax_twin.set_ylabel("$H_\mathrm{PV,max,d}$ in kW/m2")
    fig.savefig(save_path.joinpath(f"Correlation_{case_name}"))
    print("saved", save_path)
    plt.close("all")


def plot_three_cases(save_path: Path, day_of_year: int, study_path: Path, n_days=2):
    weather_configs = get_all_weather_configs()
    weather_configs = {weather_config.get_name(pretty_print=True): weather_config for weather_config in weather_configs}
    building_configs = get_all_tabula_sfh_buildings()
    building_configs = create_buildings(building_configs=building_configs, export=False, name="InputAnalysis")
    building_configs = {building_config.name: building_config for building_config in building_configs}
    cases_to_simulate = pd.read_excel(Path(__file__).absolute().parent.joinpath("SimulationCases.xlsx"),
                                      sheet_name="SimulationCases")

    dym_results_path = study_path.joinpath("DesignOptimizationResults")
    for idx, row in cases_to_simulate.iterrows():
        weather_config = weather_configs[row["weather"]]
        building_name = row["building"].replace("-", "_")
        P_pv_h, P_demand_sum_h, Q_demand_building_h, Q_demand_DHW_h = get_time_series_data_for_one_case(
            weather_config=weather_config,
            building_config=building_configs[building_name],
            interval="1H", save_path=save_path, case_name=row["case"], day_of_year=day_of_year, n_days=n_days
        )
        P_pv_d, P_demand_sum_d, Q_demand_building_d, Q_demand_DHW_d = get_time_series_data_for_one_case(
            weather_config=weather_config,
            building_config=building_configs[building_name],
            interval="1D", save_path=save_path, case_name=row["case"], day_of_year=day_of_year, n_days=n_days
        )

        case_path = dym_results_path.joinpath(
            f"{weather_config.get_name()}_B{building_name}_SingleDwelling_M_{row['case']}")
        from bes_rules.simulation_based_optimization import BESMod
        df_res = BESMod.load_design_optimization_log(file_path=case_path.joinpath("DesignOptimizerResults.xlsx"))
        idx_max_ssd = df_res.loc[:, "self_sufficiency_degree"].argmax()
        from ebcpy import TimeSeriesData
        tsd_file = TimeSeriesData(case_path.joinpath(f"iterate_{idx_max_ssd}.hdf"))
        tsd_file = tsd_file.loc[day_of_year * 86400:(day_of_year + n_days) * 86400]
        tsd_file.index -= tsd_file.index[0]
        tsd_file.index /= 3600  # to h
        tsd_file /= 1000
        from bes_rules.plotting.utils import get_figure_size
        from bes_rules.plotting import EBCColors
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=get_figure_size(1, height_factor=1.5))
        ax[0].plot(P_pv_h.index, P_pv_h.values, label="PV hourly", color="black", linestyle="-")
        ax[0].plot(P_pv_d.index, P_pv_d.values, label="PV daily", color="black", linestyle="--")
        ax[0].plot(tsd_file.index, tsd_file["outputs.electrical.gen.PElePV.value"], label="PV detailed", color="black",
                   linestyle="-.")

        ax[0].plot(P_demand_sum_h.index, P_demand_sum_h.values, label="Hourly", color=EBCColors.blue)
        ax[0].plot(P_demand_sum_d.index, P_demand_sum_d.values, label="Daily", color=EBCColors.blue, linestyle="--")
        ax[0].plot(tsd_file.index, tsd_file["outputs.hydraulic.gen.PEleHeaPum.value"] + tsd_file[
            "outputs.hydraulic.gen.PEleEleHea.value"], label="Detailed", color=EBCColors.red)
        ax[0].set_ylabel("$P_\mathrm{el}$ in kW")
        ax[0].legend(bbox_to_anchor=(0, 1.02, 1, .102), loc="lower left", ncol=2, mode="expand", borderaxespad=0)
        ax[1].plot(tsd_file.index, tsd_file["outputs.building.eneBal[1].traGain.value"], label="Detailed",
                   color=EBCColors.red)
        ax[1].plot(Q_demand_building_h.index, Q_demand_building_h.values, label="Hourly", color=EBCColors.blue)
        ax[1].plot(Q_demand_building_d.index, Q_demand_building_d.values, label="Daily", color=EBCColors.blue,
                   linestyle="--")

        ax[1].set_ylabel("$\dot{Q}_\mathrm{Dem}$ in kW")
        # ax[2].plot(tsd_file.index, tsd_file["outputs.DHW.Q_flow.value"], label="DHW", color=EBCColors.red)
        # ax[2].plot(Q_demand_DHW.index, Q_demand_DHW.values, label="Simplified", color=EBCColors.blue)
        # ax[2].set_ylabel("$\dot{Q}_\mathrm{DHW}$ in kW")
        ax[1].set_xlabel("Time in h")
        fig.tight_layout()
        fig.align_ylabels()
        name = save_path.joinpath(f"time_series_plot_{day_of_year}_{day_of_year + n_days}_case{idx}")
        fig.savefig(str(name) + ".png")
        fig.savefig(str(name) + ".pdf")
        plt.close("all")


if __name__ == '__main__':
    COLOR = "black"
    RC_PARAMS = {
        'font.size': 11,  # controls default text sizes
        'axes.titlesize': 11,  # fontsize of the axes title
        'axes.labelsize': 11,  # fontsize of the x and y labels
        'xtick.labelsize': 11,  # fontsize of the tick labels
        'ytick.labelsize': 11,  # fontsize of the tick labels
        'legend.fontsize': 11,  # legend fontsize
        'figure.titlesize': 11,  # fontsize of the figure title
        # 'text.usetex': True,
        'text.color': COLOR,
        'axes.labelcolor': COLOR,
        'xtick.color': COLOR,
        'ytick.color': COLOR,
    }

    sns.set_theme(style="whitegrid", rc=RC_PARAMS)
