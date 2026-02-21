import logging
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bes_rules import configs, STARTUP_BESMOD_MOS, BESRULES_PACKAGE_MO
from bes_rules.boundary_conditions import building, weather
from bes_rules.input_variations import InputVariations
from bes_rules.plotting.utils import get_all_results_from_config, PlotConfig
from bes_rules.plotting.utils import get_figure_size
from bes_rules.simulation_based_optimization.utils import constraints
from ebcpy import TimeSeriesData

BASE_PATH = Path(r"D:\00_temp\02_storage_VDI_presentation")
PATH_PAPER = Path(r"R:\_Dissertationen\fwu\03_Paper\01_Published\2024\VDI")
PLOT_CONFIG = PlotConfig.load_default()


VOL_NAME = "parameterStudy.VPerQFlow"
BIV_NAME = "parameterStudy.TBiv"


def run_optimization(test_only=False):
    sim_config = configs.SimulationConfig(
        startup_mos=STARTUP_BESMOD_MOS,
        model_name="BESRules.DesignOptimization.MonoenergeticVitoCal",
        sim_setup=dict(stop_time=86400 * 365, output_interval=600),
        result_names=["scalingFactor"],
        packages=[BESRULES_PACKAGE_MO],
        type="Dymola",
        recalculate=False,
        show_window=True,
        debug=False,
        extract_results_during_optimization=True,
        convert_to_hdf_and_delete_mat=False,
        equidistant_output=True,
        dymola_api_kwargs={"time_delay_between_starts": 4}
    )

    ## Optimization
    optimization_config = configs.OptimizationConfig(
        framework="doe",
        method="ffd",
        constraints=[constraints.BivalenceTemperatureGreaterNominalOutdoorAirTemperature()],
        variables=[
            configs.OptimizationVariable(
                name=BIV_NAME,
                lower_bound=273.15 - 12,
                upper_bound=273.15 + 4,
                discrete_steps=3
            ),
            configs.OptimizationVariable(
                name=VOL_NAME,
                lower_bound=5,
                upper_bound=100,
                levels=15
            )
        ],
    )
    weathers = weather.get_weather_configs_by_names(["Potsdam"])
    buildings = building.get_all_tabula_sfh_buildings(as_dict=True)
    buildings = [
        buildings["2015_standard"],
        buildings["1980_standard"],
        buildings["1918_retrofit"],
        buildings["1950_retrofit"]
    ]

    inputs_config = configs.InputsConfig(
        weathers=weathers,
        buildings=buildings,
        dhw_profiles=[{"profile": "M"}],
        evu_profiles=[configs.inputs.EVUProfile(profile="EVU_Sperre_EON"),
                      configs.inputs.EVUProfile(profile=None)]
    )
    config = configs.StudyConfig(
        base_path=BASE_PATH,
        n_cpu=11,
        name="EVUControl",
        simulation=sim_config,
        optimization=optimization_config,
        inputs=inputs_config,
        test_only=test_only,
    )
    DESOPT = InputVariations(
        config=config
    )
    # DESOPT.run()


def plot_storage_and_room_temperatures(name: str = None, storage_volumes: list = None, path: Path = None):
    if path is None:
        path = Path(
            r"N:\Forschung\EBC0878_LivingLabNRW_PractiCon_GES\Students\fwu-mha\mha-ba_Simulationsergebnisse\00_temp\01_design_optimization\Testzone\DesignOptimizationResults\TRY2015_MannLiang_Jahr_NoRetrofit1983_SingleDwelling_ProfileM_None"
        )
        t_set = "hydraulic.control.buiAndDHWCtr.heaCur.TSet"
        idx_files = [1, 9]
        storage_volumes = [5, 20]
        name = "frost_plot"
    else:
        t_set = "hydraulic.control.buiAndDHWCtr.TSetBuiSup.TSet"
        idx_files = None

    sto_top = "outputs.hydraulic.disCtrl.TStoBufTopMea"
    room_t = "outputs.building.TZone[1]"
    t_set_room = "outputs.user.TZoneSet[1]"
    variable_names = [
        sto_top,
        room_t,
        t_set,
        t_set_room,
        "systemParameters.QBui_flow_nominal[1]",
        "systemParameters.THydSup_nominal[1]"
    ]
    storage_vols = []
    if idx_files is None:
        df = pd.read_excel(path.joinpath("DesignOptimizerResults.xlsx"))
    for i, volume in enumerate(storage_volumes):
        if idx_files is None:
            idx_file = df.loc[df.loc[:, VOL_NAME] == volume]
            idx_file = min(idx_file.index)
            day = 16
        else:
            idx_file = idx_files[i]
            day = 353.5

        V = TimeSeriesData(path.joinpath(f"iterate_{idx_file}.mat"), variable_names=variable_names)
        V.index /= 3600
        V = V.loc[day * 24:(day + 1) * 24 + 1]
        V.index -= V.index[0]
        storage_vols.append(V.copy())
        # print(name, "Q", V.iloc[-1]["systemParameters.QBui_flow_nominal[1]"].values)
        # print(name, "T", V.iloc[-1]["systemParameters.THydSup_nominal[1]"].values - 273.15)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=get_figure_size(n_columns=1))
    for V, size in zip(storage_vols, storage_volumes):
        ax[0].plot(V.index, V.loc[:, sto_top] - 273.15, label=f"{size} l/kW")
    ax[0].plot(V.index, V.loc[:, t_set] - 273.15, color="black", label="Soll")
    ax[0].set_ylabel("$T_\mathrm{Speicher}$ in °C")
    for V, size in zip(storage_vols, storage_volumes):
        ax[1].plot(V.index, V.loc[:, room_t] - 273.15, label=f"{size} l/kW")
    ax[1].plot(V.index, V.loc[:, t_set_room] - 273.15, color="black", label="Soll")
    ax[1].set_ylabel("$T_\mathrm{Raum}$ in °C")
    ax[1].set_xlabel("Zeit in h")
    ax[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1))
    ax[1].set_yticks([18, 19, 20])
    # ax[1].set_xlim([0, 15])
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(PATH_PAPER.joinpath(f"temperatures_{name}.png"), bbox_inches='tight')


def plot_metric_over_storage_size(metric: str, with_evu: bool = True, switches_relative: bool = False,
                                  optimal_storage_volumes: dict = None, cost_optimal_storage_volumes: dict = None):
    study_config = configs.StudyConfig.from_json(BASE_PATH.joinpath("EVUControl", "study_config.json"))
    dfs, input_configs = get_all_results_from_config(study_config=study_config)
    swi_name = "outputs.hydraulic.gen.heaPum.numSwi"
    if switches_relative and metric != swi_name:
        raise ValueError

    same_limits = ("dTControl" not in metric) or ("dTCom" in metric and not with_evu)

    fig, ax = plt.subplots(int(len(input_configs) / 2), 1, sharex=True)
    idx_case = 0
    min_y, max_y = np.inf, -np.inf
    # Order year of construction and filter redundant inputs
    dfs_to_plot, input_configs_to_plot = {}, {}
    for df, input_config in zip(dfs, input_configs):
        if with_evu and input_config.evu_profile.profile is None:
            continue
        elif not with_evu and input_config.evu_profile.profile is not None:
            continue
        year = int(input_config.building.name.split("_")[0])
        dfs_to_plot[year] = df
        input_configs_to_plot[year] = input_config

    for year in [2015, 1980, 1950, 1918]:
        df = dfs_to_plot[year]
        input_config = input_configs_to_plot[year]
        df = PLOT_CONFIG.scale_df(df)
        for biv in df.loc[:, BIV_NAME].unique():
            df_biv = df.loc[df.loc[:, BIV_NAME] == biv]
            if switches_relative:
                min_switches = df_biv.loc[:, swi_name].min()
                values = df_biv.loc[:, swi_name] / min_switches * 100
            else:
                values = df_biv.loc[:, metric]
            ax[idx_case].plot(df_biv.loc[:, VOL_NAME], values, label=int(biv))
            min_y = min(min(values), min_y)
            max_y = max(max(values), max_y)
        ax[idx_case].text(.98, .98, f"Baujahr: {year}",
                          ha='right', va='top', transform=ax[idx_case].transAxes)
        idx_case += 1
        input_name = input_config.get_name()
        if with_evu and "dTCom" in metric:
            plot_storage_and_room_temperatures(
                name=input_name,
                storage_volumes=[5, 52.5, 100],
                path=BASE_PATH.joinpath(
                    "EVUControl",
                    "DesignOptimizationResults",
                    input_name
                )
            )
        if switches_relative:
            df_pareto_efficient = plot_pareto_front_grid(
                df=df,
                save_path=PATH_PAPER.joinpath(f"pareto_{input_name}.png"),
                objectives=["costs_total", swi_name]
            )
            V = df_pareto_efficient.loc[:, "hydraulic.distribution.parStoBuf.V"]
            QHPMax = 6733.56
            single_bivs = {}
            for idx in df_pareto_efficient.index:
                biv = df_pareto_efficient.loc[idx, BIV_NAME]
                if biv in single_bivs:
                    QHeaPumBiv_flow = single_bivs[biv]
                else:
                    QHeaPumBiv_flow = TimeSeriesData(
                        study_config.study_path.joinpath(
                            "DesignOptimizationResults", input_name, f"iterate_{idx}.mat"),
                        variable_names=["QHeaPumBiv_flow"]
                    ).to_df().iloc[-1].values[0]
                    single_bivs[biv] = QHeaPumBiv_flow
                df_pareto_efficient.loc[idx, "QHeaPumBiv_flow"] = QHeaPumBiv_flow
            idx_opt = df.loc[:, "costs_total"].argmin()
            cost_optimum = df.loc[idx_opt]
            V_opt = cost_optimum["hydraulic.distribution.parStoBuf.V"]
            df_pareto_efficient = df_pareto_efficient.loc[
                df_pareto_efficient.loc[:, BIV_NAME] == cost_optimum[BIV_NAME]
            ]
            cost_optimal_storage_volumes[input_name] = {
                "V": V_opt,
                VOL_NAME: cost_optimum[VOL_NAME],
                "4645": V_opt / df_pareto_efficient.loc[idx_opt, "QHeaPumBiv_flow"],
                "15450": V_opt / (QHPMax * cost_optimum["scalingFactor"])
            }
            optimal_storage_volumes[input_name] = {
                "V": V,
                VOL_NAME: df_pareto_efficient.loc[:, VOL_NAME],
                "4645": V / df_pareto_efficient.loc[:, "QHeaPumBiv_flow"],
                "15450": V / (QHPMax * df_pareto_efficient.loc[:, "scalingFactor"])
            }

    ax[-1].set_xlabel(PLOT_CONFIG.get_label_and_unit(VOL_NAME))
    ax[0].legend(bbox_to_anchor=(0, 1), loc="lower left", ncol=3)
    if same_limits:
        for _ax in ax:
            _ax.set_ylim([min_y, max_y])
    if switches_relative:
        fig.supylabel(f"Anstieg Schaltzyklen in %")
    else:
        fig.supylabel(PLOT_CONFIG.get_label_and_unit(metric))
    fig.tight_layout()
    fig.savefig(PATH_PAPER.joinpath(
        f"{metric}_evu={with_evu}_rel={switches_relative}.png"), bbox_inches='tight'
    )
    plt.close("all")
    return optimal_storage_volumes, cost_optimal_storage_volumes


def plot_all_combinations():
    swi_name = "outputs.hydraulic.gen.heaPum.numSwi"
    com_name = "outputs.building.dTComHea[1]"
    control_name = "outputs.building.dTControlHea[1]"
    all_optimal_storage_volumes = {}
    all_cost_optimal_storage_volumes = {}

    for with_evu in [True, False]:
        all_optimal_storage_volumes, all_cost_optimal_storage_volumes = plot_metric_over_storage_size(
            metric=swi_name, with_evu=with_evu, switches_relative=True,
            optimal_storage_volumes=all_optimal_storage_volumes,
            cost_optimal_storage_volumes=all_cost_optimal_storage_volumes
        )
        plot_metric_over_storage_size(
            metric=swi_name, with_evu=with_evu, switches_relative=False
        )
        plot_metric_over_storage_size(metric=com_name, with_evu=with_evu, switches_relative=False)
        plot_metric_over_storage_size(metric=control_name, with_evu=with_evu, switches_relative=False)

    with open(PATH_PAPER.joinpath("all_volumes.pickle"), "wb") as file:
        pickle.dump(all_optimal_storage_volumes, file)

    with open(PATH_PAPER.joinpath("all_cost_volumes.pickle"), "wb") as file:
        pickle.dump(all_cost_optimal_storage_volumes, file)

def plot_storage_volume_regression_space():
    with open(PATH_PAPER.joinpath("all_volumes.pickle"), "rb") as file:
        all_optimal_storage_volumes = pickle.load(file)
    storage_labels = {
        #"V": "$V$\n in l",
        VOL_NAME: PLOT_CONFIG.get_label_and_unit(VOL_NAME, linebreak=True),
        "4645": "$v_\mathrm{VDI 4645}$\n in l/kW",
        "15450": "$v_\mathrm{VDI 15450}$\n in l/kW",
    }
    storage_factors = {
        #"V": 1000,
        VOL_NAME: 1,
        "4645": 1000000,
        "15450": 1000000,
    }
    fig, ax = plt.subplots(len(storage_labels), 1, sharex=True,
                           figsize=get_figure_size(n_columns=1, height_factor=1.25))
    old_len = 0
    for input_name, results in all_optimal_storage_volumes.items():
        if "EVU" in input_name:
            color = "red"
        else:
            color = "blue"
        for idx, sto_type in enumerate(storage_labels.keys()):
            values = results[sto_type].values * storage_factors[sto_type]
            ax[idx].scatter(np.arange(len(values)) + old_len, values, color=color, marker="s", s=8)
        old_len += len(values)
        print(input_name, old_len)
    ax[-1].set_xlabel("Pareto-Optimale Lösungen")
    #ax[0].legend(loc="lower right")
    for idx, storage_type in enumerate(storage_labels):
        ax[idx].set_ylabel(storage_labels[storage_type])
        if storage_type == "15450":
            ax[idx].fill_between([0, old_len], [12, 12], [35, 35],
                                 edgecolor=None, alpha=0.5, facecolor="gray")
        elif storage_type == "4645":
            ax[idx].axhline(20, color="gray")
            ax[idx].fill_between([0, old_len / 2], [60, 60], [80, 80],
                                 edgecolor=None, alpha=0.5, facecolor="gray")
    fig.tight_layout()
    fig.align_ylabels()
    fig.savefig(PATH_PAPER.joinpath("regression_rule_plot.png"), bbox_inches="tight")



def plot_cost_optimal_storage_volume_regression_space():
    with open(PATH_PAPER.joinpath("all_cost_volumes.pickle"), "rb") as file:
        all_optimal_storage_volumes = pickle.load(file)
    storage_labels = {
        #"V": "$V$\n in l",
        VOL_NAME: PLOT_CONFIG.get_label(VOL_NAME),
        "4645": "$v_\mathrm{VDI 4645}$",
        "15450": "$v_\mathrm{VDI 15450}$",
    }
    storage_factors = {
        #"V": 1000,
        VOL_NAME: 1,
        "4645": 1000000,
        "15450": 1000000,
    }
    fig, ax = plt.subplots(1, 1, sharex=True,
                           figsize=get_figure_size(n_columns=1, height_factor=1))
    for input_name, results in all_optimal_storage_volumes.items():
        if "EVU" in input_name:
            color = "red"
        else:
            color = "blue"
        for idx, sto_type in enumerate(storage_labels.keys()):
            value = results[sto_type] * storage_factors[sto_type]
            ax.scatter(idx, value, color=color, marker="s", s=8)

    ax.set_ylabel("Volumen in l/kW")
    ax.set_xticks(range(len(storage_labels)))
    ax.set_xticklabels(list(storage_labels.values()))
    for idx, storage_type in enumerate(storage_labels):
        if storage_type == "15450":
            ax.fill_between([idx-0.25, idx+0.25], [12, 12], [35, 35],
                                 edgecolor=None, alpha=0.5, facecolor="gray")
        elif storage_type == "4645":
            ax.fill_between([idx-0.25, idx+0.25], [19, 19], [20, 20],
                                 edgecolor=None, alpha=0.5, facecolor="gray")

            ax.fill_between([idx-0.25, idx+0.25], [60, 60], [80, 80],
                                 edgecolor=None, alpha=0.5, facecolor="red")
    fig.tight_layout()
    fig.align_ylabels()
    fig.savefig(PATH_PAPER.joinpath("regression_distribution.png"), bbox_inches="tight")


def plot_pareto_front_grid(df, objectives: list, save_path: Path):
    assert len(objectives) == 2, "Only 2D front supported for plotting"
    from bes_rules.utils.pareto import get_pareto_efficient_points_for_df
    df = df.copy()
    print("Max discomfort:", df.loc[:, "outputs.building.dTComHea[1]"].max())
    df = df.loc[df.loc[:, "outputs.building.dTComHea[1]"] <= 73.5]
    df_pareto_efficient = get_pareto_efficient_points_for_df(df=df.copy(), objectives=objectives)
    with_biv = False
    nrows = 3 if with_biv else 2
    fig, ax = plt.subplots(
        nrows, 1,
        figsize=get_figure_size(n_columns=1, height_factor=1),
        sharex=True)
    x_obj, y_obj = objectives
    ax[0].scatter(
        df_pareto_efficient.loc[:, x_obj],
        df_pareto_efficient.loc[:, y_obj],
        color="red", marker="s"
    )
    ax[1].scatter(
        df_pareto_efficient.loc[:, x_obj],
        df_pareto_efficient.loc[:, VOL_NAME],
        color="red", marker="s"
    )
    if with_biv:
        ax[2].scatter(
            df_pareto_efficient.loc[:, x_obj],
            df_pareto_efficient.loc[:, BIV_NAME],
            color="red", marker="s"
        )
        ax[2].set_ylabel(PLOT_CONFIG.get_label_and_unit(BIV_NAME))
    ax[0].set_ylabel(PLOT_CONFIG.get_label_and_unit(y_obj))
    ax[1].set_ylabel(PLOT_CONFIG.get_label_and_unit(VOL_NAME))
    ax[-1].set_xlabel(PLOT_CONFIG.get_label_and_unit(x_obj))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close("all")
    return df_pareto_efficient


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    plt.rcParams.update(
        {"figure.figsize": [3.5, 4.2],
         "font.size": 10,
         "axes.titlesize": 10,
         "figure.dpi": 250,
         "font.family": "Arial"
         }
    )
    # plot_storage_and_room_temperatures()
    # run_optimization(test_only=False)
    plot_all_combinations()
    #plot_cost_optimal_storage_volume_regression_space()
    plot_storage_volume_regression_space()