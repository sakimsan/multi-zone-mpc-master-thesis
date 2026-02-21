import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from bes_rules import configs, RESULTS_FOLDER
from bes_rules.plotting import utils, EBCColors
from bes_rules.boundary_conditions.prices import load_dynamic_electricity_prices

from studies.use_case_1_design.influence_mpc import get_case_file_names


def compare_design_for_y_variables_vars(
        study_name: str,
        y_variables,
        save_name,
        cases: dict,
        storage_size_markers: dict
):
    x_variable = "parameterStudy.TBiv"
    plot_config = utils.load_plot_config()
    study_path = RESULTS_FOLDER.joinpath("UseCase_TBivAndV", study_name)
    study_config = configs.StudyConfig.from_json(study_path.joinpath("study_config.json"))
    dfs, input_configs = utils.get_all_results_from_config(study_config=study_config)
    fig, axes = utils.create_plots(
        plot_config=plot_config,
        x_variables=[x_variable],
        y_variables=y_variables,
    )
    for df, input_config in zip(dfs, input_configs):
        storage_sizes = list(df.loc[:, "parameterStudy.VPerQFlow"].unique())
        df = plot_config.scale_df(df)
        data = cases[input_config.modifiers[0].name]
        for storage_size in storage_sizes:
            mask_storage = df.loc[:, "parameterStudy.VPerQFlow"] == storage_size
            for _y_variable, _ax in zip(y_variables, axes[:, 0]):
                if storage_size in storage_size_markers:
                    marker = storage_size_markers[storage_size]
                    label = f"{data['label']} - {storage_size} l/kW"
                else:
                    marker = data.get("marker", None)
                    label = data["label"]
                _ax.scatter(
                    df.loc[mask_storage, x_variable], df.loc[mask_storage, _y_variable],
                    color=data["color"], marker=marker, label=label, s=10
                )

    axes[0, 0].legend(bbox_to_anchor=(0, 1), loc="lower left")
    utils.save(
        fig=fig, axes=axes,
        save_path=RESULTS_FOLDER.joinpath("UseCase_TBivAndV", study_name, save_name),
        show=False, with_legend=False, file_endings=["png"]
    )


def compare_design_for_important_variables(
        study_name: str,
        cases: dict,
        storage_size_markers: dict
):
    compare_design_for_y_variables_vars(
        study_name=study_name,
        cases=cases,
        storage_size_markers=storage_size_markers,
        save_name="costs_swi_com",
        y_variables=[
            "costs_total",
            "outputs.hydraulic.gen.heaPum.numSwi",
            "outputs.building.dTComHea[1]",
            "outputs.building.dTCtrl[1]"
        ])
    compare_design_for_y_variables_vars(
        study_name=study_name,
        cases=cases,
        storage_size_markers=storage_size_markers,
        save_name="SCOP_analysis",
        y_variables=[
            "SCOP_Sys",
            "HP_Coverage",
            "outputs.THeaPumSinMean",
            "outputs.THeaPumSouMean",
        ]
    )


def plot_required_invest_difference_inverter_on_off(study_name: str):
    from bes_rules.plotting import utils
    from bes_rules.objectives.annuity import Annuity
    from bes_rules.objectives.scop import SCOPMapping
    ann = Annuity()
    RBF = ann.get_RBF()
    c_el = ann.k_el
    plot_config = utils.load_plot_config()
    study_path = RESULTS_FOLDER.joinpath("UseCase_TBivAndV", study_name)
    study_config = configs.StudyConfig.from_json(study_path.joinpath("study_config.json"))
    dfs, input_configs = utils.get_all_results_from_config(study_config=study_config)
    df_inverter = None
    df_on_off = None
    for df, input_config in zip(dfs, input_configs):
        df = plot_config.scale_df(df)
        if input_config.modifiers[0].name == "HydraulicSeperator":
            df_inverter = df
            continue
        else:
            mask_storage = df.loc[:, "parameterStudy.VPerQFlow"] == 35
            df_on_off = df.loc[mask_storage]
            continue
    # (SCOP_onoff ** -1 - SCOP_inverter ** -1 ) * Q_Bed * c_el * RBF + K_sto < K_wp_inverter - K_wp_onoff
    # with delta_K_inverter = K_wp_inverter - K_wp_onoff:
    Q_dem_bui = SCOPMapping().building_heat_supplied
    Q_dem_dhw = SCOPMapping().dhw_heat_supplied
    Q_demand_on_off = df_on_off.loc[:, Q_dem_bui] + df_on_off.loc[:, Q_dem_dhw]
    Q_demand_inverter = df_inverter.loc[:, Q_dem_bui] + df_inverter.loc[:, Q_dem_dhw]
    costs_storage = df_on_off.loc[:, "invest_tes"].values * RBF
    delta_K_inverter_no_storage = (
            (Q_demand_on_off.values / df_on_off.loc[:, "SCOP_Sys"].values -
             Q_demand_inverter.values / df_inverter.loc[:, "SCOP_Sys"].values
             ) * c_el * RBF
    )
    delta_K_inverter = delta_K_inverter_no_storage + costs_storage
    print(f"Storage costs are {costs_storage=} €")
    x_variable = "parameterStudy.TBiv"
    fig, axes = utils.create_plots(
        plot_config=plot_config,
        x_variables=[x_variable],
        y_variables=[
            "SCOP_Sys",
            # "$Q_\mathrm{Bed}$ in kWh",
            "$\Delta K_\mathrm{Inv}$ in €"
        ],
    )

    axes[0, 0].scatter(df_inverter.loc[:, x_variable], df_inverter.loc[:, "SCOP_Sys"], s=10, color="red",
                       label="Inverter")
    axes[0, 0].scatter(df_on_off.loc[:, x_variable], df_on_off.loc[:, "SCOP_Sys"], s=10, color="blue", label="OnOff")
    axes[0, 0].set_ylim([2.6, 3.5])
    axes[0, 0].legend(bbox_to_anchor=(0, 1), loc="lower left", ncol=2)
    # axes[1, 0].scatter(df_inverter.loc[:, x_variable], Q_demand_inverter, s=10, color="red", label="Inverter")
    # axes[1, 0].scatter(df_on_off.loc[:, x_variable], Q_demand_on_off, s=10, color="blue", label="OnOff")
    # axes[1, 0].set_ylim(
    #     [
    #         min(Q_demand_on_off.min(), Q_demand_inverter.min()) * 0.7,
    #         max(Q_demand_on_off.max(), Q_demand_inverter.max()) * 1.05
    #     ])
    axes[1, 0].scatter(df_on_off.loc[:, x_variable], delta_K_inverter, s=10, color="black", label="Hyd. Weiche")
    axes[1, 0].scatter(df_on_off.loc[:, x_variable], delta_K_inverter_no_storage, s=10, color="gray", label="Speicher")
    axes[1, 0].legend(bbox_to_anchor=(0, 1), loc="lower left", ncol=2)
    # axes[2].set_ylabel("$\Delta K_\mathrm{Inv}$ in €")
    utils.save(
        fig=fig, axes=axes,
        save_path=RESULTS_FOLDER.joinpath("UseCase_TBivAndV", study_name, "InverterOnOffInvest"),
        show=True, with_legend=False, file_endings=["png"]
    )


def plot_optihorst_partload():
    study_name = "partload_optihorst"
    cases = {
        "partload": {"color": "red", "label": "$COP(n)$", "marker": "s"},
        "no_partload": {"color": "blue", "label": "$COP_\mathrm{maxes[1]}$", "marker": "^"}
    }
    storage_size_markers = {
    }
    compare_design_for_important_variables(
        study_name=study_name,
        cases=cases,
        storage_size_markers=storage_size_markers
    )


def plot_onoff_inverter(study_name):
    cases = {
        "HydraulicSeperator": {"color": "red", "label": "Inverter"},
        "OnOff": {"color": "blue", "label": "On/Off"}
    }
    storage_size_markers = {
        12: "^",
        35: "s"
    }
    compare_design_for_important_variables(
        study_name=study_name,
        cases=cases,
        storage_size_markers=storage_size_markers
    )


def plot_bars_on_off_mpc(
        variable_to_compare: str = "dynamic_costs_operation_2023",
        without_cases: list = None
):
    biv = "parameterStudy.TBiv"
    v = "parameterStudy.VPerQFlow"
    TBivs = [261.15, 267.15, 273.15]
    part_load_name = {
        True: "No minimum speed, no DHW",
        False: "Minimum speed, DHW"
    }
    case_names = list(get_case_file_names(True).keys())
    if without_cases is not None:
        for case in without_cases:
            case_names.remove(case)
    all_dfs = {
        TBiv: pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                [part_load_name.values(), case_names]
            )
        ) for TBiv in TBivs
    }
    for no_minimal_compressor_speed in [True, False]:
        cases = get_case_file_names(no_minimal_compressor_speed)
        for case in case_names:
            file = cases[case]
            df = pd.read_excel(file, index_col=0)
            for TBiv in TBivs:
                df_sub = df.loc[df.loc[:, biv] == TBiv]
                for idx, row in df_sub.iterrows():
                    all_dfs[TBiv].loc[row[v], (part_load_name[no_minimal_compressor_speed], case)] = row[variable_to_compare]

    plot_config = utils.load_plot_config()

    for TBiv, df in all_dfs.items():
        # Plotting
        fig, axes = plt.subplots(2, 2, sharex=True)

        # Set width of bars and positions of the bars
        n_bars = len(df.index)
        bar_width = 1 / n_bars * 0.9
        indices = np.arange(len(set(df.columns.get_level_values(1))))
        reference_case = "MPCDyn"
        # Plot bars for each column
        for i, load in enumerate(set(df.columns.get_level_values(0))):
            for j, col in enumerate(df.index):
                position = indices + (j - int(n_bars / 2)) * bar_width
                data = df.loc[col, load]
                x_ticks = data.index
                axes[0, i].bar(
                    position, data,
                    width=bar_width,
                    label=f"{col} {plot_config.get_variable(v).unit}"
                )
                for k, case in enumerate(x_ticks):
                    reference = df.loc[col, (load, reference_case)]
                    axes[1, i].bar(
                        position[k], (data[case] - reference) / reference * 100,
                        width=bar_width, color=EBCColors.ebc_palette_sort_2[j]
                    )

            for r in range(2):
                axes[r, i].set_xticks(indices)
                axes[r, i].set_xticklabels(x_ticks, rotation=90)
                axes[0, i].set_title(load)
                axes[1, i].set_xlabel('Regelung')

            axes[0, i].set_ylabel(plot_config.get_label_and_unit(variable_to_compare))
            axes[1, i].set_ylabel("Mehrkosten in %")
            # Add legend
        axes[0, 0].legend(loc='lower left', ncol=1)
        fig.suptitle(
            f"{plot_config.get_label(biv)} = "
            f"{round(TBiv - 273.15, 0)} "
            f"{plot_config.get_variable(biv).unit}"
        )

        # Adjust layout to prevent label cutoff
        fig.tight_layout()

    plt.show()


def plot_price_series_and_gradient_over_mpc(design: str, no_minimal_compressor_speed: bool):
    prices = load_dynamic_electricity_prices(year=2023, init_period=86400 * 2, time_step=900)
    plot_config = utils.load_plot_config()
    # Create figure with two subplots
    fig, axes = plt.subplots(5, 1, sharex=True)

    # Plot original series
    axes[0].plot(prices.index / 86400, prices.values)
    axes[0].set_ylabel('$c_\mathrm{el}$ in €/kWh')

    # Calculate and plot gradient
    gradient = prices.diff() / prices.index.to_series().diff()
    axes[1].plot(gradient.index / 86400, gradient.values)
    axes[1].set_xlabel('Time in days')
    axes[1].set_ylabel('Gradient in 1/s')
    linestyles = ["-", "--"]
    for k, case in enumerate(["MPCCon", "MPCDyn"]):
        mpc_path = get_case_file_names(no_minimal_compressor_speed)[case].parent.joinpath(f"Design_{design}.xlsx")
        df_mpc = pd.read_excel(mpc_path, index_col=0)
        df_mpc = plot_config.scale_df(df_mpc)
        for i, y in enumerate(["TBufSet", "yValSet", "outputs.hydraulic.genCtrl.yHeaPumSet"]):
            axes[2 + i].plot(df_mpc.index / 86400, df_mpc.loc[:, y],
                             label=case, color=EBCColors.ebc_palette_sort_2[k], linestyle=linestyles[k])
            axes[2 + i].set_ylabel(plot_config.get_label_and_unit(y))
    axes[2].legend()
    axes[3].legend()
    axes[4].legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    #plot_price_series_and_gradient_over_mpc(
    #    design="0", no_minimal_compressor_speed=True
    #)
    plot_bars_on_off_mpc(without_cases=["RBCOnOff", "MPCCon"])
    # plot_optihorst_partload()
    # plot_onoff_inverter("inverter_vs_onoff_hydSep")
