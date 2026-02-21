from bes_rules.configs import PlotConfig
from ebcpy import TimeSeriesData
from bes_rules.plotting import utils
from pathlib import Path


def plot_kpis_with_and_without_setback(save_path: Path):
    plot_config = PlotConfig.load_default()
    x_variable = "time"
    y_variables = [
        "outputs.electrical.dis.PEleLoa.value",
        "hydraulic.control.buiAndDHWCtr.TSetBuiSup.TSet",
        "outputs.electrical.dis.PEleLoa.integral",
        #"outputs.building.dTControlHea[1]"
        #"outputs.building.TZone[1]"
    ]
    cases = {
        "No Setback": {"file_name": "ExtremeTestNightSetBackProblems_no_setback.mat"},
        "Setback": {"file_name": "ExtremeTestNightSetBackProblems_setback_6K.mat"}
    }
    case_dfs = {}
    for case, data in cases.items():
        df = TimeSeriesData(save_path.joinpath(data["file_name"]), variable_names=y_variables).to_df()
        df.loc[:, "time"] = df.index
        df = plot_config.scale_df(df)
        case_dfs[case] = df

    fig, axes = utils.create_plots(
        plot_config=plot_config,
        y_variables=y_variables,
        x_variables=[x_variable]
    )
    axes = axes[:, 0]
    for ax, y_variable in zip(axes, y_variables):
        for case, df in case_dfs.items():
            ax.plot(df.loc[:, "time"], df.loc[:, y_variable].sort_values(ascending=False), label=case)

    utils.save(fig=fig, axes=axes, save_path=save_path.joinpath("setback_influence"), with_legend=True)


if __name__ == '__main__':
    plot_kpis_with_and_without_setback(save_path=Path(r"D:\07_offline_arbeiten\motivate_setback"))
