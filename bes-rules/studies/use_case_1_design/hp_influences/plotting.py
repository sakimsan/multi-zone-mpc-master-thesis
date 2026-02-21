import pandas as pd

from bes_rules.plotting import utils

from bes_rules import configs, RESULTS_FOLDER


def compare_plots(
        x_variable: str,
        y_variables: list,
        studies: dict,
        save_name: str
):
    plot_config = utils.load_plot_config()
    fig, axes = utils.create_plots(
        plot_config=plot_config,
        x_variables=[x_variable],
        y_variables=y_variables,
    )
    for study, data in studies.items():
        study_path = RESULTS_FOLDER.joinpath("UseCase_TBivAndV", study, "DesignOptimizerResults.xlsx")
        df = pd.read_excel(study_path, index_col=0)
        df = plot_config.scale_df(df)
        for _y_variable, _ax in zip(y_variables, axes[:, 0]):
            _ax.scatter(
                df.loc[:, x_variable], df.loc[:, _y_variable],
                color=data["color"],
                label=study
            )

    axes[0, 0].legend(bbox_to_anchor=(0, 1), loc="lower left")
    utils.save(
        fig=fig, axes=axes,
        save_path=RESULTS_FOLDER.joinpath("UseCase_TBivAndV", save_name),
        show=False, with_legend=False, file_endings=["png"]
    )


def compare_study_with_modifiers(
        x_variable: str,
        y_variables: list,
        studies: dict,
        save_name: str
):
    plot_config = utils.load_plot_config()
    len_inputs = None
    for study in studies:
        study_path = RESULTS_FOLDER.joinpath("UseCase_TBivAndV", study)

        study_config = configs.StudyConfig.from_json(study_path.joinpath("study_config.json"))
        dfs, input_configs = utils.get_all_results_from_config(study_config=study_config)
        studies[study]["dfs"] = dfs
        studies[study]["input_configs"] = input_configs
        len_inputs = len(input_configs)
    for idx in range(len_inputs):
        fig, axes = utils.create_plots(
            plot_config=plot_config,
            x_variables=[x_variable],
            y_variables=y_variables,
        )
        for study, data in studies.items():
            df = plot_config.scale_df(data["dfs"][idx])
            for _y_variable, _ax in zip(y_variables, axes[:, 0]):
                _ax.scatter(
                    df.loc[:, x_variable], df.loc[:, _y_variable],
                    color=data["color"],
                    label=study
                )

        axes[0, 0].legend(bbox_to_anchor=(0, 1), loc="lower left")
        utils.save(
            fig=fig, axes=axes,
            save_path=RESULTS_FOLDER.joinpath(
                "UseCase_TBivAndV",
                f"{save_name}_{input_configs[idx].get_name()}"
            ),
            show=False, with_legend=False, file_endings=["png"]
        )
