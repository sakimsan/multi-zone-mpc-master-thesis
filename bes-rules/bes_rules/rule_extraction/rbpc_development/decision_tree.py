import json
import os
import pathlib
import pickle
import sys

from sklearn import tree
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import train_test_split

import graphviz
import numpy as np
import matplotlib.pyplot as plt

from bes_rules.rule_extraction.rbpc_development.clustering import load_results_from_pickle
from bes_rules.rule_extraction.rbpc_development import utils
from bes_rules.plotting.utils import get_figure_size


def create_decision_tree(
        save_path: pathlib.Path,
        n_days: int,
        max_tree_depth: int,
        show_plot=False
):
    clustering_results = load_results_from_pickle(save_path)
    df = utils.load_results(save_path)
    cluster_temperature_difference_sum = utils.calculate_sum_of_overheating(clustering_results[n_days])
    cluster_names = [f"dT={dT}" for dT in np.unique(cluster_temperature_difference_sum)]
    print(f"{cluster_names=}")

    df_days = utils.split_df_into_days(df)
    # Import Feature Data
    feature_data = []
    feature_data_per_day = {}
    for idx, df_day in enumerate(df_days):
        feature_data_per_day = utils.get_features(df=df, idx_day=idx, n_days=len(df_days))
        feature_data.append(list(feature_data_per_day.values()))
    feature_data = np.array(feature_data)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, cluster_temperature_difference_sum,
        random_state=0
    )

    scores = {}
    trees = {}
    for tree_depth in range(2, max_tree_depth):

        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            max_depth=tree_depth,
            max_features=None, splitter="best",
            random_state=0
        )

        clf = clf.fit(X_train, y_train)

        _score_test_f1 = f1_score(y_test, clf.predict(X_test), average="weighted")
        _score_train_f1 = f1_score(y_train, clf.predict(X_train), average="weighted")
        _score_test_r2 = r2_score(y_test, clf.predict(X_test))
        _score_train_r2 = r2_score(y_train, clf.predict(X_train))
        _score_test_rmse = np.sqrt(np.mean(np.square(y_test - clf.predict(X_test))))
        _score_train_rmse = np.sqrt(np.mean(np.square(y_train - clf.predict(X_train))))
        dot_data = tree.export_graphviz(
            clf,
            out_file=None,
            feature_names=list(feature_data_per_day.keys()),
            class_names=cluster_names,
            filled=True, rounded=True,
            special_characters=True,
            impurity=True,
            rotate=False,
            proportion=False,
        )

        # Make Graphviz executable to prevent error
        os.environ["PATH"] += os.pathsep + r'D:\fwu\05_1graphivz\Graphviz-10.0.1-win64\bin'

        graph = graphviz.Source(dot_data, engine='dot')
        tree.plot_tree(clf)
        save_path_plot = save_path.joinpath("DT_plots")
        os.makedirs(save_path_plot, exist_ok=True)
        graph.render(save_path_plot.joinpath(f"DT_day_{n_days}_{tree_depth}"))
        plt.savefig(save_path_plot.joinpath(f"DT_day_{n_days}_{tree_depth}.png"))
        if show_plot:
            plt.show()

        scores[tree_depth] = {
            "R2": dict(zip(["train", "test"], [_score_train_f1, _score_test_f1])),
            "F1": dict(zip(["train", "test"], [_score_train_r2, _score_test_r2])),
            "RMSE": dict(zip(["train", "test"], [_score_train_rmse, _score_test_rmse]))
        }
        trees[tree_depth] = clf

    return scores, trees


def create_decision_tree_for_multiple_settings(save_path, n_days_total: int = 7):
    scores = {}
    trees = {}
    for n_days in range(2, n_days_total + 1):
        scores[n_days], trees[n_days] = create_decision_tree(save_path=save_path, n_days=n_days, max_tree_depth=10)
    with open(save_path.joinpath("DT_scores.json"), "w+") as file:
        json.dump(scores, file)
    with open(save_path.joinpath("decision_tress.pickle"), "wb+") as file:
        pickle.dump(trees, file)
    plot_train_test_scores(save_path=save_path)


def plot_train_test_scores(save_path: pathlib.Path):
    with open(save_path.joinpath("DT_scores.json"), "r") as file:
        scores = json.load(file)
    metrics = ["R2"]
    fig, ax = plt.subplots(nrows=len(scores), ncols=len(metrics), figsize=get_figure_size(n_columns=1, height_factor=1.5), sharex=True, squeeze=False)
    from bes_rules.plotting import EBCColors
    day_idx = 0
    for day, scores_td in scores.items():
        tree_depths = list(scores_td.keys())
        for score_idx, score in enumerate(metrics):
            data_train = [scores_tt[score]["train"] for scores_tt in scores_td.values()]
            data_test = [scores_tt[score]["test"] for scores_tt in scores_td.values()]
            ax[day_idx, score_idx].plot(tree_depths, data_train, color=EBCColors.red, marker="^", linestyle="-", label="Train")
            ax[day_idx, score_idx].plot(tree_depths, data_test, color=EBCColors.blue, marker="^", linestyle="-", label="Test")
            ax[day_idx, score_idx].set_title(f"{day} days")
            ax[day_idx, score_idx].set_xticks(tree_depths)
            ax[day_idx, score_idx].set_ylabel(f"${score}$ in -")
            ax[day_idx, score_idx].set_ylim(0, 1.1)
        day_idx += 1
    ax[-1, -1].legend(loc="lower right", ncol=2)
    for i in range(len(metrics)):
        ax[-1, i].set_xlabel("Tree Depth")
    fig.tight_layout()
    fig.savefig(save_path.joinpath("DT_plots", "train_test_scores.png"))
