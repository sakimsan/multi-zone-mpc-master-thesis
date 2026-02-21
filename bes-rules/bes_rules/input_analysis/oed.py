import itertools
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize

from bes_rules.configs import optimization
from bes_rules.configs.plotting import PlotConfig
from bes_rules.input_analysis import input_analysis
from bes_rules.plotting.utils import load_plot_config, get_figure_size
from bes_rules.rule_extraction import features as feature_module
from bes_rules.simulation_based_optimization import utils
from bes_rules.utils import multiprocessing_ as bes_rules_mp
from bes_rules.utils.pareto import get_pareto_efficient_points
from bes_rules import RESULTS_FOLDER

logger = logging.getLogger(__name__)


class ExperimentalDesignProblem(Problem):
    def __init__(
            self,
            n_experiments,
            n_features,
            valid_feature_combinations,
            optimal_designs: list,
            show_convergence: bool = False,
            already_performed_experiments: np.array = None
    ):
        """
        Parameters:
        -----------
        n_var : int
            Number of design variables
        valid_feature_combinations : np.ndarray
            Array of valid design combinations (n_designs x n_var)
        """
        self.optimal_designs = optimal_designs
        self.n_experiments = n_experiments
        self.n_features = n_features
        self.show_convergence = show_convergence
        if already_performed_experiments is not None:
            self.already_performed = np.hstack(
                [valid_feature_combinations[idx] for idx in already_performed_experiments]
            )
            valid_feature_combinations = get_all_except(valid_feature_combinations, already_performed_experiments)
        else:
            self.already_performed = np.array([])
        super().__init__(
            n_var=n_features * n_experiments,
            n_obj=len(optimal_designs),
            n_constr=n_experiments,  # one constraint per experiment
            xl=-1,  # Lower bound
            xu=1,  # Upper bound
            type_var=np.double
        )
        self.valid_feature_combinations = valid_feature_combinations
        self.min_distance = 1e-4
        self.obj_history = {optimal_design: np.array([]) for optimal_design in self.optimal_designs}
        self.experiment_history = {}
        for n_feat in range(self.n_features):
            for n_exp in range(self.n_experiments):
                self.experiment_history[f"{n_feat}_{n_exp}"] = np.array([])

        if self.show_convergence:
            self.fig, self.axes = plt.subplots(len(optimal_designs), 1, sharex=True)
        else:
            self.fig, self.axes = None, None

    def _evaluate(self, x, out, *args, **kwargs):
        # Number of design points
        n_points = x.shape[0]

        # Initialize objectives and constraints arrays
        f = np.zeros((n_points, self.n_obj))
        g = np.zeros((n_points, self.n_experiments))

        for i in range(n_points):
            # Find nearest valid design point
            experiments = x[i].reshape(self.n_experiments, self.n_features)
            for j in range(self.n_experiments):
                distances = np.linalg.norm(self.valid_feature_combinations - experiments[j], axis=1)
                g[i, j] = np.min(distances) - self.min_distance

            # Include already performed experiments
            experiments = np.hstack([self.already_performed, x[i]]).reshape(-1, self.n_features)
            # Compute cumulative information matrix
            FIM = get_FIM(experiments)

            for idx, optimal_design in enumerate(self.optimal_designs):
                if optimal_design == "D":
                    ret_val = -np.linalg.det(FIM)
                elif optimal_design == "E":
                    ret_val = -np.min(np.linalg.eigvals(FIM))
                elif optimal_design == "A":
                    COV = get_covariance(experiments)
                    if np.isnan(COV).any():
                        logger.error(
                            "NAN found in COV, can't get A-objective. "
                            "Indicates that features are correlated"
                        )
                        ret_val = 1e6  # Penalty for invalid designs
                    else:
                        ret_val = np.trace(COV)
                else:
                    raise NotImplementedError(f"Design {optimal_design} is not implemented")
                f[i, idx] = ret_val
        for i, optimal_design in enumerate(self.optimal_designs):
            self.obj_history[optimal_design] = np.hstack([self.obj_history[optimal_design], f[:, i]])
        for n_feat in range(self.n_features):
            for n_exp in range(self.n_experiments):
                name = f"{n_feat}_{n_exp}"
                self.experiment_history[name] = np.hstack(
                    [
                        self.experiment_history[name],
                        x[:, n_exp * self.n_features + n_feat]]
                )
        if self.show_convergence:
            for ax, optimal_design in zip(self.axes, self.optimal_designs):
                ax.cla()
                f_history = self.obj_history[optimal_design]
                ax.scatter(range(len(f_history)), f_history)
                ax.set_ylabel(optimal_design)
            plt.draw()
            self.fig.canvas.draw_idle()
            plt.pause(1e-3)
        out["F"] = f
        out["G"] = g


class ValidDesignSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        # Randomly select n_samples from valid designs
        X = []
        # First get the corner points on first run:
        if kwargs["algorithm"].pop_size == n_samples and True:
            corner_points = get_corner_points_from_valid_points(problem.valid_feature_combinations)
            if len(corner_points) > n_samples:
                raise ValueError("Your population is smaller than all the corner points, increase population.")
            for combo in itertools.combinations(corner_points, problem.n_experiments):
                X.append(np.hstack(combo))
        # Now random sample the remaining
        for _ in range(n_samples - len(X)):
            indices = np.random.choice(
                len(problem.valid_feature_combinations),
                size=problem.n_experiments,
                replace=False
            )
            X_single = np.hstack([problem.valid_feature_combinations[i] for i in indices])
            for i in range(1, problem.n_experiments):
                delta = X_single[problem.n_features * i:problem.n_features * (i + 1)] - X_single[:problem.n_features]
                if np.all(delta == 0):
                    raise ValueError("Same experiments sampled")
            X.append(X_single)
        X = np.vstack(X)
        return X


class ValidDesignMutation(Mutation):
    def __init__(self, percent_closest: float = 0.05, prob=0.9):
        super().__init__()
        self.prob = prob
        self.percent_closest = percent_closest

    def _do(self, problem, X, **kwargs):
        # For each individual that should mutate
        for i in range(len(X)):
            if np.random.random() < self.prob:
                X[i] = get_closest_valid_experiments(
                    wanted=X[i],
                    problem=problem,
                    percent_closest=self.percent_closest,
                    n_closest_to_select=1
                )[0]
        return X


class ValidDesignCrossover(Crossover):
    def __init__(self, percent_closest: float = 0.05, prob=0.9):
        super().__init__(2, 2)  # two parents, two offspring
        self.prob = prob
        self.percent_closest = percent_closest

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        # The offspring
        Y = np.full((self.n_offsprings, n_matings, n_var), None)
        for i in range(n_matings):
            if np.random.random() < self.prob:
                # Get parents
                p1, p2 = X[0, i], X[1, i]

                # Find valid designs close to midpoint
                mid = (p1 + p2) / 2
                X_selected = get_closest_valid_experiments(
                    wanted=mid,
                    problem=problem,
                    n_closest_to_select=2,  # number of neighbors which are offspring
                    percent_closest=self.percent_closest
                )
                Y[0, i], Y[1, i] = X_selected[0], X_selected[1]
            else:
                Y[0, i], Y[1, i] = X[0, i], X[1, i]
        return Y


def get_closest_valid_experiments(
        wanted: np.ndarray,
        problem: ExperimentalDesignProblem,
        percent_closest: float,
        n_closest_to_select: int
):
    # X Percent of closest to consider, at least 2
    n_closest_to_consider = max(2, int(percent_closest * len(problem.valid_feature_combinations)))

    wanted_experiments = np.reshape(wanted, (problem.n_experiments, problem.n_features))
    selected_nearest_experiments = [[] for _ in range(n_closest_to_select)]
    for i in range(problem.n_experiments):
        distances = np.linalg.norm(problem.valid_feature_combinations - wanted_experiments[i], axis=1)
        nearest_idx = np.argpartition(distances, n_closest_to_consider)[:n_closest_to_consider]  # faster than argsort
        selected = np.random.choice(nearest_idx, size=n_closest_to_select, replace=False)
        for j in range(n_closest_to_select):
            selected_nearest_experiments[j].append(problem.valid_feature_combinations[selected[j]])
    return [np.hstack(nearest_experiments) for nearest_experiments in selected_nearest_experiments]


def get_FIM(u_matrix: np.ndarray):
    if len(u_matrix.shape) == 1:
        u_matrix = np.array([u_matrix])
    return np.dot(np.transpose(u_matrix), u_matrix)


def get_covariance(u_matrix: np.ndarray):
    FIM = get_FIM(u_matrix)
    # Check if matrix is singular:
    if np.linalg.det(FIM) == 0:
        return np.NAN
    COV = np.linalg.inv(FIM)
    return COV


def get_corner_points_from_valid_points(valid_feature_combinations: np.ndarray):
    corner_points = []
    corner_min = special_argmin(valid_feature_combinations)
    corner_max = special_argmax(valid_feature_combinations)
    for col in range(valid_feature_combinations.shape[1]):
        corner_points.append(valid_feature_combinations[corner_min[col], :])
        corner_points.append(valid_feature_combinations[corner_max[col], :])

    # for col in range(valid_feature_combinations.shape[1]):
    #    arg_min = valid_feature_combinations[:, col].argmin()
    #    arg_max = valid_feature_combinations[:, col].argmax()
    #    corner_points.append(valid_feature_combinations[arg_min, :])
    #    corner_points.append(valid_feature_combinations[arg_max, :])
    for c in corner_points:
        print(c)
    return np.vstack(corner_points)


def special_argmin(arr):
    # Get dimensions
    n_rows, n_cols = arr.shape

    # Get regular argmin for each column
    min_indices = np.argmin(arr, axis=0)

    # For each column with multiple minima
    for col in range(n_cols):
        # Find indices where this column has its minimum
        min_val = arr[:, col].min()
        min_mask = arr[:, col] == min_val

        # If there are multiple minima
        if np.sum(min_mask) > 1:
            # Get the indices of multiple minima
            candidate_rows = np.where(min_mask)[0]

            # For each other column
            for other_col in range(n_cols):
                if other_col == col:
                    continue

                # Check which of our candidate rows has max/min in other column
                other_col_vals = arr[candidate_rows, other_col]
                other_col_max = arr[:, other_col].max()
                other_col_min = arr[:, other_col].min()

                # Find candidates that have max/min in other column
                max_mask = other_col_vals == other_col_max
                min_mask = other_col_vals == other_col_min
                valid_candidates = candidate_rows[max_mask | min_mask]

                # If we found valid candidates, update and break
                if len(valid_candidates) > 0:
                    min_indices[col] = valid_candidates[0]
                    break

    return min_indices


def special_argmax(arr):
    # Get dimensions
    n_rows, n_cols = arr.shape

    # Get regular argmin for each column
    max_indices = np.argmax(arr, axis=0)

    # For each column with multiple minima
    for col in range(n_cols):
        # Find indices where this column has its maximum
        max_val = arr[:, col].max()
        max_mask = arr[:, col] == max_val

        # If there are multiple maxima
        if np.sum(max_mask) > 1:
            # Get the indices of multiple maxima
            candidate_rows = np.where(max_mask)[0]

            # For each other column
            for other_col in range(n_cols):
                if other_col == col:
                    continue

                # Check which of our candidate rows has max/min in other column
                other_col_vals = arr[candidate_rows, other_col]
                other_col_max = arr[:, other_col].max()
                other_col_min = arr[:, other_col].min()

                # Find candidates that have max/min in other column
                max_mask = other_col_vals == other_col_max
                min_mask = other_col_vals == other_col_min
                valid_candidates = candidate_rows[max_mask | min_mask]

                # If we found valid candidates, update and break
                if len(valid_candidates) > 0:
                    max_indices[col] = valid_candidates[0]
                    break

    return max_indices


def get_string_differences(str1, str2):
    return [(str1[i - 2:i + 2], str2[i - 2:i + 2]) for i, (c1, c2) in enumerate(zip(str1, str2)) if c1 != c2]


def print_differences_in_duplicate_indexes(duplicate_map):
    for v in duplicate_map.values():
        for k in v:
            diff = get_string_differences(v[0], k)
            if diff:
                print(diff)


def perform_oed_for_all_inputs(
        save_path: Path,
        save_path_all_features: Path,
        features_to_consider: List[str],
        experiments_per_feature: float = 1.0,
        n_repeat: int = 1,
        plot_config: PlotConfig = None,
        just_corner_points: bool = False,
        with_cases_already_simulated: bool = False,
        cases_already_simulated: list = None
):
    df = pd.read_excel(save_path_all_features, index_col=0)
    df = df.loc[:, features_to_consider]
    duplicate_map = {}
    for name, group in df.groupby(df.columns.tolist()):
        if len(group) > 1:
            duplicate_map[group.index[0]] = group.index.tolist()
    df = df.loc[~df.duplicated()]
    optimization_config = load_optimization_config_for_features(df)
    valid_feature_combinations = utils.scale_variables(
        config=optimization_config,
        variables=df.values,
        scale_lb=-1, scale_ub=1
    )
    if with_cases_already_simulated:
        if cases_already_simulated is None:
            with open(save_path_all_features.parents[1].joinpath("cases_already_simulated.json"), "r") as file:
                cases_already_simulated = json.load(file)

        already_performed_experiments = np.array([
            np.where(df.index == case_already_simulated)[0][0]
            for case_already_simulated in cases_already_simulated
        ])
    else:
        already_performed_experiments = None

    plot_scaled_feature_space = False
    if plot_config is not None and plot_scaled_feature_space:
        plot_scaled_feature_design_space(
            valid_feature_combinations=valid_feature_combinations,
            save_path=save_path, save_name="scaled_feature_space",
            plot_config=plot_config, feature_names=features_to_consider
        )
    if just_corner_points:
        n_repeat = 1

    n_experiments = int(len(df.columns) * experiments_per_feature)

    for idx_repeat in range(n_repeat):
        if just_corner_points:
            save_name = "corner_results"
        else:
            save_name = f"OED_results_{idx_repeat}"

        os.makedirs(save_path, exist_ok=True)

        if just_corner_points:
            X_corner = get_corner_points_from_valid_points(valid_feature_combinations)
            experiments_and_relevance, _ = get_experiments_and_relevance_from_X(
                X_corner, valid_feature_combinations, 1, len(features_to_consider)
            )
        else:
            experiments_and_relevance, df_experiment_objectives, df_history = optimize_feature_design(
                n_features=len(df.columns),
                n_experiments=n_experiments,
                valid_feature_combinations=valid_feature_combinations,
                already_performed_experiments=already_performed_experiments
            )
            df_experiment_objectives.to_excel(save_path.joinpath(f"{save_name}_objectives.xlsx"))
            df_history.to_excel(save_path.joinpath(f"{save_name}_convergence.xlsx"))
            if plot_config is not None:
                plot_experiment_optimal_design_scatter(
                    df_history=df_history,
                    features_to_consider=features_to_consider,
                    save_path=save_path
                )

        input_config_names = df.index[list(experiments_and_relevance.keys())]
        df_out = df.loc[input_config_names]
        for name, relevance in zip(input_config_names, experiments_and_relevance.values()):
            df_out.loc[name, "relevance"] = relevance
            df_out.loc[name, "duplicates"] = ";".join(duplicate_map.get(name, []))

        df_out.to_excel(save_path.joinpath(f"{save_name}.xlsx"))
        if plot_config is not None:
            plot_feature_design_space(
                save_path=save_path,
                save_path_all_features=save_path_all_features,
                save_name=save_name,
                x_feature=features_to_consider[0],
                plot_config=plot_config
            )


def plot_experiment_optimal_design_scatter(
        df_history: pd.DataFrame, features_to_consider: list, save_path
):
    optimal_designs = [col for col in df_history.columns if "_" not in col]
    for optimal_design in optimal_designs:
        for n in range(len(features_to_consider)):
            plt.figure()
            plt.title(features_to_consider[n])
            plt.scatter(df_history.loc[:, f"{n}_0"], df_history.loc[:, optimal_design])
            plt.scatter(df_history.loc[:, f"{n}_1"], df_history.loc[:, optimal_design])
            plt.scatter(df_history.loc[:, f"{n}_2"], df_history.loc[:, optimal_design])
            plt.savefig(save_path.joinpath(f"scatter_{optimal_design}_{features_to_consider[n]}"))
            plt.close("all")


def plot_scaled_feature_design_space(
        valid_feature_combinations: np.ndarray,
        save_path: Path,
        feature_names: list,
        save_name: str, plot_config: PlotConfig
):
    x_feature = feature_names[0]
    n_rows = len(feature_names) - 1
    fig, axes = plt.subplots(
        n_rows, 1, sharex=True,
        figsize=get_figure_size(n_columns=1, height_factor=n_rows / 2)
    )
    for idx in range(0, len(feature_names) - 1):
        axes[idx].scatter(
            valid_feature_combinations[:, 0], valid_feature_combinations[:, idx + 1],
            marker="s", color="blue", s=2
        )
        axes[idx].set_ylabel(plot_config.get_label(feature_names[idx + 1]) + "in -")

    axes[-1].set_xlabel(plot_config.get_label(x_feature) + "in -")
    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"{save_name}.png"))
    plt.close("all")


def plot_feature_design_space(
        save_path: Path, save_path_all_features: Path,
        save_name: str, x_feature: str, plot_config: PlotConfig
):
    df = pd.read_excel(save_path_all_features, index_col=0)
    df_oed = pd.read_excel(save_path.joinpath(f"{save_name}.xlsx"), index_col=0)

    df = plot_config.scale_df(df)
    df_oed = plot_config.scale_df(df_oed)

    features = df_oed.columns
    features = features.drop("relevance")
    features = features.drop("duplicates")

    y_features = features.drop(x_feature)
    n_rows = len(y_features)
    fig, axes = plt.subplots(
        n_rows, 1, sharex=True,
        figsize=get_figure_size(n_columns=1, height_factor=n_rows / 2)
    )
    for ax, y_feature in zip(axes, y_features):
        ax.scatter(
            df.loc[:, x_feature], df.loc[:, y_feature],
            marker="s", color="blue", s=2
        )
        ax.scatter(
            df_oed.loc[:, x_feature], df_oed.loc[:, y_feature],
            marker="^", color="red", s=100
        )
        ax.set_ylabel(plot_config.get_label_and_unit(y_feature))

    axes[-1].set_xlabel(plot_config.get_label_and_unit(x_feature))
    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"{save_name}.png"))
    plt.close("all")


def plot_correlation_all_features(
        save_path: Path,
        save_path_all_features: Path,
        x_feature: str, plot_config: PlotConfig
):
    df = pd.read_excel(save_path_all_features, index_col=0)

    df = plot_config.scale_df(df)

    features = df.columns
    y_features = features.drop(x_feature)

    fig, axes = plt.subplots(
        len(y_features), 1, sharex=True,
        figsize=get_figure_size(n_columns=1, height_factor=3)
    )
    for ax, y_feature in zip(axes, y_features):
        ax.scatter(
            df.loc[:, x_feature], df.loc[:, y_feature],
            marker="s", color="blue", s=2
        )
        ax.set_ylabel(plot_config.get_label_and_unit(y_feature))

    axes[-1].set_xlabel(plot_config.get_label_and_unit(x_feature))
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close("all")


def load_optimization_config_for_features(df: pd.DataFrame):
    variables = []
    for col in df.columns:
        variables.append(
            optimization.OptimizationVariable(
                name=col,
                lower_bound=df.loc[:, col].min(),
                upper_bound=df.loc[:, col].max()
            )
        )
    return optimization.OptimizationConfig(
        framework="", method="",  # Irrelevant, just used to scale and descale variables,
        variables=variables
    )


def get_all_except(arr, exclude_idx):
    """
    Get all values from array except specified indexes

    Parameters:
    -----------
    arr : numpy array
        Input array
    exclude_idx : array-like
        Indexes to exclude

    Returns:
    --------
    numpy array with values at all indexes except exclude_idx
    """
    mask = np.ones(len(arr), dtype=bool)
    mask[exclude_idx] = False
    return arr[mask]


def optimize_feature_design(
        n_features: int,
        n_experiments: int,
        valid_feature_combinations: np.ndarray,
        optimal_designs: List[str] = None,
        already_performed_experiments: np.ndarray = None
):
    if optimal_designs is None:
        optimal_designs = ["D", "A", "E"]

    # Initialize problem
    problem = ExperimentalDesignProblem(
        n_features=n_features,
        valid_feature_combinations=valid_feature_combinations,
        n_experiments=n_experiments,
        optimal_designs=optimal_designs,
        already_performed_experiments=already_performed_experiments
    )
    pop_size = 100
    # Initialize algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=int(pop_size * 0.9),
        sampling=ValidDesignSampling(),
        mutation=ValidDesignMutation(),
        crossover=ValidDesignCrossover(),
        eliminate_duplicates=True
    )

    # Optimize
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 200),
        seed=1,
        verbose=True
    )
    df_history = pd.DataFrame({**problem.obj_history, **problem.experiment_history})
    print(f"{res.X=}")
    print(f"{res.F=}")
    if len(res.F.shape) == 1:
        # Single objective
        F = res.F
        X_pareto = np.array([res.X])
    else:
        is_pareto_efficient = get_pareto_efficient_points(objective_values=res.F)
        F = res.F[is_pareto_efficient]
        X_pareto = res.X[is_pareto_efficient]

    experiments_and_relevance, all_idx_to_experiment = get_experiments_and_relevance_from_X(
        X_pareto, valid_feature_combinations, n_experiments, n_features
    )
    experiment_index_F_map = [
        np.hstack([all_idx_to_experiment[-n_experiments:], F[j]])
        for j in range(len(X_pareto))
    ]
    df_experiment_objectives = pd.DataFrame(
        experiment_index_F_map,
        columns=[f"experiment_{i + 1}" for i in range(n_experiments)] + optimal_designs
    )
    return experiments_and_relevance, df_experiment_objectives, df_history


def get_experiments_and_relevance_from_X(X_pareto, valid_feature_combinations, n_experiments, n_features):
    all_idx_to_experiment = []
    for j in range(len(X_pareto)):
        experiments = np.reshape(X_pareto[j], (n_experiments, n_features))
        for i in range(n_experiments):
            idx_valid_combination = np.where(
                np.isclose(valid_feature_combinations, experiments[i]).all(axis=1)
            )[0]
            if len(idx_valid_combination) > 1:
                logger.info(
                    "%s are selected based on experiment %s, selecting the first occurrence",
                    len(idx_valid_combination), experiments[i]
                )
            if len(idx_valid_combination) == 0:
                logger.error("No valid design was selected, something went wrong")
            all_idx_to_experiment.append(idx_valid_combination[0])
    experiments_and_relevance = {item: count for item, count in Counter(all_idx_to_experiment).items()}
    return experiments_and_relevance, all_idx_to_experiment


def analyze_corner_combinations():
    # Grid points
    corners = np.array(list(itertools.product(*[[-1, 1]] * 2)))

    # Test all possible combinations of 3 points from corners
    for combo in itertools.combinations(corners, 3):
        experiments = np.array(combo)
        info_matrix = experiments.T @ experiments
        COV = np.linalg.inv(info_matrix)

        D = np.linalg.det(COV)
        E = np.min(np.linalg.eigvals(COV))
        A = np.trace(COV)

        print(f"\nPoints: {combo}")
        print(f"D-optimal: {D:.4f}")
        print(f"E-optimal: {E:.4f}")
        print(f"A-optimal: {A:.4f}")


def dummy_example(with_edges: bool = True):
    # Create example valid designs (replace with your actual data)
    n_features = 2
    n_valid_feature_combinations = 1000
    valid_feature_combinations = np.random.uniform(-1, 1, (n_valid_feature_combinations, n_features))
    if with_edges:
        edges_and_center = [np.array(i) for i in itertools.product(*[[-1, 0, 1]] * n_features)]
        valid_feature_combinations = np.vstack([valid_feature_combinations] + edges_and_center)
    corner_points = get_corner_points_from_valid_points(valid_feature_combinations)
    already_performed_experiments = np.array([
        np.where(np.isclose(valid_feature_combinations, corner_point).all(axis=1))[0][0]
        for corner_point in corner_points
    ])
    experiments_and_relevance, df_experiment_objectives, df_history = optimize_feature_design(
        n_features=n_features, n_experiments=2,
        valid_feature_combinations=valid_feature_combinations,
        optimal_designs=["A", "D", "E"],
        already_performed_experiments=already_performed_experiments
    )
    print(experiments_and_relevance)
    feature_design = valid_feature_combinations[list(experiments_and_relevance.keys())]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(valid_feature_combinations[:, 0], valid_feature_combinations[:, 1], marker="s", s=5, color="blue")
    ax.scatter(corner_points[:, 0], corner_points[:, 1], marker="^", s=50, color="red")
    ax.scatter(feature_design[:, 0], feature_design[:, 1], marker="s", s=50, color="red")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.show()


def load_single_practical_features(input_config, save_path: Path, with_excel: bool, name: str = "TEASER"):
    pv_directions = []
    df = pd.read_excel(
        save_path.joinpath(name, input_analysis.get_file_name(input_config, "1h", pv_directions) + ".xlsx"),
        index_col=0
    )
    # use mean as they are the same anyway
    practical_features = {
        "q_demand_total": df.loc[:, "QDemPerA"].mean(),
        "dhw_share": df.loc[:, "DHW_share"].mean()
    }
    if with_excel:
        return feature_module.get_practical_features(
            input_config=input_config,
            all_practical_features={input_config.get_name(): practical_features},
            with_custom_features=False
        )
    return practical_features


def create_all_practical_features_json(
        save_path: Path,
        save_path_excel: Path = None,
        name: str = "TEASER"
):
    with_excel = save_path_excel is not None
    input_configs = input_analysis.load_input_configs(save_path=save_path)

    all_practical_features = bes_rules_mp.execute_function_in_parallel(
        func=load_single_practical_features,
        func_kwargs=[
            {"input_config": input_config, "save_path": save_path, "with_excel": with_excel, "name": name}
            for input_config in input_configs],
        use_mp=True,
        notifier=print,
        percentage_when_to_message=10
    )
    features_in_json = [
        "q_demand_total",
        "dhw_share"
    ]
    all_practical_features_json = {
        input_config.get_name(): {
            feature_in_json: practical_features[feature_in_json] for
            feature_in_json in features_in_json
        }
        for input_config, practical_features in zip(input_configs, all_practical_features)
    }
    practical_features_path = RESULTS_FOLDER.joinpath("input_analysis", "practical_features.json")
    if os.path.exists(practical_features_path):
        with open(practical_features_path, "r") as file:
            all_existing_practical_features_json = json.load(file)
    else:
        all_existing_practical_features_json = {}
    all_existing_practical_features_json.update(all_practical_features_json)

    with open(practical_features_path, "w") as file:
        json.dump(all_existing_practical_features_json, file, indent=2)
    if not with_excel:
        return
    all_practical_features = {
        input_config.get_name(): practical_features
        for input_config, practical_features in zip(input_configs, all_practical_features)
    }
    df = pd.DataFrame(all_practical_features).transpose()
    os.makedirs(save_path_excel, exist_ok=True)
    df.to_excel(save_path_excel.joinpath("AllFeatures.xlsx"))


def plot_features_with_different_x(path, plot_config: PlotConfig):
    for X in ["q_demand_total", "TOda_nominal", "GTZ_Ti_HT", "qHeaLoa_flow"]:
        plot_correlation_all_features(
            save_path=path.joinpath(f"all_features_{X}.png"),
            save_path_all_features=path.joinpath("AllFeatures.xlsx"),
            x_feature=X, plot_config=plot_config
        )


def plot_feature_correlation_matrix_scaled_unscaled(
        save_path: Path,
        save_path_all_features: Path,
        plot_config: PlotConfig,
        method: str = "pearson",
        features_to_drop: list = None
):
    # or spearman
    df = pd.read_excel(save_path_all_features, index_col=0)
    if features_to_drop is not None:
        df = df.drop(features_to_drop, axis=1)
    df_unscaled = df.corr(method=method).pow(2)
    df = plot_config.scale_df(df).pow(2)
    df_scaled = df.corr(method=method)

    fig, axes = plt.subplots(
        1, 2, sharex=True,
        figsize=get_figure_size(n_columns=2, height_factor=1)
    )
    labels = [plot_config.get_label(feature) for feature in df_scaled.index]
    for ax, _df, title in zip(axes, [df_unscaled, df_scaled], ["SI", "Scaled"]):
        ax.matshow(_df)
        ax.set_title(title)
        ax.set_yticks(range(len(labels)))
        ax.set_xticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels, rotation=90)
    # fig.colorbar()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close("all")


def plot_feature_correlation_matrix(
        save_path: Path,
        save_path_all_features: Path,
        plot_config: PlotConfig,
        method: str = "pearson",
        features_to_drop: list = None
):
    # or spearman
    df = pd.read_excel(save_path_all_features, index_col=0)
    if features_to_drop is not None:
        df = df.drop(features_to_drop, axis=1)
    df = plot_config.scale_df(df)
    # TODO-Assumption: Use pow as direction is irrelevant due to a in Innovizatiob
    df_scaled = df.corr(method=method).pow(2)

    fig, ax = plt.subplots(
        1, 1, sharex=True,
        figsize=get_figure_size(n_columns=1.2, height_factor=1)
    )
    labels = [plot_config.get_label(feature) for feature in df_scaled.index]
    cax = ax.matshow(df_scaled)
    ax.set_yticks(range(len(labels)))
    ax.set_xticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticklabels(labels, rotation=90)
    cbar = fig.colorbar(cax)
    cbar.set_label("$R^2$ in -")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close("all")


def create_all_feature_plots(path: Path):
    plot_config = load_plot_config(language="de")
    plot_features_with_different_x(path, plot_config)
    features_to_drop = ["dhw_share"]
    for method in ["pearson", "spearman", "kendall"]:
        plot_feature_correlation_matrix(
            save_path=path.joinpath(f"features_correlation_{method}.png"),
            save_path_all_features=path.joinpath("AllFeatures.xlsx"),
            method=method, plot_config=plot_config, features_to_drop=features_to_drop
        )
        plot_feature_correlation_matrix_scaled_unscaled(
            save_path=path.joinpath(f"features_correlation_{method}_scaled_unscaled.png"),
            save_path_all_features=path.joinpath("AllFeatures.xlsx"),
            method=method, plot_config=plot_config, features_to_drop=features_to_drop
        )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    dummy_example()
