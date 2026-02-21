import pandas as pd
from ebcpy import Optimizer
from typing import Type, List
import numpy as np
from bes_rules.rule_extraction.regression.regressors import Regressor

deviation_metrics = {
    "mean": np.mean,
    "max": np.max,
    "RMSE": lambda x: np.mean(np.array(x) ** 2) ** 0.5
}


class ObjectiveMetricRegression(Optimizer):

    def __init__(self, cd=None, **kwargs):
        """Instantiate class parameters"""
        self.feature_values = kwargs.pop("feature_values")
        self.objectives = kwargs.pop("objectives")
        self.design_values = kwargs.pop("design_values")
        self.metric = kwargs.pop("metric")
        self.regressor: Type[Regressor] = kwargs.get("regressor")
        assert self.metric in deviation_metrics
        super().__init__(cd=cd, **kwargs)

    def obj(self, xk, *args):
        deviations, _ = get_deviation_from_optimum(
            regressor=self.regressor,
            feature_values=self.feature_values,
            objectives=self.objectives,
            design_values=self.design_values,
            parameters=xk
        )
        return deviations[self.metric]

    def mp_obj(self, x, *args):
        raise NotImplementedError


def minimize_deviation_from_optimum(
        regressor: Type[Regressor],
        feature_values: np.ndarray,
        design_values: np.ndarray,
        objectives: np.ndarray,
        metric: str
):
    warm_start_parameters = regressor.get_parameters(x=feature_values, y=design_values)
    bounds = []
    for r in warm_start_parameters:
        if r < 0:
            bounds.append((r * 1.9, r * 0.1))
        elif r == 0:
            bounds.append((-1, 1))
        else:
            bounds.append((r * 0.1, r * 1.9))

    optimizer = ObjectiveMetricRegression(
        feature_values=feature_values,
        objectives=objectives,
        design_values=design_values,
        bounds=bounds,
        metric=metric,
        regressor=regressor
    )
    kwargs_diff_evo = dict(framework="scipy_differential_evolution", method="best1bin", popsize=50)
    result = optimizer.optimize(**kwargs_diff_evo)
    regressor.get_equation_string(
        x=plot_config.get_label(objective_variable_name),
        parameters=result.x
    )


def get_deviation_from_optimum(
        optimal_design_regressions: np.ndarray,
        objective_values: List[pd.Series]
):
    objective_deviations = []
    # TODO: Add constraint
    for optimal_design_regression_idx, objective_values_idx in zip(
            optimal_design_regressions, objective_values
    ):
        objective_values_idx = objective_values_idx[~pd.isna(objective_values_idx)]
        minimum = np.min(objective_values_idx)
        design_values_idx = objective_values_idx.index
        if (
                optimal_design_regression_idx < design_values_idx.min()
                or optimal_design_regression_idx > design_values_idx.max()
        ):
            # TODO: How to deal with extrapolation?
            objective_regression = objective_values_idx.max()
        else:
            objective_regression = np.interp(
                optimal_design_regression_idx,
                design_values_idx,
                objective_values_idx.values
            )
        objective_percent_deviation = (objective_regression - minimum) / minimum * 100
        objective_deviations.append(objective_percent_deviation)
    objective_deviations = np.array(objective_deviations)
    return {
        metric: deviation_metric(objective_deviations)
        for metric, deviation_metric in deviation_metrics.items()
    }, objective_deviations
