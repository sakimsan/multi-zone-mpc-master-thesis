import itertools
from typing import List

import pandas as pd


def get_optima_subset(
        df: pd.DataFrame,
        optimization_variables: List[str],
        metrics: List[str],
        variables_to_extract: list = None
):
    """
    Extract the optimal values for each combination of optimization_variables.

    :param pd.DataFrame df:
        DataFrame with result data
    :param List[str] optimization_variables:
        Optimization variables to extract optima for
    :param List[str] metrics:
        Metric to extract optimum for
    :param variables_to_extract:
        Only include this set of variables in the optimum returned.
        If None, all DataFrame columns are returned.

    :return: optima
        Dictionary with optima for each of the required metrics.
    """
    if variables_to_extract is None:
        variables_to_extract = df.columns
    optima = {metric: [] for metric in metrics}
    for unique_variable_values in itertools.product(*[df.loc[:, var].unique() for var in optimization_variables]):
        filtered_df = df.copy()
        for variable_name, variable_value in zip(optimization_variables, unique_variable_values):
            filtered_df = filtered_df.loc[filtered_df.loc[:, variable_name] == variable_value]
        for metric in metrics:
            # Directly using argmax only returns one index, we want all argmax in case of multiple optima.
            mask_argmin = filtered_df.loc[:, metric] == filtered_df.loc[:, metric].min()
            optimum = filtered_df.loc[mask_argmin, variables_to_extract]
            if len(optimum) > 0:
                optima[metric].append(optimum)
    return optima
