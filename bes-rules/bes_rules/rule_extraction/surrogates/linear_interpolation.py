import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from bes_rules.plotting.utils import PlotConfig
from bes_rules.rule_extraction.surrogates import Surrogate
from bes_rules.rule_extraction.surrogates.plotting import plot_surrogate_quality

logger = logging.getLogger(__name__)


class LinearInterpolationSurrogate(Surrogate):

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df

    def predict(
            self,
            design_variables: Dict[str, np.ndarray],
            metrics: List[str],
            save_path_plot: Path,
            plot_config: PlotConfig
    ) -> pd.DataFrame:
        df_interpolated = self.df.copy()

        variables_to_interpolate = []
        mask_constant_variables = np.array([True] * len(self.df))
        for design_variable, values in design_variables.items():
            if len(set(values)) > 1:
                variables_to_interpolate.append(design_variable)
            else:
                df_interpolated = df_interpolated.loc[df_interpolated.loc[:, design_variable] == values[0]]
                mask_constant_variables = (
                                                  mask_constant_variables &
                                                  (self.df.loc[:, design_variable] == values[0]).values
                )
        if len(variables_to_interpolate) > 1:
            raise ValueError("Currently, only linear interpolation of one axis is allowed")
        variable_to_interpolate = variables_to_interpolate[0]
        df_interpolated = df_interpolated.loc[:, metrics + [variable_to_interpolate]]

        values = design_variables[variable_to_interpolate]
        df_interpolated = df_interpolated.set_index(variable_to_interpolate)
        # real_values = df_interpolated.index.copy()
        try:
            df_interpolated = df_interpolated.reindex(
                list(set(list(values) + list(df_interpolated.index)))).sort_index()
        except ValueError as err:
            raise err
        df_interpolated = df_interpolated.interpolate(limit_area='inside').loc[values]
        mask_no_extrapolation = ~df_interpolated.isnull().any(axis=1)
        if save_path_plot is not None:
            plot_surrogate_quality(
                df_simulation=self.df.loc[mask_constant_variables],
                df_interpolated=df_interpolated,
                save_path=save_path_plot,
                plot_config=plot_config
            )
        return df_interpolated.loc[:, metrics]
