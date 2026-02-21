import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bes_rules.plotting.utils import PlotConfig
from bes_rules.rule_extraction.surrogates import Surrogate
from bes_rules.rule_extraction.surrogates.plotting import plot_surrogate_quality

logger = logging.getLogger(__name__)


class BayesSurrogate(Surrogate):

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(df=df, **kwargs)
        self.df = df
        self.metric_hyperparameters = kwargs["metric_hyperparameters"]
        self.metric_gp = {}
        self.optimization_variables_order = None
        for metric, hyperparameters in self.metric_hyperparameters.items():
            optimization_variables = list(hyperparameters.keys())
            kernel = Matern(
                length_scale=[hyperparameters[optimization_variable]
                              for optimization_variable in optimization_variables],
                nu=2.5)
            if self.optimization_variables_order is None:
                self.optimization_variables_order = optimization_variables
            else:
                if self.optimization_variables_order != optimization_variables:
                    raise ValueError("Order of optimization_variables does not match across hyperparameters")

            # Initialize Gaussian Process with the selected kernel
            self.metric_gp[metric] = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        self.fit(df=self.df)

    def fit(self, df: pd.DataFrame):
        for metric, gp in self.metric_gp.items():
            design_variables_order = self.metric_hyperparameters[metric]
            X_predict = []
            for design_variable in design_variables_order:
                X_predict.append(df.loc[:, design_variable].values)
            X_predict = np.array(X_predict).T
            y_real = df.loc[:, metric]
            #idx = [0, 200, 400]
            #gp.fit(X_predict[idx], y_real[idx])
            gp.fit(X_predict, y_real)

    def predict(
            self,
            design_variables: Dict[str, np.ndarray],
            metrics: List[str],
            save_path_plot: Path = None,
            plot_config: PlotConfig = None
    ) -> pd.DataFrame:
        # Order is the same for all metrics, see __init__
        X_predict = []
        for design_variable in self.optimization_variables_order:
            if design_variable not in design_variables:
                raise KeyError(f"{design_variable} not present in design_variables to predict")
            X_predict.append(design_variables[design_variable])
        multi_index = pd.MultiIndex.from_arrays(X_predict, names=self.optimization_variables_order)
        X_predict = np.array(X_predict).T
        df_predict = pd.DataFrame(index=multi_index)
        df_std = df_predict.copy()
        for metric in metrics:
            if metric not in self.metric_hyperparameters:
                raise KeyError("Given metric not in hyperparameters, can't predict")
            y_predict, y_std = self.metric_gp[metric].predict(X_predict, return_std=True)
            df_predict[metric] = y_predict
            df_std[metric] = y_std * 3  # 3 sigma, 99.7 %
        if save_path_plot is not None:
            plot_surrogate_quality(
                df_simulation=self.df,
                df_interpolated=df_predict,
                save_path=save_path_plot,
                plot_config=plot_config,
                df_std=df_std,
                plot_surface=False
            )
        return df_predict


if __name__ == '__main__':
    from bes_rules import RESULTS_FOLDER
    from bes_rules.rule_extraction.innovization import mesh_arrays
    with open(RESULTS_FOLDER.joinpath("BayesHyperparameters", "best_hyperparameters.json"), "r") as file:
        PARAS = json.load(file)
    df_path = RESULTS_FOLDER.joinpath(
            "RE_Journal", "BESCtrl", "DesignOptimizationResults",
            "TRY2015_536322100078_Jahr_B1994_retrofit_SingleDwelling_M_South",
            "DesignOptimizerResults.xlsx")
    SURROGATE = BayesSurrogate(
        df=pd.read_excel(df_path, index_col=0),
        metric_hyperparameters=PARAS
    )
    design_variables = {
        "parameterStudy.TBiv": np.linspace(-9, 4, 100) + 273.15,
        "parameterStudy.VPerQFlow": np.linspace(5, 5, 1),
        # "parameterStudy.ShareOfPEleNominal": np.ones(100)
    }
    design_values = mesh_arrays(list(design_variables.values()))
    flat_design_variables = {var: design_values[:, idx] for idx, var in enumerate(design_variables)}

    print(SURROGATE.predict(
        metrics=["SCOP_Sys"],
        design_variables=flat_design_variables,
        save_path_plot=RESULTS_FOLDER.joinpath("BayesHyperparameters", "quality.png")
    ))
