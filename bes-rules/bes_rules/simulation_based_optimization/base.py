import logging
import os
import pathlib
from abc import abstractmethod, ABC
from collections import namedtuple
from itertools import product
from warnings import simplefilter

import openpyxl
import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from bayes_opt import acquisition
from bes_rules.simulation_based_optimization.utils.custom_bayes import CustomBayesianOptimization
from ebcpy import Optimizer, DymolaAPI
from bes_rules.simulation_based_optimization.utils import constraints

from bes_rules.configs import InputConfig, OptimizationConfig
from bes_rules.simulation_based_optimization.utils import (
    apply_constraints,
    descale_variables,
    get_simulation_input_from_variables
)

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class BaseSurrogateBuilder(Optimizer, ABC):

    def __init__(self,
                 working_directory: pathlib.Path,
                 optimization_config: OptimizationConfig,
                 test_only: bool = False
                 ):
        super().__init__(cd=working_directory, bounds=[(var.lower_bound, var.upper_bound) for var in optimization_config.variables])
        self.optimization_config = optimization_config
        self.test_only = test_only

    def _choose_framework(self, framework):
        """
        Adds design of experiments to ebcpy real
        optimization options.

        :param str framework:
            String for selection of the relevant function. Supported options are:
            - scipy_minimize
            - dlib_minimize
            - scipy_differential_evolution
            - pymoo
            - DOE
        """
        if framework.lower() == "doe":
            return self._doe, True
        if framework.lower() == "bayes":
            return self._bayes_optimization, True
        return super()._choose_framework(framework=framework)

    def _bayes_optimization(self, method, n_cpu=1, **kwargs):
        old_choices = kwargs.get("old_choices", False)
        n_iter = kwargs["n_iter"]
        hyperparameters = kwargs["hyperparameters"][self.optimization_config.objective_names[0]]
        assert len(self.optimization_config.objective_names) == 1, "Only SO is supported"

        # Try to handle constraints already here
        if self.optimization_config.constraints:
            new_optimization_config = self.optimization_config.model_copy()
            for constraint in self.optimization_config.constraints:
                if isinstance(constraint, constraints.BivalenceTemperatureGreaterNominalOutdoorAirTemperature):
                    for variable in new_optimization_config.variables:
                        if variable.name == "parameterStudy.TBiv":
                            variable.lower_bound = max(self.input_config.weather.TOda_nominal, variable.lower_bound)
                elif isinstance(constraint, constraints.HydraulicSeperatorConstraint):
                    new_variables = []
                    for variable in new_optimization_config.variables:
                        if variable.name != "parameterStudy.VPerQFlow":
                            new_variables.append(variable)
                    new_optimization_config.variables = new_variables
            self.optimization_config = new_optimization_config

        # Get hyperparameters
        optimization_variables = {}
        length_scales = []
        for variable in self.optimization_config.variables:
            optimization_variables[variable.name.replace(".", "_")] = (0, 1)
            length_scales.append(hyperparameters[variable.name])

        # Setup bayes
        if old_choices:
            from bayes_opt import UtilityFunction
            acquisition_function = UtilityFunction(kind="ei", xi=1e-1)
        else:
            acquisition_function = acquisition.ProbabilityOfImprovement(
                xi=0.1,
                exploration_decay=0
            )
        optimizer = CustomBayesianOptimization(
            f=self._target_function_adaptor_to_besrules,
            pbounds=optimization_variables,
            allow_duplicate_points=kwargs.get("allow_duplicate_points", True),
            acquisition_function=acquisition_function
        )
        if old_choices:
            optimizer._gp = GaussianProcessRegressor(
                C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=optimizer._random_state,
            )
        else:
            kernel = Matern(
                length_scale=length_scales,
                nu=2.5
            )
            optimizer._gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)

        optimizer.maximize(init_points="central_points", n_iter=n_iter)
        return None  # Result is not important, saved in self.obj anyways

    def _target_function_adaptor_to_besrules(self, **kwargs):
        return self.obj(xk=list(kwargs.values()))

    def _create_ffd(self):
        vars = self.optimization_config.variables
        samples = []  # (num_samples, num_variables)
        _product = []
        for var in vars:
            if var.discrete_values:
                _product.append(
                    np.array(
                        [(value - var.lower_bound) / (var.upper_bound - var.lower_bound)
                         for value in var.discrete_values]
                    )
                )
            elif var.discrete_steps > 0:
                _product.append(np.arange(0, 1, var.discrete_steps / (var.upper_bound - var.lower_bound)))
            else:
                _product.append(np.linspace(0, 1, var.levels))
        if self.test_only:
            _product = [(0, 1) for _ in vars]

        for vars_values in product(*_product):
            samples.append(list(vars_values))
        return np.array(samples)

    def _doe(self, method, n_cpu=1, **kwargs):
        """
        Perform Design of Experiments on the given parameter space

        :param method:
        :param n_cpu:
            Number of cpu's to use.
        :param kwargs:
            Further settings required for DOE lib.
        """
        if method == "ffd":
            samples = self._create_ffd()
        else:
            raise ValueError("Given method is not supported!")
        try:
            f_res = self.mp_obj(x=samples)
            if not self.optimization_config.objective_names:
                return  # No result for ffd without explicit objective
            x_min = np.array(samples)[np.argmin(f_res), :]
            f_res_min = np.min(f_res)
            res_tuple = namedtuple("res_tuple", "x fun")
            res = res_tuple(x=x_min, fun=f_res_min)
            return res
        except (KeyboardInterrupt, Exception) as error:
            # pylint: disable=inconsistent-return-statements
            self._handle_error(error)

    def run(self):
        if all([
            self.optimization_config.framework != "doe",
            not self.optimization_config.objective_names
        ]):
            raise ValueError("Must set at least one objective_name for frameworks other than doe!")
        self.optimize(
            framework=self.optimization_config.framework,
            method=self.optimization_config.method,
            **self.optimization_config.solver_settings
        )


class SurrogateBuilder(BaseSurrogateBuilder):
    """
    Class to perform a design optimization
    using dynamic simulations as model.
    """

    def __init__(
            self,
            config: "StudyConfig",
            input_config: InputConfig,
            sim_api: DymolaAPI = None,
            use_mp: bool = True,
            **kwargs
    ):
        from bes_rules.configs import StudyConfig  # Add type hint here to avoid circular import
        self.config: StudyConfig = config
        self.input_config = input_config
        self._log_path = self.create_and_get_log_path(
            base_path=pathlib.Path(self.config.study_path),
            study_name=input_config.get_name()
        )
        self.use_mp = use_mp
        super().__init__(working_directory=self._log_path.parent,
                         optimization_config=self.config.optimization,
                         test_only=self.config.test_only)
        self.sim_api = sim_api

    @staticmethod
    @abstractmethod
    def start_simulation_api(config, **kwargs) -> DymolaAPI:
        raise NotImplementedError

    @staticmethod
    def create_and_get_log_path(base_path: pathlib.Path, study_name: str, create: bool = True):
        path = base_path.joinpath("DesignOptimizationResults", study_name, "DesignOptimizerResults.xlsx")
        if create:
            os.makedirs(path.parent, exist_ok=True)
        return path

    def get_result_names(self):
        return list(set(
            self.config.get_additional_result_names() +
            self.config.simulation.get_result_names() +
            self.sim_api.result_names
        ))

    def get_log_path(self):
        return self._log_path

    def obj(self, xk, *args):
        """Directly use mp_obj function."""
        return self.mp_obj(x=[xk], *args)[0]

    def mp_obj(self, x, *args):
        x = np.array(x)  # If a framework uses lists instead of arrays

        if os.path.exists(self._log_path):
            _log_df = self.load_design_optimization_log(file_path=self._log_path)
        else:
            _log_df = pd.DataFrame(
                columns=["iterate"]
            ).set_index("iterate")

        # Descale from (0, 1) to normal bounds:
        x_descaled = descale_variables(config=self.optimization_config, variables=x)
        x_descaled = apply_constraints(
            config=self.optimization_config,
            variables=x_descaled,
            input_config=self.input_config
        )
        if x_descaled.size == 0:
            logger.error("No variables left to optimize after constraint!")
            return []
        if self.config.test_only and len(x_descaled) > 4:
            x_descaled = x_descaled[:2]
        parameters = []
        for idx, _x in enumerate(x_descaled):
            single_parameters = get_simulation_input_from_variables(
                values=_x,
                variables=self.optimization_config.variables
            )
            _log_df.loc[self._counter + idx, single_parameters.keys()] = single_parameters.values()
            parameters.append(single_parameters)

        results, _log_df = self.simulate(parameters=parameters, log_df=_log_df)

        for idx, result_kpis in enumerate(results):
            if result_kpis is None:
                _log_df.loc[self._counter + idx] = np.NAN
            else:
                _log_df.loc[self._counter + idx, result_kpis.keys()] = result_kpis.values()

        # Add objectives
        for obj in self.config.objectives:
            _log_df = obj.calc(df=_log_df, input_config=self.input_config)

        # Save log-results as excel:
        self.save_design_optimization_log(file_path=self._log_path, df=_log_df)
        # Duplicate with different name one folder up to allow multiple opens in Excel
        self.save_design_optimization_log(
            file_path=self.config.study_path.joinpath(self.input_config.get_name() + ".xlsx"),
            df=_log_df
        )

        if not self.optimization_config.objective_names:
            self._counter += len(results)
            return []

        objective_names = self.optimization_config.objective_names
        # Calc objective(s):
        weightings = self.optimization_config.weightings
        objective_values = []

        for idx in range(len(results)):
            values = []
            for name in objective_names:
                values.append(_log_df.loc[self._counter + idx, name])
            if self.optimization_config.multi_objective:
                objective_values.append(values)
            else:
                objective_value = 0
                for value, weighting in zip(values, weightings):
                    objective_value += value * weighting
                objective_values.append(objective_value)

        self._counter += len(results)
        return objective_values

    @abstractmethod
    def simulate(self, parameters, log_df):
        raise NotImplementedError

    @staticmethod
    def load_design_optimization_log(file_path: pathlib.Path) -> pd.DataFrame:
        df = pd.read_excel(
            file_path, sheet_name="DesignOptimization",
            index_col=[0],
            header=[0, 1]
        )
        if "OptimizationVariables" in df.columns.get_level_values(0):
            return df
        # Else
        return pd.read_excel(
            file_path, sheet_name="DesignOptimization",
            index_col=[0]
        )

    @staticmethod
    def save_design_optimization_log(file_path: pathlib.Path, df: pd.DataFrame):
        df = df.reset_index()
        df.to_excel(file_path, sheet_name="DesignOptimization", index=False)
        book = openpyxl.load_workbook(file_path)
        sheet = book["DesignOptimization"]
        if "OptimizationVariables" in df.columns.get_level_values(0):
            sheet.delete_rows(3, 1)
        book.save(file_path)
