from pathlib import Path

import numpy as np
import pandas as pd
from ebcpy import DymolaAPI

from bes_rules.configs import InputConfig
from bes_rules.simulation_based_optimization import supervisory_control
import logging

from bes_rules.simulation_based_optimization.milp.external_control import run_external_control

logger = logging.getLogger(__name__)


class MILPBasedOptimizer(supervisory_control.BaseSupervisoryControl):

    def __init__(self, config, input_config: InputConfig, sim_api: DymolaAPI = None, **kwargs):
        super().__init__(config=config, input_config=input_config, sim_api=sim_api, **kwargs)
        self._predictive_control_function = kwargs["predictive_control_function"]
        self._predictive_control_options = kwargs["predictive_control_options"]
        self.extra_model_parameters = kwargs.get("extra_model_parameters", {})
        self.variables = kwargs["variables"]
        self.external_control_function = run_external_control

    def get_function_inputs_for_parameters(
            self,
            design_parameters: dict,
            bes_parameters: dict,
            fmu_path: Path,
            predictions: pd.DataFrame,
            save_path: Path,
            state_result_names: list,
            output_result_names: list
    ):
        predictions.loc[:, "P_PV"] = predictions.loc[:, "P_el_pv_raw"] * design_parameters.get("parameterStudy.f_design", 1)
        model_parameters = {**bes_parameters, **self.extra_model_parameters}

        return dict(
            parameter=design_parameters,
            time_series_inputs=predictions,
            fmu_path=fmu_path,
            save_path=save_path,
            output_result_names=output_result_names,
            state_result_names=state_result_names,
            start_time=self.config.simulation.sim_setup.get("start_time", 0),
            stop_time=self.config.simulation.sim_setup["stop_time"] + self.config.simulation.init_period,
            output_interval=self.config.simulation.sim_setup["output_interval"],
            external_control=self._predictive_control_function,
            model_parameters=model_parameters,
            variables=self.variables,
            **self._predictive_control_options,
        )

