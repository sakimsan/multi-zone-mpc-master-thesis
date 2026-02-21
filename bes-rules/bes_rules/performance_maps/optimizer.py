from pathlib import Path

import pandas as pd
import numpy as np

from bes_rules.simulation_based_optimization.base import BaseSurrogateBuilder
from bes_rules.simulation_based_optimization.utils import descale_variables
from bes_rules.configs import OptimizationConfig
from bes_rules.performance_maps import plotting


class PerformanceMapGenerator(BaseSurrogateBuilder):

    def __init__(
            self,
            working_directory: Path,
            optimization_config: OptimizationConfig,
            generate_vclibpy_map_function: callable,
            generate_map_kwargs: dict
    ):
        super().__init__(working_directory=working_directory, optimization_config=optimization_config, test_only=False)
        self.generate_vclibpy_map = generate_vclibpy_map_function
        self.generate_map_kwargs = generate_map_kwargs

    def mp_obj(self, x, *args):
        x = np.array(x)  # If a framework uses lists instead of arrays
        x_descaled = descale_variables(config=self.optimization_config, variables=x)
        # Merge to dict:
        variables_mp = []
        results = []
        objective_values = []

        for x_single in x_descaled:
            variables_mp.append(dict(zip(self.optimization_config.get_variable_names(), x_single)))
        for idx, variables in enumerate(variables_mp):
            self.logger.info("Running combination %s of %s", idx + 1, len(variables_mp))
            y, extra_results = self.generate_vclibpy_map(
                variables,
                save_path=self.cd.joinpath(f"ParameterCombination_{idx}"),
                **self.generate_map_kwargs
            )
            objective_values.append(y)
            results.append({**variables, "objective": y, **extra_results})
        pd.DataFrame(results).to_excel(self.cd.joinpath(f"ParameterStudyResults_{self.cd.name}.xlsx"))
        result_df = pd.DataFrame(results)
        best_combination = result_df["objective"].idxmin()
        self.logger.info(
            "Best parameter combination: %s with objective: %s ",
            best_combination, result_df["objective"].min()
        )
        # Plotting the best run for all T_cons:
        for T_con in result_df.loc[best_combination, "T_con"]:
            plotting.optimization_plot(
                T_con=int(T_con),
                parameter_combination=best_combination,
                optimizer_path=self.cd,
                frosting=result_df.loc[best_combination, "frosting"]
            )
        return objective_values
