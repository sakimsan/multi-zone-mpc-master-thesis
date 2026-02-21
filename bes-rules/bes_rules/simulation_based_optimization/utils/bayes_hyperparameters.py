import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import os
import json

logger = logging.getLogger(__name__)


def tune_hyperparameters(
        variables: dict,
        number_of_iterations: int,
        path_training_data: Path,
        save_path: Path) -> dict:
    """
    Tune Gaussian Process hyperparameters using RandomizedSearchCV.

    Parameters:
    variables (dict): A dictionary containing variable names as keys and their respective
                      hyperparameter ranges (length scales) as values.
    optimization_variables (list): A list of column names to be used as input features.
    number_of_iterations (int): Number of iterations for the RandomizedSearchCV.
    path_training_data (Path): Path to the Excel file containing training data.
    save_path (Path): Path to save the results and figures.

    Returns:
    dict: A dictionary containing the best hyperparameters for each variable.
    """
    best_hyperparams_dict = {}
    os.makedirs(save_path, exist_ok=True)
    i_obj = 0
    # Loop through each variable defined in the input dictionary
    for objective, opt_variables_length_scale in variables.items():
        logger.info("Tune objective %s/%s: %s", i_obj + 1, len(variables), objective)
        i_obj += 1
        optimization_variables = list(opt_variables_length_scale.keys())

        # Read training data from Excel
        df = pd.read_excel(path_training_data)

        # Use the provided optimization variable names
        X = df[optimization_variables].values
        y = df[objective].values

        # Extract length scales dynamically from the variable dictionary
        length_scales = []
        for opt_variable_length_scale in opt_variables_length_scale.values():
            length_scales.append(opt_variable_length_scale[0])

        # Define the Gaussian Process kernel
        kernel = Matern(length_scale=length_scales, nu=2.5)

        # Define the parameter distribution for the Randomized Search
        param_dist = {
            "kernel__length_scale": uniform(
                loc=length_scales,
                scale=[scale[1] - scale[0] for scale in opt_variables_length_scale.values()]
            )
        }

        # Initialize the Gaussian Process Regressor
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)

        # Perform Randomized Search for hyperparameter tuning
        randomized_search = RandomizedSearchCV(gp, scoring="neg_mean_squared_error",
                                               param_distributions=param_dist,
                                               n_iter=number_of_iterations, cv=5)
        randomized_search.fit(X, y)

        hyperparams = randomized_search.cv_results_["params"]
        scores = randomized_search.cv_results_["mean_test_score"]  # Negative mean squared error

        # Prepare results DataFrame
        results_df = pd.DataFrame({
            **{
                optimization_variable: [params["kernel__length_scale"][idx] for params in hyperparams]
                for idx, optimization_variable in enumerate(optimization_variables)
            },
            "Mean Squared Error": -scores
        })

        # Save results to Excel
        output_filename = f"{objective}.xlsx"
        output_file_path = save_path.joinpath(output_filename)
        results_df.to_excel(output_file_path, index=False)

        # Extract the best hyperparameters
        best_hyperparams = hyperparams[randomized_search.best_index_]
        best_hyperparams_dict[objective] = {
            optimization_variable: best_hyperparams["kernel__length_scale"][idx]
            for idx, optimization_variable in enumerate(optimization_variables)
        }

        # Plot the results in 3D for hyperparameters vs MSE
        for i in range(1, len(optimization_variables)):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                results_df[optimization_variables[0]],
                results_df[optimization_variables[i]],
                results_df["Mean Squared Error"],
                c=results_df["Mean Squared Error"],
                cmap="viridis",
                alpha=0.7
            )

            ax.set_xlabel(optimization_variables[0])
            ax.set_ylabel(optimization_variables[i])
            ax.set_zlabel("Mean Squared Error")
            ax.set_title(f"MSE vs {optimization_variables[0]} and {optimization_variables[i]}")

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Mean Squared Error")

            # Save the 3D scatter plot
            hyperparam_plot_filename = f"{objective}_3Dscatter_{optimization_variables[i]}.png"
            hyperparam_plot_file_path = save_path.joinpath(hyperparam_plot_filename)
            plt.savefig(hyperparam_plot_file_path, dpi=300)
            plt.close()  # Close the figure to prevent display

    json_output_path = save_path.joinpath("best_hyperparameters.json")
    with open(json_output_path, "w") as json_file:
        json.dump(best_hyperparams_dict, json_file, indent=4)

    return best_hyperparams_dict


def autotune_hyperparameters(
        objectives: list,
        optimization_variables: list,
        path_training_data: Path,
        save_path: Path,
        max_iterations: int = 5,
        percentage_to_include: int = 20
):
    default_scales = [1, 10000]
    default_length_scales = {variable: default_scales for variable in optimization_variables}
    variables = {
        objective: default_length_scales
        for objective in objectives
    }
    percentage_convergence_step = percentage_to_include / (max_iterations - 1)
    for i in range(max_iterations):
        logger.info("Iteration %s/%s: %s", i + 1, max_iterations, variables)
        best_hyp = tune_hyperparameters(
            variables=variables,
            number_of_iterations=500,
            path_training_data=path_training_data,
            save_path=save_path.joinpath(f"iter_{i}")
        )
        # Update variables:
        for variable, paras in best_hyp.items():
            for opt_var, best_value in paras.items():
                variables[variable][opt_var] = [
                    best_value * (1 - percentage_to_include / 100),
                    best_value * (1 + percentage_to_include / 100)
                ]
        percentage_to_include -= percentage_convergence_step
        percentage_to_include = max(percentage_to_include, 1)

    with open(save_path.joinpath("best_hyperparameters.json"), "w") as json_file:
        json.dump(best_hyp, json_file, indent=4)
    print(best_hyp)


if __name__ == "__main__":
    from bes_rules import RESULTS_FOLDER
    logging.basicConfig(level="INFO")
    # Example usage
    autotune_hyperparameters(
        objectives=[
            "SCOP_Sys",
            "costs_total",
            "outputs.hydraulic.gen.heaPum.numSwi",
        ],
        optimization_variables=[
            "parameterStudy.TBiv",
            "parameterStudy.VPerQFlow",
            #"parameterStudy.ShareOfPEleNominal"
        ],
        path_training_data=RESULTS_FOLDER.joinpath(
            "RE_Journal", "BESCtrl", "DesignOptimizationResults",
            "TRY2015_536322100078_Jahr_B1994_retrofit_SingleDwelling_M_South",
            "DesignOptimizerResults.xlsx"),
        save_path=RESULTS_FOLDER.joinpath("BayesHyperparameters")
    )
