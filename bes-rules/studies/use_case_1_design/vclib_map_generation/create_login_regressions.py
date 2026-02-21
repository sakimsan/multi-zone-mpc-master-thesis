import numpy as np

from bes_rules import DATA_PATH
import pandas as pd


def create_all_regressions():
    for variable in ["eta_glob", "lambda_h"]:
        create_regression(variable=variable)


def create_regression(variable):
    path = DATA_PATH.joinpath("map_generation", "login_efficiencies")
    df = pd.read_excel(path.joinpath(variable + ".xlsx"), header=[0, 1, 2])
    T_eva_in = [-3, 2, 7, 12]
    # TODO: Convert T_con to p_con and T_eva to p_eva
    all_T_con = []
    all_n = []
    all_T_eva = []
    y = []
    for T_con_out in [35, 50, 65]:
        for T_eva in T_eva_in:
            n = df.loc[:, (T_con_out, T_eva, "n")].values
            is_na = np.isnan(n)
            n = n[~is_na]
            all_n.extend(n)
            y.extend(df.loc[~is_na, (T_con_out, T_eva, variable)].values)
            all_T_con.extend([T_con_out] * len(n))
            all_T_eva.extend([T_eva] * len(n))
    y = np.array(y)
    from bes_rules.utils import function_fit
    print(f"# Second order {variable}")
    variables = function_fit.create_variables_n_degree(2, all_n, all_T_con, all_T_eva)
    function_fit.fit_linear_regression(variables=variables, y=y, show_plot=True)
    print(f"# Third order {variable}")
    variables = function_fit.create_variables_n_degree(3, all_n, all_T_con, all_T_eva)
    function_fit.fit_linear_regression(variables=variables, y=y, show_plot=True)


if __name__ == '__main__':
    create_all_regressions()
