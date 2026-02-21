import os
import pandas as pd
import numpy as np
from pathlib import Path


def compare(fmu_path, dymola_path):
    fmu_path = Path(fmu_path)
    dym_path = Path(dymola_path)
    for file in os.listdir(fmu_path):
        fmu_xlsx = fmu_path.joinpath(file, "DesignOptimizerResults.xlsx")
        dym_xlsx = dym_path.joinpath(file, "DesignOptimizerResults.xlsx")
        if not os.path.exists(dym_xlsx):
            print("Dymola counterpart does not exist for study:", file)
        if not os.path.exists(fmu_xlsx):
            print("FMU result does not exist for study:", file)
        df_fmu = pd.read_excel(fmu_xlsx)
        df_dym = pd.read_excel(dym_xlsx)
        errors = {}
        for col in df_dym.columns:
            if col not in df_fmu:
                print(file, col, "not in fmu results")
            if isinstance(df_fmu[col].values[0], str):
                continue
            dym_data = df_dym[col]
            fmu_data = df_fmu[col]
            if np.any(fmu_data != dym_data):
                mask = df_dym[col] != 0
                if not np.any(mask):
                    continue
                err_arr = np.abs((df_dym.loc[mask, col] - df_fmu.loc[mask, col])/df_dym.loc[mask, col] * 100)
                if err_arr.max() > 1:
                    errors[col] = (max(err_arr), sum(err_arr)/len(df_dym[col]))
        print(errors)

if __name__ == '__main__':
    compare(
        fmu_path=r"D:\00_temp\01_design_optimization\TestFMUCases\DesignOptimizationResults",
        dymola_path=r"D:\00_temp\01_design_optimization\Test180Cases\DesignOptimizationResults"
    )
