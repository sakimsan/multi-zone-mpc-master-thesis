from pathlib import Path
import os

import pandas as pd
import numpy as np
from ebcpy import TimeSeriesData
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt

from bes_rules import DATA_PATH
from bes_rules.utils.functions import argmean


def create_electricity_profiles_for_occupants(
        save_path: Path,
        number_of_occupants: int
):

    df_grid = pd.read_excel(
        DATA_PATH.joinpath("Kerber_Vorstadtnetz.xlsx"),
        sheet_name=f"Kerber Netz Altbau",
        index_col=0
    )
    mask = df_grid.loc[:, "Anzahl Bewohner"] == number_of_occupants
    if not np.any(mask):
        raise KeyError(f"Number of occupants '{number_of_occupants}' not found in Kerber Netz")
    idx = argmean(df_grid.loc[mask, "Jahresenergiebedarf [kWh/a]"])
    csv_path = DATA_PATH.joinpath("electricity_profiles", f"elec_{idx}.csv")
    tsd = TimeSeriesData(csv_path, sep=",")
    tsd.to_float_index()
    tsd.loc[:, "electricity in W"] = tsd.loc[:, ("Wirkleistung [kW]", "raw")] * 1000  # to W
    os.makedirs(save_path.parent, exist_ok=True)
    convert_tsd_to_modelica_txt(
        tsd,
        table_name="ElectricityGains",
        columns=["electricity in W"],
        save_path_file=save_path
    )


if __name__ == '__main__':
    for N_OCC in range(2, 50):
        try:
            create_electricity_profiles_for_occupants(
                save_path=DATA_PATH.joinpath("electricity_profiles", f"{N_OCC}_occupants.txt"),
                number_of_occupants=N_OCC
            )
        except KeyError as err:
            print(err)
