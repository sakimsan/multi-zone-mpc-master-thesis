from ebcpy import TimeSeriesData
from pathlib import Path
from bes_rules import RESULTS_FOLDER
from studies_ssr.sfh_mpc_hom_monovalent_spawn.run_mpc_fullStudy import study_names
import pandas as pd
import numpy as np
import json

buf_sizes = [12, 23.5, 35]


def get_heated_zone_names(with_sum: bool = False, with_TZoneAreaWeighted: bool = False):
    zone_names = [
        "1_living",
        "2_hobby",
        "3_corridor",
        "4_bath",
        "5_kitchen",
        "6_bed",
        "7_child",
        "8_corridor",
        "9_bath",
        "10_child"
    ]
    if with_sum:
        zone_names.append("sum")
    if with_TZoneAreaWeighted:
        zone_names.append("TZoneAreaWeighted")
    return zone_names


def get_result_file_names(base_path: Path = None, study_name_list: list = None):
    result_xlsx_names = []
    for idx, buf_size in enumerate(buf_sizes):
        result_xlsx_names.append(f"Design_{idx}.xlsx")
    result_file_names = []
    for i, study_name in enumerate(study_name_list):
        result_file_names.append([])
        for result_mat_name in result_xlsx_names:
            result_file_names[i].append(base_path.joinpath(
                study_name,
                "DesignOptimizationResults",
                "TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai_",
                result_mat_name)
            )

    return result_file_names


def get_discomfort_dict(
        zone_names: list,
        start_time: int = 0,
        stop_time: int = 0,
        result_files: list = None
):
    discomfort_dict = {}
    result_names = [f"Design_{idx}" for idx, buf_size in enumerate(buf_sizes)]
    columns = [f"{buf_size}" for buf_size in buf_sizes]

    # Ein Dictionary zur Speicherung der geladenen Daten
    loaded_data = {}

    # Erste Schleife: Lade alle benötigten Daten einmal
    for grid_constraint in study_names:
        print(f"Loading data for {grid_constraint}")
        for i, column in enumerate(columns):
            result_path = find_correct_path(result_files, grid_constraint, result_names[i])
            # Lade Daten nur einmal und speichere sie im Dictionary
            key = (grid_constraint, column)
            loaded_data[key] = pd.read_excel(result_path, index_col=0)

    # Zweite Schleife: Verarbeite die Daten für jede Zone
    for i, zone_name in enumerate(zone_names):
        df_full = pd.DataFrame(index=study_names, columns=columns)

        for grid_constraint in study_names:
            for column in columns:
                # Verwende die bereits geladenen Daten
                df = loaded_data[(grid_constraint, column)]

                if zone_name == "sum":
                    df_discomfort = df.loc[:,
                                     [(f"discomfortKh[{zone}].y") for zone in range(1, i + 1)]].sum(axis=1)
                elif zone_name == "TZoneAreaWeighted":
                    df_discomfort = df.loc[:, (f"discomfortKhTZoneAreaWeighted.y")]
                else:
                    df_discomfort = df.loc[:, (f"discomfortKh[{i + 1}].y")]

                discomfort_period = df_discomfort.loc[stop_time] - df_discomfort.loc[start_time]
                if hasattr(discomfort_period, 'iloc'):
                    discomfort_period = discomfort_period.iloc[0]
                df_full.loc[grid_constraint, column] = float(discomfort_period)

        discomfort_dict[zone_name] = df_full.copy()  # Explizites Kopieren

    # Speicher freigeben
    del loaded_data

    return discomfort_dict


def find_correct_path(path_lists, name1, name2):
    for path_list in path_lists:
        for path in path_list:
            if name1 in str(path) and name2 in str(path):
                return path


# Save discomfort dict
def save_discomfort_dict_json(discomfort_dict, filename_path='discomfort_data.json'):
    json_dict = {}
    for zone, df in discomfort_dict.items():
        json_dict[zone] = df.to_dict()

    with open(filename_path, 'w') as f:
        json.dump(json_dict, f)
    print(f"Data successfully saved. Path: {filename_path}")


if __name__ == "__main__":
    # get discomfort for all zones and in sum for all simulated szenarios and save the discomfort results in a dict. You have to do it only once
    zone_name_list = get_heated_zone_names(with_sum=True, with_TZoneAreaWeighted=True)
    result_file_name_list = get_result_file_names(base_path=RESULTS_FOLDER.joinpath("SFH_MPCRom_monovalent_spawn"),
                                                  study_name_list=study_names)
    discomf_dict = get_discomfort_dict(zone_names=zone_name_list,
                                       start_time=24 * 24 * 3600,
                                       stop_time=31 * 24 * 3600 - 3600,
                                       result_files=result_file_name_list)
    save_discomfort_dict_json(discomfort_dict=discomf_dict,
                              filename_path=RESULTS_FOLDER.joinpath("SFH_MPCRom_monovalent_spawn",
                                                                    "discomfort_data.json"))
