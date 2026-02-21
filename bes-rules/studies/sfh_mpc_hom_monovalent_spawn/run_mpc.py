import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json
from pathlib import Path
from typing import Iterable, Dict, Any, List

from bes_rules import RESULTS_FOLDER, PC_SPECIFIC_SETTINGS
from bes_rules import configs
from bes_rules.input_variations import run_input_variations
from bes_rules.simulation_based_optimization import AgentLibMPC
from studies_ssr.sfh_mpc_hom_monovalent_spawn import MPC_UTILS_PATH, base_design_optimization
from studies_ssr.sfh_mpc_hom_monovalent_spawn.buf_influences import compare_buf_sizes
from studies_ssr.sfh_mpc_hom_monovalent_spawn.plotting import plot_mpc
from bes_rules.simulation_based_optimization.agentlib_mpc.get_idf_data import get_idf_data

HOM_predictor: bool = True

def manipulate_predictions(df):
    df.loc[:, "P_el_demand"] = (
            df.loc[:, "internal_gains_convective"] +
            df.loc[:, "internal_gains_radiative"]
    )
    return df


def run_mpc(
        n_cpu,
        with_dynamic_prices: bool,
        no_minimal_compressor_speed: bool,
        inverter_uses_storage: bool,
        with_custom_modifiers: bool,
        existing_zone_record_path: str = None
):
    time_step = 3600

    # Anpassen!
    study_name = "".join([
        _get_study_name(no_minimal_compressor_speed),
        "_Nachtabsenkung_TARP_Max"
    ])


    if HOM_predictor:
        mapping_predictions = {
            "electricalGrid.PElecGen": "P_el_pv_raw",
            "electrical.generation.f_design[1]": "PVDesignSize",
            "outputs.weather.TDryBul": "T_amb",
            "hydraulic.control.buiAndDHWCtr.TSetBuiSup.TSet": "THeaCur",

            "building.ZoneWindowsTotalHeatRateNonSolar2.y[1]": "ZoneWindowsTotalHeatRate_livingroom",
            "building.ZoneWindowsTotalHeatRateNonSolar2.y[2]": "ZoneWindowsTotalHeatRate_hobby",
            "building.ZoneWindowsTotalHeatRateNonSolar2.y[3]": "ZoneWindowsTotalHeatRate_corridor",
            "building.ZoneWindowsTotalHeatRateNonSolar2.y[4]": "ZoneWindowsTotalHeatRate_wcstorage",
            "building.ZoneWindowsTotalHeatRateNonSolar2.y[5]": "ZoneWindowsTotalHeatRate_kitchen",
            "building.ZoneWindowsTotalHeatRateNonSolar2.y[6]": "ZoneWindowsTotalHeatRate_bedroom",
            "building.ZoneWindowsTotalHeatRateNonSolar2.y[7]": "ZoneWindowsTotalHeatRate_children",
            "building.ZoneWindowsTotalHeatRateNonSolar2.y[8]": "ZoneWindowsTotalHeatRate_corridor2",
            "building.ZoneWindowsTotalHeatRateNonSolar2.y[9]": "ZoneWindowsTotalHeatRate_bath",
            "building.ZoneWindowsTotalHeatRateNonSolar2.y[10]": "ZoneWindowsTotalHeatRate_children2",
            "building.ZoneWindowsTotalHeatRateNonSolar2.y[11]": "ZoneWindowsTotalHeatRate_attic",


            "building.ZoneInsideFaceSolarRadiationHeatGainRate6.y": "Q_RadSol_bedroom",
            "building.ZoneInsideFaceSolarRadiationHeatGainRate1.y": "Q_RadSol_livingroom",
            "building.ZoneInsideFaceSolarRadiationHeatGainRate5.y": "Q_RadSol_kitchen",
            "building.ZoneInsideFaceSolarRadiationHeatGainRate2.y": "Q_RadSol_hobby",
            "building.ZoneInsideFaceSolarRadiationHeatGainRate4.y": "Q_RadSol_wcstorage",
            "building.ZoneInsideFaceSolarRadiationHeatGainRate3.y": "Q_RadSol_corridor",
            "building.ZoneInsideFaceSolarRadiationHeatGainRate7.y": "Q_RadSol_children",
            "building.ZoneInsideFaceSolarRadiationHeatGainRate8.y": "Q_RadSol_corridor2",
            "building.ZoneInsideFaceSolarRadiationHeatGainRate9.y": "Q_RadSol_bath",
            "building.ZoneInsideFaceSolarRadiationHeatGainRate10.y": "Q_RadSol_children2",
            "building.ZoneInsideFaceSolarRadiationHeatGainRate11.y": "Q_RadSol_attic",

            "building.useProBus.TZoneSet[1]": "TSetOneZone_livingroom",
            "building.useProBus.TZoneSet[2]": "TSetOneZone_hobby",
            "building.useProBus.TZoneSet[3]": "TSetOneZone_corridor",
            "building.useProBus.TZoneSet[4]": "TSetOneZone_wcstorage",
            "building.useProBus.TZoneSet[5]": "TSetOneZone_kitchen",
            "building.useProBus.TZoneSet[6]": "TSetOneZone_bedroom",
            "building.useProBus.TZoneSet[7]": "TSetOneZone_children",
            "building.useProBus.TZoneSet[8]": "TSetOneZone_corridor2",
            "building.useProBus.TZoneSet[9]": "TSetOneZone_bath",
            "building.useProBus.TZoneSet[10]": "TSetOneZone_children2",


            "building.TempExtWallNode1[1].y": "T_preTemWall_OuterWall_wcstorage",
            "building.TempExtWallNode1[2].y": "T_preTemWall_OuterWall2_wcstorage",
            "building.TempExtWallNode1[3].y": "T_preTemWall_OuterWall_children",
            "building.TempExtWallNode1[4].y": "T_preTemWall_OuterWall2_children",
            "building.TempExtWallNode1[5].y": "T_preTemWall_OuterWall_bath",
            "building.TempExtWallNode1[6].y": "T_preTemWall_OuterWall2_bath",
            "building.TempExtWallNode1[7].y": "T_preTemWall_OuterWall_attic",
            "building.TempExtWallNode1[8].y": "T_preTemWall_OuterWall2_attic",
            "building.TempExtWallNode1[9].y": "T_preTemWall_OuterWall_bedroom",
            "building.TempExtWallNode1[10].y": "T_preTemWall_OuterWall2_bedroom",
            "building.TempExtWallNode1[11].y": "T_preTemWall_OuterWall_hobby",
            "building.TempExtWallNode1[12].y": "T_preTemWall_OuterWall2_hobby",
            "building.TempExtWallNode1[13].y": "T_preTemWall_OuterWall_livingroom",
            "building.TempExtWallNode1[14].y": "T_preTemWall_OuterWall2_livingroom",
            "building.TempExtWallNode1[15].y": "T_preTemWall_OuterWall_corridor",
            "building.TempExtWallNode1[16].y": "T_preTemWall_OuterWall_corridor2",
            "building.TempExtWallNode1[17].y": "T_preTemWall_OuterWall_kitchen",
            "building.TempExtWallNode1[18].y": "T_preTemWall_OuterWall2_kitchen",
            "building.TempExtWallNode1[19].y": "T_preTemWall_OuterWall_children2",
            "building.TempExtWallNode1[20].y": "T_preTemWall_OuterWall2_children2",

            "building.TempRoofNode1[1].y": "T_preTemRoof_Dach_attic",
            "building.TempRoofNode1[2].y": "T_preTemRoof_Dach2_attic",
            "building.TempRoofNode1[3].y": "T_preTemRoof_Dach3_attic",

            "building.TempGroundFloorNode1[1].y": "T_preTemFloor_GroundFloor_wcstorage",
            "building.TempGroundFloorNode1[2].y": "T_preTemFloor_GroundFloor_hobby",
            "building.TempGroundFloorNode1[3].y": "T_preTemFloor_GroundFloor_livingroom",
            "building.TempGroundFloorNode1[4].y": "T_preTemFloor_GroundFloor_corridor",
            "building.TempGroundFloorNode1[5].y": "T_preTemFloor_GroundFloor_kitchen",

            "hydraulic.control.buiAndDHWCtr.TSetBuiSupSGReady.datRea.y[1]": "SGReadySignal",


            "const[1].k": "schedule_light",
            "const[2].k": "schedule_dev",
            "const[3].k": "schedule_human"
        }
    else:
        mapping_predictions = {
            "electrical.generation.outBusGen.PElePV.value": "P_el_pv_raw",
            "parameterStudy.f_design": "PVDesignSize",
            "outputs.weather.TDryBul": "T_amb",
            "hydraulic.control.buiAndDHWCtr.TSetBuiSup.TSet": "THeaCur",
            "building.thermalZone[1].ROM.radHeatSol[1].Q_flow": "Q_RadSol_or_1",
            "building.thermalZone[1].ROM.radHeatSol[2].Q_flow": "Q_RadSol_or_2",
            "building.thermalZone[1].ROM.radHeatSol[3].Q_flow": "Q_RadSol_or_3",
            "building.thermalZone[1].ROM.radHeatSol[4].Q_flow": "Q_RadSol_or_4",
            "building.thermalZone[1].lights.uRel": "schedule_light",
            "building.thermalZone[1].machinesSenHea.uRel": "schedule_dev",
            "building.thermalZone[1].humanSenHeaDependent.uRel": "schedule_human",
            "building.thermalZone[1].preTemWall.T": "T_preTemWall",
            "building.thermalZone[1].preTemWin.T": "T_preTemWin",
            "building.thermalZone[1].preTemRoof.T": "T_preTemRoof",
            "building.thermalZone[1].preTemFloor.T": "T_preTemFloor",
            "hydraulic.control.buiAndDHWCtr.TSetBuiSupSGReady.datRea.y[1]": "SGReadySignal"
            #"building.thermalZone[1].ventCont.redFac": "redFac",
        }

    def manipulate_predictions(df: pd.DataFrame):
        if HOM_predictor:
            #df.loc[:, "T_preTemWall"] = df.loc[:, "T_preTemWall"] + 273.15
            #df.loc[:, "T_preTemRoof"] = df.loc[:, "T_preTemRoof"] + 273.15
            #df.loc[:, "T_preTemFloor"] = df.loc[:, "T_preTemFloor"] + 273.15

            return df
        else:
            df.loc[:, "Q_RadSol"] = (
                    df.loc[:, "Q_RadSol_or_1"] +
                    df.loc[:, "Q_RadSol_or_2"] +
                    df.loc[:, "Q_RadSol_or_3"] +
                    df.loc[:, "Q_RadSol_or_4"]
            )
            df.drop(columns=["Q_RadSol_or_1", "Q_RadSol_or_2", "Q_RadSol_or_3", "Q_RadSol_or_4"], inplace=True)
            return df

    if HOM_predictor:
        surrogate_builder_kwargs = dict(
            predictive_control_options=dict(
                mpc_module=MPC_UTILS_PATH.joinpath("agent_modules/automated_files/mpc.json"),
                predictor_module=MPC_UTILS_PATH.joinpath("agent_modules/automated_files/predictor.json"),
                simulator_module=MPC_UTILS_PATH.joinpath("agent_modules/automated_files/simulator_fmu.json"),
                model_path=MPC_UTILS_PATH.joinpath("model_hom_alpha.py"),
                mpc_parameters={},
                save_mpc_results=False,
                save_mpc_stats=True
            ),
            # Anpassen?!
            mapping_prediction_defaults={
                "P_el_pv_raw": 0,
                "PVDesignSize": 1,
                "schedule_light": 0,
                "schedule_dev": 0,
                "schedule_human": 0
            },
            manipulate_predictions=manipulate_predictions,
            mapping_predictions=mapping_predictions,
            hom=True,
            with_custom_modifiers=with_custom_modifiers
        )
    else:
        surrogate_builder_kwargs = dict(
            predictive_control_options=dict(
                mpc_module=MPC_UTILS_PATH.joinpath("agent_modules/automated_files/mpc.json"),
                predictor_module=MPC_UTILS_PATH.joinpath("agent_modules/predictor.json"),
                simulator_module=MPC_UTILS_PATH.joinpath("agent_modules/automated_files/simulator_fmu.json"),
                model_path=MPC_UTILS_PATH.joinpath("model_hom_alpha.py"),
                mpc_parameters={},
                save_mpc_results=False,
                save_mpc_stats=True
            ),
            # Anpassen?!
            mapping_prediction_defaults={
                "P_el_pv_raw": 0,
                "PVDesignSize": 1,
            },
            recalculate_predictions=True,
            manipulate_predictions=manipulate_predictions,
            mapping_predictions=mapping_predictions,
            hom=True,
            with_custom_modifiers=with_custom_modifiers
        )


    sim_config = base_design_optimization.get_simulation_config(
        #model="MonovalentVitoCal_ROM_2h",
        model="MonovalentVitoCal_HOM_Nachtabsenkung",
        model_hom="MonovalentVitoCal_HOM_MPC_yValSet",
        # Anpassen!
        start_time=22 * 24 * 3600,
        n_days=9,
        init_period=0,
        time_step=time_step,
        debug=True,
        convert_to_hdf_and_delete_mat=False,
        recalculate=True,
        equidistant_output=True,
        variables_to_save=dict(
            states=True,
            derivatives=True,
            inputs=True,
            outputs=True,
            auxiliaries=True,
        )
    )

    optimization_config = compare_buf_sizes.get_optimization_config()
    # Anpassen!
    inputs_config = compare_buf_sizes.get_inputs_config(
        no_dhw=True,
        inverter_uses_storage=inverter_uses_storage,
        no_minimal_compressor_speed=no_minimal_compressor_speed,
        with_start_losses=False,
        only_inverter=True,
        hom=True,
        existing_zone_record_path=existing_zone_record_path
    )
    if with_dynamic_prices:
        inputs_config.prices = [{"year": None}, {"year": 2023}, {"year": 2019}, {"year": 2023, "only_wholesale_price": True}]

    config = configs.StudyConfig(
        base_path=RESULTS_FOLDER.joinpath("SPAWN_Studies"),
        n_cpu=n_cpu,
        name=study_name,
        simulation=sim_config,
        optimization=optimization_config,
        inputs=inputs_config,
        objectives=[],
        test_only=False
    )
    run_input_variations(
        config=config, run_inputs_in_parallel=False,
        surrogate_builder_class=AgentLibMPC,
        **surrogate_builder_kwargs
    )
    #calculate_with_dynamic_prices(study_name)


def _get_study_name(no_minimal_compressor_speed: bool):
    return "mpc_perfect" if no_minimal_compressor_speed else "mpc_design"


if __name__ == "__main__":
    logging.basicConfig(level="INFO")


    run_mpc(
        n_cpu=1,
        with_dynamic_prices=False,
        inverter_uses_storage=True,
        no_minimal_compressor_speed=False,
        with_custom_modifiers=False,
        existing_zone_record_path=f"{PC_SPECIFIC_SETTINGS['BESGRICONOP_PACKAGE_MO']}/Modelica/BESGriConOp/Studies/SFH/MPCModelROM/HeatDemandCISBAT/RecordsCollection/bim2sim_teaserExport_4elements_CISBAT_atticUnheated_calibratedColdPeriod_higherCRoof"
    )
