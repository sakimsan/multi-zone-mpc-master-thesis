import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from bes_rules.performance_maps.datasheet_deviation import calc_deviation_to_data_sheet

from vclibpy.components.heat_exchangers import moving_boundary_Tm
from vclibpy.components.heat_exchangers import heat_transfer
from vclibpy.components.expansion_valves import Bernoulli
from vclibpy.components.heat_exchangers.heat_transfer import vdi_atlas_air_to_wall
from vclibpy.flowsheets.ihx import IHX, IHX_NTU
import custom_compressor

from bes_rules import DATA_PATH
from bes_rules.configs import OptimizationConfig, OptimizationVariable
from bes_rules.performance_maps.optimizer import PerformanceMapGenerator
from bes_rules.performance_maps.utils import read_data_sheet
from bes_rules.performance_maps import custom_automation

logger = logging.getLogger(__name__)


def run_vclibpy_map(
        variables: dict,
        save_path: Path,
        weight_cop=0.5,
        weight_q_max=0.25,
        weight_q_nom=0.15,
        weight_q_min=0.1,
        frosting: bool = False,
        step_n: float = 0.1,
        weighting_T_con: dict = None,
        evaporator_values_to_ignore: list = None,
        condenser_values_to_ignore: list = None,
        re_simulate_vclib: bool = True
):
    ### Für VDI Berechnungsmethode ###
    geometry_parameters = vdi_atlas_air_to_wall.AirSourceHeatExchangerGeometry(**{
        "lambda_R": 236,  # kupfer
        "t_q": 0.025,  # Achsabstand der Rohre quer zur Luftrichtung in m
        "t_l": 0.023,  # Achsabstand der Rohre in Luftrichtung in m; t_l = für nur eine Rohrreihe
        "tiefe": 0.07,  # Tiefe der Rippe gesamt
        # "d_a": 7.0e-3,  # Äußerer Rohrdurchmesser gemessen an Verdampfer OptiHorst
        "d_a": 8.0e-3,  # Äußerer Rohrdurchmesser gemessen an Verdampfer Vitocal
        # "d_a": 12.0e-3,  # assumption
        # "d_i": 5.0e-3,  # Innener Rohrdurchmesser gemessen an Verdampfer OptiHorst
        "d_i": 6.0e-3,  # Innener Rohrdurchmesser gemessen an Verdampfer Vitocal
        # "d_i": 10.0e-3,  # assumption
        "n_Rohre": 150,  # Anzahl Rohre
        "n_Rippen": 350,  # anzahl Rippen
        "a": 2.2e-3,  # Abstand zwischen 2 Rippen
        "dicke_rippe": 0.05e-3,  # nicht messbar, übernommen von vorher
        "laenge": 0.75,  # länge des berippten Bereichs
        "hoehe": 1.2,  # Höhe des berippten bereichs
    })
    use_new_ihx = True
    if use_new_ihx:
        from studies.use_case_1_design.vclib_map_generation.new_ihx import IHXNew, IHX_SinglePhase
        ihx_class = IHX_SinglePhase
        ihx_flowsheet = IHXNew
    else:
        ihx_class = IHX_NTU
        ihx_flowsheet = IHX

    condenser = moving_boundary_Tm.MovingBoundaryTmCondenser(
        A=3,  # Geschätzt an Außeneinheit
        secondary_medium="water",
        flow_type="counter",
        ratio_outer_to_inner_area=1,
        two_phase_heat_transfer=heat_transfer.constant.ConstantTwoPhaseHeatTransfer(alpha=variables.get("WÜK_TP", 827.94)),
        gas_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=1200),
        wall_heat_transfer=heat_transfer.wall.WallTransfer(lambda_=236, thickness=2e-3),
        liquid_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=1500),
        secondary_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=variables["WÜK_Wasser"])
    )
    evaporator = moving_boundary_Tm.MovingBoundaryTmEvaporator(
        A=88,  # Gemessen an Außeneinheit
        secondary_medium="air",
        flow_type="counter",
        ratio_outer_to_inner_area=36,  # Berechnet nach VDI Wärmeatlas und Messwerten
        two_phase_heat_transfer=heat_transfer.constant.ConstantTwoPhaseHeatTransfer(alpha=variables.get("WÜK_TP", 542.57)),
        gas_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=1200),
        wall_heat_transfer=heat_transfer.wall.WallTransfer(lambda_=236, thickness=2e-3),
        liquid_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=1500),
        secondary_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=variables["WÜK_Luft"])
    )
    expansion_valve = Bernoulli(A=0.1)

    eta_mech_options = list(custom_compressor.get_eta_mech_cases().keys())
    eta_mech_name = eta_mech_options[int(variables["eta_mech_name"])]

    compressor_names = {
        0: {"name": "c10", "c10_name": "10C_WHP07600_corr", "regression": False},
        1: {"name": "c10", "c10_name": "10C_WHP07600_corr", "regression": True},
        2: {"name": "c10", "c10_name": "10C_WHP07600", "regression": False},
        3: {"name": "c10", "c10_name": "10C_WHP07600", "regression": True},
        4: {"name": "login", "degree_fit": 3},
        5: {"name": "login", "degree_fit": 2},
    }
    idx_compressor = int(variables["compressor_name"])
    compressor_data = compressor_names[idx_compressor]
    logger.info(f"Getting compressor {idx_compressor}: {compressor_names[idx_compressor]} with {eta_mech_name=}")
    if compressor_data["name"] == "login":
        compressor = custom_compressor.get_login_compressor(
            eta_mech_name=eta_mech_name,
            degree_fit=compressor_data["degree_fit"],
            scaling_factor=variables["scaling_factor_10C"],
        )
    else:
        compressor = custom_compressor.get_vitocal_compressor(
            c10_name=compressor_data["c10_name"],
            eta_mech_name=eta_mech_name,
            scaling_factor=variables["scaling_factor_10C"],
            regression=compressor_data["regression"]
        )

    ihx = ihx_class(
        A=variables["A_IHX"],
        wall_heat_transfer=heat_transfer.wall.WallTransfer(lambda_=236, thickness=2e-3),
        alpha_high_side=600,
        alpha_low_side=600,
        #dT_pinch_min=2,
        flow_type="counter"
    )
    expansion_valve_high = Bernoulli(A=100e-6)
    flowsheet = ihx_flowsheet(
        evaporator=evaporator,
        condenser=condenser,
        fluid="Propane",
        compressor=compressor,
        ihx=ihx,
        expansion_valve_high=expansion_valve_high,
        expansion_valve_low=expansion_valve
    )
    datasheet_df = read_data_sheet(
        file_path=DATA_PATH.joinpath("map_generation", "Vitocal_13kW_Datenblatt.xlsx"),
        evaporator_values_to_ignore=evaporator_values_to_ignore,
        condenser_values_to_ignore=condenser_values_to_ignore
    )
    T_eva_in_ar = datasheet_df.loc[:, "T_eva_in"].unique() + 273.15
    T_con_ar = datasheet_df.loc[:, "T_con_out"].unique() + 273.15
    n_ar = np.arange(0.2, 1.01, step_n)

    #T_eva_in_ar = [253.15]
    #T_con_ar = [308.15]
    #n_ar = [0.7, 0.75]

    # Only use_condenser_inlet = False was tested, however, inlet should work as well, just check results
    use_condenser_inlet = False
    os.makedirs(save_path, exist_ok=True)
    from vclibpy.algorithms import Iteration
    algorithm = Iteration(
        dT_start_guess=10,
        raise_errors=True,
        #max_err=0.01,
        #max_err_dT_min=0,
        show_iteration=False,
        #save_path_plots=save_path
    )
    save_path_csv = save_path.joinpath(f"{flowsheet.flowsheet_name}_{flowsheet.fluid}.csv")
    if re_simulate_vclib or not os.path.exists(save_path_csv):
        _, save_path_csv = custom_automation.full_factorial_map_generation(
            flowsheet=flowsheet,
            save_path=save_path,
            T_con_ar=T_con_ar,
            T_eva_in_ar=T_eva_in_ar,
            n_ar=n_ar,
            use_multiprocessing=True,
            datasheet_df=datasheet_df,
            use_condenser_inlet=use_condenser_inlet,
            save_plots=False,
            algorithm=algorithm,
            dT_eva_superheating=variables["dT_superheating"],
            dT_con_subcooling=variables["dT_subcooling"],
            raise_errors=True,
        )
    # Use a unique name to open different files in Excel
    save_path_deviation = save_path.joinpath(f"Deviations_{save_path.name.split('_')[-1]}.xlsx")

    y = calc_deviation_to_data_sheet(
        datasheet_df=datasheet_df,
        vclibpy_df=pd.read_csv(save_path_csv, index_col=0, sep=","),
        save_path=save_path_deviation,
        weight_cop=weight_cop,
        weight_q_max=weight_q_max,
        weight_q_nom=weight_q_nom,
        weight_q_min=weight_q_min,
        frosting=frosting,
        weighting_T_con=weighting_T_con,
        use_condenser_inlet=use_condenser_inlet
    )

    # Store some extra helpful stuff
    extra_results = {
        "save_path_deviation": save_path_deviation,
        "weight_cop": weight_cop,
        "weight_q_max": weight_q_max,
        "weight_q_nom": weight_q_nom,
        "weight_q_min": weight_q_min,
        "frosting": frosting,
        "T_con": T_con_ar - 273.15
    }
    return y, extra_results


if __name__ == "__main__":
    import logging

    logging.basicConfig(level="INFO")
    pd.set_option("display.max_columns", None)  # to show the entire dataframe in the "run" window

    parameter_study_config = OptimizationConfig(
        framework="doe",
        method="ffd",
        variables=[
            OptimizationVariable(
                name="dT_superheating",
                lower_bound=5,
                upper_bound=5,
                levels=1
            ),
            OptimizationVariable(
                name="A_IHX",
                lower_bound=0.001,
                upper_bound=0.02,
                levels=3
            ),
            OptimizationVariable(
                name="dT_subcooling",
                lower_bound=0,
                upper_bound=0,
                levels=1
            ),
            OptimizationVariable(
                name="WÜK_TP",
                lower_bound=3000,
                upper_bound=3000,
                levels=1
            ),
            OptimizationVariable(
                name="WÜK_Luft",
                lower_bound=250,
                upper_bound=500,
                levels=2
            ),
            OptimizationVariable(
                name="WÜK_Wasser",
                lower_bound=3500,
                upper_bound=3500,
                levels=1
            ),
            OptimizationVariable(
                name="scaling_factor_10C",
                lower_bound=1.5,
                upper_bound=1.5,
                levels=1
            ),
            OptimizationVariable(
                name="eta_mech_name",
                lower_bound=1,  # New eta_mech
                upper_bound=1,
                levels=1
            ),
            # see compressor_names for info on meaning
            OptimizationVariable(
                name="compressor_name",
                lower_bound=3,
                upper_bound=3 ,
                levels=1
            ),
        ],
    )

    from bes_rules import RESULTS_FOLDER

    KWARGS = dict(
        weighting_T_con={
            35: 2 / 14,
            45: 4 / 14,
            55: 4 / 14,
            65: 0,#3 / 14,
            70: 0,# / 14
        },
        evaporator_values_to_ignore=[30., 35.],
        frosting=True,
        step_n=0.05,
        re_simulate_vclib=False
    )
    map_generator = PerformanceMapGenerator(
        optimization_config=parameter_study_config,
        working_directory=RESULTS_FOLDER.joinpath("vitocal", "map_ihx_new"),
        generate_vclibpy_map_function=run_vclibpy_map,
        generate_map_kwargs=KWARGS
    )
    map_generator.run()
