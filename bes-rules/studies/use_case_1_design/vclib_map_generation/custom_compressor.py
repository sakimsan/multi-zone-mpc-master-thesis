import pathlib

from bes_rules import DATA_PATH, RESULTS_FOLDER
from eta_mech import rotary_eta_mech_c10_injected, rotary_eta_mech_injected
from vclibpy.components.compressors import Compressor, rotary
from vclibpy import Inputs, FlowsheetState, media, RelativeCompressorSpeedControl, HeatExchangerInputs
from vclibpy.components.compressors import TenCoefficientCompressor, DataSheetCompressor
from eta_mech import get_eta_mech_cases

def load_nested_json(main_json_path):
    import json
    import os
    # Load the main JSON file
    with open(main_json_path, 'r') as f:
        main_data = json.load(f)

    # Get the directory of the main JSON file
    base_dir = os.path.dirname(main_json_path)

    # Create a new dictionary to store the results
    result = {}

    # Iterate through each key-value pair in the main JSON
    for key, filename in main_data.items():
        # Construct the full path to the referenced JSON file
        file_path = os.path.join(base_dir, key, filename)

        # Load the referenced JSON file
        with open(file_path, 'r') as f:
            file_data = json.load(f)

        # Add the loaded data to the result dictionary
        result[key] = file_data

    return result


def get_optihorst_compressor(med_prop: media.MedProp, config_name: str = "EN_MEN412_Linear"):
    import sys, socket
    if socket.gethostname() == "Laptop-EBC221":
        vclibpy_validation_path = pathlib.Path(r"D:\02_Paper\vclibpy\vclibpy_validation\opti_horst")
    else:
        vclibpy_validation_path = pathlib.Path(r"E:\02_Paper\01_vclibpy\vclibpy_validation\opti_horst")
    sys.path.insert(0, str(vclibpy_validation_path.joinpath("models")))

    from compressor import CalibratedCompressorOptiHorst
    model_config = load_nested_json(vclibpy_validation_path.joinpath("model_configs", f"{config_name}.json"))
    compressor = CalibratedCompressorOptiHorst(N_max=110, V_h=42.3e-6, model_config=model_config)
    compressor.med_prop = med_prop
    return compressor


def get_rotary_compressor(med_prop: media.MedProp):
    compressor = rotary.RotaryCompressor(N_max=110, V_h=42.3e-6)
    compressor.med_prop = med_prop
    return compressor


def get_login_compressor(
        eta_mech_name: str,
        med_prop: media.MedProp = None,
        scaling_factor: float = 1,
        degree_fit: int = 3
):
    from login_compressor import LoginCompressor
    V_h_datasheet = 30.7e-6
    compressor = LoginCompressor(
        V_h=V_h_datasheet*scaling_factor,
        N_max=120,
        degree_fit=degree_fit,
        assumed_eta_mech=get_eta_mech_cases()[eta_mech_name]
    )
    if med_prop is not None:
        compressor.med_prop = med_prop
    return compressor


def get_vitocal_compressor(
        eta_mech_name: str,
        c10_name: str,
        regression: bool,
        scaling_factor: float = 1,
        med_prop: media.MedProp = None
):
    V_h_datasheet = 30.7e-6
    N_max_datasheet = 120   # 7200
    n_min_datasheet = 900 / 7200
    # T_sh_datasheet = 35 - 7.2  # This "rated condition" leads to lambda_h > 1.05...
    T_sh_datasheet = 10
    if regression:
        compressor = DataSheetCompressor(
            N_max=N_max_datasheet,
            V_h=V_h_datasheet * scaling_factor,
            datasheet=RESULTS_FOLDER.joinpath("vitocal", "plots_c10", f"new_regressions_{c10_name}_{eta_mech_name}.csv"),
            extrapolate="linear"
        )
    else:
        compressor = TenCoefficientCompressor(
            N_max=N_max_datasheet,
            V_h=V_h_datasheet,
            datasheet=DATA_PATH.joinpath("map_generation", c10_name + ".xlsx"),
            sheet_name="Tabelle1",
            T_sh=T_sh_datasheet,
            assumed_eta_mech=get_eta_mech_cases()[eta_mech_name],
            capacity_definition="cooling",
            scaling_factor=scaling_factor,
            extrapolate="hold"
        )
    if med_prop is not None:
        compressor.med_prop = med_prop
    return compressor


def calc_compressor(T_eva_in: float, n: float, T_con_out: float, compressor: Compressor):
    fs_state = FlowsheetState()
    dT_pinch = 3
    dT_sh = 10
    inputs = Inputs(
        control=RelativeCompressorSpeedControl(n=n, dT_eva_superheating=dT_sh, dT_con_subcooling=0),
        evaporator=HeatExchangerInputs(T_in=T_eva_in),
        condenser=HeatExchangerInputs(T_out=T_con_out),
    )
    try:
        p_inlet = compressor.med_prop.calc_state("TQ", T_eva_in - dT_pinch - dT_sh, 1).p
        state_inlet = compressor.med_prop.calc_state("PT", p_inlet, T_eva_in - dT_pinch)
        p_outlet = compressor.med_prop.calc_state("TQ", T_con_out - dT_pinch, 0).p
        fs_state.set(name="PI", value=p_outlet / p_inlet)
        compressor.state_inlet = state_inlet
        compressor.calc_state_outlet(p_outlet=p_outlet, inputs=inputs, fs_state=fs_state)
        compressor.calc_m_flow(inputs=inputs, fs_state=fs_state)
        compressor.calc_electrical_power(inputs=inputs, fs_state=fs_state)
        fs_state.set(
            name="eta_glob", value=fs_state.get("eta_is").value * fs_state.get("eta_mech").value,
            unit="%", description="Global compressor efficiency"
        )
    except Exception as err:
        raise err
    return fs_state
