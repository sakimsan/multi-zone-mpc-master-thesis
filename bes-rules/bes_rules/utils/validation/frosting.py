"""Module with functions to validate frosting"""
import os

import matplotlib.pyplot as plt
from ebcpy import TimeSeriesData, DymolaAPI
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt
from pathlib import Path
import pandas as pd
import numpy as np
from ebcpy.preprocessing import convert_datetime_index_to_float_index
import json
from bes_rules import BESRULES_PACKAGE_MO, RESULTS_FOLDER
from bes_rules.utils import validation
from bes_rules.plotting import utils
from bes_rules import STARTUP_BESMOD_MOS
from aixcalibuha import Calibrator, TunerParas, Goals, CalibrationClass, SobolAnalyzer

SAVE_PATH = RESULTS_FOLDER.joinpath("FrostingValidation")
FROSTING_PATH = BESRULES_PACKAGE_MO.parent.joinpath("Resources", "Validation")


def run_validation():
    validation_cases = get_validation_cases()
    validation_cases = validation.simulate(validation_cases, sim_setup=dict(stop_time=86400 * 3, output_interval=5),
                                           save_path=SAVE_PATH)
    plot_results(validation_cases)


def plot_results(validation_cases: list):
    variables_to_plot = [
        ["TConOutMea", "heaPum.refCyc.sigBus.TConOutMea"],
        ["PEleCom", "heaPum.refCyc.sigBus.PEleMea"],
        "heaPum.refCyc.sigBus.icefacHPMea",
    ]
    validation.plot_results_plotly(
        validation_cases=validation_cases,
        variables_to_plot=variables_to_plot,
        custom_plot_config={},
        save_path=SAVE_PATH
    )


def plot_errors_over_inputs():
    x_variables = [
        "heaPum.refCyc.sigBus.yMea",
        "heaPum.refCyc.sigBus.TConInMea",
        "heaPum.refCyc.sigBus.TEvaInMea",
        "heaPum.refCyc.sigBus.mConMea_flow"
    ]

    y_errors = [
        ["TConOutMea", "heaPum.refCyc.sigBus.TConOutMea"],
        ["PEleCom", "heaPum.refCyc.sigBus.PEleMea"]
    ]
    plot_config = utils.load_plot_config()
    plot_config.update_config({"variables": validation.CUSTOM_PLOT_CONFIG})
    variables = validation.flatten_nested_list(y_errors) + x_variables + ["sigBus1.hea"]
    all_results = {}

    df_stats = pd.DataFrame()

    res_path = SAVE_PATH.joinpath(SAVE_PATH, "validation_case_results.json")
    with open(res_path, "r") as file:
        validation_cases = json.load(file)

    for validation_case in validation_cases:
        validation_case = validation.ValidationCase(**validation_case)
        df = TimeSeriesData(validation_case.result_path,
                            variable_names=variables).to_df()

        df = plot_config.scale_df(df)
        df.index /= 60  # Convert to minutes
        # Filter first 3 min of init in simulation
        df = df.loc[3:]
        # Filter defrost operation
        df = df.loc[df.loc[:, "sigBus1.hea"] == 1]  # If fan is off, hp is either off or in defrost

        from ebcpy.utils.statistics_analyzer import StatisticsAnalyzer
        for y_error in y_errors:
            df_stats.loc[validation_case.name, y_error[0]] = StatisticsAnalyzer.calc_rmse(
                meas=df.loc[:, y_error[0]],
                sim=df.loc[:, y_error[1]]
            )
        all_results[validation_case.name] = {variable: df.loc[:, variable].values
                                             for variable in variables}
    df_stats.to_excel(SAVE_PATH.joinpath(f"RMSE_stats.xlsx"))
    validation.plot_error_hist(
        all_results=all_results,
        x_variables=x_variables,
        y_errors=y_errors,
        save_path=SAVE_PATH.joinpath(f".png"),
        plot_config=plot_config
    )


def get_validation_cases():
    return [
        validation.ValidationCase(model_name="BESRules.Validation.Defrost.LiangAndZhuCalibrated",
                                  name="LiangAndZhuCalibrated"),
        validation.ValidationCase(model_name="BESRules.Validation.Defrost.LiangAndZhu", name="LiangAndZhu"),
        validation.ValidationCase(model_name="BESRules.Validation.Defrost.NoFrosting", name="NoFrosting"),
        validation.ValidationCase(model_name="BESRules.Validation.Defrost.RocatelloCOPCorrection",
                                  name="RocatelloCOPCorrection"),
        validation.ValidationCase(model_name="BESRules.Validation.Defrost.CICO", name="CICO"),
        validation.ValidationCase(model_name="BESRules.Validation.Defrost.Li", name="Li"),
        validation.ValidationCase(model_name="BESRules.Validation.Defrost.AfjeiAndWetter", name="AfjeiAndWetter"),
    ]


def convert_tables_from_csv_to_hdf(with_plot: bool):
    csv_path = Path(r"R:\_Dissertationen\fwu\06_Diss\Kalibrierung_fwu\_csv")

    frosting_zone_days = {
        "mild": "25",
        "moderate": "24",
        "severe": "23"
    }
    columns_kk = {
        "HYD_Bank8_Temp_extRL": {"name": "TConOutMea", "factor": 1, "offset": 273.15},
        "HYD_Bank8_Temp_extVL": {"name": "TConInMea", "factor": 1, "offset": 273.15},
        "HYD_Bank8_Vdot": {"name": "mCon_flow", "factor": 0.997 / 3600, "offset": 0},
        "KK_rlFeuchte_KK": {"name": "relHum", "factor": 1 / 100, "offset": 0},
        "Leistung_Messung_1_Gesamtleistung": {"name": "PEleMeaTot", "factor": 1, "offset": 0},
    }
    columns_hp = {
        "Defrosting_ActiveDirectDefrost": {"name": "hea", "factor": -1, "offset": -1},
        "StateMachine_Pel_Fan": {"name": "PEleMeaFan", "factor": 1000, "offset": 0},
        "StateMachine_Pel_Inv": {"name": "PEleMeaInv", "factor": 1000, "offset": 0},
        "StateMachine_T_Air_In": {"name": "TEvaInMea", "factor": 1, "offset": 273.15},
        "StateMachine_rps_Comp": {"name": "ySet", "factor": 1 / 110, "offset": 0},
        "WAAGEN_Waage2_Masse": {"name": "mIceWaage", "factor": 1, "offset": 0},
    }
    clean_df_sum = None
    for zone_day, num in frosting_zone_days.items():
        plot_path = csv_path.joinpath("plots", zone_day)
        os.makedirs(plot_path, exist_ok=True)
        path_hp = csv_path.joinpath(f"data{num}_HP.csv")
        path_kk = csv_path.joinpath(f"data{num}_KK.csv")
        tsd_hp = TimeSeriesData(path_hp).to_df()
        tsd_kk = TimeSeriesData(path_kk).to_df()
        assert np.all(tsd_hp.index == tsd_kk.index)
        clean_df = pd.DataFrame(index=tsd_hp.index)
        for tsd, columns in zip([tsd_kk, tsd_hp], [columns_kk, columns_hp]):
            for col, data in columns.items():
                clean_df.loc[:, data["name"]] = (tsd.loc[:, col] + data["offset"]) * data["factor"]
                if not with_plot:
                    continue
                plt.figure()
                plt.plot(tsd.loc[:, col])
                plt.title(col)
                plt.savefig(plot_path.joinpath(f"{col}.png"))
                plt.close("all")
        clean_df = convert_datetime_index_to_float_index(df=clean_df)
        if clean_df_sum is None:
            clean_df_sum = clean_df
        else:
            clean_df.index += clean_df_sum.index[-1]
            clean_df_sum = pd.concat([clean_df_sum, clean_df])
    convert_tsd_to_modelica_txt(
        tsd=clean_df_sum,
        table_name="FrostingData",
        save_path_file=FROSTING_PATH.joinpath(f"frosting.txt")
    )


def calibrate_liang_and_zhu():
    save_path = RESULTS_FOLDER.joinpath("FrostingCalibrationIceFac")
    calibration_class = setup_aixcalibuha()
    dym_api = get_sim(save_path)

    kwargs_scipy_dif_evo = {"maxiter": 30,
                            "popsize": 5,
                            "mutation": (0.5, 1),
                            "recombination": 0.7,
                            "seed": None,
                            "polish": True,
                            "init": 'latinhypercube',
                            "atol": 0}
    kwargs_calibrator = {
        "timedelta": 0,
        "save_files": True,
        "verbose_logging": True,
        "show_plot": False,
        "create_tsd_plot": False,
        "save_tsd_plot": False,
        "show_plot_pause_time": 1e-3,
        "plot_file_type": "png",
        "fail_on_error": False,
        "ret_val_on_error": 10,
        # For this example, let's keep the runtime low
        "max_itercount": 500
    }
    modelica_calibrator = Calibrator(
        working_directory=save_path,
        sim_api=dym_api,
        calibration_class=calibration_class,
        **kwargs_calibrator)
    result = modelica_calibrator.calibrate(
        framework="scipy_differential_evolution",
        method="best1bin",
        **kwargs_scipy_dif_evo
    )


def get_sim(save_path):
    return DymolaAPI(
        model_name="BESRules.Validation.Defrost.LiangAndZhuAixCaliBuHA",
        mos_script_pre=STARTUP_BESMOD_MOS,
        packages=[BESRULES_PACKAGE_MO],
        show_window=True,
        debug=True,
        equidistant_output=True,
        n_cpu=1,
        working_directory=save_path
    )


def run_sa():
    save_path = RESULTS_FOLDER.joinpath("FrostingCalibration_SA")
    calibration_class = setup_aixcalibuha()
    dym_api = get_sim(save_path)
    sen_analyzer = SobolAnalyzer(
        sim_api=dym_api,
        num_samples=50,
        calc_second_order=True,
        working_directory=save_path,
        save_files=True,
        load_files=False,
        savepath_sim=save_path.joinpath('files'),
        suffix_files='mat'
    )
    sen_analyzer.run(
        calibration_classes=calibration_class,
        verbose=True,
        use_first_sim=True,
        plot_result=True,
        save_results=True,
        load_sim_files=True,
        n_cpu=5
    )


def setup_aixcalibuha():
    tuner_paras = {
        "corCoeffSev": [0.01, 1],
        "corCoeffMod": [0.01, 1],
        "corCoeffMil": [0.01, 1],
        "mIce_max": [1, 6],
        #"den_min": [1, 100],
        "facSevMil": [1, 10],
        "timeSev": [5*60, 120*60],
        #"COP_constant": [3, 15]
    }

    tuner_paras = TunerParas(
        names=list(tuner_paras.keys()),
        initial_values=[(v[0] + v[1]) / 2 for v in tuner_paras.values()],
        bounds=list(tuner_paras.values())
    )
    data_path = FROSTING_PATH.joinpath(f"frosting.txt")
    meas_target_data = TimeSeriesData(pd.read_csv(data_path, sep="\t", skiprows=[0, 1], index_col=0))
    data_path_ice_fac = FROSTING_PATH.joinpath(f"iceFac.txt")
    meas_target_data_iceFac = TimeSeriesData(pd.read_csv(data_path_ice_fac, sep="\t", skiprows=[0, 1], index_col=0))
    def resample(t: TimeSeriesData):
        t.to_datetime_index()
        t.clean_and_space_equally(desired_freq="5s")
        t.to_float_index()
        return t
    meas_target_data = resample(meas_target_data)
    meas_target_data_iceFac = resample(meas_target_data_iceFac)
    meas_target_data.loc[:, ("iceFac", "raw")] = meas_target_data_iceFac.loc[:meas_target_data.index[-1], "iceFac"].values
    meas_target_data.loc[:, ("dQConDefGoal", "raw")] = 0
    variable_names = {
        # "TConOut": {"meas": "TConOutMea", "sim": "sigBus1.TConOutMea"},
        "iceFac": {"meas": "iceFac", "sim": "sigBus1.icefacHPMea"},
        # #"dQConDef": {"meas": "dQConDefGoal", "sim": "dQConDef.y"}
    }

    statistical_measure = "RMSE"
    goals = Goals(
        meas_target_data=meas_target_data,
        variable_names=variable_names,
        statistical_measure=statistical_measure,
        #weightings=[0.5, 0.5]
    )

    relevant_intervals = get_on_intervals(series=meas_target_data.loc[90300:, ("hea", "raw")])

    return CalibrationClass(
        name="hea",
        start_time=relevant_intervals[0][0], stop_time=relevant_intervals[-1][-1],
        goals=goals, tuner_paras=tuner_paras,
        relevant_intervals=relevant_intervals
    )


def get_on_intervals(series: pd.Series, threshold=0.5):
    """
    Extract time intervals where device is on.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with float time index and signal column
    signal_column : str
        Name of the column containing the on/off signal
    threshold : float, default 0.5
        Threshold to determine on state (signal > threshold is considered on)

    Returns:
    --------
    list of tuples
        List of (start_time, stop_time) intervals where device is on
    """
    # Convert signal to boolean
    is_on = series > threshold

    # Find changes in state
    state_changes = np.diff(is_on.astype(int))
    change_points = np.where(state_changes != 0)[0]

    # Get timestamps of changes
    times = series.index.values
    intervals = []

    # Handle case where device starts in ON state
    if is_on.iloc[0]:
        intervals.append((times[0], times[change_points[0] + 1]))
        change_points = change_points[1:]

    # Create intervals from change points
    for i in range(0, len(change_points) - 1, 2):
        start_idx = change_points[i] + 1
        stop_idx = change_points[i + 1] + 1
        intervals.append((times[start_idx], times[stop_idx]))

    # Handle case where device ends in ON state
    if len(change_points) % 2 == 1:
        intervals.append((times[change_points[-1] + 1], times[-1]))

    return intervals


def create_interpolated_table():
    """
    Create an equally sampled table from 2D data.

    Parameters:
    -----------
    data : DataFrame
        DataFrame with columns ['Time in min', 'TFro in K', 'rho in kg/m3']
    tfro_points : int, default=50
        Number of equally spaced points for TFro interpolation

    Returns:
    --------
    DataFrame
        Table with equally sampled TFro as index and Time as columns
    """
    path = Path(r"R:\_Dissertationen\fwu\06_Diss\Modellierung\Validierung\Kalibrierung_fwu")
    data = pd.read_excel(path.joinpath("Korn_density_frost.xlsx"))
    tfro_points = 16
    # Get unique time points
    time_points = data['Time in min'].unique()
    # Create equally spaced TFro points
    tfro_min = data['TFro in K'].min()
    tfro_max = data['TFro in K'].max()
    tfro_equal = np.linspace(tfro_min, tfro_max, tfro_points)

    # Initialize result DataFrame
    result = pd.DataFrame(index=tfro_equal)

    # Interpolate for each time point
    for t in time_points:
        time_data = data[data['Time in min'] == t]
        # Sort by TFro to ensure proper interpolation
        time_data = time_data.sort_values('TFro in K')

        # Interpolate rho values for equal TFro points
        rho_interp = np.interp(tfro_equal,
                               time_data['TFro in K'],
                               time_data['rho in kg/m3'])

        result[f'{int(t)}'] = rho_interp
    print(result)
    result.to_excel(path.joinpath("Korn_modelica_2d.xlsx"))


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def plot_logistic_interactive():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)  # Make room for sliders

    # Initial parameters
    k_init = 1.0
    x0_init = 0.0

    # Create x data
    x = np.linspace(-40, 40, 1000)

    # Logistic function
    def logistic(x, k, x0):
        return 1 / (1 + np.exp(-k * (x - x0)))

    # Initial plot
    line, = ax.plot(x, logistic(x, k_init, x0_init), 'b-', lw=2)

    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Logistic Function: y = 1/(1 + e^(-k(x-x0)))')

    # Create sliders
    ax_k = plt.axes([0.1, 0.1, 0.65, 0.03])
    ax_x0 = plt.axes([0.1, 0.05, 0.65, 0.03])
    s_k = Slider(ax_k, 'k (steepness)', 0.1, 5.0, valinit=k_init)
    s_x0 = Slider(ax_x0, 'x0 (midpoint)', -5.0, 5.0, valinit=x0_init)

    # Update function
    def update(val):
        k = s_k.val
        x0 = s_x0.val
        line.set_ydata(logistic(x, k, x0))
        fig.canvas.draw_idle()

    # Register update function with sliders
    s_k.on_changed(update)
    s_x0.on_changed(update)

    plt.show()


def plot_empirical_ice_factor():
    path = RESULTS_FOLDER.joinpath("FrostingCalibration", "NoFrosting.mat")
    df = TimeSeriesData(path).to_df()
    var = "heaPum.refCyc.sigBus.yMea"
    QConMea_name = "eff.QUse_flow"
    QConSim_name = "heaPum.QCon_flow"
    PEleMea_name = "PEleCom"
    PEleSim_name = "sigBus1.PEleMea"
    hea_name = "sigBus1.hea"
    # Create mask for valid data points
    mask = df[hea_name] > 0.5  # Start with non-zero points. All below 5 % speed is off

    filter_time_after_defrost = 60 * 20  # Min. 15 min
    #filter_time_after_defrost = 0

    # Find where control variable becomes non-zero
    transitions = (df[hea_name] > 0) & (df[hea_name].shift(1) == 0)
    transition_times = df.loc[transitions].index

    # Exclude startup_delay seconds after each transition
    for t_start in transition_times:
        mask = mask & ~((df.index >= t_start) &
                        (df.index <= t_start + filter_time_after_defrost))

    transitions_to_defrost = (df[hea_name] == 0) & (df[hea_name].shift(1) > 0)
    transitions_to_defrost_times = df.loc[transitions_to_defrost].index
    filter_time_before_defrost = 5
    # Exclude startup_delay seconds after each transition
    for t_start in transitions_to_defrost_times:
        mask = mask & ~((df.index >= t_start - filter_time_before_defrost) &
                        (df.index <= t_start))

    dMIce = cast_local_max_to_sections(df, "mIceWaage", mask)
    dMIce[mask] = np.nan
    df.loc[~mask] = np.nan
    df.loc[:, "dMIce"] = dMIce
    df.index /= 3600
    mask.index /= 3600
    QConMea = df.loc[:, QConMea_name]
    QConSim = add_diff_to_section(df=df, sim=QConSim_name, mea=QConMea_name, mask=mask)
    #QConSim = df.loc[:, QConSim_name]
    PEleMea = df.loc[:, PEleMea_name]
    PEleSim = add_diff_to_section(df=df, sim=PEleSim_name, mea=PEleMea_name, mask=mask)
    #PEleSim = df.loc[:, PEleSim_name]
    QEvaMea = QConMea - PEleMea
    QEvaSim = QConSim - PEleSim
    iceFac = QEvaMea / QEvaSim
    fig, axes = plt.subplots(4, 1, sharex=True)
    axes[0].plot(QConMea.index, QConMea, label="Mea", color="blue")
    axes[0].plot(QConSim.index, QConSim, label="Sim", color="red", linestyle="--")
    axes[0].set_ylabel("QCon")
    axes[1].plot(PEleMea.index, PEleMea, label="Mea", color="blue")
    axes[1].plot(PEleSim.index, PEleSim, label="Sim", color="red", linestyle="--")
    axes[1].set_ylabel("PEle")
    axes[2].plot(QEvaMea.index, QEvaMea, label="Mea", color="blue")
    axes[2].plot(QEvaSim.index, QEvaSim, label="Sim", color="red", linestyle="--")
    axes[2].set_ylabel("QEva")
    axes[3].plot(iceFac.index, iceFac, label="Mea", color="blue")
    axes[3].plot(df.loc[:, "dMIce"].index, df.loc[:, "dMIce"] / 3, color="red")
    axes[3].axhline(1, color="black", linestyle="--")
    axes[3].set_ylabel("iceFac")
    axes[3].set_ylim([0, 1.5])
    # Export iceFac to modelica txt
    iceFac[iceFac < 0.3] = 0.3
    iceFac[iceFac > 1] = 1
    iceFac[~mask] = 1
    iceFac.index *= 3600
    iceFac = iceFac[~iceFac.index.duplicated(keep='first')]
    convert_tsd_to_modelica_txt(
        tsd=pd.DataFrame({"iceFac": iceFac.values}, index=iceFac.index),
        table_name="FrostingData",
        save_path_file=FROSTING_PATH.joinpath(f"iceFac.txt")
    )
    plt.show()


def cast_local_max_to_sections(df, column_name, mask):
    # Create a copy to avoid modifying the original
    result = df[column_name].copy()

    # Find the boundaries of False sections
    # This creates groups where True sections start a new group
    groups = mask.ne(mask.shift()).cumsum()

    # For each False section, replace all values with the local maximum
    for group_id in groups[~mask].unique():
        section_mask = groups == group_id
        result.loc[section_mask] = df.loc[section_mask, column_name].max() - df.loc[section_mask, column_name].min()

    return result


def add_diff_to_section(df, sim, mea, mask):
    # Create a copy to avoid modifying the original
    result = df[sim].copy()

    # Find the boundaries of False sections
    # This creates groups where True sections start a new group
    groups = mask.ne(mask.shift()).cumsum()

    # For each False section, replace all values with the local maximum
    for group_id in groups[mask].unique():
        section_mask = groups == group_id
        values_to_mean = 60
        result.loc[section_mask] += (
                df.loc[section_mask, mea].values[:values_to_mean].mean() -
                df.loc[section_mask, sim].values[:values_to_mean].mean()
        )

    return result



if __name__ == '__main__':
    # convert_tables_from_csv_to_hdf(with_plot=False)
    # run_validation()

    #calibrate_liang_and_zhu()
    #run_sa()
    #create_interpolated_table()


    # Run the function
    plot_logistic_interactive()
    #plot_empirical_ice_factor()
