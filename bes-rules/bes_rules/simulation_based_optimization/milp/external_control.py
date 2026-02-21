import logging
from pathlib import Path

import pandas as pd
from agentlib.models.fmu_model import FmuModel
from ebcpy import TimeSeriesData

from bes_rules.rule_extraction.rbpc_development import utils
from bes_rules.simulation_based_optimization.besmod import plot_result
from bes_rules.utils import create_or_append_list

logger = logging.getLogger(__name__)


def start_fmu(
        fmu_path: Path,
        parameters: dict,
        output_result_names: list,
        state_result_names: list,
        variables: dict,
):
    def merge_lists(l1, l2):
        return list(set(list(l1) + list(l2)))

    parameter_names = merge_lists(parameters.keys(), variables.get("parameter_names", []))

    output_names = merge_lists(output_result_names, variables.get("output_names", []))
    state_names = merge_lists(state_result_names, variables.get("state_names", []))

    return FmuModel(
        path=fmu_path, extract_fmu=False,
        parameters=[dict(name=var) for var in parameter_names],
        inputs=[dict(name=var) for var in variables.get("input_names", [])],
        outputs=[dict(name=var) for var in output_names],
        states=[dict(name=var) for var in state_names]
    )


def run_external_control(
        parameter: dict,
        time_series_inputs: pd.DataFrame,
        fmu_path: Path,
        save_path: Path,
        state_result_names: list,
        output_result_names: list,
        start_time: int,
        stop_time: int,
        output_interval: float,
        external_control: callable,
        model_parameters: dict,
        variables: dict,
        control_horizon: int = 24,
        prediction_horizon: int = 24,
        closed_loop: bool = True,
        with_dhw: bool = False,
        get_df: bool = False,
        solver_kwargs: dict = None,
        minimal_part_load_heat_pump: float = 0.25,
        check_results_correctness: bool = True
):
    fmu = start_fmu(
        fmu_path=fmu_path,
        parameters=parameter,
        state_result_names=state_result_names,
        output_result_names=output_result_names,
        variables=variables
    )

    logger.info("Simulating parameter combination %s", parameter)
    for key, value in parameter.items():
        fmu.set_parameter_value(key, value)
    try:
        fmu.system.reset()
    except OSError:
        logger.info("Can't reset FMI Instance due to access violation error. This should not be an issue.")

    fmu.initialize(t_start=start_time,
                   t_stop=stop_time)

    # start time in hours (Stunde des Jahres)
    total_runtime = (stop_time - start_time) / 3600
    start_time_in_h = start_time / 3600
    time_step = output_interval / 3600  # step size in hours
    t_sample = time_step * 3600

    simulation_settings = {
        'prediction_horizon': prediction_horizon,
        'control_horizon': control_horizon,
        'time_step': time_step,
        'start_time': start_time_in_h,
        'overall_stop_time': stop_time / 3600,
        'total_runtime': total_runtime,
        'safe_start_time': start_time_in_h,  # only for saving
    }

    if control_horizon > simulation_settings['prediction_horizon']:
        raise ValueError('Control Horizon has to be smaller than the prediction horizon')

    end = 0
    # Load Parameters
    save_results = {}
    TBufSetInit = None
    T_DHW_Init = model_parameters.get("T_DHW_init", 273.15 + 60)

    # Start Loop of EasyModell and set temperatures as new init temperatures for each time step
    # Alle 4 Stunden für MPC und alle 24h für RBPC!
    n_iterations = int(simulation_settings['total_runtime'] / simulation_settings['control_horizon'])
    for iteration in range(n_iterations):
        logger.info("============ iteration %s of %s ============", iteration, n_iterations - 1)
        logger.info('New start time is: %s', simulation_settings['start_time'])
        logger.info('Optimization is running....')
        if iteration > 0:
            if closed_loop:
                TBufSetInit = fmu.get_state("hydraulic.control.sigBusDistr.TStoBufTopMea").value
                if with_dhw:
                    T_DHW_Init = fmu.get_state("hydraulic.control.sigBusDistr.TStoDHWTopMea").value
            else:
                if not len(save_results["TBufSet"]) == end + 1:
                    raise IndexError("Something in the result handling went wrong, "
                                     "time-steps and result length do not match.")
                TBufSetInit = save_results['TBufSet'][end]
                if with_dhw:
                    T_DHW_Init = save_results['T_DHW'][end]
        logger.info(f"in loop: {TBufSetInit=}, {iteration=}")
        results_optim = external_control(
            simulation_settings=simulation_settings,
            solver_kwargs=solver_kwargs,
            time_series_inputs=time_series_inputs,
            model_parameters=model_parameters,
            iteration=iteration,
            TBufSetInit=TBufSetInit,
            T_DHW_Init=T_DHW_Init,
            with_dhw=with_dhw,
            minimal_part_load_heat_pump=minimal_part_load_heat_pump,
            save_path_result=save_path
        )

        logger.info('Simulation is running....')
        fmu_inputs = fmu.get_input_names()
        matching_inputs = set(fmu_inputs).intersection(list(results_optim.keys()))
        # 16 x 15min Zeitschritte für MPC und 96x für RBPC übergeben
        for idx in range(int(simulation_settings['control_horizon'] / simulation_settings['time_step'])):
            if closed_loop:
                control_step_failed = False
                for input_name in matching_inputs:
                    value = results_optim[input_name][idx]
                    if value is None:
                        control_step_failed = True
                        break
                    fmu.set_input_value(input_name, value)
                if control_step_failed:
                    # Disable all external controls
                    for input_name in fmu_inputs:
                        if input_name.startswith("act"):
                            fmu.set_input_value(input_name, False)

                t_start = simulation_settings['start_time'] * 3600 + idx * t_sample
                fmu.do_step(
                    t_start=t_start,
                    t_sample=t_sample
                )
                for var in fmu.variables:
                    save_results = create_or_append_list(save_results, var.name, var.value)
            for res, value in results_optim.items():
                save_results = create_or_append_list(save_results, res, value[idx])

        simulation_settings['start_time'] = simulation_settings['start_time'] + simulation_settings['control_horizon']
        end = int(simulation_settings['control_horizon'] * (1 / time_step)) * (iteration + 1) - 1

    df = pd.DataFrame(save_results)
    df.index += (start_time_in_h * 3600)
    df.index *= 3600 * time_step
    if "Q_Sto_Energy" in df.columns and check_results_correctness:
        df["QStoDischarge"] = (df["Q_Sto_Energy"] - df.shift()["Q_Sto_Energy"]) / (time_step * 3600)
        utils.check_results_correctness(df)
    df.to_excel(save_path)
    if get_df:
        return df
    return save_path
