import logging
import math
import pathlib

import numpy as np
from pyomo.environ import (
    Var,
    ConcreteModel, Constraint, Objective, minimize, SolverFactory
)
from pyomo.core import NonNegativeReals, Reals, NonPositiveReals, Binary
from pyomo.opt import SolverStatus, TerminationCondition

from bes_rules.utils import create_or_append_list
from bes_rules.utils.functions import calculate_storage_surface_area

logger = logging.getLogger(__name__)


def heating_curve(TOutdoorAir):
    THeaCur = 293 + (55 - 20) / (-12 - 20) * (TOutdoorAir - 293.15)
    if isinstance(TOutdoorAir, float):
        return max(291.15, THeaCur)
    THeaCur[THeaCur < 293.15] = 293.15
    return THeaCur


def create_model(simulation_settings, time_series_inputs, model_parameters, iteration, TBufSetInit, T_DHW_Init, with_dhw,
                 minimal_part_load_heat_pump) -> ConcreteModel:
    # Set parameter
    start_time = int(simulation_settings["start_time"] / simulation_settings["time_step"])
    prediction_horizon = simulation_settings["prediction_horizon"]
    total_runtime = int(simulation_settings["total_runtime"])
    time = list(range(int(prediction_horizon / simulation_settings["time_step"])))  # 24h/0.25h --> läuft von: 0-95
    seconds_in_hour = 3600  # in [s/h]!!!
    time_step = simulation_settings["time_step"]
    start_time_hour = int(start_time * time_step)

    rho_water = 995  # [kg/m^3]
    cp_water = 4184  # [J/kgK]

    # Parameters
    # Heat pump scaling
    m_flow_HP = model_parameters["hydraulic.generation.m_flow_nominal[1]"]
    scalingFactor = model_parameters["scalingFactor"]
    # Heating Rod
    eta_HR = model_parameters["hydraulic.generation.parEleHea.eta"]
    # Buffer
    V_Sto = model_parameters["hydraulic.distribution.parStoBuf.V"]
    QBuf_loss_flow = model_parameters["hydraulic.distribution.parStoBuf.QLoss_flow"]  # in W
    dTBuf_nominal = (
            model_parameters["hydraulic.distribution.parStoBuf.T_m"] -
            model_parameters["hydraulic.distribution.parStoBuf.TAmb"]
    )
    UABuf = QBuf_loss_flow / dTBuf_nominal
    sInsBuf = model_parameters["hydraulic.distribution.parStoBuf.sIns"]
    # DHW
    VStoDHW = model_parameters["hydraulic.distribution.parStoDHW.V"]
    QDHW_loss_flow = model_parameters["hydraulic.distribution.parStoDHW.QLoss_flow"]  # in W
    dTDHW_nominal = (
            model_parameters["hydraulic.distribution.parStoDHW.T_m"] -
            model_parameters["hydraulic.distribution.parStoDHW.TAmb"]
    )
    UADHW = QDHW_loss_flow / dTDHW_nominal
    sInsDHW = model_parameters["hydraulic.distribution.parStoDHW.sIns"]

    Q_HR_max = model_parameters["hydraulic.generation.eleHea.Q_flow_nominal"]

    # Set Heat storage parameters
    TBufSet_max = model_parameters.get("TBufSet_max", 95 + 273.15)  # [K]
    m_Sto_water = V_Sto * rho_water  # [kg]
    TBufSet_Env = model_parameters.get("TBufSet_Env", 18 + 273.15)  # [K]
    A_Sto = calculate_storage_surface_area(
        V=V_Sto,
        h_d=model_parameters.get("h_d_ratio_sto", 2),
        sIns=sInsBuf,
    )
    U_Sto = UABuf / A_Sto

    # Set DHW Parameters:
    V_DHW = VStoDHW  # [m^3]
    m_DHW_water = V_DHW * rho_water  # [kg]
    T_DHW_Max = model_parameters.get("T_DHW_Max", 95 + 273.15)  # [K]
    A_DHW = calculate_storage_surface_area(
        h_d= model_parameters.get("h_d_ratio_dhw", 2),
        V=V_DHW,
        sIns=sInsDHW
    )
    U_DHW = UADHW / A_DHW  # [W/K]
    T_DHW_Min = model_parameters.get("T_DHW_Min", 273.15 + 18)  # [K]
    T_DHW_Soll = model_parameters.get("T_DHW_Soll", 273.15 + 50)  # [K]

    # Set Consumer parameter
    T_Hou_Gre = model_parameters.get("T_Hou_Gre", 273.15 + 15)  # [K]
    T_VL_min = model_parameters.get("T_Hou_VL_min", 273.15 + 10)  # neu von mir [K]
    T_HP_max = model_parameters.get("T_Hou_VL_max", 273.15 + 70)  # neu von mir [K]

    def interpolate_if_necessary(_values, _time_step):
        from ebcpy.preprocessing import get_df_index_frequency_mean_and_std
        frequency, std = get_df_index_frequency_mean_and_std(_values.index)
        if std > 0:
            logger.warning("Input predictions are not equally sampled, index has a std greater zero.")
        _time_step_predictions = frequency / 3600
        length_predictions_h = _values.index[-1] / 3600
        if _time_step_predictions - _time_step != 0:
            x_time_step = np.arange(0.0, length_predictions_h + 0.1, _time_step)
            x_time_step_predictions = np.arange(0.0, length_predictions_h + 0.1, _time_step_predictions)
            return np.interp(x_time_step, x_time_step_predictions, _values)
        return _values.values

    # Inputs: P_PV, T_Air, P_EL_Dem, Q_Hou_Dem, c_grid, c_feed_in, Q_DHW_Dem
    P_PV = time_series_inputs.loc[:, "P_PV"]  # [W]
    T_outdoor_air = time_series_inputs.loc[:, "T_Air"]  # [K]  Außentemperatur
    P_EL_Dem = time_series_inputs.loc[:, "P_El_Dem"]  # [W]
    Q_house_demand_input = time_series_inputs.loc[:, "Q_Hou_Dem"]  # [W] Heat Demand of House from simulation
    c_grid = time_series_inputs.loc[:, "c_grid"] / 1000  # [€/Wh]
    c_feed_in = time_series_inputs.loc[:, "c_feed_in"] / 1000  # [€/Wh]
    THeaCur = time_series_inputs.loc[:, "THeaCur"]  # [K] Heating Curve
    if with_dhw:
        Q_DHW_demand = time_series_inputs.loc[:, "Q_DHW_Dem"] / time_step  # [W].
        # Convert DHW demand based on number of people:
        mean_per_day = sum(Q_DHW_demand[:int(24 / time_step)])
        dhw_Wh_per_person = 1.45 * 1000
        n_persons = 4
        Q_DHW_demand *= dhw_Wh_per_person * n_persons / mean_per_day
        Q_DHW_demand = interpolate_if_necessary(Q_DHW_demand, time_step)

    # Interpolation of hourly Input-Data: von Stunde zu time_step
    # 366 Tage * 4 Einträge --> Alle 15 Minuten: +1Tag um +1Tag in die Zukunft zu schauen.

    P_PV = interpolate_if_necessary(P_PV, time_step)
    T_outdoor_air = interpolate_if_necessary(T_outdoor_air, time_step)
    THeaCur = interpolate_if_necessary(THeaCur, time_step)
    P_EL_Dem = interpolate_if_necessary(P_EL_Dem, time_step)
    Q_house_demand_input = interpolate_if_necessary(Q_house_demand_input, time_step)
    c_grid = interpolate_if_necessary(c_grid, time_step)
    c_feed_in = interpolate_if_necessary(c_feed_in, time_step)

    # Most inputs are not part of the model but should be saved for analysis
    inputs_to_store_in_df = {
        "P_PV": P_PV,
        "T_outdoor_air": T_outdoor_air,
        "THeaCur": THeaCur,
        "P_EL_Dem": P_EL_Dem,
        "c_grid": c_grid,
        "c_feed_in": c_feed_in
    }
    if with_dhw:
        inputs_to_store_in_df["Q_DHW_demand"] = Q_DHW_demand

    # Calculation of Mean-Temperature of this day and the next day
    for i in range(0, total_runtime, int(24 / time_step)):
        start_index = start_time
        start_index_next = start_index + 96  # "4 Tage in der Zukunft bezogen auf start_time"
        if start_time_hour % 24 != 0:
            start_index -= int((start_time_hour % 24) / time_step)
            start_index_next -= int(((start_time_hour + 24) % 24) / time_step)
        T_Mean = sum(T_outdoor_air[start_index: start_index + int(24 / time_step)]) / int(24 / time_step)
        # What happens here at Day 365?
        T_Mean_Next = sum(T_outdoor_air[start_index_next: start_index_next + int(24 / time_step)]) / int(24 / time_step)

    if TBufSetInit is None:
        TBufSetInit = THeaCur[start_time] + 0.2  # To avoid infeasible starts

    model = ConcreteModel()
    model.Q_Hou = Var(time, within=NonNegativeReals, name="Q_Hou")  # , initialize=initials_test["Q_Hou"])
    model.Q_HP = Var(time, within=NonNegativeReals, name="Q_HP")
    model.Q_HP_Max = Var(time, within=NonNegativeReals, name="Q_HP_Max")
    if minimal_part_load_heat_pump > 0:
        model.Q_HP_Min = Var(time, within=NonNegativeReals, name="Q_HP_Min")
    model.Q_HR = Var(time, within=NonNegativeReals, name="Q_HR", bounds=(0, Q_HR_max))
    # Q_house_demand=Q_hou+Q_penalty
    model.Q_Penalty = Var(time, within=NonNegativeReals, name="Q_Penalty")
    model.Q_Sto_Loss = Var(time, within=Reals, name="Q_Sto_Loss")
    model.Q_Sto_Energy = Var(time, within=Reals, name="Q_Sto_Energy")
    model.Q_Sto_Power_max = Var(time, within=NonNegativeReals, name="Q_Sto_Power_max")
    model.Q_house_demand = Var(time, within=Reals, name="Q_house_demand")
    # minimale/maximale VL-Temperatur eingeführt
    model.T_VL_HP = Var(time, within=NonNegativeReals, name="T_VL_HP", bounds=(T_VL_min, T_HP_max))
    # --> jetzt neu, da T_VL-Erhöhung nicht bestraft wird!
    model.T_VL_HR = Var(time, within=NonNegativeReals, name="T_VL_HR", bounds=(T_VL_min, 373.15))
    model.T_RL_HP = Var(time, within=NonNegativeReals, name="T_RL_HP", bounds=(283.15, 368.15))
    # Pufferspeichertemperatur and boundaries
    model.TBufSet = Var(time, within=NonNegativeReals, bounds=(TBufSet_Env, TBufSet_max), name="TBufSet")

    model.P_EL_HP = Var(time, within=NonNegativeReals, name="P_EL_HP")
    model.P_EL_HR = Var(time, within=NonNegativeReals, name="P_EL_HR", bounds=(0, Q_HR_max / eta_HR))
    # =Bedarf(Haus+WP+HR)-PV
    model.P_EL = Var(time, within=Reals, name="P_EL")
    model.costs_total_ph = Var(within=Reals, name="costs_total_ph", initialize=0)
    model.c_penalty = Var(time, within=NonNegativeReals, name="c_penalty")
    model.c_revenue = Var(time, within=NonPositiveReals, name="c_revenue")
    model.c_el_power = Var(time, within=NonNegativeReals, name="c_el_power")
    model.H_off = Var(time, within=Binary, name="H_off")
    model.H_PS = Var(time, within=Binary, name="H_PS")
    model.COP_HP = Var(time, within=NonNegativeReals, name="COP_HP")
    # If PV>EL-Demand
    model.No_Feed_In = Var(time, within=Binary, name="No_Feed_In")
    # VL-RL
    if with_dhw:
        model.Mode = Var(time, within=Reals, name="Mode")
        model.Q_DHW_Loss = Var(time, within=NonNegativeReals, name="Q_DHW_Loss")
        # DHW-Temperatur, boundaries?
        model.T_DHW = Var(time, within=NonNegativeReals, name="T_DHW")
        model.Q_DHW_Dem = Var(time, within=NonNegativeReals, name="Q_DHW_Dem")
        # =mode3:T_VL=65GradC --> Allgemeiner Modus DHW
        model.H_DHW = Var(time, within=Binary, name="H_DHW")
        model.Q_DHW_Max = Var(time, within=NonNegativeReals, name="Q_DHW_Max")
        # DHW_Penalty == DHW_Low, ist 1 oder 0
        model.DHW_Penalty = Var(time, within=Binary, name="DHW_Penalty")
        model.Q_Penalty_DHW = Var(time, within=NonNegativeReals, name="Q_Penalty_DHW")

    def minimal_storage_temperature_no_heating_but_next_day_heating(m: ConcreteModel, t: int):
        if t >= 22:
            return m.TBufSet[t] >= 273.15 + 32  # pre-heating to avoid infeasibility
        return m.TBufSet[t] >= TBufSet_Env

    def minimal_storage_temperature_heating(m: ConcreteModel, t: int):
        # Mindesthauswärmebedarf T_VL_HK
        return m.TBufSet[t] >= THeaCur[t + start_time]

    def minimal_storage_temperature_no_heating(m: ConcreteModel, t: int):
        # Untergrenze für TBufSet: TBufSet_Env (18GradC), wenn nächster Tag auch warm genug ist
        return m.TBufSet[t] >= TBufSet_Env

    def house_demand_no_heating(m: ConcreteModel, t: int):
        # Read T-Mean for each time-step and Set Q-Hou_Dem = 0, if T_Mean> T_Heiz_Grenz: [W]
        return m.Q_house_demand[t] == 0

    def house_demand_heating(m: ConcreteModel, t: int):
        # Read T-Mean for each time-step and Set Q-Hou_Dem = 0, if T_Mean> T_Heiz_Grenz: [W]
        return m.Q_house_demand[t] == Q_house_demand_input[t + start_time]

    def heating_mode(m: ConcreteModel, t: int):
        # Constraint to set Mode: [-]
        # PS oder DHW oder off
        if with_dhw:
            return m.H_off[t] + m.H_PS[t] + m.H_DHW[t] == 1
        return m.H_off[t] + m.H_PS[t] == 1

    def storage_limit_based_on_mode(m: ConcreteModel, t: int):
        # Constraint that TBufSet/T_DHW <= TVL_HP: [K]
        if with_dhw:
            return m.H_PS[t] * m.TBufSet[t] + m.H_DHW[t] * m.T_DHW[t] + m.H_off[t] * TBufSet_Env <= m.TBufSet[t]
        return m.H_PS[t] * m.TBufSet[t] + m.H_off[t] * TBufSet_Env <= m.TBufSet[t]

    def actual_heat_pump_thermal_power(m: ConcreteModel, t: int):
        # Real HP-Heating: [W]
        return m.Q_HP[t] == m_flow_HP * cp_water * (m.T_VL_HP[t] - m.T_RL_HP[t])

    def return_temperature_to_heat_pump(m: ConcreteModel, t: int):
        # Connect RL-Temperature: [K]
        if with_dhw:
            return m.T_RL_HP[t] == m.H_off[t] * m.TBufSet[t] + m.H_PS[t] * m.TBufSet[t] + m.H_DHW[t] * m.T_DHW[t]
        return m.T_RL_HP[t] == m.TBufSet[t]

    def model_electric_heater_off(m: ConcreteModel, t: int):
        # Constraint1 to set Mode On [-]
        return m.H_off[t] * (m.TBufSet[t] - m.T_VL_HP[t]) == 0

    def mode_heat_pump_off(m: ConcreteModel, t: int):
        # Constraint2 to set Mode On [-]
        return m.H_off[t] * (m.T_VL_HP[t] - m.T_RL_HP[t]) == 0

    def COP(m: ConcreteModel, t: int):
        # COP [-] from 2D linear Regression (Data from Optihorst) (Score: 0.9263990436920592)
        # Old from optihorst
        # return (m.COP_HP[t] == 3.853293030788175 + 0.028072655651649094 * (
        #        T_Input[t + start_time] - 273.15) - 0.04565155302180306 * (m.T_VL_HP[t] - 273.15))
        TAir = T_outdoor_air[t + start_time]
        return m.COP_HP[t] == (
                965.902231300392 +
                -9.533246340114223 * TAir +
                -1.029123761692837 * m.T_VL_HP[t] +
                0.008767379505684366 * TAir * m.T_VL_HP[t] +
                0.029481697344765898 * TAir ** 2 +
                -1.899325852627404e-05 * TAir ** 2 * m.T_VL_HP[t] +
                -2.6599338912753304e-05 * TAir ** 3
        )

    def calculate_maximal_heat_pump_thermal_power(m: ConcreteModel, t: int):
        TAir = T_outdoor_air[t + start_time]
        return m.Q_HP_Max[t] == (1 - m.H_off[t]) * (
                3189231.4110641507 +
                -32220.980830563 * TAir +
                -1685.3455766577608 * m.T_VL_HP[t] +
                10.644665614582552 * TAir * m.T_VL_HP[t] +
                109.27161006239132 * TAir ** 2 +
                -0.016922003609194197 * TAir ** 2 * m.T_VL_HP[t] +
                -0.12370531548537612 * TAir ** 3
        ) * scalingFactor

    def calculate_minimal_heat_pump_thermal_power(m: ConcreteModel, t: int):
        return m.Q_HP_Min[t] == m.Q_HP_Max[t] * minimal_part_load_heat_pump

    def electrical_power_heat_pump(m: ConcreteModel, t: int):
        # Actual Power from HP: [W] from 2D linear Regression (Data from Optihorst) (Score=0.959665133341799)
        # Old: OptiHorst
        # return (
        #         m.P_EL_HP[t] == (
        #         (1 - m.H_off[t]) * (
        #         3712.992051892565 - 1579.6883107073077 * m.COP_HP[t] +
        #         0.4485744366446532 * m.Q_HP[t]
        # )))
        TAir = T_outdoor_air[t + start_time]
        TCon = m.T_VL_HP[t]
        COP = m.COP_HP[t]
        Q = m.Q_HP[t] / scalingFactor
        return m.P_EL_HP[t] == (1 - m.H_off[t]) * (
                34869.87960221601 +
                -1443.1498146679442 * TAir +
                729.6600500786593 * TCon +
                13.171961611049136 * Q +
                -4403.21602733943 * COP +
                -5.274876390743887 * TAir * TCon +
                -0.08605133504737798 * TAir * Q +
                15.069049149089173 * TAir * COP +
                9.165964285742943 * TAir ** 2 +
                0.009604671358728301 * TAir ** 2 * TCon +
                0.00014481416630599483 * TAir ** 2 * Q +
                -0.00012029652514154305 * TAir ** 2 * COP +
                -0.015974100471851202 * TAir ** 3
        ) * scalingFactor

    def minimal_heat_pump_thermal_power(m: ConcreteModel, t: int):
        # Constraint to minimize the HP-Power: not below 25% of Q_nom: [W]
        return m.Q_HP[t] >= m.Q_HP_Min[t]

    def maximal_heat_pump_thermal_power(m: ConcreteModel, t: int):
        # Constraint to minimize the HP-Power: not above Q_max: [W]
        return m.Q_HP[t] <= m.Q_HP_Max[t]

    def display_heat_pump_mode(m: ConcreteModel, t: int):
        # Constraint to Display the Mode: [-]. 0: off, 1: PS, 2: DHW
        return m.Mode[t] == (0 * m.H_off[t]) + (1 * m.H_PS[t]) + (2 * m.H_DHW[t])

    def actual_electric_heater_thermal_power(m: ConcreteModel, t: int):
        # Real HR-Heating: [W]
        return m.Q_HR[t] == (m_flow_HP * cp_water * (m.T_VL_HR[t] - m.T_VL_HP[t]))

    def electric_power_heating_rod(m: ConcreteModel, t: int):
        # Demand of el. Power by HR: [W]
        return m.P_EL_HR[t] == (m.Q_HR[t] / eta_HR)

    def constraint_actual_heating_demand(m: ConcreteModel, t: int):
        # Constraint to Limit the Q_Hou to the demand: [W] (Q_house_demand oben * time_step und Q_hou auch!)
        return m.Q_Hou[t] <= m.Q_house_demand[t]

    def maximum_heat_flow_from_storage(m: ConcreteModel, t: int):
        # Limit Q_Hou to the maximum Storage Power, as the Storage Power depends
        # on TBufSet and provides the Temperature for Heating: [W]
        return m.Q_Hou[t] <= m.Q_Sto_Power_max[t]

    def penalty_heat_flow_building(m: ConcreteModel, t: int):
        # Define Penalty if Q_Hou can not be reached: [W]
        # Housedemand(Input) = Strafe + Q_hou(MPC)
        return m.Q_Hou[t] + m.Q_Penalty[t] == m.Q_house_demand[t]

    def space_heating_storage_model(m: ConcreteModel, t: int):
        # Calculation of Temperature in Storage in current time step: Temperature in [K], but EB in: [Wh/h=W]
        if t == 0:
            TBufSet_last = TBufSetInit
        else:
            TBufSet_last = m.TBufSet[t - 1]
        m_cp_div_time = m_Sto_water * cp_water / (seconds_in_hour * time_step)
        heat_generation = (m.Q_HR[t] + m.Q_HP[t]) * m.H_PS[t]
        return m.TBufSet[t] == TBufSet_last + (heat_generation - m.Q_Hou[t] - m.Q_Sto_Loss[t]) / m_cp_div_time

    def space_heating_storage_energy(m: ConcreteModel, t: int):
        # Calculation of Energy in Storage [J]
        return m.Q_Sto_Energy[t] == (
                m_Sto_water * cp_water * (m.TBufSet[t] - TBufSet_Env)
        )

    def space_heating_storage_losses(m: ConcreteModel, t: int):
        # Calculation of Heat-Loss during storage time: [W]
        return m.Q_Sto_Loss[t] == U_Sto * A_Sto * (m.TBufSet[t] - TBufSet_Env)

    def maximum_storage_power(m: ConcreteModel, t: int):
        # Maximum usable Power by Storage: [W] --> Wird oben mit Q_hou [W] verglichen.
        return (m.Q_Sto_Power_max[t] == (
                m_Sto_water * cp_water * (m.TBufSet[t] - TBufSet_Env) /
                (seconds_in_hour * time_step)))  # T_sto-32GradC(usable)

    def dhw_storage_losses(m: ConcreteModel, t: int):
        # Calculation of Heat-Loss in DHW-Storage: [W].
        return m.Q_DHW_Loss[t] == U_DHW * A_DHW * (m.T_DHW[t] - TBufSet_Env)

    def dhw_storage_model(m: ConcreteModel, t: int):
        # Energy-Balance of DHW-Storage: [W]
        if t >= 1:
            return ((m.T_DHW[t] - m.T_DHW[t - 1]) * m_DHW_water * cp_water / (seconds_in_hour * time_step) == (
                    (m.Q_HR[t] + m.Q_HP[t]) * m.H_DHW[t]) - Q_DHW_demand[t + start_time] - m.Q_DHW_Loss[t])
        if iteration == 0:
            return m.T_DHW[t] == T_DHW_Init
        return (
                ((m.T_DHW[t] - T_DHW_Init) * m_DHW_water * cp_water) / (seconds_in_hour * time_step) == (
                (m.Q_HR[t] + m.Q_HP[t]) * m.H_DHW[t]) - Q_DHW_demand[t + start_time] - m.Q_DHW_Loss[t]
        )

    def minimal_dhw_temperature(m: ConcreteModel, t: int):
        return m.T_DHW[t] >= T_DHW_Min

    def activate_dhw_penalty(m: ConcreteModel, t: int):
        # Sobald T_DHW<T_Soll: muss T_penalty = 1.
        return (T_DHW_Soll - m.T_DHW[t]) * (1 - m.DHW_Penalty[t]) <= 0

    def deactivate_dhw_penalty(m: ConcreteModel, t: int):
        # Sobald T_DHW>T_soll muss T_penalty = 0 werden.
        return (T_DHW_Soll - m.T_DHW[t]) * m.DHW_Penalty[t] >= 0

    def dhw_penalty(m: ConcreteModel, t: int):
        # Q_penalty für Kostenfnkt: [W] wenn DHW_penalty == 1
        return (
                m.Q_Penalty_DHW[t] == m.DHW_Penalty[t] * m_DHW_water *
                cp_water * (T_DHW_Soll - m.T_DHW[t]) / (seconds_in_hour * time_step)
        )

    def maximal_dhw_storage_tempeature(m: ConcreteModel, t: int):
        # Temperature in DHW-Storage never above T_HP_VL_Max: [K]
        return m.T_DHW[t] <= T_DHW_Max

    def power_balance(m: ConcreteModel, t: int):
        # Sum of all electrical demands and generation
        return m.P_EL[t] == (m.P_EL_HR[t] + P_EL_Dem[t + start_time] + m.P_EL_HP[t]) - P_PV[t + start_time]

    def activate_feed_in_binary(m: ConcreteModel, t: int):
        # Binary Variable to know when there is more PV-Generation than EL-Demand: ==0
        # wenn PV>Dem und ==1 wenn Dem>PV. Wenn möglich werden Speicher voll
        # geladen und danach erst eingespeist!
        return m.P_EL[t] * m.No_Feed_In[t] >= 0

    def deactivate_feed_in_binary(m: ConcreteModel, t: int):
        return m.P_EL[t] * (1 - m.No_Feed_In[t]) <= 0

    def costs_electricity(m: ConcreteModel, t: int):
        # Calculation of Cost for Power: [€]
        return m.c_el_power[t] == m.No_Feed_In[t] * c_grid[t + start_time] * m.P_EL[t] * time_step

    def revenue_for_power(m: ConcreteModel, t: int):
        # Calculation of Revenue for PV-Power: [€]
        return m.c_revenue[t] == (1 - m.No_Feed_In[t]) * c_feed_in[t + start_time] * m.P_EL[t] * time_step

    def costs_of_penalty(m: ConcreteModel, t: int):
        # Calculation of Penalty-Costs due to Discomfort: [€]
        c_comfort = c_grid[t + start_time] * 20  # Costs twenty times as much as the electricity
        if with_dhw:
            return m.c_penalty[t] == (m.Q_Penalty[t] + m.Q_Penalty_DHW[t]) * c_comfort * time_step
        return m.c_penalty[t] == m.Q_Penalty[t] * c_comfort * time_step

    def cost_in_prediction_horizon(m: ConcreteModel, t: int):
        # Calculation of sum of all costs per prediction-horizon: [€]
        return m.costs_total_ph == sum(m.c_el_power[t] + m.c_revenue[t] + m.c_penalty[t] for t in time)

    def objective_rule(m: ConcreteModel):
        return m.costs_total_ph

    constraints = [
        heating_mode,
        storage_limit_based_on_mode,
        actual_heat_pump_thermal_power,
        actual_electric_heater_thermal_power,
        return_temperature_to_heat_pump,
        model_electric_heater_off,
        mode_heat_pump_off,
        COP,
        electrical_power_heat_pump,
        electric_power_heating_rod,
        calculate_maximal_heat_pump_thermal_power,
        maximal_heat_pump_thermal_power,
        constraint_actual_heating_demand,
        maximum_heat_flow_from_storage,
        penalty_heat_flow_building,
        space_heating_storage_model,
        space_heating_storage_energy,
        space_heating_storage_losses,
        maximum_storage_power,
        power_balance,
        activate_feed_in_binary,
        deactivate_feed_in_binary,
        costs_electricity,
        revenue_for_power,
        costs_of_penalty,
        cost_in_prediction_horizon
    ]
    if minimal_part_load_heat_pump > 0:
        constraints.append(calculate_minimal_heat_pump_thermal_power)
        constraints.append(minimal_heat_pump_thermal_power)
    if with_dhw:
        constraints.extend([
            maximal_dhw_storage_tempeature,
            dhw_storage_losses,
            dhw_storage_model,
            minimal_dhw_temperature,
            activate_dhw_penalty,
            deactivate_dhw_penalty,
            dhw_penalty,
            display_heat_pump_mode
        ])
    # Define what happens, if the Mean Temperature of a Day is above the Heating Limit Temperature(15GradC):
    if T_Mean >= T_Hou_Gre:
        constraints.append(house_demand_no_heating)
        if T_Mean_Next >= T_Hou_Gre:
            constraints.append(minimal_storage_temperature_no_heating)
        else:  # Am nächsten Tag muss wieder geheizt werden
            constraints.append(minimal_storage_temperature_no_heating_but_next_day_heating)
    else:
        constraints.append(minimal_storage_temperature_heating)
        constraints.append(house_demand_heating)

    for constraint in constraints:
        model.__setattr__(constraint.__name__, Constraint(
            time, rule=constraint, name=constraint.__name__
        ))

    model.total_costs = Objective(rule=objective_rule, sense=minimize, name="Minimize total costs")

    return model, inputs_to_store_in_df


def solve_model(model: ConcreteModel, solver_kwargs: dict, save_path: pathlib.Path, model_id: str):
    ####################--- 9. Set Up of Solver ---####################
    solver = SolverFactory("gurobi")
    solver.options["Presolve"] = 1
    solver.options["mipgap"] = solver_kwargs.get("MIP_gap", 0.03)
    solver.options["TimeLimit"] = solver_kwargs.get("TimeLimit", 30)
    solver.options["DualReductions"] = 0
    # neu
    # solver.options["check_constraints"] = 1

    # write an ILP file to print the IIS
    solver_parameters = "ResultFile=" + save_path.joinpath(f"model_{model_id}.ilp").absolute().as_posix()

    result = solver.solve(model, warmstart=True, tee=True, symbolic_solver_labels=True, options_string=solver_parameters)

    if (result.solver.status == SolverStatus.ok) and (
            result.solver.termination_condition == TerminationCondition.optimal):
        # model.display()
        logger.info("Model successfully solved.")
    elif (
            result.solver.termination_condition == TerminationCondition.infeasible or result.solver.termination_condition == TerminationCondition.other):
        logger.error("Model is infeasible. Check Constraints")
        with open(save_path.joinpath(f"model_{model_id}_pp.txt"), "w+") as file:
            model.pprint(ostream=file)
    else:
        logger.warning("Solver status is: %s", result.solver.status)
        logger.warning("TerminationCondition is: %s", result.solver.termination_condition)

    model_results = {}
    for k, v in model.__dict__.items():
        if not isinstance(v, Var):
            continue
        try:
            _ = v[0]
            model_results[k] = v
        except Exception:
            continue
    return model_results


def run_milp_model(
        simulation_settings,
        solver_kwargs,
        time_series_inputs,
        model_parameters,
        iteration,
        TBufSetInit,
        T_DHW_Init,
        with_dhw,
        minimal_part_load_heat_pump,
        save_path_result: pathlib.Path
):
    results_horizon = int((simulation_settings["control_horizon"] / simulation_settings["time_step"]))
    model, inputs_to_store_in_df = create_model(simulation_settings, time_series_inputs, model_parameters, iteration,
                                                TBufSetInit, T_DHW_Init, with_dhw, minimal_part_load_heat_pump)
    if solver_kwargs is None:
        solver_kwargs = {}

    model_results = solve_model(
        model=model, solver_kwargs=solver_kwargs,
        save_path=save_path_result.parent, model_id=save_path_result.stem + f"iteration={iteration}"
    )
    res_control_horizon = {}
    start_time = int(simulation_settings["start_time"] / simulation_settings["time_step"])
    for t in range(results_horizon):
        for var, model_var in model_results.items():
            _value = model_var[t].value
            res_control_horizon = create_or_append_list(res_control_horizon, var, _value)

        for key, var in inputs_to_store_in_df.items():
            res_control_horizon = create_or_append_list(res_control_horizon, key, var[t + start_time])

        if with_dhw:
            mode = model_results["Mode"][t].value
            res_control_horizon = create_or_append_list(res_control_horizon, "actExtBufCtrl", mode == 1)
            res_control_horizon = create_or_append_list(res_control_horizon, "actExtDHWCtrl", mode == 2)
        else:
            res_control_horizon = create_or_append_list(res_control_horizon, "actExtBufCtrl", True)
            res_control_horizon = create_or_append_list(res_control_horizon, "actExtDHWCtrl", False)
    return res_control_horizon
