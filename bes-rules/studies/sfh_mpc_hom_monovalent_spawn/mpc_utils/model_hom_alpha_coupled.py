"""
Model taken from here
https://git-ce.rwth-aachen.de/ebc/projects/ACS_EBC0022_ERANET_I-Greta_bsc/optimization/-/tree/fwu-pkr
"""

import logging
import casadi as ca
from typing import List, Any,  Dict
import pandas as pd
from pydantic import ConfigDict


from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from bes_rules.simulation_based_optimization.agentlib_mpc.calc_resistances import calc_resistances
from bes_rules.simulation_based_optimization.agentlib_mpc.get_idf_data import get_idf_data
from bes_rules.simulation_based_optimization.agentlib_mpc import component_models

logger = logging.getLogger(__name__)

# storage layers
N_LAYER = 4

multizone_coupled: bool = True
HOM_predictor: bool = True
test_case: bool = False
calc_resistances_new: bool = True

# zones
#zones = ["livingroom", "kitchen", "hobby", "wcstorage", "corridor", "bedroom", "children", "corridor2", "bath", "children2", "attic"]
zones = ["livingroom", "kitchen", "hobby", "wcstorage", "corridor", "bedroom", "children", "corridor2", "bath", "attic", "children2"]
#zones = ["one", "two"]


class MPCModelConfig(CasadiModelConfig):
    model_config = ConfigDict(validate_assignment=True, extra="allow")
    bes_parameters: dict
    zone_parameters: dict
    #test_case: bool = False

    nLayer: int = N_LAYER
    #zones: dict = zones

    if test_case:
        inputs: List[CasadiInput] = [
            # controls
            CasadiInput(name="TBufSet", value=323.15, unit="K",
                        description="Set temperature of buffer storage, sink temperature of heat pump"),
            # CasadiInput(name="TSetOneZone", value=293.15, unit="K",
            #             description="Set room temperature for one zone model"),
            # CasadiInput(name="yValSet", value=0, unit="-",
            #             description="set point for electric heater (not used, only for interest)", lb=0, ub=1
            #             ),
        ] + [
            CasadiInput(name=f"TSetOneZone_{z}", value=293.15, unit="K", description=f"Setpoint temp in {z}") for z in zones
        ] + [
            CasadiInput(name=f"yValSet_{z}", value=0, unit="-", description=f"Valve control in {z}", lb=0, ub=1) for z in zones
        ] + [
            CasadiInput(name="yEleHeaSet", value=0, unit="-",
                        description="set point for electric heater (not used, only for interest)", lb=0, ub=1
                        ),
            #CasadiState(name="s_valve_pos_lb", value=0, unit="-",
            #            description="allowing to close the valve"),

            # disturbances
            CasadiInput(name="P_el_pv", value=0, unit="W",
                        description="electricity produced by photovoltaic unit"),
            CasadiInput(name="Q_RadSol", value=0, unit="W",
                        description="Radiative solar heat for all orientations"),
            CasadiInput(name="T_amb", value=273.15, unit="K",
                        description="Ambient temperature on the outside"),
            CasadiInput(name="THeaCur", value=293.15, unit="K",
                        description="Heat curve temperature"),
            #CasadiInput(name="redFac", value=1, unit="-",
            #            description="redFac"),
            CasadiInput(name="T_preTemWin", value=294.15, unit="K",
                        description="Outdoor surface temperature of window"),
            CasadiInput(name="T_preTemWall", value=294.15, unit="K",
                        description="Outdoor surface temperature of wall"),
            CasadiInput(name="T_preTemRoof", value=294.15, unit="K",
                        description="Outdoor surface temperature of roof"),
            CasadiInput(name="T_preTemFloor", value=286.15, unit="K",
                        description="Outdoor surface temperature of floor"),

            CasadiInput(name="schedule_human", value=0.1, unit="",
                        description="schedule of humans"),
            CasadiInput(name="schedule_dev", value=0, unit="",
                        description="schedule of devices"),
            CasadiInput(name="schedule_light", value=0, unit="",
                        description="schedule of light"),
            CasadiInput(name="SGReadySignal", value=2, unit="",
                        description="SGReady signal for heat pump operation"),
            # costs
            #CasadiInput(name="c_feed_in", value=0.082, unit="€/kWh",
            #            description="Power Price"),
            #CasadiInput(name="c_grid", value=0.3606, unit="€/kWh",
            #            description="Power Price"),
        ]

        states: List[CasadiState] = [
            # differential
            #CasadiState(name="TBuf", value=323.15, unit="K",
            #            description="Temperature of buffer storage"),
            #CasadiState(name="yValState", value=1, unit="-",
            #            description="state variable of vale position"),
            #CasadiState(name="yValSet", value=0, unit="-",
            #            description="set value of valve position"),

        ] + [
            CasadiState(name=f"T_Air_{z}", value=293.15, unit="K", description=f"Air temp in {z}") for z in zones
        ] + [
            CasadiState(name=f"T_IntWall_{z}", value=293.15, unit="K", description=f"Inner wall temp in {z}") for z in zones
        ] + [
            CasadiState(name=f"T_ExtWall_{z}", value=293.15, unit="K", description=f"Outer wall temp in {z}") for z in zones
        ] + [
            CasadiState(name=f"T_Roof_{z}", value=293.15, unit="K", description=f"Roof temp in {z}") for z in zones
        ] + [
            CasadiState(name=f"T_Floor_{z}", value=293.15, unit="K", description=f"Floor temp in {z}") for z in zones
        ] + [
            # slack variables
        ] + [
            CasadiState(name=f"TSetOneZoneDiff_lb_slack_{z}", value=0, unit="K",
                        description=f"Slack variable for TSetOneZone - T_Air in {z} (lower bound)") for z in zones
        ] + [
            CasadiState(name=f"TSetOneZoneDiff_ub_slack_{z}", value=0, unit="K",
                        description=f"Slack variable for TSetOneZone - T_Air in {z} (upper bound)")  for z in zones
        ] + [
            CasadiState(name=f"TAir_ub_slack_{z}", value=0, unit="K",
                        description=f"Slack variable for upper air temp in {z}") for z in zones
        ] + [
            CasadiState(name=f"TAir_lb_slack_{z}", value=0, unit="K",
                        description=f"Slack variable for lower air temp in {z}") for z in zones
        ] + [
            CasadiState(name="TTraSup_slack", value=0, unit="K",
                        description="Slack variable for heater temperature"),
            CasadiState(name="THeaPumSup_slack", value=0, unit="K",
                        description="Slack variable for heat pump sink temperature"),
            CasadiState(name="QHeaPum_flow_slack", value=0, unit="W",
                        description="Slack variable for heat pump"),
            CasadiState(name="PEleHeaPum_slack", value=0, unit="W",
                        description="Slack variable for heat pump power"),
            #CasadiState(name="yValSet_slack", value=0, unit="-",
            #            description="slack variable for yValSet"),
            CasadiState(name="s_Pel_feed", value=0, unit="-",
                        description="slack variable for Pel balance"),
            #CasadiState(name="T_Air_next_predicted", value=293.15, unit="K",
            #            description="Predicted air temperature for next time step"),

            CasadiState(name="P_el_feed_into_grid", value=0, unit="W",
                        description="electricity feed-in"),
            CasadiState(name="P_el_feed_from_grid", value=0, unit="W",
                        description="electricity feed_out"),
            #CasadiState(name="s_valve_pos_ub", value=0, unit="-",
            #            description="slack variable for valve positions greater than 1"),

        ] + [
            # storage differential
            CasadiState(name=f'T_TES_{n}', value=323.15, unit="K") for n in range(1, N_LAYER + 1)
        ] + [
            # storage slack
            CasadiState(name=f's_T_TES_{n}', value=0, unit="K") for n in range(1, N_LAYER + 1)
        ]

        parameters: List[CasadiParameter] = [
            #CasadiParameter(name="nLayer", value=4, unit="", description="Storage Layer"),
            CasadiParameter(name="AZone", unit="m^2",
                            description="zone area"),
            CasadiParameter(name="VAir", unit="m^3",
                            description="Air volume of thermal zone"),
            CasadiParameter(name="air_rho", unit="kg/m**3",
                            description="density of air"),
            CasadiParameter(name="air_cp", unit="J/kg*K",
                            description="thermal capacity of air"),
            CasadiParameter(name="CAir", unit="J/kg*K",
                            description="Heat capacities of air"),
            CasadiParameter(name="CRoof", unit="J/K",
                            description="Heat capacities of roof"),
            CasadiParameter(name="CExt", unit="J/K",
                            description="Heat capacities of exterior walls"),
            CasadiParameter(name="CInt", unit="J/K",
                            description="Heat capacities of interior walls"),
            CasadiParameter(name="CFloor", unit="J/K",
                            description="Heat capacities of floor"),
            CasadiParameter(name="hConRoofOut", unit="W/(m^2*K)",
                            description="Roof's convective coefficient of heat transfer (outdoor)"),
            CasadiParameter(name="hConRoof", unit="W/(m^2*K)",
                            description="Roof's convective coefficient of heat transfer (indoor)"),
            CasadiParameter(name="RRoof", unit="K/W",
                            description="Resistances of roof, from inside to outside"),
            CasadiParameter(name="RRoofRem", unit="K/W",
                            description="Resistance of remaining resistor between capacity n and outside"),
            CasadiParameter(name="hConExt", unit="W/(m^2*K)",
                            description="External walls convective coefficient of heat transfer (indoor)"),
            CasadiParameter(name="RExt", unit="K/W",
                            description="Resistances of external walls, from inside to middle of wall"),
            CasadiParameter(name="hConWallOut", unit="W/(m^2*K)",
                            description="External walls convective coefficient of heat transfer (outdoor)"),
            CasadiParameter(name="RExtRem", unit="K/W",
                            description="Resistances of external walls, from middle of wall to outside"),
            CasadiParameter(name="hConInt", unit="W/(m^2*K)",
                            description="Internal walls convective coefficient of heat transfer (indoor)"),
            CasadiParameter(name="RInt", unit="K/W",
                            description="Resistances of internal walls, from inside to outside"),
            CasadiParameter(name="hConWin", unit="W/(m^2*K)",
                            description="Windows convective coefficient of heat transfer (indoor)"),
            CasadiParameter(name="hConWinOut", unit="W/(m^2*K)",
                            description="Windows convective coefficient of heat transfer (outdoor)"),
            CasadiParameter(name="RWin", unit="K/W",
                            description="Resistances of windows, from inside to outside"),
            CasadiParameter(name="hConFloor", unit="W/(m^2*K)",
                            description="Floor convective coefficient of heat transfer (indoor)"),
            CasadiParameter(name="RFloor", unit="K/W",
                            description="Resistances of floor, from inside to outside"),
            CasadiParameter(name="RFloorRem", unit="K/W",
                            description="Resistance of floor mass to outside"),
            CasadiParameter(name="hRad", unit="W/(m^2*K)",
                            description="Coefficient of heat transfer for linearized radiation exchange between walls"),
            CasadiParameter(name="hRadRoof", unit="W/(m^2*K)",
                            description="Coefficient of heat transfer for linearized radiation for roof"),
            CasadiParameter(name="hRadWall", unit="W/(m^2*K)",
                            description="Coefficient of heat transfer for linearized radiation for walls"),
            CasadiParameter(name="gWin", unit="-",
                            description="Total energy transmittance of windows"),
            CasadiParameter(name="ratioWinConRad", unit="-",
                            description="Ratio for windows between convective and radiation emission"),
            CasadiParameter(name="AExttot", unit="m^2",
                            description="total external walls area"),
            CasadiParameter(name="AInttot", unit="m^2",
                            description="total internal walls area"),
            CasadiParameter(name="AWintot", unit="m^2",
                            description="total window area"),
            CasadiParameter(name="AFloortot", unit="m^2",
                            description="total floor area"),
            CasadiParameter(name="ARooftot", unit="m^2",
                            description="total roof area"),
            CasadiParameter(name="ATransparent", unit="m^2",
                            description="total transparent area"),
            CasadiParameter(name="activityDegree", unit="met",
                            description="activity Degree of people in met"),
            CasadiParameter(name="specificPeople", unit="1/m^2",
                            description="people per square meter"),
            CasadiParameter(name="ratioConvectiveHeatPeople", unit="-",
                            description="Ratio of convective heat from overall heat output for people"),
            CasadiParameter(name="internalGainsMachinesSpecific", unit="W",
                            description="Heat Flux of machines"),
            CasadiParameter(name="ratioConvectiveHeatMachines", unit="-",
                            description="Ratio of convective heat from overall heat output for machines"),
            CasadiParameter(name="lightingPowerSpecific", unit="W/m^2",
                            description="Heat flux of lighting"),
            CasadiParameter(name="ratioConvectiveHeatLighting", unit="-",
                            description="Ratio of convective heat from overall heat output for lights"),
            CasadiParameter(name="nEle_heater", unit="-",
                            description="Number of elements in heater"),
            CasadiParameter(name="V_Ele_heater", unit="m^3",
                            description="Volume of one heater element"),
            CasadiParameter(name="UA_heater", unit="W/K^n",
                            description="Product of heat transfer coefficient and area of heater"),
            CasadiParameter(name="UA_heater_Ele", unit="W/K^n",
                            description="Product of heat transfer coefficient and area of heater element"),
            CasadiParameter(name="n_heater_exp", unit="-",
                            description="Heater exponent n"),
            CasadiParameter(name="fraRad", unit="-",
                            description="Radiative heat fraction of heater"),
            CasadiParameter(name="mTra_flow_nominal", unit="kg/s",
                            description="Mass flow through heater"),
            CasadiParameter(name="valve_leakage", unit="-",
                            description="Valve leakage"),
            CasadiParameter(name="cp_water", unit="J/kg*K",
                            description="specific heat capacity of water"),
            CasadiParameter(name="rho_water", unit="kg/m^3",
                            description="density of water in heating cycle"),
            CasadiParameter(name="scalingFactor", unit="-",
                            description="scaling factor from bes-rules for heat pump"),
            # costs
            CasadiParameter(name="c_feed_in", value=0.082, unit="€/kWh",
                       description="Power Price"),
            CasadiParameter(name="c_grid", value=0.3606, unit="€/kWh",
            #CasadiParameter(name="c_grid", value=0.0, unit="€/kWh",
                       description="Power Price"),
            # comfort
            CasadiParameter(name="T_Air_ub", value=296.15, unit="K",
                            description="Upper boundary for T_Air", ),
            CasadiParameter(name="T_Air_lb", value=293.15, unit="K",
                            description="Lower boundary for T_Air", ),
            CasadiParameter(name="c_comf_upper", value=1, unit="ct/sK^2",
                            description="weighting factor for discomfort costs"),
            #CasadiParameter(name="c_comf_lower", value=1, unit="ct/sK^2",
            CasadiParameter(name="c_comf_lower", value=1, unit="ct/sK^2",
                            description="weighting factor for discomfort costs"),


            # temperature bounds
            CasadiParameter(name="T_TES_lb", value=293.15, unit="K",
                            description="lower temp. bound for all TES layers"),
            CasadiParameter(name="T_TES_ub", value=343.15, unit="K",
                            description="upper temp. bound for all TES layers"),
            CasadiParameter(name="scale_obj", value=3600, unit="K",
                            description="scale penalties of obj function to reach objective values around 1")
        ]

        outputs: List[CasadiOutput] = [
            CasadiOutput(name="Tamb", value=273.15, unit="K", description="Ambient temperature output"),
            CasadiOutput(name="Qdot_hp", value=0, unit="W"),
            CasadiOutput(name="P_el_hp", value=0, unit="W"),
            CasadiOutput(name="P_el_demand", value=0, unit="W"),
            CasadiOutput(name="P_el_demand_devices", value=0, unit="W"),
            CasadiOutput(name="P_el_demand_lights", value=0, unit="W"),
            CasadiOutput(name="P_pv", value=0, unit="W"),
            CasadiOutput(name="P_el_feed", value=0, unit="W"),
            CasadiOutput(name="T_lb", value=0, unit="K")
        ] + [
          # air heat balance
            CasadiOutput(name=f"Qdot_Air_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_int_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_ext_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_roof_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_floor_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_win_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_gain_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_sol", value=0, unit="W")
        ] + [
            CasadiOutput(name=f"Qdot_Air_heater_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_{z}", value=0, unit="W") for z in zones
        ] + [
            # internal wall
            CasadiOutput(name=f"T_IntWall_sur_out_{z}", value=0, unit="K") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_IntWall_sur_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_IntWall_{z}", value=0, unit="W") for z in zones
        ] + [
            # external wall
            CasadiOutput(name=f"T_ExtWall_sur_out_{z}", value=0, unit="K") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_ExtWall_sur_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"T_ExtWall_pre_{z}", value=0, unit="K") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_ExtWall_pre_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_ExtWall_{z}", value=0, unit="W") for z in zones
        ] + [
          # roof
            CasadiOutput(name=f"T_Roof_sur_out_{z}", value=0, unit="K") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Roof_sur_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"T_Roof_pre_{z}", value=0, unit="K") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Roof_pre_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Roof_{z}", value=0, unit="W") for z in zones
        ] + [
          # floor
            CasadiOutput(name=f"T_Floor_sur_out_{z}", value=0, unit="K") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Floor_sur_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"T_Floor_pre_{z}", value=0, unit="K") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Floor_pre_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Floor_{z}", value=0, unit="W") for z in zones
        ] + [
            # windows
            CasadiOutput(name=f"T_Win_sur_out_{z}", value=0, unit="K") for z in zones
        ] + [
          # heating system
            CasadiOutput(name=f"valve_actual_out_{z}", value=0, unit="-") for z in zones
        ] + [
            CasadiOutput(name=f"mTra_flow_out_{z}", value=0, unit="kg/s") for z in zones
        ] + [
            CasadiOutput(name="TTraSup", value=273.15, unit="K")
        ] + [
            CasadiOutput(name=f"TTraRet_{z}", value=273.15, unit="K") for z in zones
        ] + [
            CasadiOutput(name=f"QTra_flow_out_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"QTraCon_flow_out_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"QTraRad_flow_out_{z}", value=0, unit="W") for z in zones
        ] + [



            # internal gains
            CasadiOutput(name=f"Q_RadSol_or_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_conv_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_conv_machines_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_conv_lights_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_conv_humans_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_rad_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_rad_machines_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_rad_lights_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_rad_humans_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_machines_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_lights_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_humans_{z}", value=0, unit="W") for z in zones
        ] + [
          # ventilation
            CasadiOutput(name=f"ventRate_airExc_{z}", value=0, unit="m^3/s") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_airExc_{z}", value=0, unit="W") for z in zones
        ]
    else:
        inputs: List[CasadiInput] = [
            # controls
            CasadiInput(name="TBufSet", value=323.15, unit="K",
                        description="Set temperature of buffer storage, sink temperature of heat pump"),
        ] + [
            CasadiInput(name=f"yValSet_{z}", value=0, unit="-", description=f"Valve control in {z}", lb=0, ub=1) for z in zones
        ] + [
            CasadiInput(name="yEleHeaSet", value=0, unit="-",
                        description="set point for electric heater (not used, only for interest)", lb=0, ub=1
                        ),


            # disturbances
            CasadiInput(name="P_el_pv", value=0, unit="W",
                        description="electricity produced by photovoltaic unit"),

            CasadiInput(name="T_amb", value=273.15, unit="K",
                        description="Ambient temperature on the outside"),
            CasadiInput(name="THeaCur", value=293.15, unit="K",
                        description="Heat curve temperature"),

            CasadiInput(name="schedule_human", value=0.1, unit="",
                        description="schedule of humans"),
            CasadiInput(name="schedule_dev", value=0, unit="",
                        description="schedule of devices"),
            CasadiInput(name="schedule_light", value=0, unit="",
                        description="schedule of light"),
            CasadiInput(name="SGReadySignal", value=2, unit="",
                        description="SGReady signal for heat pump operation"),
        ]

        states: List[CasadiState] = [
            # differential
        ] + [
            CasadiState(name=f"T_Air_{z}", value=293.15, unit="K", description=f"Air temp in {z}") for z in zones
        ] + [

            # slack variables
        ] + [
            CasadiState(name=f"TAir_ub_slack_{z}", value=0, unit="K",
                        description=f"Slack variable for upper air temp in {z}") for z in zones
        ] + [
            CasadiState(name=f"TAir_lb_slack_{z}", value=0, unit="K",
                        description=f"Slack variable for lower air temp in {z}") for z in zones
        ] + [
            CasadiState(name="TTraSup_slack", value=0, unit="K",
                        description="Slack variable for heater temperature"),
            CasadiState(name="THeaPumSup_slack", value=0, unit="K",
                        description="Slack variable for heat pump sink temperature"),
            CasadiState(name="QHeaPum_flow_slack", value=0, unit="W",
                        description="Slack variable for heat pump"),
            CasadiState(name="PEleHeaPum_slack", value=0, unit="W",
                        description="Slack variable for heat pump power"),
            CasadiState(name="s_Pel_feed", value=0, unit="-",
                        description="slack variable for Pel balance"),
            CasadiState(name="P_el_feed_into_grid", value=0, unit="W",
                        description="electricity feed-in"),
            CasadiState(name="P_el_feed_from_grid", value=0, unit="W",
                        description="electricity feed_out"),
        ] + [
            # storage differential
            CasadiState(name=f'T_TES_{n}', value=323.15, unit="K") for n in range(1, N_LAYER + 1)
        ] + [
            # storage slack
            CasadiState(name=f's_T_TES_{n}', value=0, unit="K") for n in range(1, N_LAYER + 1)
        ]

        parameters: List[CasadiParameter] = [
            #CasadiParameter(name="nLayer", value=4, unit="", description="Storage Layer"),
            CasadiParameter(name="AZone", unit="m^2",
                            description="zone area"),
            CasadiParameter(name="VAir", unit="m^3",
                            description="Air volume of thermal zone"),
            CasadiParameter(name="air_rho", unit="kg/m**3",
                            description="density of air"),
            CasadiParameter(name="air_cp", unit="J/kg*K",
                            description="thermal capacity of air"),
            CasadiParameter(name="CAir", unit="J/kg*K",
                            description="Heat capacities of air"),
            CasadiParameter(name="CRoof", unit="J/K",
                            description="Heat capacities of roof"),
            CasadiParameter(name="CExt", unit="J/K",
                            description="Heat capacities of exterior walls"),
            CasadiParameter(name="CInt", unit="J/K",
                            description="Heat capacities of interior walls"),
            CasadiParameter(name="CFloor", unit="J/K",
                            description="Heat capacities of floor"),
            CasadiParameter(name="hConRoofOut", unit="W/(m^2*K)",
                            description="Roof's convective coefficient of heat transfer (outdoor)"),
            CasadiParameter(name="hConRoof", unit="W/(m^2*K)",
                            description="Roof's convective coefficient of heat transfer (indoor)"),
            CasadiParameter(name="RRoof", unit="K/W",
                            description="Resistances of roof, from inside to outside"),
            CasadiParameter(name="RRoofRem", unit="K/W",
                            description="Resistance of remaining resistor between capacity n and outside"),
            CasadiParameter(name="hConExt", unit="W/(m^2*K)",
                            description="External walls convective coefficient of heat transfer (indoor)"),
            CasadiParameter(name="RExt", unit="K/W",
                            description="Resistances of external walls, from inside to middle of wall"),
            CasadiParameter(name="hConWallOut", unit="W/(m^2*K)",
                            description="External walls convective coefficient of heat transfer (outdoor)"),
            CasadiParameter(name="RExtRem", unit="K/W",
                            description="Resistances of external walls, from middle of wall to outside"),
            CasadiParameter(name="hConInt", unit="W/(m^2*K)",
                            description="Internal walls convective coefficient of heat transfer (indoor)"),
            CasadiParameter(name="RInt", unit="K/W",
                            description="Resistances of internal walls, from inside to outside"),
            CasadiParameter(name="hConWin", unit="W/(m^2*K)",
                            description="Windows convective coefficient of heat transfer (indoor)"),
            CasadiParameter(name="hConWinOut", unit="W/(m^2*K)",
                            description="Windows convective coefficient of heat transfer (outdoor)"),
            CasadiParameter(name="RWin", unit="K/W",
                            description="Resistances of windows, from inside to outside"),
            CasadiParameter(name="hConFloor", unit="W/(m^2*K)",
                            description="Floor convective coefficient of heat transfer (indoor)"),
            CasadiParameter(name="RFloor", unit="K/W",
                            description="Resistances of floor, from inside to outside"),
            CasadiParameter(name="RFloorRem", unit="K/W",
                            description="Resistance of floor mass to outside"),
            CasadiParameter(name="hRad", unit="W/(m^2*K)",
                            description="Coefficient of heat transfer for linearized radiation exchange between walls"),
            CasadiParameter(name="hRadRoof", unit="W/(m^2*K)",
                            description="Coefficient of heat transfer for linearized radiation for roof"),
            CasadiParameter(name="hRadWall", unit="W/(m^2*K)",
                            description="Coefficient of heat transfer for linearized radiation for walls"),
            CasadiParameter(name="gWin", unit="-",
                            description="Total energy transmittance of windows"),
            CasadiParameter(name="ratioWinConRad", unit="-",
                            description="Ratio for windows between convective and radiation emission"),
            CasadiParameter(name="AExttot", unit="m^2",
                            description="total external walls area"),
            CasadiParameter(name="AInttot", unit="m^2",
                            description="total internal walls area"),
            CasadiParameter(name="AWintot", unit="m^2",
                            description="total window area"),
            CasadiParameter(name="AFloortot", unit="m^2",
                            description="total floor area"),
            CasadiParameter(name="ARooftot", unit="m^2",
                            description="total roof area"),
            CasadiParameter(name="ATransparent", unit="m^2",
                            description="total transparent area"),
            CasadiParameter(name="activityDegree", unit="met",
                            description="activity Degree of people in met"),
            CasadiParameter(name="specificPeople", unit="1/m^2",
                            description="people per square meter"),
            CasadiParameter(name="ratioConvectiveHeatPeople", unit="-",
                            description="Ratio of convective heat from overall heat output for people"),
            CasadiParameter(name="internalGainsMachinesSpecific", unit="W",
                            description="Heat Flux of machines"),
            CasadiParameter(name="ratioConvectiveHeatMachines", unit="-",
                            description="Ratio of convective heat from overall heat output for machines"),
            CasadiParameter(name="lightingPowerSpecific", unit="W/m^2",
                            description="Heat flux of lighting"),
            CasadiParameter(name="ratioConvectiveHeatLighting", unit="-",
                            description="Ratio of convective heat from overall heat output for lights"),
            CasadiParameter(name="nEle_heater", unit="-",
                            description="Number of elements in heater"),
            CasadiParameter(name="V_Ele_heater", unit="m^3",
                            description="Volume of one heater element"),

            CasadiParameter(name="n_heater_exp", unit="-",
                            description="Heater exponent n"),
            CasadiParameter(name="fraRad", unit="-",
                            description="Radiative heat fraction of heater"),
            CasadiParameter(name="valve_leakage", unit="-",
                            description="Valve leakage"),
            CasadiParameter(name="cp_water", unit="J/kg*K",
                            description="specific heat capacity of water"),
            CasadiParameter(name="rho_water", unit="kg/m^3",
                            description="density of water in heating cycle"),
            CasadiParameter(name="scalingFactor", unit="-",
                            description="scaling factor from bes-rules for heat pump"),
            # costs
            CasadiParameter(name="c_feed_in", value=0.082, unit="€/kWh",
                       description="Power Price"),
            CasadiParameter(name="c_grid", value=0.3606, unit="€/kWh",
            #CasadiParameter(name="c_grid", value=0.0, unit="€/kWh",
                       description="Power Price"),
            # comfort
            CasadiParameter(name="T_Air_ub", value=297.15, unit="K",
                            description="Upper boundary for T_Air", ),
            CasadiParameter(name="T_Air_lb", value=293.15, unit="K",
                            description="Lower boundary for T_Air", ),
            CasadiParameter(name="c_comf_upper", value=1, unit="ct/sK^2",
                            description="weighting factor for discomfort costs"),
            #CasadiParameter(name="c_comf_lower", value=1, unit="ct/sK^2",
            CasadiParameter(name="c_comf_lower", value=1, unit="ct/sK^2",
                            description="weighting factor for discomfort costs"),


            # temperature bounds
            CasadiParameter(name="T_TES_lb", value=293.15, unit="K",
                            description="lower temp. bound for all TES layers"),
            CasadiParameter(name="T_TES_ub", value=343.15, unit="K",
                            description="upper temp. bound for all TES layers"),
            CasadiParameter(name="scale_obj", value=3600, unit="K",
                            description="scale penalties of obj function to reach objective values around 1")
        ]


        outputs: List[CasadiOutput] = [
            CasadiOutput(name="Tamb", value=273.15, unit="K", description="Ambient temperature output"),
            CasadiOutput(name="Qdot_hp", value=0, unit="W"),
            CasadiOutput(name="P_el_hp", value=0, unit="W"),
            CasadiOutput(name="P_el_demand", value=0, unit="W"),
            CasadiOutput(name="P_el_demand_devices", value=0, unit="W"),
            CasadiOutput(name="P_el_demand_lights", value=0, unit="W"),
            CasadiOutput(name="P_pv", value=0, unit="W"),
            CasadiOutput(name="P_el_feed", value=0, unit="W"),
            CasadiOutput(name="T_lb", value=0, unit="K")
        ] + [
          # air heat balance
            CasadiOutput(name=f"Qdot_Air_sol_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_heater_{z}", value=0, unit="W") for z in zones
        ] + [
            # ventilation
            CasadiOutput(name=f"ventRate_airExc_{z}", value=0, unit="m^3/s") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_airExc_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_positiv_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Qdot_Air_negativ_{z}", value=0, unit="W") for z in zones
        ] + [



        ] + [
          # heating system
            CasadiOutput(name=f"valve_actual_out_{z}", value=0, unit="-") for z in zones
        ] + [
            CasadiOutput(name=f"mTra_flow_out_{z}", value=0, unit="kg/s") for z in zones
        ] + [
            CasadiOutput(name="TTraSup", value=273.15, unit="K")
        ] + [
            CasadiOutput(name=f"TTraRet_{z}", value=273.15, unit="K") for z in zones
        ] + [
            CasadiOutput(name=f"QTra_flow_out_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"QTraCon_flow_out_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"QTraRad_flow_out_{z}", value=0, unit="W") for z in zones
        ] + [


            # internal gains
            # CasadiOutput(name=f"Q_RadSol_or_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_conv_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_conv_machines_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_conv_lights_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_conv_humans_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_rad_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_rad_machines_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_rad_lights_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_rad_humans_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_machines_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_lights_{z}", value=0, unit="W") for z in zones
        ] + [
            CasadiOutput(name=f"Q_IntGains_humans_{z}", value=0, unit="W") for z in zones
        ]


        if HOM_predictor:
            inputs = inputs + [CasadiInput(name=f"ZoneWindowsTotalHeatRate_{z}", value=0, unit="W",description="Zone Windows Total Heat Rate") for z in zones]
            outputs = outputs + [CasadiOutput(name=f"ZoneWindowsTotalHeatRate_or_{z}", value=0, unit="W") for z in zones]
            inputs = inputs + [CasadiInput(name=f"Q_RadSol_{z}", value=0, unit="W", description="Radiative solar heat for all orientations") for z in zones]
            outputs = outputs + [CasadiOutput(name=f"Q_RadSol_or_{z}", value=0, unit="W") for z in zones]

            inputs = inputs + [CasadiInput(name=f"TSetOneZone_{z}", value=0, unit="K", description="TSetOneZone for each zone") for z in zones]
            outputs = outputs + [CasadiOutput(name=f"TSetOneZone_{z}", value=0, unit="K") for z in zones]

            parameters = parameters + [CasadiParameter(name=f"mTra_flow_nominal_{z}", value=0, unit="kg/s",
                            description="Mass flow through heater") for z in zones]

            parameters = parameters + [CasadiParameter(name=f"UA_heater_{z}", value=0, unit="W/K^n",
                            description="Product of heat transfer coefficient and area of heater") for z in zones]

            parameters = parameters + [CasadiParameter(name=f"UA_heater_Ele_{z}", value=0, unit="W/K^n",
                            description="Product of heat transfer coefficient and area of heater element") for z in zones]

        else:
            inputs = inputs + [CasadiInput(name="Q_RadSol", value=0, unit="W", description="Radiative solar heat for all orientations")]
            outputs = outputs + [CasadiOutput(name=f"Q_RadSol_or_{z}", value=0, unit="W") for z in zones]

            inputs = inputs + [CasadiInput(name="T_preTemWin", value=294.15, unit="K", description="Outdoor surface temperature of window")]
            outputs = outputs + [CasadiOutput(name=f"T_Win_sur_out_{z}", value=0, unit="K") for z in zones]

            parameters = parameters + [CasadiParameter(name="mTra_flow_nominal", unit="kg/s",
                            description="Mass flow through heater")]
            parameters = parameters + [CasadiParameter(name="UA_heater", unit="W/K^n",
                            description="Product of heat transfer coefficient and area of heater")]
            parameters = parameters + [CasadiParameter(name="UA_heater_Ele", unit="W/K^n",
                            description="Product of heat transfer coefficient and area of heater element")]



        material: pd.DataFrame
        zone_construction: pd.DataFrame
        windows: pd.DataFrame

        material, zone_construction, windows = get_idf_data()

        zone_name: str
        for zone_name in zones:

            dach: list
            RoofName: str
            RoofArea: str
            roof: tuple[str, float]

            dach = list(zone_construction.loc[zone_name].filter(regex="^Dach").items())

            for roof in dach:
                if roof[1] > 0:
                    RoofName, RoofArea = roof

                    inputs = inputs + [CasadiInput(name=f"T_preTemRoof_{RoofName}_{zone_name}", value=294.15, unit="K", description="Outdoor surface temperature of roof")]

                    states = states + [CasadiState(name=f"T_Roof_{RoofName}_1_{zone_name}", value=293.15, unit="K", description=f"Roof temp in {zone_name}") ]
                    states = states + [CasadiState(name=f"T_Roof_{RoofName}_2_{zone_name}", value=293.15, unit="K", description=f"Roof temp in {zone_name}")]

                    outputs = (outputs + [
                                            CasadiOutput(name=f"T_Roof_sur_out_{RoofName}_{zone_name}", value=0, unit="K")
                                        ] )
                    #            + [
                    #                         CasadiOutput(name=f"Qdot_Roof_sur_{RoofName}_1_{zone_name}", value=0, unit="W")
                    #                     ] + [
                    #                         CasadiOutput(name=f"T_Roof_pre_{RoofName}_1_{zone_name}", value=0, unit="K")
                    #                     ] + [
                    #                         CasadiOutput(name=f"Qdot_Roof_pre_{RoofName}_1_{zone_name}", value=0, unit="W")
                    #                     ] + [
                    #                         CasadiOutput(name=f"Qdot_Roof_{RoofName}_1_{zone_name}", value=0, unit="W")
                    #                     ])
                    # outputs = outputs + [
                    #                         CasadiOutput(name=f"T_Roof_sur_out_{RoofName}_2_{zone_name}", value=0, unit="K")
                    #                     ] + [
                    #                         CasadiOutput(name=f"Qdot_Roof_sur_{RoofName}_2_{zone_name}", value=0, unit="W")
                    #                     ] + [
                    #                         CasadiOutput(name=f"T_Roof_pre_{RoofName}_2_{zone_name}", value=0, unit="K")
                    #                     ] + [
                    #                         CasadiOutput(name=f"Qdot_Roof_pre_{RoofName}_2_{zone_name}", value=0, unit="W")
                    #                     ] + [
                    #                         CasadiOutput(name=f"Qdot_Roof_{RoofName}_2_{zone_name}", value=0, unit="W")
                    #                     ]

                else:
                    continue

            erdboden: list
            GroundFloorName: str
            GroundFloorArea: str
            groundfloor: tuple[str, float]


            erdboden = list(zone_construction.loc[zone_name].filter(regex="^GroundFloor").items())
            for groundfloor in erdboden:
                if groundfloor[1] > 0:
                    GroundFloorName, GroundFloorArea = groundfloor

                    inputs = inputs + [CasadiInput(name=f"T_preTemFloor_{GroundFloorName}_{zone_name}", value=286.15, unit="K", description="Outdoor surface temperature of floor")]

                    states = states + [CasadiState(name=f"T_Floor_{GroundFloorName}_1_{zone_name}", value=293.15, unit="K", description=f"Floor temp in {zone_name}") ]
                    states = states + [CasadiState(name=f"T_Floor_{GroundFloorName}_2_{zone_name}", value=293.15, unit="K", description=f"Floor temp in {zone_name}") ]

                    outputs = (outputs + [
                                            CasadiOutput(name=f"T_Floor_sur_out_{GroundFloorName}_{zone_name}", value=0, unit="K")
                                        ])
                               # + [
                               #              CasadiOutput(name=f"Qdot_Floor_sur_{GroundFloorName}_1_{zone_name}", value=0, unit="W")
                               #          ] + [
                               #              CasadiOutput(name=f"T_Floor_pre_{GroundFloorName}_1_{zone_name}", value=0, unit="K")
                               #          ] + [
                               #              CasadiOutput(name=f"Qdot_Floor_pre_{GroundFloorName}_1_{zone_name}", value=0, unit="W")
                               #          ] + [
                               #              CasadiOutput(name=f"Qdot_Floor_{GroundFloorName}_1_{zone_name}", value=0, unit="W")
                               #          ])

                    #outputs = (outputs + [
                     #                       CasadiOutput(name=f"T_Floor_sur_out_{GroundFloorName}_2_{zone_name}", value=0, unit="K")
                      #                  ])
                               # + [
                               #              CasadiOutput(name=f"Qdot_Floor_sur_{GroundFloorName}_2_{zone_name}", value=0, unit="W")
                               #          ] + [
                               #              CasadiOutput(name=f"T_Floor_pre_{GroundFloorName}_2_{zone_name}", value=0, unit="K")
                               #          ] + [
                               #              CasadiOutput(name=f"Qdot_Floor_pre_{GroundFloorName}_2_{zone_name}", value=0, unit="W")
                               #          ] + [
                               #              CasadiOutput(name=f"Qdot_Floor_{GroundFloorName}_2_{zone_name}", value=0, unit="W")
                               #          ])
                else:
                    continue

            innenboden: list
            InnerFloorName: str
            InnerFloorArea: str
            innerfloor: tuple[str, float]

            innenboden = list(zone_construction.loc[zone_name].filter(regex="^InnerFloor").items())
            for innerfloor in innenboden:
                if innerfloor[1] > 0:
                    InnerFloorName, InnerFloorArea = innerfloor

                    states = states + [CasadiState(name=f"T_Floor_{InnerFloorName}_{zone_name}", value=293.15, unit="K", description=f"Floor temp in {zone_name}") ]

                    outputs = (outputs + [
                                             CasadiOutput(name=f"T_Floor_sur_out_{InnerFloorName}_{zone_name}", value=0, unit="K")
                                        ])
                                        # + [
                                        #      CasadiOutput(name=f"Qdot_Floor_sur_{InnerFloorName}_{zone_name}", value=0, unit="W")
                                        # ] + [
                                        #      CasadiOutput(name=f"Qdot_Floor_{InnerFloorName}_{zone_name}", value=0, unit="W")
                                        # ])
                else:
                    continue

            decke: list
            InnerFloorName: str
            InnerFloorArea: str
            ceiling: tuple[str, float]

            decke = list(zone_construction.loc[zone_name].filter(regex="^Decke").items())
            for ceiling in decke:
                if ceiling[1] > 0:
                    InnerFloorName, InnerFloorArea = ceiling


                    states = states + [CasadiState(name=f"T_Floor_{InnerFloorName}_{zone_name}", value=293.15, unit="K", description=f"Floor temp in {zone_name}") ]

                    outputs = outputs + [
                                            CasadiOutput(name=f"T_Floor_sur_out_{InnerFloorName}_{zone_name}", value=0, unit="K")
                                        ]
                                        # + [
                                        #     CasadiOutput(name=f"Qdot_Floor_sur_{InnerFloorName}_{zone_name}", value=0, unit="W")
                                        # ] + [
                                        #     CasadiOutput(name=f"Qdot_Floor_{InnerFloorName}_{zone_name}", value=0, unit="W")
                                        # ]
                else:
                    continue


            innenwand: list
            InnerWallName: str
            InnerWallArea: str
            innerwall: tuple[str, float]


            innenwand = list(zone_construction.loc[zone_name].filter(regex="^InnerWall").items())
            for innerwall in innenwand:
                if innerwall[1] > 0:
                    InnerWallName, InnerWallArea = innerwall


                    states = states + [CasadiState(name=f"T_IntWall_{InnerWallName}_{zone_name}", value=293.15, unit="K", description=f"Inner wall temp in {zone_name}") ]

                    outputs = (outputs + [
                                            CasadiOutput(name=f"T_IntWall_sur_out_{InnerWallName}_{zone_name}", value=0, unit="K")
                                        ])
                                        # + [
                                        #     CasadiOutput(name=f"Qdot_IntWall_sur_{InnerWallName}_{zone_name}", value=0, unit="W")
                                        # ] + [
                                        #     CasadiOutput(name=f"Qdot_IntWall_{InnerWallName}_{zone_name}", value=0, unit="W")
                                        # ])
                else:
                    continue


            aussenwand: list
            OuterWallName: str
            OuterWallArea: str
            outerwall: tuple[str, float]

            aussenwand = list(zone_construction.loc[zone_name].filter(regex="^OuterWall").items())
            for outerwall in aussenwand:
                if outerwall[1] > 0:
                    OuterWallName, OuterWallArea = outerwall


                    inputs = inputs + [CasadiInput(name=f"T_preTemWall_{OuterWallName}_{zone_name}", value=294.15, unit="K", description="Outdoor surface temperature of wall")]


                    states = states + [CasadiState(name=f"T_ExtWall_{OuterWallName}_1_{zone_name}", value=293.15, unit="K", description=f"Outer wall temp in {zone_name}") ]
                    states = states + [CasadiState(name=f"T_ExtWall_{OuterWallName}_2_{zone_name}", value=293.15, unit="K", description=f"Outer wall temp in {zone_name}") ]

                    outputs = outputs + [
                                            CasadiOutput(name=f"T_ExtWall_sur_out_{OuterWallName}_{zone_name}", value=0, unit="K")]

                    #                     ] + [
                    #                         CasadiOutput(name=f"Qdot_ExtWall_sur_{OuterWallName}_{zone_name}", value=0, unit="W")
                    #                     ] + [
                    #                         CasadiOutput(name=f"T_ExtWall_pre_{OuterWallName}_{zone_name}", value=0, unit="K")
                    #                     ] + [
                    #                         CasadiOutput(name=f"Qdot_ExtWall_pre_{OuterWallName}_{zone_name}", value=0, unit="W")
                    #                     ] + [
                    #                         CasadiOutput(name=f"Qdot_ExtWall_{OuterWallName}_1_{zone_name}", value=0, unit="W")
                    #                     ]
                    # outputs = outputs + [
                    #                         CasadiOutput(name=f"Qdot_ExtWall_{OuterWallName}_2_{zone_name}", value=0, unit="W")
                    #                     ]
                else:
                    continue




class MPC(CasadiModel):
    config: MPCModelConfig

    def setup_system(self):
        #bes_parameters = custom_alter_model_parameters(self.config.bes_parameters)
        bes_parameters = self.config.bes_parameters
        zone_parameters = self.config.zone_parameters
        self.nLayer: int = N_LAYER
        self.zones: list = zones
        self.multizone_coupled: bool = multizone_coupled
        self.HOM_predictor: bool = HOM_predictor
        self.calc_resistances_new: bool = calc_resistances_new


        def get_other_state(self, kind: str, elem_name: str, other_zone: str):
            # kind ∈ {"Floor","IntWall"}
            state_name = f"T_{kind}_{elem_name}_{other_zone}"
            mx = self._states.get(state_name)
            if mx is None:
                logger.warning(f"Gegenstate {state_name} nicht gefunden – nutze 293.15 K")
                return ca.MX(293.15)
            return mx

        def make_symmetric_pairs(pairs):
            adj = {}
            for (zA, eA), (zB, eB) in pairs:
                adj[(zA, eA)] = (zB, eB)
                adj[(zB, eB)] = (zA, eA)
            return adj

        if self.multizone_coupled:
            floor_pairs = make_symmetric_pairs([
                (("attic", "InnerFloor"), ("children2", "Decke")),
                (("attic", "InnerFloor2"), ("bath", "Decke")),
                (("attic", "InnerFloor3"), ("corridor2", "Decke")),
                (("attic", "InnerFloor4"), ("bedroom", "Decke")),
                (("attic", "InnerFloor5"), ("children", "Decke")),
                (("wcstorage", "Decke"), ("bath", "InnerFloor")),
                (("children", "InnerFloor"), ("hobby", "Decke")),
                (("bedroom", "InnerFloor"), ("livingroom", "Decke")),
                (("corridor", "Decke"), ("corridor2", "InnerFloor")),
                (("kitchen", "Decke"), ("children2", "InnerFloor")),
            ])
            iwall_pairs = make_symmetric_pairs([
                (("bedroom", "InnerWall"), ("children2", "InnerWall")),
                (("bedroom", "InnerWall3"), ("corridor2", "InnerWall4")),
                (("children", "InnerWall"), ("corridor2", "InnerWall")),
                (("children", "InnerWall2"), ("bedroom", "InnerWall2")),
                (("corridor2", "InnerWall3"), ("children2", "InnerWall3")),
                (("bath", "InnerWall"), ("children2", "InnerWall2")),
                (("bath", "InnerWall2"), ("corridor2", "InnerWall2")),
                (("livingroom", "InnerWall"), ("corridor", "InnerWall4")),
                (("livingroom", "InnerWall2"), ("kitchen", "InnerWall2")),
                (("hobby", "InnerWall"), ("corridor", "InnerWall")),

                (("hobby", "InnerWall2"), ("livingroom", "InnerWall3")),
                (("wcstorage", "InnerWall2"), ("kitchen", "InnerWall")),

                (("corridor", "InnerWall3"), ("kitchen", "InnerWall3")),
                (("wcstorage", "InnerWall"), ("corridor", "InnerWall2")),

            ])
        else:
            floor_pairs = {}
            iwall_pairs = {}

        self.test_case: bool = False

        if self.test_case:
            # building envelope
            # Get areas as python variable (not casadi)
            ainttot = sum(zone_parameters['AInt']) if isinstance(zone_parameters['AInt'], list) else zone_parameters['AInt']
            aexttot = sum(zone_parameters['AExt']) if isinstance(zone_parameters['AExt'], list) else zone_parameters['AExt']
            arooftot = sum(zone_parameters['ARoof']) if isinstance(zone_parameters['ARoof'], list) else zone_parameters[
                'ARoof']
            afloortot = sum(zone_parameters['AFloor']) if isinstance(zone_parameters['AFloor'], list) else zone_parameters[
                'AFloor']
            awintot = sum(zone_parameters['AWin']) if isinstance(zone_parameters['AWin'], list) else zone_parameters['AWin']
            zone_parameters['AInttot'] = ainttot
            zone_parameters['AExttot'] = aexttot
            zone_parameters['ARooftot'] = arooftot
            zone_parameters['AFloortot'] = afloortot
            zone_parameters['AWintot'] = awintot
            area_tot = ainttot + aexttot + arooftot + afloortot + awintot

            # split factor for internal radiation (internal gains, radiator)
            split_rad_int = {
                'int': ainttot / area_tot,
                'ext': aexttot / area_tot,
                'roof': arooftot / area_tot,
                'floor': afloortot / area_tot,
                'win': 0}

            # split factors for solar radiation
            split_rad_sol = {
                'int': bes_parameters["split_int_sol"],
                'ext': bes_parameters["split_ext_sol"],
                'roof': bes_parameters["split_roof_sol"],
                'floor': bes_parameters["split_floor_sol"],
                'win': 0}

            # calc resistances
            coeff_dict = calc_resistances(
                zone_parameters=zone_parameters,
                split_rad_int=split_rad_int,
                split_rad_sol=split_rad_sol
            )

            # s = pd.Series(coeff_dict).sort_index()
            # print(s.to_string(max_rows=None))  # nur für diesen Aufruf
            # # oder global:
            # pd.set_option('display.max_rows', None)
            # pd.set_option('display.max_colwidth', None)
            # pd.set_option('display.width', None)  # automatische Breite, weniger Umbruch
            # print(s.to_string())

            #
            # # Liste der gewünschten Keys:
            # wanted_keys = [
            #     'hConWin',
            #     'hRad',
            #     'hConWallOut',
            #     'hRadWall',
            #     'RExtRem',
            #     'hConRoofOut',
            #     'hRadRoof',
            #     'RRoofRem',
            #     'RFloorRem',
            #     'hConWinOut',
            #     'RWin',
            # ]
            #
            # def get_ci(d: dict, key: str, default=None):
            #     """Case-insensitiver Dict-Zugriff."""
            #     key_l = key.lower()
            #     for k, v in d.items():
            #         if k.lower() == key_l:
            #             return v
            #     return default
            #
            # def print_params(d: dict, keys: list[str]):
            #     """Gibt alle gewünschten Parameter direkt aus (case-insensitiv)."""
            #     for k in keys:
            #         v = get_ci(d, k, default="<nicht gesetzt>")
            #         print(f"{k} = {v}")
            #
            # # ---- Ausgaben ----
            # print_params(zone_parameters, wanted_keys)
            #
            # # Dein Beispiel: direkter Zugriff (case-insensitiv) und print
            # print(get_ci(zone_parameters, 'hconwin', default="<nicht gesetzt>"))

        self.test_case: bool = False

        coeff_dict: dict = {}

        material: pd.DataFrame
        zone_construction: pd.DataFrame
        windows: pd.DataFrame

        material, zone_construction, windows = get_idf_data()

        self.TTraSup = self.get_T_layer(self.nLayer)

        self.constraints = []
        self.house_zones = {}

        self.floor = {}
        self.extwall = {}
        self.intwall = {}
        self.roof = {}

        self._zone_coeff = {}

        for name in zones:
            if self.test_case:
                self.house_zones[name] = component_models.single_zone_alpha(model=self, name=name, coeff_dict=coeff_dict)
                self.floor[name] = component_models.GroundFloor(model=self, zone1=name, coeff_dict=coeff_dict, material=material, element_construction=zone_construction, windows=windows)
                self.intwall[name] = component_models.InnerWall(model=self, zone1=name, coeff_dict=coeff_dict, material=material, element_construction=zone_construction, windows=windows)
                self.roof[name] = component_models.Roof(model=self, zone1=name, coeff_dict=coeff_dict, material=material, element_construction=zone_construction, windows=windows)
                self.extwall[name] = component_models.OuterWall(model=self, zone1=name, coeff_dict=coeff_dict, material=material, element_construction=zone_construction, windows=windows)

                self.house_zones[name].set_zone()

            else:
                self.house_zones[name] = component_models.single_zone_alpha(model=self, name=name, coeff_dict=coeff_dict, material=material, zone_construction = zone_construction)

                dach = list(zone_construction.loc[name].filter(regex="^Dach").items())
                for roof in dach:
                    if roof[1] > 0:
                        self.roof[f"{name}_{roof[0]}"] = component_models.Roof(model=self, zone1=name, coeff_dict=coeff_dict,
                                                                material=material,
                                                                element_construction=roof,
                                                                windows=windows)
                    else:
                        continue

                erdboden = list(zone_construction.loc[name].filter(regex="^GroundFloor").items())
                for groundfloor in erdboden:
                    if groundfloor[1] > 0:
                        self.floor[f"{name}_{groundfloor[0]}"] = component_models.GroundFloor(model=self, zone1=name, coeff_dict=coeff_dict,
                                                                        material=material,
                                                                        element_construction=groundfloor,
                                                                        windows=windows)
                    else:
                        continue

                innenboden = list(zone_construction.loc[name].filter(regex="^InnerFloor").items())
                for innerfloor in innenboden:
                    if innerfloor[1] > 0:

                        T_other = 293.15
                        if self.multizone_coupled:
                            pair = floor_pairs.get((name, innerfloor[0]))
                            if pair:
                                other_zone, other_elem = pair
                                T_other = get_other_state(self, "Floor", other_elem, other_zone)

                        self.floor[f"{name}_{innerfloor[0]}"] = component_models.InnerFloor(
                            model=self, zone1=name, coeff_dict=coeff_dict,
                            material=material, element_construction=innerfloor, windows=windows,
                            T_other_side=T_other
                        )

                    else:
                        continue

                decke = list(zone_construction.loc[name].filter(regex="^Decke").items())
                for ceiling in decke:
                    if ceiling[1] > 0:

                        T_other = 293.15
                        if self.multizone_coupled:
                            pair = floor_pairs.get((name, ceiling[0]))
                            if pair:
                                other_zone, other_elem = pair
                                T_other = get_other_state(self, "Floor", other_elem, other_zone)

                        self.floor[f"{name}_{ceiling[0]}"] = component_models.Ceiling(
                            model=self, zone1=name, coeff_dict=coeff_dict,
                            material=material, element_construction=ceiling, windows=windows,
                            T_other_side=T_other
                        )

                    else:
                        continue

                innenwand = list(zone_construction.loc[name].filter(regex="^InnerWall").items())
                for innerwall in innenwand:
                    if innerwall[1] > 0:

                        T_other = 293.15
                        if self.multizone_coupled:
                            pair = iwall_pairs.get((name, innerwall[0]))
                            if pair:
                                other_zone, other_elem = pair
                                T_other = get_other_state(self, "IntWall", other_elem, other_zone)

                        self.intwall[f"{name}_{innerwall[0]}"] = component_models.InnerWall(
                            model=self, zone1=name, coeff_dict=coeff_dict,
                            material=material, element_construction=innerwall, windows=windows,
                            T_other_side=T_other
                        )

                    else:
                        continue

                aussenwand = list(zone_construction.loc[name].filter(regex="^OuterWall").items())
                for outerwall in aussenwand:
                    if outerwall[1] > 0:
                        self.extwall[f"{name}_{outerwall[0]}"] = component_models.OuterWall(model=self, zone1=name, coeff_dict=coeff_dict,
                                                                        material=material,
                                                                        element_construction=outerwall,
                                                                        windows=windows)
                    else:
                        continue


                self.house_zones[name].set_zone()






        self._outputs.get("TTraSup").alg = self.TTraSup

        # Supply temperature control
        THeaPumSup = self.TBufSet
        #THeaPumSup = self._inputs.get("TBufSet")


        # buffer storage
        # Zonenmischer

        #mTra_flow_total = sum(zone.mTra_flow for zone in self.house_zones.values())
        #TTraRet_total = (sum(zone.mTra_flow * zone.TTraOut for zone in self.house_zones.values()) / mTra_flow_total)

        eps = 1e-9
        mTra_flow_total = sum(zone.mTra_flow for zone in self.house_zones.values())  # MX
        num = sum(zone.mTra_flow * zone.TTraOut for zone in self.house_zones.values())  # MX

        TTraRet_total = ca.if_else(
            mTra_flow_total > eps,  # MX (bool)
            num / mTra_flow_total,  # MX
            self.TTraSup.sym,  # MX (!) statt CasadiState
            True  # short_circuit
        )



        # increase of power
        #mHeaPum_flow = 1 * bes_parameters["hydraulic.generation.m_flow_nominal[1]"] * (self.SGReadySignal - 1)
        mHeaPum_flow = 1 * bes_parameters["hydraulic.generation.m_flow_nominal[1]"]
        Q_storage_loss = component_models.storage_n_layer(
            casadi_model=self,
            bes_parameters=bes_parameters,
            mTra_flow=mTra_flow_total,
            mGen_flow=mHeaPum_flow,
            TTraRet=TTraRet_total,
            TGenSup=THeaPumSup
        )

        # heat pump
        TStoRet = self.get_T_layer(1)

        QHeaPum_flow_max, QHeaPum_flow, PEleHeaPum, PEleEleHea = component_models.vitocal250_with_ideal_heater(
            casadi_model=self, bes_parameters=bes_parameters,
            THeaPumIn=TStoRet,
            THeaPumSup=THeaPumSup,
            THeaPumSou=self.T_amb,
            mHeaPum_flow=mHeaPum_flow
        )

        # electric energy
        P_el_demand_devices = self.AZone * self.internalGainsMachinesSpecific * self.schedule_dev
        P_el_demand_lights = self.AZone * self.lightingPowerSpecific * self.schedule_light
        P_el_demand = P_el_demand_devices + P_el_demand_lights
        P_el_feed = P_el_demand + PEleHeaPum + PEleEleHea - self.P_el_pv

        # Constraints: List[(lower bound, function, upper bound)]
        # best practise: hard constraints -> soft with high penalty

        self.constraints += [
            # heatpump
            (273.15 + 30, THeaPumSup + self.THeaPumSup_slack, 273.15 + 70),
            (0, QHeaPum_flow + self.QHeaPum_flow_slack, QHeaPum_flow_max),
            # increase of power
            (0, PEleHeaPum + self.PEleHeaPum_slack, 1 * 6770 * self.scalingFactor),
            # electric power
            (0, P_el_feed + self.P_el_feed_into_grid - self.P_el_feed_from_grid + self.s_Pel_feed, 0),
            (0, self.P_el_feed_into_grid, ca.inf),
            (0, self.P_el_feed_from_grid, ca.inf)
        ]

        # Objective function

        C_el = self.c_grid * PEleHeaPum / 1000  # €/kWh * 1h/3600s * W / 1000

        # without attic
        self.zones = ["livingroom", "kitchen", "hobby", "wcstorage", "corridor", "bedroom", "children", "corridor2", "bath",
                 "children2"]

        C_comf = 0

        for zone in self.zones:
            slack_ub = self._states.get(f"TAir_ub_slack_{zone}")
            slack_lb = self._states.get(f"TAir_lb_slack_{zone}")
            C_comf += self.c_comf_upper * slack_ub ** 2 + self.c_comf_lower * slack_lb ** 2


        cost_terms = [
                # increase of power
                10 * C_el / self.scale_obj,
                C_comf / self.scale_obj,
                #1 / self.scale_obj * self.yValSet ** 2,

                # 10 * 100 / self.scale_obj * self.TTraSup_slack ** 2,
                10 * 1 / self.scale_obj * self.THeaPumSup_slack ** 2,

                # increase of power
                10 * 100 / self.scale_obj * self.QHeaPum_flow_slack ** 2,
                10 * 100 / self.scale_obj * self.PEleHeaPum_slack ** 2,
                #10000 / self.scale_obj * self.s_valve_pos_ub ** 2,
                #10000 / self.scale_obj * self.s_valve_pos_lb ** 2,
                #100 / self.scale_obj * self.s_Pel_feed ** 2
            ]


        #for n in range(1, self.config.nLayer + 1):
        #    s_T_TES_n = self._states.get(f"s_T_TES_{n}")
        #    cost_terms.append(100 / self.scale_obj * s_T_TES_n ** 2)

        objective = sum(cost_terms)


        # Outputs
        # not needed to work
        # all for comparison with sim
        self.Tamb.alg = self.T_amb

        self.Qdot_hp.alg = QHeaPum_flow
        self.P_el_hp.alg = PEleHeaPum

        self.P_el_demand.alg = P_el_demand
        self.P_el_demand_devices.alg = P_el_demand_devices
        self.P_el_demand_lights.alg = P_el_demand_lights

        self.P_pv.alg = self.P_el_pv
        self.P_el_feed.alg = P_el_feed

        self.T_lb.alg = self.T_Air_lb

        self.dump_ode_equations_txt()
        self.dump_bounded_constraints_txt()

        return objective

    def get_T_layer(self, _n):
        return self._states.get(f"T_TES_{_n}")

    def dump_ode_equations_txt(self, filename1="ode_equations1.txt", filename2="alg_equations1.txt", filename3="inputs.txt"):
        """
        Exportiere alle .ode-Zuweisungen aus self._states als Klartextdatei.
        """
        odes = []
        for name, state in self._states.items():
            if state.ode is not None:
                odes.append((name, state.ode))
            else:
                odes.append((name, "⚠️ KEINE ZUWEISUNG"))

        with open(filename1, "w", encoding="utf-8") as f:
            for name, expr in odes:
                f.write(f"{name}.ode = {expr}\n\n")


        outputs = []
        for name, output in self._outputs.items():
            if output.alg is not None:
                outputs.append((name, output.alg))
            else:
                outputs.append((name, "⚠️ KEINE ZUWEISUNG"))

        with open(filename2, "w", encoding="utf-8") as f:
            for name, expr in outputs:
                f.write(f"{name}.alg = {expr}\n\n")

        print(f"✅ Gleichungen als Text gespeichert")

        inputs = []
        for name, input in self._inputs.items():
            if input is not None:
                inputs.append((name, input))
            else:
                inputs.append((name, "⚠️ KEINE ZUWEISUNG"))

        with open(filename3, "w", encoding="utf-8") as f:
            for name, expr in inputs:
                f.write(f"{name} = {expr}\n\n")

        print(f"✅ Gleichungen als Text gespeichert")


    def dump_bounded_constraints_txt(self, filename="bounded_constraints1.txt"):
        import os
        os.makedirs("debug_output", exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            if not hasattr(self, "constraints") or not self.constraints:
                f.write("⚠️ Keine Constraints gefunden.\n")
                print("⚠️ Keine Constraints gefunden.")
                return

            for i, con in enumerate(self.constraints):
                if isinstance(con, tuple) and len(con) == 3:
                    lb, expr, ub = con
                    f.write(f"Constraint {i}:\n")
                    f.write(f"  Lower Bound: {lb}\n")
                    f.write(f"  Expression: {expr}\n")
                    f.write(f"  Upper Bound: {ub}\n\n")
                else:
                    f.write(f"Constraint {i}: ⚠️ Kein 3-Tupel: {con}\n\n")

        print(f"✅ {len(self.constraints)} Constraints gespeichert in '{filename}'")
