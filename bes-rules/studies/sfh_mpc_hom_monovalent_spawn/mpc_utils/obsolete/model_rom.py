"""
Model taken from here
https://git-ce.rwth-aachen.de/ebc/projects/ACS_EBC0022_ERANET_I-Greta_bsc/optimization/-/tree/fwu-pkr
"""

import logging
import casadi as ca
from typing import List
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
from bes_rules.simulation_based_optimization.agentlib_mpc import component_models

logger = logging.getLogger(__name__)

# storage layers
N_LAYER = 4

def custom_alter_model_parameters(bes_parameters: dict):
    mass_dict = {
        "1": "split_ext_sol",
        "2": "split_win_sol",
        "3": "split_int_sol",
        "4": "split_floor_sol",
        "5": "split_roof_sol"
    }
    n_sides = 4
    for num, split in mass_dict.items():
        bes_parameters[split] = sum(
            bes_parameters[f"building.thermalZone[1].ROM.thermSplitterSolRad.splitFactor[{num},{i}]"]
            for i in range(1, n_sides + 1)
        ) / n_sides

    return bes_parameters


class MPCModelConfig(CasadiModelConfig):
    model_config = ConfigDict(validate_assignment=True, extra="allow")
    bes_parameters: dict
    zone_parameters: dict

    nLayer: int = N_LAYER

    inputs: List[CasadiInput] = [
        # controls
        CasadiInput(name="TBufSet", value=323.15, unit="K",
                    description="Set temperature of buffer storage, sink temperature of heat pump"),
        CasadiInput(name="TSetOneZone", value=293.15, unit="K",
                    description="Set room temperature for one zone model", lb=0, ub=1),
        CasadiInput(name="yEleHeaSet", value=0, unit="-",
                    description="set point for electric heater (not used, only for interest)", lb=0, ub=1
                    ),
        #CasadiState(name="s_valve_pos_lb", value=0, unit="-",
        #            description="allowing to close the valve"),
        CasadiInput(name="yValSet", value=0, unit="-",
                    description="set point for electric heater (not used, only for interest)", lb=0, ub=1
                    ),

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
        CasadiState(name="T_Air", value=293.15, unit="K",
                    description="Temperature of zone"),
        CasadiState(name="T_IntWall", value=293.15, unit="K",
                    description="Inner wall temperature"),
        CasadiState(name="T_ExtWall", value=293.15, unit="K",
                    description="Outer wall temperature"),
        CasadiState(name="T_Roof", value=293.15, unit="K",
                    description="Temperature of roof"),
        CasadiState(name="T_Floor", value=293.15, unit="K",
                    description="Temperature of floor"),
        #CasadiState(name="yValSet", value=0, unit="-",
        #            description="set value of valve position"),

        # slack variables
        CasadiState(name="TAir_ub_slack", value=0, unit="K",
                    description="Slack variable for temperature"),
        CasadiState(name="TAir_lb_slack", value=0, unit="K",
                    description="Slack variable for temperature"),
        CasadiState(name="TSetOneZoneDiff_lb_slack", value=0, unit="K",
                    description="Slack variable for temperature diff TSetOneZone - T_Air"),
        CasadiState(name="TSetOneZoneDiff_ub_slack", value=0, unit="K",
                    description="Slack variable for temperature diff TSetOneZone - T_Air"),
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
        CasadiState(name=f'T_TES_{n}', value=310, unit="K") for n in range(1, N_LAYER + 1)
    ] + [
        # storage slack
        CasadiState(name=f's_T_TES_{n}', value=0, unit="K") for n in range(1, N_LAYER + 1)
    ]

    parameters: List[CasadiParameter] = [
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
                   description="Power Price"),
        # comfort
        CasadiParameter(name="T_Air_ub", value=297.15, unit="K",
                        description="Upper boundary for T_Air", ),
        CasadiParameter(name="T_Air_lb", value=293.15, unit="K",
                        description="Lower boundary for T_Air", ),
        CasadiParameter(name="c_comf_upper", value=1, unit="ct/sK^2",
                        description="weighting factor for discomfort costs"),
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
        CasadiOutput(name="Tamb", value=273.15, unit="K",
                     description="Ambient temperature output"),

        CasadiOutput(name="Qdot_Air", value=0, unit="W"),
        CasadiOutput(name="Qdot_Air_int", value=0, unit="W"),
        CasadiOutput(name="Qdot_Air_ext", value=0, unit="W"),
        CasadiOutput(name="Qdot_Air_roof", value=0, unit="W"),
        CasadiOutput(name="Qdot_Air_floor", value=0, unit="W"),
        CasadiOutput(name="Qdot_Air_win", value=0, unit="W"),
        CasadiOutput(name="Qdot_Air_gain", value=0, unit="W"),
        CasadiOutput(name="Qdot_Air_sol", value=0, unit="W"),
        CasadiOutput(name="Qdot_Air_heater", value=0, unit="W"),

        CasadiOutput(name="T_IntWall_sur", value=0, unit="K"),
        CasadiOutput(name="Qdot_IntWall_sur", value=0, unit="W"),
        CasadiOutput(name="Qdot_IntWall", value=0, unit="W"),

        CasadiOutput(name="T_ExtWall_sur", value=0, unit="K"),
        CasadiOutput(name="Qdot_ExtWall_sur", value=0, unit="W"),
        CasadiOutput(name="T_ExtWall_pre", value=0, unit="K"),
        CasadiOutput(name="Qdot_ExtWall_pre", value=0, unit="W"),
        CasadiOutput(name="Qdot_ExtWall", value=0, unit="W"),

        CasadiOutput(name="T_Roof_sur", value=0, unit="K"),
        CasadiOutput(name="Qdot_Roof_sur", value=0, unit="W"),
        CasadiOutput(name="T_Roof_pre", value=0, unit="K"),
        CasadiOutput(name="Qdot_Roof_pre", value=0, unit="W"),
        CasadiOutput(name="Qdot_Roof", value=0, unit="W"),

        CasadiOutput(name="T_Floor_sur", value=0, unit="K"),
        CasadiOutput(name="Qdot_Floor_sur", value=0, unit="W"),
        CasadiOutput(name="T_Floor_pre", value=0, unit="K"),
        CasadiOutput(name="Qdot_Floor_pre", value=0, unit="W"),
        CasadiOutput(name="Qdot_Floor", value=0, unit="W"),

        CasadiOutput(name="valve_actual", value=0, unit="-"),
        CasadiOutput(name="mTra_flow", value=0, unit="kg/s"),
        CasadiOutput(name="TTraSup", value=273.15, unit="K"),
        CasadiOutput(name="TTraRet", value=273.15, unit="K"),
        CasadiOutput(name="QTra_flow", value=0, unit="W"),
        CasadiOutput(name="QTraCon_flow", value=0, unit="W"),
        CasadiOutput(name="QTraRad_flow", value=0, unit="W"),

        CasadiOutput(name="Q_RadSol_or", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_conv", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_conv_machines", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_conv_lights", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_conv_humans", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_rad", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_rad_machines", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_rad_lights", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_rad_humans", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_machines", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_lights", value=0, unit="W"),
        CasadiOutput(name="Q_IntGains_humans", value=0, unit="W"),

        CasadiOutput(name="ventRate_airExc", value=0, unit="m^3/s"),
        CasadiOutput(name="Qdot_airExc", value=0, unit="W"),

        CasadiOutput(name="Qdot_hp", value=0, unit="W"),
        CasadiOutput(name="P_el_hp", value=0, unit="W"),

        CasadiOutput(name="P_el_demand", value=0, unit="W"),
        CasadiOutput(name="P_el_demand_devices", value=0, unit="W"),
        CasadiOutput(name="P_el_demand_lights", value=0, unit="W"),

        CasadiOutput(name="P_pv", value=0, unit="W"),
        CasadiOutput(name="P_el_feed", value=0, unit="W"),

        CasadiOutput(name="T_lb", value=0, unit="K"),
    ]


class MPC(CasadiModel):
    config: MPCModelConfig

    def setup_system(self):
        bes_parameters = custom_alter_model_parameters(self.config.bes_parameters)
        zone_parameters = self.config.zone_parameters

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

        # internal gains
        q_humans = ((0.865 - (0.025 * (self.T_Air - 273.15))) * (
                self.activityDegree * 58 * 1.8) + 35) * self.specificPeople * self.AZone * self.schedule_human
        q_humans_conv = q_humans * self.ratioConvectiveHeatPeople
        q_humans_rad = q_humans * (1 - self.ratioConvectiveHeatPeople)

        q_devices = self.AZone * self.internalGainsMachinesSpecific * self.schedule_dev
        q_devices_conv = q_devices * self.ratioConvectiveHeatMachines
        q_devices_rad = q_devices * (1 - self.ratioConvectiveHeatMachines)

        q_lights = self.AZone * self.lightingPowerSpecific * self.schedule_light
        q_lights_conv = q_lights * self.ratioConvectiveHeatLighting
        q_lights_rad = q_lights * (1 - self.ratioConvectiveHeatLighting)

        q_ig_conv = q_humans_conv + q_devices_conv + q_lights_conv
        q_ig_rad = q_humans_rad + q_devices_rad + q_lights_rad

        # natural air exchange
        # see documentation aixlib: ventRate = f(people activity, outside temperature, inside temperature, leakage)
        # baseACH = 0.2
        #maxUserACH = 1
        #userACH = self.schedule_human * maxUserACH
        #ventRate = baseACH + userACH * self.redFac  # simplification as welL in modelica model

        # constant ventRate of 0.5
        ventRate = 0.5
        Qdot_vent = ventRate * self.VAir * self.air_rho * self.air_cp * (self.T_amb - self.T_Air) * (1 / 3600)

        # thermal transmittance
        # Air
        k_int_air = self.hConInt * self.AInttot
        k_ext_air = self.hConExt * self.AExttot
        k_roof_air = self.hConRoof * self.ARooftot
        k_floor_air = self.hConFloor * self.AFloortot
        k_win_air = self.hConWin * self.AWintot

        # Interior Walls
        k_int = 1 / self.RInt

        # Exterior Walls
        k_amb_ext = 1 / (1 / ((self.hConWallOut + self.hRadWall) * self.AExttot) + self.RExtRem)
        k_ext = 1 / self.RExt

        # Roof
        k_amb_roof = 1 / (1 / ((self.hConRoofOut + self.hRadRoof) * self.ARooftot) + self.RRoofRem)
        k_roof = 1 / self.RRoof

        # Floor
        k_amb_floor = 1 / self.RFloorRem
        k_floor = 1 / self.RFloor

        # Solar radiation to walls (approximated)
        Q_RadSol_air = (self.Q_RadSol / (self.gWin * (
                1 - self.ratioWinConRad) * self.ATransparent) * self.gWin * self.ratioWinConRad * self.ATransparent)

        # calculate valve position and mass flow according to P-controller
        #valve_ctrl_k = bes_parameters["hydraulic.control.valCtrl.PI[1].k"]
        #error = self.TSetOneZone - self.T_Air
        #   try simple P-ctrl
        #p_ctrl_output = valve_ctrl_k * error
        #valve_output = p_ctrl_output + self.s_valve_pos_lb  # set limits [0,1]
        #tau_valve = 10   # time constant in s
        #   DGL
        #self.yValSet.ode = (1 / tau_valve) * (valve_output - self.yValSet)   # consider constraints of pi_ctrl [0,1]. If pi_ctrl_output < 0, yValSet --> 0. If pi_ctrl_output > 1, yValSet --> 1. Otherwise yValSet --> pi_ctrl_output
        #self.yValState.ode = 1/300 * (self.yValSet - self.yValState)
        valve_actual = self.valve_leakage + self.yValSet * (1 - self.valve_leakage)
        mTra_flow = self.mTra_flow_nominal * valve_actual

        # heater
        # assumption stationary energy balance
        # no delay by volume elements/thermal inertia
        # sim: simple radiator model (EN442-2), no delay between buffer storage and heater
        # heater
        TTraSup = self.get_T_layer(self.config.nLayer)
        TTraOut = component_models.radiator_no_exponent_outlet_temperature(
            casadi_model=self,
            TTraSup=TTraSup,
            T_Air=self.T_Air,
            mTra_flow=mTra_flow,
        )
        QTra_flow = mTra_flow * self.cp_water * (TTraSup - TTraOut)
        QTraRad_flow = QTra_flow * self.fraRad
        QTraCon_flow = QTra_flow * (1 - self.fraRad)

        # Supply temperature control
        THeaPumSup = self.TBufSet

        # buffer storage
        mHeaPum_flow = bes_parameters["hydraulic.generation.m_flow_nominal[1]"] * (self.SGReadySignal - 1)
        Q_storage_loss = component_models.storage_n_layer(
            casadi_model=self,
            bes_parameters=bes_parameters,
            mTra_flow=mTra_flow,
            mGen_flow=mHeaPum_flow,
            TTraRet=TTraOut,
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

        # Calculate Surface Temperature of components
        T_IntWall_sur = (coeff_dict['T_int_sur']['T_Air'] * self.T_Air +
                         coeff_dict['T_int_sur']['T_int'] * self.T_IntWall +
                         coeff_dict['T_int_sur']['T_ext'] * self.T_ExtWall +
                         coeff_dict['T_int_sur']['T_roof'] * self.T_Roof +
                         coeff_dict['T_int_sur']['T_floor'] * self.T_Floor +
                         coeff_dict['T_int_sur']['T_preTemWin'] * self.T_preTemWin +
                         coeff_dict['T_int_sur']['QTraRad_flow'] * QTraRad_flow +
                         coeff_dict['T_int_sur']['Q_RadSol'] * self.Q_RadSol +
                         coeff_dict['T_int_sur']['q_ig_rad'] * q_ig_rad)
        T_ExtWall_sur = (coeff_dict['T_ext_sur']['T_Air'] * self.T_Air +
                         coeff_dict['T_ext_sur']['T_int'] * self.T_IntWall +
                         coeff_dict['T_ext_sur']['T_ext'] * self.T_ExtWall +
                         coeff_dict['T_ext_sur']['T_roof'] * self.T_Roof +
                         coeff_dict['T_ext_sur']['T_floor'] * self.T_Floor +
                         coeff_dict['T_ext_sur']['T_preTemWin'] * self.T_preTemWin +
                         coeff_dict['T_ext_sur']['QTraRad_flow'] * QTraRad_flow +
                         coeff_dict['T_ext_sur']['Q_RadSol'] * self.Q_RadSol +
                         coeff_dict['T_ext_sur']['q_ig_rad'] * q_ig_rad)
        T_Roof_sur = (coeff_dict['T_roof_sur']['T_Air'] * self.T_Air +
                      coeff_dict['T_roof_sur']['T_int'] * self.T_IntWall +
                      coeff_dict['T_roof_sur']['T_ext'] * self.T_ExtWall +
                      coeff_dict['T_roof_sur']['T_roof'] * self.T_Roof +
                      coeff_dict['T_roof_sur']['T_floor'] * self.T_Floor +
                      coeff_dict['T_roof_sur']['T_preTemWin'] * self.T_preTemWin +
                      coeff_dict['T_roof_sur']['QTraRad_flow'] * QTraRad_flow +
                      coeff_dict['T_roof_sur']['Q_RadSol'] * self.Q_RadSol +
                      coeff_dict['T_roof_sur']['q_ig_rad'] * q_ig_rad)
        T_Floor_sur = (coeff_dict['T_floor_sur']['T_Air'] * self.T_Air +
                       coeff_dict['T_floor_sur']['T_int'] * self.T_IntWall +
                       coeff_dict['T_floor_sur']['T_ext'] * self.T_ExtWall +
                       coeff_dict['T_floor_sur']['T_roof'] * self.T_Roof +
                       coeff_dict['T_floor_sur']['T_floor'] * self.T_Floor +
                       coeff_dict['T_floor_sur']['T_preTemWin'] * self.T_preTemWin +
                       coeff_dict['T_floor_sur']['QTraRad_flow'] * QTraRad_flow +
                       coeff_dict['T_floor_sur']['Q_RadSol'] * self.Q_RadSol +
                       coeff_dict['T_floor_sur']['q_ig_rad'] * q_ig_rad)
        T_Win_sur = (coeff_dict['T_win_sur']['T_Air'] * self.T_Air +
                     coeff_dict['T_win_sur']['T_int'] * self.T_IntWall +
                     coeff_dict['T_win_sur']['T_ext'] * self.T_ExtWall +
                     coeff_dict['T_win_sur']['T_roof'] * self.T_Roof +
                     coeff_dict['T_win_sur']['T_floor'] * self.T_Floor +
                     coeff_dict['T_win_sur']['T_preTemWin'] * self.T_preTemWin +
                     coeff_dict['T_win_sur']['QTraRad_flow'] * QTraRad_flow +
                     coeff_dict['T_win_sur']['Q_RadSol'] * self.Q_RadSol +
                     coeff_dict['T_win_sur']['q_ig_rad'] * q_ig_rad)

        self.T_IntWall.ode = (1 / self.CInt) * (
                (T_IntWall_sur - self.T_IntWall) * k_int
        )

        self.T_ExtWall.ode = (1 / self.CExt) * (
                (T_ExtWall_sur - self.T_ExtWall) * k_ext
                + (self.T_preTemWall - self.T_ExtWall) * k_amb_ext
        )

        self.T_Roof.ode = (1 / self.CRoof) * (
                (T_Roof_sur - self.T_Roof) * k_roof
                + (self.T_preTemRoof - self.T_Roof) * k_amb_roof
        )

        self.T_Floor.ode = (1 / self.CFloor) * (
                (T_Floor_sur - self.T_Floor) * k_floor
                + (self.T_preTemFloor - self.T_Floor) * k_amb_floor
        )

        self.T_Air.ode = (1 / self.CAir) * (
                (T_IntWall_sur - self.T_Air) * k_int_air
                + (T_ExtWall_sur - self.T_Air) * k_ext_air
                + (T_Roof_sur - self.T_Air) * k_roof_air
                + (T_Floor_sur - self.T_Air) * k_floor_air
                + (T_Win_sur - self.T_Air) * k_win_air
                + q_ig_conv
                + Q_RadSol_air
                + QTraCon_flow
                + Qdot_vent
        )

        # Predicted Air temperature in the next time step
        T_Air_next_predicted = self.T_Air + self.scale_obj * self.T_Air.ode

        # electric energy
        P_el_demand_devices = self.AZone * self.internalGainsMachinesSpecific * self.schedule_dev
        P_el_demand_lights = self.AZone * self.lightingPowerSpecific * self.schedule_light
        P_el_demand = P_el_demand_devices + P_el_demand_lights
        P_el_feed = P_el_demand + PEleHeaPum + PEleEleHea - self.P_el_pv

        # Constraints: List[(lower bound, function, upper bound)]
        # best practise: hard constraints -> soft with high penalty
        self.constraints = [
            # T_air
            (-ca.inf, self.T_Air - self.TAir_ub_slack, self.T_Air_ub),
            (self.T_Air_lb, self.T_Air + self.TAir_lb_slack, ca.inf),

            # TSetOneZone
            (0, self.TSetOneZone - T_Air_next_predicted + self.TSetOneZoneDiff_lb_slack, ca.inf),
            (-ca.inf, self.TSetOneZone - T_Air_next_predicted - self.TSetOneZoneDiff_ub_slack, 0),

            # heater
            (self.T_Air, TTraSup + self.TTraSup_slack, ca.inf),

            # heatpump
            # todo: check whether THeaCur is necessary as lower bound
            (273.15 + 30, THeaPumSup + self.THeaPumSup_slack, 273.15 + 70),
            # (0, QHeaPum_flow + self.QHeaPum_flow_slack, 11100),
            (0, QHeaPum_flow + self.QHeaPum_flow_slack, QHeaPum_flow_max),
            (0, PEleHeaPum + self.PEleHeaPum_slack, 6770 * self.scalingFactor),

            # valve
            #(-ca.inf, valve_output + self.s_valve_pos_ub, 0.99999),
            #(0.00001, valve_output + self.s_valve_pos_lb, ca.inf),

            # electric power
            (0, P_el_feed + self.P_el_feed_into_grid - self.P_el_feed_from_grid + self.s_Pel_feed, 0),
            # split into in and out to avoid if-else in costs
            (0, self.P_el_feed_into_grid, ca.inf),
            (0, self.P_el_feed_from_grid, ca.inf)
        ]

        #for n in range(1, self.config.nLayer + 1):
        #    T_TES_n = self._states.get(f"T_TES_{n}")
        #    s_T_TES_n = self._states.get(f"s_T_TES_{n}")
        #    self.constraints.append((self.T_TES_lb, T_TES_n + s_T_TES_n, self.T_TES_ub))

        # Objective function
        #C_el_feed_from_grid = self.c_grid / 3600 * PEleHeaPum / 1000  # €/kWh * 1h/3600s * W / 1000
        #C_el_feed_into_grid = self.c_feed_in / 3600 * self.P_el_feed_into_grid / 1000  # €/kWh * 1h/3600s * W / 1000
        #C_el = (C_el_feed_from_grid - C_el_feed_into_grid)  # €

        C_el = self.c_grid * PEleHeaPum / 1000  # €/kWh * 1h/3600s * W / 1000

        C_comf_upper = self.c_comf_upper * self.TAir_ub_slack ** 2
        C_comf_lower = self.c_comf_lower * self.TAir_lb_slack ** 2
        C_comf = C_comf_upper + C_comf_lower

        cost_terms = [
                1 * C_el / self.scale_obj * 1,
                C_comf / self.scale_obj,
                #1 / self.scale_obj * self.yValSet ** 2,
                100 / self.scale_obj * self.TTraSup_slack ** 2,
                1 / self.scale_obj * self.THeaPumSup_slack ** 2,
                100 / self.scale_obj * self.QHeaPum_flow_slack ** 2,
                100 / self.scale_obj * self.PEleHeaPum_slack ** 2,
                1 / self.scale_obj * self.TSetOneZoneDiff_ub_slack ** 2,
                1 / self.scale_obj * self.TSetOneZoneDiff_lb_slack ** 2,
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

        self.Qdot_Air_int.alg = (T_IntWall_sur - self.T_Air) * k_int_air
        self.Qdot_Air_ext.alg = (T_ExtWall_sur - self.T_Air) * k_ext_air
        self.Qdot_Air_roof.alg = (T_Roof_sur - self.T_Air) * k_roof_air
        self.Qdot_Air_floor.alg = (T_Floor_sur - self.T_Air) * k_floor_air
        self.Qdot_Air_win.alg = (T_Win_sur - self.T_Air) * k_win_air
        self.Qdot_Air_gain.alg = q_ig_conv
        self.Qdot_Air_sol.alg = Q_RadSol_air
        self.Qdot_Air_heater.alg = QTraCon_flow
        self.Qdot_Air.alg = (
                (T_IntWall_sur - self.T_Air) * k_int_air
                + (T_ExtWall_sur - self.T_Air) * k_ext_air
                + (T_Roof_sur - self.T_Air) * k_roof_air
                + (T_Floor_sur - self.T_Air) * k_floor_air
                + (T_Win_sur - self.T_Air) * k_win_air
                + q_ig_conv
                + Q_RadSol_air
                + QTraCon_flow
                + Qdot_vent
        )

        self.T_IntWall_sur.alg = T_IntWall_sur
        self.Qdot_IntWall_sur.alg = (T_IntWall_sur - self.T_IntWall) * k_int
        self.Qdot_IntWall.alg = (T_IntWall_sur - self.T_IntWall) * k_int

        self.T_ExtWall_sur.alg = T_ExtWall_sur
        self.Qdot_ExtWall_sur.alg = (T_ExtWall_sur - self.T_ExtWall) * k_ext
        self.T_ExtWall_pre.alg = self.T_preTemWall
        self.Qdot_ExtWall_pre.alg = (self.T_preTemWall - self.T_ExtWall) * k_amb_ext
        self.Qdot_ExtWall.alg = self.Qdot_ExtWall_sur + self.Qdot_ExtWall_pre

        self.T_Roof_sur.alg = T_Roof_sur
        self.Qdot_Roof_sur.alg = (T_Roof_sur - self.T_Roof) * k_roof
        self.T_Roof_pre.alg = self.T_preTemRoof
        self.Qdot_Roof_pre.alg = (self.T_preTemRoof - self.T_Roof) * k_amb_roof
        self.Qdot_Roof.alg = self.Qdot_Roof_sur + self.Qdot_Roof_pre

        self.T_Floor_sur.alg = T_Floor_sur
        self.Qdot_Floor_sur.alg = (T_Floor_sur - self.T_Floor) * k_floor
        self.T_Floor_pre.alg = self.T_preTemFloor
        self.Qdot_Floor_pre.alg = (self.T_preTemFloor - self.T_Floor) * k_amb_floor
        self.Qdot_Floor.alg = self.Qdot_Floor_sur + self.Qdot_Floor_pre

        self.valve_actual.alg = valve_actual
        self.mTra_flow.alg = mTra_flow
        self.TTraSup.alg = TTraSup
        self.TTraRet.alg = TTraOut
        self.QTra_flow.alg = QTra_flow
        self.QTraCon_flow.alg = QTraCon_flow
        self.QTraRad_flow.alg = QTraRad_flow

        self.Q_RadSol_or.alg = self.Q_RadSol
        self.Q_IntGains.alg = q_ig_conv + q_ig_rad
        self.Q_IntGains_conv.alg = q_ig_conv
        self.Q_IntGains_conv_machines.alg = q_devices_conv
        self.Q_IntGains_conv_lights.alg = q_lights_conv
        self.Q_IntGains_conv_humans.alg = q_humans_conv
        self.Q_IntGains_rad.alg = q_ig_rad
        self.Q_IntGains_rad_machines.alg = q_devices_rad
        self.Q_IntGains_rad_lights.alg = q_lights_rad
        self.Q_IntGains_rad_humans.alg = q_humans_rad
        self.Q_IntGains_machines.alg = q_devices
        self.Q_IntGains_lights.alg = q_lights
        self.Q_IntGains_humans.alg = q_humans

        self.ventRate_airExc.alg = ventRate
        self.Qdot_airExc.alg = Qdot_vent

        self.Qdot_hp.alg = QHeaPum_flow
        self.P_el_hp.alg = PEleHeaPum

        self.P_el_demand.alg = P_el_demand
        self.P_el_demand_devices.alg = P_el_demand_devices
        self.P_el_demand_lights.alg = P_el_demand_lights

        self.P_pv.alg = self.P_el_pv
        self.P_el_feed.alg = P_el_feed

        self.T_lb.alg = self.T_Air_lb

        return objective

    def get_T_layer(self, _n):
        return self._states.get(f"T_TES_{_n}")