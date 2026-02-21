from agentlib_mpc.models.casadi_model import CasadiModel
import casadi as ca


def zone_test(
        casadi_model: CasadiModel,
        bes_parameters: dict,
        coeff_dict: dict,
        QTraCon_flow,
        QTraRad_flow

):
    q_humans = ((0.865 - (0.025 * (casadi_model.T_Air - 273.15))) * (
            casadi_model.activityDegree * 58 * 1.8) + 35) * casadi_model.specificPeople * casadi_model.AZone * casadi_model.schedule_human
    q_humans_conv = q_humans * casadi_model.ratioConvectiveHeatPeople
    q_humans_rad = q_humans * (1 - casadi_model.ratioConvectiveHeatPeople)

    q_devices = casadi_model.AZone * casadi_model.internalGainsMachinesSpecific * casadi_model.schedule_dev
    q_devices_conv = q_devices * casadi_model.ratioConvectiveHeatMachines
    q_devices_rad = q_devices * (1 - casadi_model.ratioConvectiveHeatMachines)

    q_lights = casadi_model.AZone * casadi_model.lightingPowerSpecific * casadi_model.schedule_light
    q_lights_conv = q_lights * casadi_model.ratioConvectiveHeatLighting
    q_lights_rad = q_lights * (1 - casadi_model.ratioConvectiveHeatLighting)

    q_ig_conv = q_humans_conv + q_devices_conv + q_lights_conv
    q_ig_rad = q_humans_rad + q_devices_rad + q_lights_rad

    ventRate = 0.5
    Qdot_vent = ventRate * casadi_model.VAir * casadi_model.air_rho * casadi_model.air_cp * (
                casadi_model.T_amb - casadi_model.T_Air) * (1 / 3600)

    # thermal transmittance
    # Air
    k_int_air = casadi_model.hConInt * casadi_model.AInttot
    k_ext_air = casadi_model.hConExt * casadi_model.AExttot
    k_roof_air = casadi_model.hConRoof * casadi_model.ARooftot
    k_floor_air = casadi_model.hConFloor * casadi_model.AFloortot
    k_win_air = casadi_model.hConWin * casadi_model.AWintot

    # Interior Walls
    k_int = 1 / casadi_model.RInt

    # Exterior Walls
    k_amb_ext = 1 / (
                1 / ((casadi_model.hConWallOut + casadi_model.hRadWall) * casadi_model.AExttot) + casadi_model.RExtRem)
    k_ext = 1 / casadi_model.RExt

    # Roof
    k_amb_roof = 1 / (1 / (
                (casadi_model.hConRoofOut + casadi_model.hRadRoof) * casadi_model.ARooftot) + casadi_model.RRoofRem)
    k_roof = 1 / casadi_model.RRoof

    # Floor
    k_amb_floor = 1 / casadi_model.RFloorRem
    k_floor = 1 / casadi_model.RFloor

    # Solar radiation to walls (approximated)
    Q_RadSol_air = (casadi_model.Q_RadSol / (casadi_model.gWin * (
            1 - casadi_model.ratioWinConRad) * casadi_model.ATransparent) * casadi_model.gWin * casadi_model.ratioWinConRad * casadi_model.ATransparent)

    # Calculate Surface Temperature of components
    T_IntWall_sur = (coeff_dict['T_int_sur']['T_Air'] * casadi_model.T_Air +
                     coeff_dict['T_int_sur']['T_int'] * casadi_model.T_IntWall +
                     coeff_dict['T_int_sur']['T_ext'] * casadi_model.T_ExtWall +
                     coeff_dict['T_int_sur']['T_roof'] * casadi_model.T_Roof +
                     coeff_dict['T_int_sur']['T_floor'] * casadi_model.T_Floor +
                     coeff_dict['T_int_sur']['T_preTemWin'] * casadi_model.T_preTemWin +
                     coeff_dict['T_int_sur']['QTraRad_flow'] * QTraRad_flow +
                     coeff_dict['T_int_sur']['Q_RadSol'] * casadi_model.Q_RadSol +
                     coeff_dict['T_int_sur']['q_ig_rad'] * q_ig_rad)
    T_ExtWall_sur = (coeff_dict['T_ext_sur']['T_Air'] * casadi_model.T_Air +
                     coeff_dict['T_ext_sur']['T_int'] * casadi_model.T_IntWall +
                     coeff_dict['T_ext_sur']['T_ext'] * casadi_model.T_ExtWall +
                     coeff_dict['T_ext_sur']['T_roof'] * casadi_model.T_Roof +
                     coeff_dict['T_ext_sur']['T_floor'] * casadi_model.T_Floor +
                     coeff_dict['T_ext_sur']['T_preTemWin'] * casadi_model.T_preTemWin +
                     coeff_dict['T_ext_sur']['QTraRad_flow'] * QTraRad_flow +
                     coeff_dict['T_ext_sur']['Q_RadSol'] * casadi_model.Q_RadSol +
                     coeff_dict['T_ext_sur']['q_ig_rad'] * q_ig_rad)
    T_Roof_sur = (coeff_dict['T_roof_sur']['T_Air'] * casadi_model.T_Air +
                  coeff_dict['T_roof_sur']['T_int'] * casadi_model.T_IntWall +
                  coeff_dict['T_roof_sur']['T_ext'] * casadi_model.T_ExtWall +
                  coeff_dict['T_roof_sur']['T_roof'] * casadi_model.T_Roof +
                  coeff_dict['T_roof_sur']['T_floor'] * casadi_model.T_Floor +
                  coeff_dict['T_roof_sur']['T_preTemWin'] * casadi_model.T_preTemWin +
                  coeff_dict['T_roof_sur']['QTraRad_flow'] * QTraRad_flow +
                  coeff_dict['T_roof_sur']['Q_RadSol'] * casadi_model.Q_RadSol +
                  coeff_dict['T_roof_sur']['q_ig_rad'] * q_ig_rad)
    T_Floor_sur = (coeff_dict['T_floor_sur']['T_Air'] * casadi_model.T_Air +
                   coeff_dict['T_floor_sur']['T_int'] * casadi_model.T_IntWall +
                   coeff_dict['T_floor_sur']['T_ext'] * casadi_model.T_ExtWall +
                   coeff_dict['T_floor_sur']['T_roof'] * casadi_model.T_Roof +
                   coeff_dict['T_floor_sur']['T_floor'] * casadi_model.T_Floor +
                   coeff_dict['T_floor_sur']['T_preTemWin'] * casadi_model.T_preTemWin +
                   coeff_dict['T_floor_sur']['QTraRad_flow'] * QTraRad_flow +
                   coeff_dict['T_floor_sur']['Q_RadSol'] * casadi_model.Q_RadSol +
                   coeff_dict['T_floor_sur']['q_ig_rad'] * q_ig_rad)
    T_Win_sur = (coeff_dict['T_win_sur']['T_Air'] * casadi_model.T_Air +
                 coeff_dict['T_win_sur']['T_int'] * casadi_model.T_IntWall +
                 coeff_dict['T_win_sur']['T_ext'] * casadi_model.T_ExtWall +
                 coeff_dict['T_win_sur']['T_roof'] * casadi_model.T_Roof +
                 coeff_dict['T_win_sur']['T_floor'] * casadi_model.T_Floor +
                 coeff_dict['T_win_sur']['T_preTemWin'] * casadi_model.T_preTemWin +
                 coeff_dict['T_win_sur']['QTraRad_flow'] * QTraRad_flow +
                 coeff_dict['T_win_sur']['Q_RadSol'] * casadi_model.Q_RadSol +
                 coeff_dict['T_win_sur']['q_ig_rad'] * q_ig_rad)

    casadi_model.T_IntWall.ode = (1 / casadi_model.CInt) * (
            (T_IntWall_sur - casadi_model.T_IntWall) * k_int
    )

    casadi_model.T_ExtWall.ode = (1 / casadi_model.CExt) * (
            (T_ExtWall_sur - casadi_model.T_ExtWall) * k_ext
            + (casadi_model.T_preTemWall - casadi_model.T_ExtWall) * k_amb_ext
    )

    casadi_model.T_Roof.ode = (1 / casadi_model.CRoof) * (
            (T_Roof_sur - casadi_model.T_Roof) * k_roof
            + (casadi_model.T_preTemRoof - casadi_model.T_Roof) * k_amb_roof
    )

    casadi_model.T_Floor.ode = (1 / casadi_model.CFloor) * (
            (T_Floor_sur - casadi_model.T_Floor) * k_floor
            + (casadi_model.T_preTemFloor - casadi_model.T_Floor) * k_amb_floor
    )

    casadi_model.T_Air.ode = (1 / casadi_model.CAir) * (
            (T_IntWall_sur - casadi_model.T_Air) * k_int_air
            + (T_ExtWall_sur - casadi_model.T_Air) * k_ext_air
            + (T_Roof_sur - casadi_model.T_Air) * k_roof_air
            + (T_Floor_sur - casadi_model.T_Air) * k_floor_air
            + (T_Win_sur - casadi_model.T_Air) * k_win_air
            + q_ig_conv
            + Q_RadSol_air
            + QTraCon_flow
            + Qdot_vent
    )

    # Predicted Air temperature in the next time step
    T_Air_next_predicted = casadi_model.T_Air + casadi_model.scale_obj * casadi_model.T_Air.ode

    casadi_model.constraints = [
        # T_air
        (-ca.inf, casadi_model.T_Air - casadi_model.TAir_ub_slack, casadi_model.T_Air_ub),
        (casadi_model.T_Air_lb, casadi_model.T_Air + casadi_model.TAir_lb_slack, ca.inf),

        # TSetOneZone
        (0, casadi_model.TSetOneZone - T_Air_next_predicted + casadi_model.TSetOneZoneDiff_lb_slack, ca.inf),
        (-ca.inf, casadi_model.TSetOneZone - T_Air_next_predicted - casadi_model.TSetOneZoneDiff_ub_slack, 0),

    ]

    return q_ig_conv, q_ig_rad, Qdot_vent


'''
    casadi_model.Q_RadSol_or.alg = casadi_model.Q_RadSol
    casadi_model.Q_IntGains.alg = q_ig_conv + q_ig_rad
    casadi_model.Q_IntGains_conv.alg = q_ig_conv
    casadi_model.Q_IntGains_conv_machines.alg = q_devices_conv
    casadi_model.Q_IntGains_conv_lights.alg = q_lights_conv
    casadi_model.Q_IntGains_conv_humans.alg = q_humans_conv
    casadi_model.Q_IntGains_rad.alg = q_ig_rad
    casadi_model.Q_IntGains_rad_machines.alg = q_devices_rad
    casadi_model.Q_IntGains_rad_lights.alg = q_lights_rad
    casadi_model.Q_IntGains_rad_humans.alg = q_humans_rad
    casadi_model.Q_IntGains_machines.alg = q_devices
    casadi_model.Q_IntGains_lights.alg = q_lights
    casadi_model.Q_IntGains_humans.alg = q_humans
'''

