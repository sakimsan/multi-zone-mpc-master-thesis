from agentlib_mpc.models.casadi_model import CasadiModel
from bes_rules.simulation_based_optimization.agentlib_mpc.get_idf_data import get_idf_data
import casadi as ca
import pandas as pd
from typing import Union, Tuple


class BuildingElements:

    def __init__(self, model: CasadiModel, zone1: str, **kwargs):
        self.m = model
        self.zone1 = zone1

        self.coeff_dict = kwargs.get("coeff_dict", None)
        self.material = kwargs.get("material", None)
        self.element_construction = kwargs.get("element_construction", None)
        self.windows = kwargs.get("windows", None)

        self.type = kwargs.get("type", None)


        self.multizone_coupled = self.m.multizone_coupled
        self.HOM_predictor = self.m.HOM_predictor
        self.calc_resistances_new = self.m.calc_resistances_new
        self.test_case = self.m.test_case

        self.compute_transmittance()
        if self.test_case:
            self.compute_surface_temperatures()


        if not self.m.HOM_predictor:
            self.T_Win_sur = self.compute_dynamic_T_sur("T_win_sur", self.zone1)
            self.var_outputs_zone("T_Win_sur_out").alg = self.T_Win_sur



    def var_parameter(self, base):
        value = self.m._parameters.get(f"{base}")
        if value is None:
            raise ValueError(f"Parameter '{base}' wurde nicht im Modell registriert.")
        return value

    def var_inputs(self, base):
        name = f"{base}"
        value = self.m._inputs.get(name)
        if value is None:
            raise ValueError(f"Input '{name}' wurde nicht im Modell registriert.")
        return value

    def var_inputs_zone(self, base):
        name = f"{base}_{self.zone1}"
        value = self.m._inputs.get(name)
        if value is None:
            raise ValueError(f"Zonen-Input '{name}' wurde nicht im Modell registriert.")
        return value

    def var_states_zone(self, base):
        name = f"{base}_{self.zone1}"
        value = self.m._states.get(name)
        if value is None:
            raise ValueError(f"Zonen-State '{name}' wurde nicht im Modell registriert.")
        return value

    def var_states(self, base):
        name = f"{base}"
        value = self.m._states.get(name)
        if value is None:
            raise ValueError(f"State '{name}' wurde nicht im Modell registriert.")
        return value

    def var_outputs_zone(self, base):
        name = f"{base}_{self.zone1}"
        value = self.m._outputs.get(name)
        if value is None:
            raise ValueError(f"Zonen-Output '{name}' wurde nicht im Modell registriert.")
        # todo: ggf.: print(name, id(value))
        return value

    def var_outputs(self, base):
        name = f"{base}"
        value = self.m._outputs.get(name)
        if value is None:
            raise ValueError(f"Output '{name}' wurde nicht im Modell registriert.")
        return value

    def zone_mapping(self, base):
        return f'{base}_{self.zone1}'

    def compute_surface_temperatures(self):
        m = self.m
        cd = self.coeff_dict

        def T_sur_test():
            self.T_IntWall_sur = (cd['T_int_sur']['T_Air'] * self.var_states_zone("T_Air") +
                             cd['T_int_sur']['T_int'] * self.var_states_zone("T_IntWall") +
                             cd['T_int_sur']['T_ext'] * self.var_states_zone("T_ExtWall") +
                             cd['T_int_sur']['T_roof'] * self.var_states_zone("T_Roof") +
                             cd['T_int_sur']['T_floor'] * self.var_states_zone("T_Floor") +
                             cd['T_int_sur']['T_preTemWin'] * self.var_inputs("T_preTemWin") +
                             cd['T_int_sur']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                             cd['T_int_sur']['Q_RadSol'] * self.var_inputs("Q_RadSol") +
                             cd['T_int_sur']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)
            self.T_ExtWall_sur = (cd['T_ext_sur']['T_Air'] * self.var_states_zone("T_Air") +
                             cd['T_ext_sur']['T_int'] * self.var_states_zone("T_IntWall") +
                             cd['T_ext_sur']['T_ext'] * self.var_states_zone("T_ExtWall") +
                             cd['T_ext_sur']['T_roof'] * self.var_states_zone("T_Roof") +
                             cd['T_ext_sur']['T_floor'] * self.var_states_zone("T_Floor") +
                             cd['T_ext_sur']['T_preTemWin'] * self.var_inputs("T_preTemWin") +
                             cd['T_ext_sur']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                             cd['T_ext_sur']['Q_RadSol'] * self.var_inputs("Q_RadSol") +
                             cd['T_ext_sur']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)
            self.T_Roof_sur = (cd['T_roof_sur']['T_Air'] * self.var_states_zone("T_Air") +
                                cd['T_roof_sur']['T_int'] * self.var_states_zone("T_IntWall") +
                                cd['T_roof_sur']['T_ext'] * self.var_states_zone("T_ExtWall") +
                                cd['T_roof_sur']['T_roof'] * self.var_states_zone("T_Roof") +
                                cd['T_roof_sur']['T_floor'] * self.var_states_zone("T_Floor") +
                                cd['T_roof_sur']['T_preTemWin'] * self.var_inputs("T_preTemWin") +
                                cd['T_roof_sur']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                                cd['T_roof_sur']['Q_RadSol'] * self.var_inputs("Q_RadSol") +
                                cd['T_roof_sur']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)
            self.T_Floor_sur = (cd['T_floor_sur']['T_Air'] * self.var_states_zone("T_Air") +
                           cd['T_floor_sur']['T_int'] * self.var_states_zone("T_IntWall") +
                           cd['T_floor_sur']['T_ext'] * self.var_states_zone("T_ExtWall") +
                           cd['T_floor_sur']['T_roof'] * self.var_states_zone("T_Roof") +
                           cd['T_floor_sur']['T_floor'] * self.var_states_zone("T_Floor") +
                           cd['T_floor_sur']['T_preTemWin'] * self.var_inputs("T_preTemWin") +
                           cd['T_floor_sur']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                           cd['T_floor_sur']['Q_RadSol'] * self.var_inputs("Q_RadSol") +
                           cd['T_floor_sur']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)
            self.T_Win_sur = (cd['T_win_sur']['T_Air'] * self.var_states_zone("T_Air") +
                         cd['T_win_sur']['T_int'] * self.var_states_zone("T_IntWall") +
                         cd['T_win_sur']['T_ext'] * self.var_states_zone("T_ExtWall") +
                         cd['T_win_sur']['T_roof'] * self.var_states_zone("T_Roof") +
                         cd['T_win_sur']['T_floor'] * self.var_states_zone("T_Floor") +
                         cd['T_win_sur']['T_preTemWin'] * self.var_inputs("T_preTemWin") +
                         cd['T_win_sur']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                         cd['T_win_sur']['Q_RadSol'] * self.var_inputs("Q_RadSol") +
                         cd['T_win_sur']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)

            return self.T_IntWall_sur, self.T_ExtWall_sur, self.T_Roof_sur, self.T_Floor_sur, self.T_Win_sur

        self.T_IntWall_sur, self.T_ExtWall_sur, self.T_Roof_sur, self.T_Floor_sur, self.T_Win_sur = T_sur_test()



        self.var_outputs_zone("T_Floor_sur_out").alg = self.T_Floor_sur
        self.var_outputs_zone("T_ExtWall_sur_out").alg = self.T_ExtWall_sur
        self.var_outputs_zone("T_IntWall_sur_out").alg = self.T_IntWall_sur
        self.var_outputs_zone("T_Roof_sur_out").alg = self.T_Roof_sur
        self.var_outputs_zone("T_Win_sur_out").alg = self.T_Win_sur

    def compute_dynamic_T_sur(self, surfacename: str, name: str):

        if self.calc_resistances_new:

            """
            NEU: per-Instanz-Oberflächentemperatur aus dem per-Zone coeff_dict.
            Erwartet:
              - self.element_construction = (instanz_name, fläche) z.B. ('OuterWall2', 8.58)
              - coeff_dict pro Zone mit Key 'T_s_<Instanz>' (z.B. 'T_s_OuterWall2')
            Mappt Symbolnamen aus dem coeff_dict auf deine CasADi-Variablen:
              'T_Air'              -> State  T_Air_<zone>
              'QTraRad_flow'       -> Output QTraRad_flow_out_<zone>.alg
              'q_ig_rad'           -> Output Q_IntGains_rad_<zone>.alg
              'Q_RadSol'           -> Input  Q_RadSol_<zone> (fallback global Q_RadSol)
              alle anderen Strings -> States <Symbol>_<zone> (z.B. T_ExtWall_OuterWall2_1_<zone>)
            """
            # 1) coeff_dict für diese Zone holen (m._zone_coeff gespeichert)
            zone_cd = self.coeff_dict or getattr(self.m, "_zone_coeff", {}).get(self.zone1, None)
            if zone_cd is None:
                raise ValueError(f"Kein coeff_dict für Zone '{self.zone1}' gefunden (self.calc_resistances_new=True).")

            # 2) Instanznamen aus element_construction ziehen
            if not self.element_construction or len(self.element_construction) < 1:
                raise ValueError("element_construction fehlt oder ist leer (erwarte z.B. ('OuterWall2', 8.58)).")
            inst_name = self.element_construction[0]  # 'OuterWall2', 'Dach3', 'Decke', 'GroundFloor', ...

            key = f"T_s_{inst_name}"
            if key not in zone_cd:
                # hilfreiche Fehlermeldung mit ein paar verfügbaren Keys
                some_keys = list(zone_cd.keys())[:10]
                raise KeyError(f"'{key}' nicht im coeff_dict. Verfügbar (Auszug): {some_keys}")

            coeffs = zone_cd[key]

            # 3) Helper zum Auflösen eines Symbolnamens -> CasADi-Variable/Expr
            def _resolve_symbol(sym_name: str):
                if sym_name == "T_Air":
                    return self.var_states_zone("T_Air")
                elif sym_name == "QTraRad_flow":
                    return self.var_outputs_zone("QTraRad_flow_out").alg
                elif sym_name == "q_ig_rad":
                    return self.var_outputs_zone("Q_IntGains_rad").alg
                elif sym_name == "Q_RadSol":
                    return self.var_inputs_zone("Q_RadSol")
                else:
                    # Bauteil-Knotentemperaturen kommen ohne Zonen-Suffix im coeff_dict
                    # -> State "<sym_name>_<zone>"
                    return self.var_states_zone(sym_name)

            # 4) Lineare Kombination zusammenbauen
            expr = float(coeffs.get("const", 0.0))
            for sym_name, k in coeffs.items():
                if sym_name == "const" or k == 0.0:
                    continue
                expr = expr + k * _resolve_symbol(sym_name)

            return expr

        else:
            cd = getattr(self.m, "_zone_coeff", {}).get(self.zone1, None)


            Surface_T = cd[f'{surfacename}']['T_Air'] * self.var_states_zone("T_Air")

            material: pd.DataFrame
            zone_construction: pd.DataFrame
            windows: pd.DataFrame

            material, zone_construction, windows = get_idf_data()


            numDach = 0
            tempAreaTot = 0
            AreaTot = 0
            dach = list(zone_construction.loc[name].filter(regex="^Dach").items())
            for roof in dach:
                if roof[1] > 0:
                    RoofName, RoofArea = roof

                    numDach += 1
                    tempAreaTot += self.var_states_zone(f"T_Roof_{RoofName}_1") * RoofArea
                    AreaTot += RoofArea
                else:
                    continue
            if AreaTot > 0:
                Surface_T = Surface_T + cd[f'{surfacename}']['T_roof'] * tempAreaTot/AreaTot



            numErdboden = 0
            tempAreaTot = 0
            AreaTot = 0
            erdboden = list(zone_construction.loc[name].filter(regex="^GroundFloor").items())
            for groundfloor in erdboden:
                if groundfloor[1] > 0:
                    GroundFloorName, GroundFloorArea = groundfloor

                    numErdboden += 1
                    tempAreaTot += self.var_states_zone(f"T_Floor_{GroundFloorName}_1") * GroundFloorArea
                    AreaTot += GroundFloorArea
                else:
                    continue
            if AreaTot > 0:
                Surface_T = Surface_T + cd[f'{surfacename}']['T_floor'] * tempAreaTot / AreaTot


            numInnenboden = 0
            tempAreaTot = 0
            AreaTot = 0
            innenboden = list(zone_construction.loc[name].filter(regex="^InnerFloor").items())
            for innerfloor in innenboden:
                if innerfloor[1] > 0:
                    InnerFloorName, InnerFloorArea = innerfloor

                    numInnenboden += 1
                    tempAreaTot += self.var_states_zone(f"T_Floor_{InnerFloorName}") * InnerFloorArea
                    AreaTot += InnerFloorArea
                else:
                    continue
            if AreaTot > 0:
                Surface_T = Surface_T + cd[f'{surfacename}']['T_floor'] * tempAreaTot / AreaTot


            numDecke = 0
            tempAreaTot = 0
            AreaTot = 0
            decke = list(zone_construction.loc[name].filter(regex="^Decke").items())
            for ceiling in decke:
                if ceiling[1] > 0:
                    InnerFloorName, InnerFloorArea = ceiling

                    numDecke += 1
                    tempAreaTot += self.var_states_zone(f"T_Floor_{InnerFloorName}") * InnerFloorArea
                    AreaTot += InnerFloorArea
                else:
                    continue
            if AreaTot > 0:
                Surface_T = Surface_T + cd[f'{surfacename}']['T_roof'] * tempAreaTot / AreaTot


            numInnenwand = 0
            tempAreaTot = 0
            AreaTot = 0
            innenwand = list(zone_construction.loc[name].filter(regex="^InnerWall").items())
            for innerwall in innenwand:
                if innerwall[1] > 0:
                    InnerWallName, InnerWallArea = innerwall

                    numInnenwand += 1
                    tempAreaTot += self.var_states_zone(f"T_IntWall_{InnerWallName}") * InnerWallArea
                    AreaTot += InnerWallArea
                else:
                    continue
            if AreaTot > 0:
                Surface_T = Surface_T + cd[f'{surfacename}']['T_int'] * tempAreaTot / AreaTot


            numAussenwand = 0
            tempAreaTot = 0
            AreaTot = 0
            aussenwand = list(zone_construction.loc[name].filter(regex="^OuterWall").items())
            for outerwall in aussenwand:
                if outerwall[1] > 0:
                    OuterWallName, OuterWallArea = outerwall

                    numAussenwand += 1
                    tempAreaTot += self.var_states_zone(f"T_ExtWall_{OuterWallName}_1") * OuterWallArea
                    AreaTot += OuterWallArea
                else:
                    continue
            if AreaTot > 0:
                Surface_T = Surface_T + cd[f'{surfacename}']['T_ext'] * tempAreaTot / AreaTot


            if self.m.HOM_predictor:
                if self.type == "InnerFloor":
                    Surface_T = (Surface_T +
                                 (cd[f'{surfacename}']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                                 cd[f'{surfacename}']['Q_RadSol'] * self.var_inputs_zone("Q_RadSol"))/numInnenboden +
                                 cd[f'{surfacename}']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)
                elif self.type == "GroundFloor":
                    Surface_T = ((Surface_T +
                                  cd[f'{surfacename}']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                                  cd[f'{surfacename}']['Q_RadSol'] * self.var_inputs_zone("Q_RadSol")) /numErdboden +
                                 cd[f'{surfacename}']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)
                elif self.type == "Ceiling":
                    Surface_T = (Surface_T +
                                 (cd[f'{surfacename}']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                                 cd[f'{surfacename}']['Q_RadSol'] * self.var_inputs_zone("Q_RadSol"))/numDecke +
                                 cd[f'{surfacename}']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)
                elif self.type == "OuterWall":
                    Surface_T = (Surface_T +
                                 (cd[f'{surfacename}']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                                 cd[f'{surfacename}']['Q_RadSol'] * self.var_inputs_zone("Q_RadSol"))/numAussenwand +
                                 cd[f'{surfacename}']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)
                elif self.type == "InnerWall":
                    Surface_T = (Surface_T +
                                 (cd[f'{surfacename}']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                                 cd[f'{surfacename}']['Q_RadSol'] * self.var_inputs_zone("Q_RadSol"))/numInnenwand +
                                 cd[f'{surfacename}']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)
                elif self.type == "Roof":
                    Surface_T = (Surface_T +
                                 (cd[f'{surfacename}']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                                 cd[f'{surfacename}']['Q_RadSol'] * self.var_inputs_zone("Q_RadSol"))/numDach +
                                 cd[f'{surfacename}']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)



            else:
                Surface_T = (Surface_T + cd[f'{surfacename}']['T_preTemWin'] * self.var_inputs("T_preTemWin") +
                             cd[f'{surfacename}']['QTraRad_flow'] * self.var_outputs_zone("QTraRad_flow_out").alg +
                             cd[f'{surfacename}']['Q_RadSol'] * self.var_inputs("Q_RadSol")/10 +
                             cd[f'{surfacename}']['q_ig_rad'] * self.var_outputs_zone("Q_IntGains_rad").alg)

            return Surface_T





    def compute_transmittance(self):
        # heat transfer coefficient
        self.k_int_air = self.var_parameter("hConInt") * self.var_parameter("AInttot")
        self.k_ext_air = self.var_parameter("hConExt") * self.var_parameter("AExttot")
        self.k_roof_air = self.var_parameter("hConRoof") * self.var_parameter("ARooftot")
        self.k_floor_air = self.var_parameter("hConFloor") * self.var_parameter("AFloortot")
        self.k_win_air = self.var_parameter("hConWin") * self.var_parameter("AWintot")

        # thermal conductivity coefficient
        self.k_int = 1 / self.var_parameter("RInt")
        self.k_amb_ext = 1 / (1 / (
                    (self.var_parameter("hConWallOut") + self.var_parameter("hRadWall")) * self.var_parameter(
                "AExttot")) + self.var_parameter("RExtRem"))
        self.k_ext = 1 / self.var_parameter("RExt")
        self.k_amb_roof = 1 / (1 / (
                    (self.var_parameter("hConRoofOut") + self.var_parameter("hRadRoof")) * self.var_parameter(
                "ARooftot")) + self.var_parameter("RRoofRem"))
        self.k_roof = 1 / self.var_parameter("RRoof")
        self.k_amb_floor = 1 / self.var_parameter("RFloorRem")
        self.k_floor = 1 / self.var_parameter("RFloor")

class GroundFloor(BuildingElements):
    def __init__(self, model: CasadiModel, zone1: str, coeff_dict: dict, material: pd, element_construction: Tuple[str, float], windows: pd):
        super().__init__(model, zone1, coeff_dict=coeff_dict, material=material, element_construction=element_construction, windows=windows, type="GroundFloor")
        self.set_equation()



    def set_equation(self):
        m = self.m

        if self.test_case:
            self.var_states_zone("T_Floor").ode = (1 / self.var_parameter("CFloor")) * (
                    (self.T_Floor_sur - self.var_states_zone("T_Floor")) * self.k_floor +
                    (self.var_inputs("T_preTemFloor") - self.var_states_zone("T_Floor")) * self.k_amb_floor)

            self.var_outputs_zone("Qdot_Floor_sur").alg = (self.T_Floor_sur - self.var_states_zone(
                "T_Floor")) * self.k_floor
            self.var_outputs_zone("T_Floor_pre").alg = self.var_inputs("T_preTemFloor")
            self.var_outputs_zone("Qdot_Floor_pre").alg = (self.var_inputs("T_preTemFloor") - self.var_states_zone(
                "T_Floor")) * self.k_amb_floor
            self.var_outputs_zone("Qdot_Floor").alg = (
                    (self.T_Floor_sur - self.var_states_zone("T_Floor")) * self.k_floor +
                    (self.var_inputs("T_preTemFloor") - self.var_states_zone("T_Floor")) * self.k_amb_floor)

        else:
            cd = self.coeff_dict

            match = self.material[self.material["Construction"].str.contains("GroundFloor", case=False)]
            if not match.empty:
                self.capacity_GroundFloor = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_GroundFloor = match.iloc[0]["R_total [m²K/W]"]
                self.k_amb_GroundFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
                self.k_int_GroundFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]


            else:
                raise Exception("IDF Data of GroundFloor invalid")

            self.GroundFloorName, self.GroundFloorArea = self.element_construction

            self.T_Floor_sur = self.compute_dynamic_T_sur("T_floor_sur", self.zone1)
            self.var_outputs_zone(f"T_Floor_sur_out_{self.GroundFloorName}").alg = self.T_Floor_sur


            self.var_states_zone(f"T_Floor_{self.GroundFloorName}_1").ode = (1 / (self.capacity_GroundFloor * self.GroundFloorArea)) * (
                (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.GroundFloorName}_1")) * (1/(self.resistance_GroundFloor / self.GroundFloorArea)) +
                (self.var_states_zone(f"T_Floor_{self.GroundFloorName}_2") - self.var_states_zone(f"T_Floor_{self.GroundFloorName}_1")) * (1/(self.resistance_GroundFloor / self.GroundFloorArea))
            )

            self.var_states_zone(f"T_Floor_{self.GroundFloorName}_2").ode = (1 / (self.capacity_GroundFloor * self.GroundFloorArea)) * (
                (self.var_states_zone(f"T_Floor_{self.GroundFloorName}_1") - self.var_states_zone(f"T_Floor_{self.GroundFloorName}_2")) * (1/(self.resistance_GroundFloor / self.GroundFloorArea)) +
                (self.var_inputs_zone(f"T_preTemFloor_{self.GroundFloorName}") - self.var_states_zone(f"T_Floor_{self.GroundFloorName}_2")) * (1/(self.k_amb_GroundFloor / self.GroundFloorArea))
            )



            #self.var_outputs_zone(f"Qdot_Floor_sur_{self.GroundFloorName}_1").alg = (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.GroundFloorName}")) * (1/(self.resistance_GroundFloor / self.GroundFloorArea))
            # self.var_outputs_zone(f"T_Floor_pre_{self.GroundFloorName}_1").alg = self.var_inputs("T_preTemFloor")
            # self.var_outputs_zone(f"Qdot_Floor_pre_{self.GroundFloorName}").alg = (self.var_inputs("T_preTemFloor") - self.var_states_zone(
            #     f"T_Floor_{self.GroundFloorName}")) * (1/(self.k_amb_GroundFloor / self.GroundFloorArea))
            # self.var_outputs_zone(f"Qdot_Floor_{self.GroundFloorName}").alg = (
            #         (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.GroundFloorName}")) * (1/(self.resistance_GroundFloor / self.GroundFloorArea)) +
            #         (self.var_inputs("T_preTemFloor") - self.var_states_zone(f"T_Floor_{self.GroundFloorName}")) * (1/(self.k_amb_GroundFloor / self.GroundFloorArea)))



class InnerFloor(BuildingElements):
    def __init__(self, model: CasadiModel, zone1: str, coeff_dict: dict, material: pd, element_construction: Tuple[str, float], windows:pd, **kwargs):
        super().__init__(model, zone1, coeff_dict=coeff_dict, material=material, element_construction=element_construction, windows=windows, type="InnerFloor")
        self.T_other_side=kwargs.get("T_other_side", None)
        if self.T_other_side is None: self.T_other_side = 293.15

        self.set_equation()

    def set_equation(self):
        if self.test_case:
            self.var_states_zone("T_Floor").ode = (1 / self.var_parameter("CFloor")) * (
                    (self.T_Floor_sur - self.var_states_zone("T_Floor")) * self.k_floor +
                    (self.var_inputs("T_preTemFloor") - self.var_states_zone("T_Floor")) * self.k_amb_floor
            )

            self.var_outputs_zone("Qdot_Floor_sur").alg = (self.T_Floor_sur - self.var_states_zone(
                "T_Floor")) * self.k_floor
            self.var_outputs_zone("T_Floor_pre").alg = self.var_inputs("T_preTemFloor")
            self.var_outputs_zone("Qdot_Floor_pre").alg = (self.var_inputs("T_preTemFloor") - self.var_states_zone(
                "T_Floor")) * self.k_amb_floor
            self.var_outputs_zone("Qdot_Floor").alg = (
                    (self.T_Floor_sur - self.var_states_zone("T_Floor")) * self.k_floor +
                    (self.var_inputs("T_preTemFloor") - self.var_states_zone("T_Floor")) * self.k_amb_floor
            )


        else:
            match = self.material[self.material["Construction"].str.contains("InnerFloor", case=False)]
            if not match.empty:
                self.capacity_InnerFloor = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_InnerFloor = match.iloc[0]["R_total [m²K/W]"]
                self.k_amb_InnerFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
                self.k_int_InnerFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of InnerFloor invalid")
            self.InnerFloorName, self.InnerFloorArea = self.element_construction


            self.T_Floor_sur = self.compute_dynamic_T_sur("T_floor_sur", self.zone1)

            self.var_outputs_zone(f"T_Floor_sur_out_{self.InnerFloorName}").alg = self.T_Floor_sur

            if self.multizone_coupled:
                self.var_states_zone(f"T_Floor_{self.InnerFloorName}").ode = (1 / (self.capacity_InnerFloor * self.InnerFloorArea)) * (
                        (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1 / (self.resistance_InnerFloor / self.InnerFloorArea)) +
                        (self.T_other_side - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1 / (self.k_int_InnerFloor / self.InnerFloorArea))
                )

            else:
                self.T_other_side = 293.15
                self.var_states_zone(f"T_Floor_{self.InnerFloorName}").ode = (1 / (self.capacity_InnerFloor * self.InnerFloorArea)) * (
                    (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1/(self.resistance_InnerFloor / self.InnerFloorArea))
                )


            # self.var_outputs_zone(f"Qdot_Floor_sur_{self.InnerFloorName}").alg = (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1/(self.resistance_InnerFloor / self.InnerFloorArea))
            # self.var_outputs_zone(f"Qdot_Floor_{self.InnerFloorName}").alg = (
            #     (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1/(self.resistance_InnerFloor / self.InnerFloorArea)) +
            #     (self.T_other_side - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1/(self.k_amb_InnerFloor / self.InnerFloorArea))
            # )


class Ceiling(BuildingElements):
    def __init__(self, model: CasadiModel, zone1: str, coeff_dict: dict, material: pd, element_construction: Tuple[str, float], windows:pd, **kwargs):
        super().__init__(model, zone1, coeff_dict=coeff_dict, material=material, element_construction=element_construction, windows=windows, type="Ceiling")
        self.T_other_side=kwargs.get("T_other_side", None)
        if self.T_other_side is None: self.T_other_side = 293.15


        self.set_equation()

    def set_equation(self):
        if self.test_case:
            self.var_states_zone("T_Floor").ode = (1 / self.var_parameter("CFloor")) * (
                    (self.T_Floor_sur - self.var_states_zone("T_Floor")) * self.k_floor +
                    (self.var_inputs("T_preTemFloor") - self.var_states_zone("T_Floor")) * self.k_amb_floor
            )

            self.var_outputs_zone("Qdot_Floor_sur").alg = (self.T_Floor_sur - self.var_states_zone(
                "T_Floor")) * self.k_floor
            self.var_outputs_zone("T_Floor_pre").alg = self.var_inputs("T_preTemFloor")
            self.var_outputs_zone("Qdot_Floor_pre").alg = (self.var_inputs("T_preTemFloor") - self.var_states_zone(
                "T_Floor")) * self.k_amb_floor
            self.var_outputs_zone("Qdot_Floor").alg = (
                    (self.T_Floor_sur - self.var_states_zone("T_Floor")) * self.k_floor +
                    (self.var_inputs("T_preTemFloor") - self.var_states_zone("T_Floor")) * self.k_amb_floor
            )


        else:
            match = self.material[self.material["Construction"].str.contains("InnerFloor", case=False)]
            if not match.empty:
                self.capacity_InnerFloor = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_InnerFloor = match.iloc[0]["R_total [m²K/W]"]
                self.k_amb_InnerFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
                self.k_int_InnerFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of InnerFloor invalid")
            self.InnerFloorName, self.InnerFloorArea = self.element_construction


            self.T_Floor_sur = self.compute_dynamic_T_sur("T_floor_sur", self.zone1)

            self.var_outputs_zone(f"T_Floor_sur_out_{self.InnerFloorName}").alg = self.T_Floor_sur

            if self.multizone_coupled:
                self.var_states_zone(f"T_Floor_{self.InnerFloorName}").ode = (1 / (self.capacity_InnerFloor * self.InnerFloorArea)) * (
                        (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1 / (self.resistance_InnerFloor / self.InnerFloorArea)) +
                        (self.T_other_side - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1 / (self.k_int_InnerFloor / self.InnerFloorArea))
                )

            else:
                self.T_other_side =  293.15
                self.var_states_zone(f"T_Floor_{self.InnerFloorName}").ode = (1 / (self.capacity_InnerFloor * self.InnerFloorArea)) * (
                    (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1/(self.resistance_InnerFloor / self.InnerFloorArea))
                )


            # self.var_outputs_zone(f"Qdot_Floor_sur_{self.InnerFloorName}").alg = (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1/(self.resistance_InnerFloor / self.InnerFloorArea))
            # self.var_outputs_zone(f"Qdot_Floor_{self.InnerFloorName}").alg = (
            #     (self.T_Floor_sur - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1/(self.resistance_InnerFloor / self.InnerFloorArea)) +
            #     (self.T_other_side - self.var_states_zone(f"T_Floor_{self.InnerFloorName}")) * (1/(self.k_amb_InnerFloor / self.InnerFloorArea))
            # )


class OuterWall(BuildingElements):
    def __init__(self, model: CasadiModel, zone1: str, coeff_dict: dict, material: pd, element_construction: Tuple[str, float], windows:pd, **kwargs):
        super().__init__(model, zone1, coeff_dict=coeff_dict, material=material, element_construction=element_construction, windows=windows, type="OuterWall")

        self.set_equation()


    def set_equation(self):
        if self.test_case:

            self.var_states_zone("T_ExtWall").ode = (1 / self.var_parameter("CExt")) * (
                    (self.T_ExtWall_sur - self.var_states_zone("T_ExtWall")) * self.k_ext +
                    (self.var_inputs("T_preTemWall") - self.var_states_zone("T_ExtWall")) * self.k_amb_ext
            )


            self.var_outputs_zone("Qdot_ExtWall_sur").alg = (self.T_ExtWall_sur - self.var_states_zone("T_ExtWall")) * self.k_ext
            self.var_outputs_zone("T_ExtWall_pre").alg = self.var_inputs("T_preTemWall")
            self.var_outputs_zone("Qdot_ExtWall_pre").alg = (self.var_inputs("T_preTemWall") - self.var_states_zone(
                "T_ExtWall")) * self.k_amb_ext
            self.var_outputs_zone("Qdot_ExtWall").alg = (
                    (self.T_ExtWall_sur - self.var_states_zone("T_ExtWall")) * self.k_ext +
                    (self.var_inputs("T_preTemWall") - self.var_states_zone("T_ExtWall")) * self.k_amb_ext
            )


        else:
            cd = self.coeff_dict



            match = self.material[self.material["Construction"].str.contains("OuterWall", case=False)]
            if not match.empty:
                self.capacity_OuterWall = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_OuterWall = match.iloc[0]["R_total [m²K/W]"]
                self.k_amb_OuterWall = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
                self.k_int_OuterWall = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of OuterWall invalid")

            self.OuterWallName, self.OuterWallArea = self.element_construction

            self.T_ExtWall_sur = self.compute_dynamic_T_sur("T_ext_sur", self.zone1)

            self.var_outputs_zone(f"T_ExtWall_sur_out_{self.OuterWallName}").alg = self.T_ExtWall_sur

            self.var_states_zone(f"T_ExtWall_{self.OuterWallName}_1").ode = (1 / (self.capacity_OuterWall * self.OuterWallArea)) * (
                (self.T_ExtWall_sur - self.var_states_zone(f"T_ExtWall_{self.OuterWallName}_1")) * (1/(self.resistance_OuterWall / self.OuterWallArea)) +
                (self.var_states_zone(f"T_ExtWall_{self.OuterWallName}_2") - self.var_states_zone(f"T_ExtWall_{self.OuterWallName}_1")) * (1/(self.resistance_OuterWall / self.OuterWallArea))
            )

            self.var_states_zone(f"T_ExtWall_{self.OuterWallName}_2").ode = (1 / (self.capacity_OuterWall * self.OuterWallArea)) * (
                (self.var_states_zone(f"T_ExtWall_{self.OuterWallName}_1") - self.var_states_zone(f"T_ExtWall_{self.OuterWallName}_2")) * (1/(self.resistance_OuterWall / self.OuterWallArea)) +
                (self.var_inputs_zone(f"T_preTemWall_{self.OuterWallName}") - self.var_states_zone(f"T_ExtWall_{self.OuterWallName}_2")) * (1/(self.k_amb_OuterWall / self.OuterWallArea))
            )


            # self.var_outputs_zone(f"Qdot_ExtWall_sur_{self.OuterWallName}").alg = (self.T_ExtWall_sur - self.var_states_zone(f"T_ExtWall_{self.OuterWallName}")) * (1/(self.resistance_OuterWall / self.OuterWallArea))
            # self.var_outputs_zone(f"T_ExtWall_pre_{self.OuterWallName}").alg = self.var_inputs("T_preTemWall")
            # self.var_outputs_zone(f"Qdot_ExtWall_pre_{self.OuterWallName}").alg = (self.var_inputs("T_preTemWall") - self.var_states_zone(f"T_ExtWall_{self.OuterWallName}")) * (1/(self.k_amb_OuterWall / self.OuterWallArea))
            # self.var_outputs_zone(f"Qdot_ExtWall_{self.OuterWallName}").alg = (
            #     (self.T_ExtWall_sur - self.var_states_zone(f"T_ExtWall_{self.OuterWallName}")) * (1/(self.resistance_OuterWall / self.OuterWallArea)) +
            #     (self.var_inputs("T_preTemWall") - self.var_states_zone(f"T_ExtWall_{self.OuterWallName}")) * (1/(self.k_amb_OuterWall / self.OuterWallArea))
            # )

        
class InnerWall(BuildingElements):
    def __init__(self, model: CasadiModel, zone1: str, coeff_dict: dict, material: pd, element_construction: Tuple[str, float], windows:pd, **kwargs):
        super().__init__(model, zone1, coeff_dict=coeff_dict, element_construction=element_construction, material=material, windows=windows, type="InnerWall")
        self.T_other_side=kwargs.get("T_other_side", None)
        if self.T_other_side is None: self.T_other_side = 293.15

        self.set_equation()

    def set_equation(self):
        if self.test_case:
            self.var_states_zone("T_IntWall").ode = (1 / self.var_parameter("CInt")) * (self.T_IntWall_sur - self.var_states_zone("T_IntWall")) * self.k_int

            self.var_outputs_zone("Qdot_IntWall_sur").alg = (self.T_IntWall_sur - self.var_states_zone("T_IntWall")) * self.k_int
            self.var_outputs_zone("Qdot_IntWall").alg = (self.T_IntWall_sur - self.var_states_zone("T_IntWall")) * self.k_int

        else:
            cd = self.coeff_dict


            match = self.material[self.material["Construction"].str.contains("InnerWall", case=False)]
            if not match.empty:
                self.capacity_InnerWall = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_InnerWall = match.iloc[0]["R_total [m²K/W]"]
                self.k_InnerWall = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of InnerWall invalid")

            self.InnerWallName, self.InnerWallArea = self.element_construction


            self.T_IntWall_sur = self.compute_dynamic_T_sur("T_int_sur", self.zone1)


            self.var_outputs_zone(f"T_IntWall_sur_out_{self.InnerWallName}").alg = self.T_IntWall_sur

            if self.multizone_coupled:
                self.var_states_zone(f"T_IntWall_{self.InnerWallName}").ode = (1 / (self.capacity_InnerWall * self.InnerWallArea)) * ((self.T_IntWall_sur - self.var_states_zone(f"T_IntWall_{self.InnerWallName}"))
                                                                              * (1 / (self.resistance_InnerWall / self.InnerWallArea))
                                                                              + (self.T_other_side - self.var_states_zone(f"T_IntWall_{self.InnerWallName}"))
                                                                              * (1 / (self.k_InnerWall / self.InnerWallArea)))
            else:
                self.T_other_side =  293.15
                self.var_states_zone(f"T_IntWall_{self.InnerWallName}").ode = (1 / (self.capacity_InnerWall * self.InnerWallArea)) * ((self.T_IntWall_sur - self.var_states_zone(f"T_IntWall_{self.InnerWallName}"))
                                                                              * (1 / (self.resistance_InnerWall / self.InnerWallArea))
                                                                              )


            # self.var_outputs_zone(f"Qdot_IntWall_sur_{self.InnerWallName}").alg = (self.T_IntWall_sur - self.var_states_zone(f"T_IntWall_{self.InnerWallName}")) * (1 / (self.k_InnerWall / self.InnerWallArea))
            # self.var_outputs_zone(f"Qdot_IntWall_{self.InnerWallName}").alg = ((self.T_IntWall_sur - self.var_states_zone(f"T_IntWall_{self.InnerWallName}"))
            #                                                                   * (1 / (self.resistance_InnerWall / self.InnerWallArea))
            #                                                                   + (self.T_other_side - self.var_states_zone(f"T_IntWall_{self.InnerWallName}"))
            #                                                                   * (1 / (self.k_InnerWall / self.InnerWallArea)))


class Roof(BuildingElements):
    def __init__(self, model: CasadiModel, zone1: str, coeff_dict: dict, material: pd, element_construction: Tuple[str, float], windows:pd, **kwargs):
        super().__init__(model, zone1, coeff_dict=coeff_dict, material=material, element_construction=element_construction, windows=windows, type="Roof")

        self.set_equation()

    def set_equation(self):
        if self.test_case:
            self.var_states_zone("T_Roof").ode = (1 / self.var_parameter("CRoof")) * (
                    (self.T_Roof_sur - self.var_states_zone("T_Roof")) * self.k_roof +
                    (self.var_inputs("T_preTemRoof") - self.var_states_zone("T_Roof")) * self.k_amb_roof
            )


            self.var_outputs_zone("Qdot_Roof_sur").alg = (self.T_Roof_sur - self.var_states_zone(
                "T_Roof")) * self.k_roof
            self.var_outputs_zone("T_Roof_pre").alg = self.var_inputs("T_preTemRoof")
            self.var_outputs_zone("Qdot_Roof_pre").alg = (self.var_inputs("T_preTemRoof") - self.var_states_zone(
                "T_Roof")) * self.k_amb_roof
            self.var_outputs_zone("Qdot_Roof").alg = (
                    (self.T_Roof_sur - self.var_states_zone("T_Roof")) * self.k_roof +
                    (self.var_inputs("T_preTemRoof") - self.var_states_zone("T_Roof")) * self.k_amb_roof
            )


        else:
            cd = self.coeff_dict


            match = self.material[self.material["Construction"].str.contains("Roof", case=False)]
            if not match.empty:
                self.capacity_Roof = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_Roof = match.iloc[0]["R_total [m²K/W]"]
                self.k_amb_Roof = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
                self.k_int_Roof = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of Roof invalid")

            self.RoofName, self.RoofArea = self.element_construction

            self.T_Roof_sur = self.compute_dynamic_T_sur("T_roof_sur", self.zone1)

            self.var_outputs_zone(f"T_Roof_sur_out_{self.RoofName}").alg = self.T_Roof_sur


            self.var_states_zone(f"T_Roof_{self.RoofName}_1").ode = (1 / (self.capacity_Roof * self.RoofArea)) * (
                (self.T_Roof_sur - self.var_states_zone(f"T_Roof_{self.RoofName}_1")) * (1/(self.resistance_Roof / self.RoofArea)) +
                (self.var_states_zone(f"T_Roof_{self.RoofName}_2") - self.var_states_zone(f"T_Roof_{self.RoofName}_1")) * (1/(self.resistance_Roof / self.RoofArea))
            )

            self.var_states_zone(f"T_Roof_{self.RoofName}_2").ode = (1 / (self.capacity_Roof * self.RoofArea)) * (
                (self.var_states_zone(f"T_Roof_{self.RoofName}_1") - self.var_states_zone(f"T_Roof_{self.RoofName}_2")) * (1/(self.resistance_Roof / self.RoofArea)) +
                (self.var_inputs_zone(f"T_preTemRoof_{self.RoofName}") - self.var_states_zone(f"T_Roof_{self.RoofName}_2")) * (1/(self.k_amb_Roof / self.RoofArea))
            )

            # self.var_outputs_zone(f"Qdot_Roof_sur_{self.RoofName}").alg = (self.T_Roof_sur - self.var_states_zone(f"T_Roof_{self.RoofName}")) * (1/(self.resistance_Roof / self.RoofArea))
            # self.var_outputs_zone(f"T_Roof_pre_{self.RoofName}").alg = self.var_inputs("T_preTemRoof")
            # self.var_outputs_zone(f"Qdot_Roof_pre_{self.RoofName}").alg = (self.var_inputs("T_preTemRoof") - self.var_states_zone(f"T_Roof_{self.RoofName}")) * (1/(self.k_amb_Roof / self.RoofArea))
            # self.var_outputs_zone(f"Qdot_Roof_{self.RoofName}").alg = (
            #     (self.T_Roof_sur - self.var_states_zone(f"T_Roof_{self.RoofName}")) * (1/(self.resistance_Roof / self.RoofArea)) +
            #     (self.var_inputs("T_preTemRoof") - self.var_states_zone(f"T_Roof_{self.RoofName}")) * (1/(self.k_amb_Roof / self.RoofArea))
            # )
