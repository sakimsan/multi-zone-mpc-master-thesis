from agentlib_mpc.models.casadi_model import CasadiModel
from bes_rules.simulation_based_optimization.agentlib_mpc import component_models
from bes_rules.simulation_based_optimization.agentlib_mpc.calc_resistances_zone_specific import calc_resistances_zone_specific
from bes_rules.simulation_based_optimization.agentlib_mpc.calc_resistances_zone_element_specific import calc_resistances_zone_element_specific
#from bes_rules.simulation_based_optimization.agentlib_mpc.new_calc_resistances import calc_resistances_zone_element_specific

import casadi as ca

class single_zone_alpha:
    def __init__(self, model: CasadiModel, name: str, coeff_dict: dict, material, zone_construction):
        self.m = model
        self.name = name
        self.coeff_dict = coeff_dict
        self.material = material
        self.zone_construction = zone_construction
        self.HOM_predictor = self.m.HOM_predictor
        self.calc_resistances_new = self.m.calc_resistances_new


        if self.calc_resistances_new:
            self.m._zone_coeff[self.name] = calc_resistances_zone_element_specific(
                zone=self.name,
                material=self.material,
                zone_construction=self.zone_construction,
            )
        else:
            self.m._zone_coeff[self.name] = calc_resistances_zone_specific(name=self.name, HOM_Predictor=self.HOM_predictor)

            if self.m.test_case:
                self.compute_transmittance()
            else:
                self.k_win_air = self.var_parameter("hConWin") * self.var_parameter("AWintot")

        self.compute_internal_gains()
        self.compute_ventilation_losses()


        self.compute_QRadSol_air()

        self.radiator()


    def set_zone(self):
        self.assign_differential_equations()
        self.define_constraints()
        self.set_outputs()

    def var_parameter(self, base):
        value = self.m._parameters.get(f"{base}")
        if value is None:
            raise ValueError(f"Parameter '{base}' wurde nicht im Modell registriert.")
        return value
    
    def var_parameter_zone(self, base):
        name = f"{base}_{self.name}"
        value = self.m._parameters.get(name)
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
        name = f"{base}_{self.name}"
        value = self.m._inputs.get(name)
        if value is None:
            raise ValueError(f"Zonen-Input '{name}' wurde nicht im Modell registriert.")
        return value

    def var_states_zone(self, base):
        name = f"{base}_{self.name}"
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
        name = f"{base}_{self.name}"
        value = self.m._outputs.get(name)
        if value is None:
            raise ValueError(f"Zonen-Output '{name}' wurde nicht im Modell registriert.")
        return value

    def var_outputs(self, base):
        name = f"{base}"
        value = self.m._outputs.get(name)
        if value is None:
            raise ValueError(f"Output '{name}' wurde nicht im Modell registriert.")
        return value

    def zone_mapping(self, base):
        return f'{base}_{self.name}'
    
    def compute_QRadSol_air(self):
        if not self.m.HOM_predictor:
            self.Q_RadSol_air = (self.var_inputs("Q_RadSol") / (self.var_parameter("gWin") * (1 - self.var_parameter("ratioWinConRad")) * self.var_parameter("ATransparent")) * self.var_parameter("gWin") * self.var_parameter("ratioWinConRad") * self.var_parameter("ATransparent"))
            self.var_outputs_zone("Qdot_Air_sol").alg = self.Q_RadSol_air
            self.var_outputs_zone("Q_RadSol_or").alg = self.var_inputs("Q_RadSol")
        else:
            self.Q_RadSol_air = (self.var_inputs_zone("Q_RadSol") / (1 - self.var_parameter("ratioWinConRad")) * self.var_parameter("ratioWinConRad"))
            self.var_outputs_zone("Qdot_Air_sol").alg = self.Q_RadSol_air
            self.var_outputs_zone("Q_RadSol_or").alg = self.var_inputs_zone("Q_RadSol")
    
    def radiator(self):
        m = self.m
        self.valve_actual = self.var_parameter("valve_leakage") + self.var_inputs_zone("yValSet") * (1 - self.var_parameter("valve_leakage"))


        if self.name == "attic":
            self.mTra_flow = 1e-10
        else:
            if self.HOM_predictor:
                self.mTra_flow = self.var_parameter_zone("mTra_flow_nominal") * self.valve_actual
            else:
                self.mTra_flow = self.var_parameter("mTra_flow_nominal") / 10 * self.valve_actual

        def radiator_no_exponent_outlet_temperature(
                casadi_model: CasadiModel,
                TTraSup: float,
                mTra_flow: float,
                T_Air
        ):
            # assumption stationary energy balance
            # no delay by volume elements/thermal inertia
            # sim: simple radiator model (EN442-2), no delay between buffer storage and heater
            # from energy balance and heat transfer

            #if self.name == "attic":
            #    return T_Air + (TTraSup - T_Air) * ca.exp(-casadi_model.UA_heater / (mTra_flow * casadi_model.cp_water))
            #else:
            #    return T_Air + (TTraSup - T_Air) * ca.exp(-casadi_model.UA_heater/10 / (mTra_flow * casadi_model.cp_water))
            return T_Air + (TTraSup - T_Air) * ca.exp(
                - self.var_parameter_zone("UA_heater") / (mTra_flow * casadi_model.cp_water))


        self.TTraOut = radiator_no_exponent_outlet_temperature(
            casadi_model=m,
            TTraSup=m.TTraSup,
            T_Air=self.var_states_zone("T_Air"),
            mTra_flow=self.mTra_flow
        )


        self.QTra_flow = self.mTra_flow * self.var_parameter("cp_water") * (m.TTraSup - self.TTraOut)
        self.QTraRad_flow = self.QTra_flow * self.var_parameter("fraRad")
        self.QTraCon_flow = self.QTra_flow * (1 - self.var_parameter("fraRad"))


        self.var_outputs_zone("Qdot_Air_heater").alg = self.QTraCon_flow
        self.var_outputs_zone("valve_actual_out").alg = self.valve_actual
        self.var_outputs_zone("mTra_flow_out").alg = self.mTra_flow
        self.var_outputs_zone("TTraRet").alg = self.TTraOut
        self.var_outputs_zone("QTra_flow_out").alg = self.QTra_flow
        self.var_outputs_zone("QTraCon_flow_out").alg = self.QTraCon_flow
        self.var_outputs_zone("QTraRad_flow_out").alg = self.QTraRad_flow

    def compute_internal_gains(self):
        self.q_humans = ((0.865 - (0.025 * (self.var_states_zone("T_Air") - 273.15))) * (
                self.var_parameter("activityDegree") * 58 * 1.8) + 35) * self.var_parameter(
            "specificPeople") * self.var_parameter("AZone") * self.var_inputs("schedule_human")
        self.q_humans_conv = self.q_humans * self.var_parameter("ratioConvectiveHeatPeople")
        self.q_humans_rad = self.q_humans * (1 - self.var_parameter("ratioConvectiveHeatPeople"))

        self.q_devices = self.var_parameter("AZone") * self.var_parameter(
            "internalGainsMachinesSpecific") * self.var_inputs("schedule_dev")
        self.q_devices_conv = self.q_devices * self.var_parameter("ratioConvectiveHeatMachines")
        self.q_devices_rad = self.q_devices * (1 - self.var_parameter("ratioConvectiveHeatMachines"))

        self.q_lights = self.var_parameter("AZone") * self.var_parameter("lightingPowerSpecific") * self.var_inputs(
            "schedule_light")
        self.q_lights_conv = self.q_lights * self.var_parameter("ratioConvectiveHeatLighting")
        self.q_lights_rad = self.q_lights * (1 - self.var_parameter("ratioConvectiveHeatLighting"))

        self.q_ig_conv = self.q_humans_conv + self.q_devices_conv + self.q_lights_conv
        self.q_ig_rad = self.q_humans_rad + self.q_devices_rad + self.q_lights_rad

        self.var_outputs_zone("Q_IntGains").alg = self.q_ig_conv + self.q_ig_rad
        self.var_outputs_zone("Q_IntGains_conv").alg = self.q_ig_conv
        self.var_outputs_zone("Q_IntGains_conv_machines").alg = self.q_devices_conv
        self.var_outputs_zone("Q_IntGains_conv_lights").alg = self.q_lights_conv
        self.var_outputs_zone("Q_IntGains_conv_humans").alg = self.q_humans_conv
        self.var_outputs_zone("Q_IntGains_rad").alg = self.q_ig_rad
        self.var_outputs_zone("Q_IntGains_rad_machines").alg = self.q_devices_rad
        self.var_outputs_zone("Q_IntGains_rad_lights").alg = self.q_lights_rad
        self.var_outputs_zone("Q_IntGains_rad_humans").alg = self.q_humans_rad
        self.var_outputs_zone("Q_IntGains_machines").alg = self.q_devices
        self.var_outputs_zone("Q_IntGains_lights").alg = self.q_lights
        self.var_outputs_zone("Q_IntGains_humans").alg = self.q_humans
    
    def compute_ventilation_losses(self):

        if not self.name == "attic":
            volumes = list(self.zone_construction.loc[self.name].filter(regex=r"^Volume").items())
            for volume in volumes:
                if volume[1] > 0:
                    self.VolumeName, self.Volume = volume

            self.ventRate = 0.5
            self.Qdot_vent = self.ventRate * self.Volume * self.var_parameter("air_rho") * self.var_parameter("air_cp") * (self.var_inputs("T_amb") - self.var_states_zone("T_Air")) * (1 / 3600)
            self.var_outputs_zone("ventRate_airExc").alg = self.ventRate
            self.var_outputs_zone("Qdot_airExc").alg = self.Qdot_vent

        else:
            volumes = list(self.zone_construction.loc[self.name].filter(regex=r"^Volume").items())
            for volume in volumes:
                if volume[1] > 0:
                    self.VolumeName, self.Volume = volume

            self.ventRate = 1
            self.Qdot_vent = self.ventRate * self.Volume * self.var_parameter("air_rho") * self.var_parameter("air_cp") * (self.var_inputs("T_amb") - self.var_states_zone("T_Air")) * (1 / 3600)
            self.var_outputs_zone("ventRate_airExc").alg = self.ventRate
            self.var_outputs_zone("Qdot_airExc").alg = self.Qdot_vent

    
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
    
    def assign_differential_equations(self):
        m = self.m

        if self.m.test_case:
            self.var_states_zone("T_Air").ode = (1 / self.var_parameter("CAir")) * (
                (self.var_outputs_zone("T_IntWall_sur_out").alg - self.var_states_zone("T_Air")) * self.k_int_air +
                (self.var_outputs_zone("T_ExtWall_sur_out").alg - self.var_states_zone("T_Air")) * self.k_ext_air +
                (self.var_outputs_zone("T_Roof_sur_out").alg - self.var_states_zone("T_Air")) * self.k_roof_air +
                (self.var_outputs_zone("T_Floor_sur_out").alg - self.var_states_zone("T_Air")) * self.k_floor_air +
                (self.var_outputs_zone("T_Win_sur_out").alg - self.var_states_zone("T_Air")) * self.k_win_air +
                self.q_ig_conv + self.var_outputs_zone("Qdot_Air_sol").alg + self.QTraCon_flow + self.Qdot_vent
            )

        else:

            # Saubere Formulierung der T_Air-ODE nur mit Konvektion Luft <-> Innenoberflächen

            # Hilfsfunktion: hA = A / R_conv_in  (R_conv_in in [m²K/W], A in [m²]) → hA in [W/K]
            def hA(area, r_conv_m2K_per_W):
                return area / r_conv_m2K_per_W

            # 1) Luftzustand initialisieren
            self.var_states_zone("T_Air").ode = 0
            self.var_outputs_zone("Qdot_Air_negativ").alg = 0

            # --- ROOF (Dach) -----------------------------------------------------------------
            dach = list(self.zone_construction.loc[self.name].filter(regex=r"^Dach").items())
            match = self.material[self.material["Construction"].str.contains("Roof", case=False)]
            if not match.empty:
                self.capacity_Roof = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_Roof = match.iloc[0]["R_total [m²K/W]"]
                self.k_amb_Roof = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
                self.k_int_Roof = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of Roof invalid")

            for roof in dach:
                if roof[1] > 0:
                    self.RoofName, self.RoofArea = roof
                    # Verwendung der vorhandenen innenseitigen Oberfläche gemäß deinem Naming: *_sur_out_...
                    T_surf_roomside = self.var_outputs_zone(f"T_Roof_sur_out_{self.RoofName}").alg
                    self.var_states_zone("T_Air").ode += hA(self.RoofArea, self.k_int_Roof) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))

                    self.var_outputs_zone("Qdot_Air_negativ").alg += hA(self.RoofArea, self.k_int_Roof) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))



            # --- GROUND FLOOR (Boden gegen Erdreich) -----------------------------------------
            erdboden = list(self.zone_construction.loc[self.name].filter(regex=r"^GroundFloor").items())
            match = self.material[self.material["Construction"].str.contains("GroundFloor", case=False)]
            if not match.empty:
                self.capacity_GroundFloor = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_GroundFloor = match.iloc[0]["R_total [m²K/W]"]
                self.k_amb_GroundFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
                self.k_int_GroundFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of GroundFloor invalid")

            for groundfloor in erdboden:
                if groundfloor[1] > 0:
                    self.GroundFloorName, self.GroundFloorArea = groundfloor
                    T_surf_roomside = self.var_outputs_zone(f"T_Floor_sur_out_{self.GroundFloorName}").alg

                    self.var_states_zone("T_Air").ode += hA(self.GroundFloorArea, self.k_int_GroundFloor) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))

                    self.var_outputs_zone("Qdot_Air_negativ").alg += hA(self.GroundFloorArea, self.k_int_GroundFloor) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))



            # --- INNER FLOOR (Innenboden) ----------------------------------------------------
            innenboden = list(self.zone_construction.loc[self.name].filter(regex=r"^InnerFloor").items())
            match = self.material[self.material["Construction"].str.contains("InnerFloor", case=False)]
            if not match.empty:
                self.capacity_InnerFloor = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_InnerFloor = match.iloc[0]["R_total [m²K/W]"]
                self.k_amb_InnerFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
                self.k_int_InnerFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of InnerFloor invalid")

            for innerfloor in innenboden:
                if innerfloor[1] > 0:
                    self.InnerFloorName, self.InnerFloorArea = innerfloor
                    T_surf_roomside = self.var_outputs_zone(f"T_Floor_sur_out_{self.InnerFloorName}").alg

                    self.var_states_zone("T_Air").ode += hA(self.InnerFloorArea, self.k_int_InnerFloor) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))

                    self.var_outputs_zone("Qdot_Air_negativ").alg += hA(self.InnerFloorArea, self.k_int_InnerFloor) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))

            # --- DECKE (Ceiling; Zuordnung wie bei dir: Material = InnerFloor) --------------
            decke = list(self.zone_construction.loc[self.name].filter(regex=r"^Decke").items())
            match = self.material[self.material["Construction"].str.contains("InnerFloor", case=False)]
            if not match.empty:
                self.capacity_InnerFloor = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_InnerFloor = match.iloc[0]["R_total [m²K/W]"]
                self.k_amb_InnerFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
                self.k_int_InnerFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of InnerFloor invalid (Ceiling)")

            for ceiling in decke:
                if ceiling[1] > 0:
                    self.InnerFloorName, self.InnerFloorArea = ceiling
                    # Naming analog zu deinem Floor-Schema; falls du separate Ceiling-Outputs hast, bitte hier anpassen.
                    T_surf_roomside = self.var_outputs_zone(f"T_Floor_sur_out_{self.InnerFloorName}").alg

                    self.var_states_zone("T_Air").ode += hA(self.InnerFloorArea, self.k_int_InnerFloor) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))

                    self.var_outputs_zone("Qdot_Air_negativ").alg += hA(self.InnerFloorArea, self.k_int_InnerFloor) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))

            # --- INNER WALL (Innenwände) -----------------------------------------------------
            innenwand = list(self.zone_construction.loc[self.name].filter(regex=r"^InnerWall").items())
            match = self.material[self.material["Construction"].str.contains("InnerWall", case=False)]
            if not match.empty:
                self.capacity_InnerWall = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_InnerWall = match.iloc[0]["R_total [m²K/W]"]
                self.k_int_InnerWall = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of InnerWall invalid")

            for innerwall in innenwand:
                if innerwall[1] > 0:
                    self.InnerWallName, self.InnerWallArea = innerwall
                    T_surf_roomside = self.var_outputs_zone(f"T_IntWall_sur_out_{self.InnerWallName}").alg
                    self.var_states_zone("T_Air").ode += hA(self.InnerWallArea, self.k_int_InnerWall) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))

                    self.var_outputs_zone("Qdot_Air_negativ").alg += hA(self.InnerWallArea, self.k_int_InnerWall) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))

            # --- OUTER WALL (Außenwände, innenseitige Konvektion) ---------------------------
            aussenwand = list(self.zone_construction.loc[self.name].filter(regex=r"^OuterWall").items())
            match = self.material[self.material["Construction"].str.contains("OuterWall", case=False)]
            if not match.empty:
                self.capacity_OuterWall = match.iloc[0]["C_total [J/K·m²]"]
                self.resistance_OuterWall = match.iloc[0]["R_total [m²K/W]"]
                self.k_amb_OuterWall = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
                self.k_int_OuterWall = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
            else:
                raise Exception("IDF Data of OuterWall invalid")

            for outerwall in aussenwand:
                if outerwall[1] > 0:
                    self.OuterWallName, self.OuterWallArea = outerwall
                    T_surf_roomside = self.var_outputs_zone(f"T_ExtWall_sur_out_{self.OuterWallName}").alg
                    self.var_states_zone("T_Air").ode += hA(self.OuterWallArea, self.k_int_OuterWall) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))

                    self.var_outputs_zone("Qdot_Air_negativ").alg += hA(self.OuterWallArea, self.k_int_OuterWall) * (
                                T_surf_roomside - self.var_states_zone("T_Air"))

            if m.HOM_predictor:
                self.var_states_zone("T_Air").ode += (
                        self.q_ig_conv
                        + self.var_inputs_zone("ZoneWindowsTotalHeatRate")
                        + self.var_outputs_zone("Qdot_Air_sol").alg
                        + self.QTraCon_flow
                        + self.Qdot_vent
                )
                self.var_outputs_zone("Qdot_Air_negativ").alg += self.var_inputs_zone("ZoneWindowsTotalHeatRate")
                self.var_outputs_zone("Qdot_Air_positiv").alg = self.q_ig_conv  + self.var_outputs_zone("Qdot_Air_sol").alg + self.QTraCon_flow
                self.var_outputs_zone("ZoneWindowsTotalHeatRate_or").alg = self.var_inputs_zone("ZoneWindowsTotalHeatRate")

            else:
                # --- WINDOW (Fenster: innere Scheibenoberfläche über *_sur_out) -----------------
                # Falls du k_win_air bereits als hA (W/K) modellierst, nutze direkt k_win_air; ansonsten hA(A, R_conv_in)
                T_win_roomside = self.var_outputs_zone("T_Win_sur_out").alg
                self.var_states_zone("T_Air").ode += self.k_win_air/10 * (T_win_roomside - self.var_states_zone("T_Air"))
                # --- REIN KONVEKTIVE QUELLEN ----------------------------------------------------
                self.var_states_zone("T_Air").ode += (
                        self.q_ig_conv
                        + self.var_outputs_zone("Qdot_Air_sol").alg / 10
                        + self.QTraCon_flow
                        + self.Qdot_vent
                )

                self.var_outputs_zone("Qdot_Air_positiv").alg = self.q_ig_conv + self.var_outputs_zone(
                    "Qdot_Air_sol").alg + self.QTraCon_flow

                self.var_outputs_zone("Qdot_Air_negativ").alg += (self.var_outputs_zone(
                    "T_Win_sur_out").alg - self.var_states_zone("T_Air")) * self.k_win_air



            # --- Normierung auf Wärmekapazität der Luft ------------------------------------

            volumes = list(self.zone_construction.loc[self.name].filter(regex=r"^Volume").items())
            for volume in volumes:
                if volume[1] > 0:
                    self.VolumeName, self.Volume = volume

            self.var_states_zone("T_Air").ode = (self.var_states_zone("T_Air").ode) * (1 / (self.var_parameter("air_cp") * self.var_parameter("air_rho") * self.Volume))



    def define_constraints(self):
        m = self.m

        if m.HOM_predictor:
            if not self.name == "attic":
                
                # Slacks >= 0
                m.constraints += [(0.0, self.var_states_zone("TAir_ub_slack"), ca.inf)]
                m.constraints += [(0.0, self.var_states_zone("TAir_lb_slack"), ca.inf)]
                m.constraints += [(0.0, self.var_states("TTraSup_slack"),    ca.inf)]

                # T_Air <= TTraSup + TTraSup_slack
                m.constraints += [(-ca.inf, self.var_states_zone("T_Air") - (m.TTraSup + self.var_states("TTraSup_slack")), 0.0)]


                
                #m.constraints += [(-ca.inf, self.var_states_zone("T_Air") - self.var_states_zone("TAir_ub_slack") - (self.var_inputs_zone("TSetOneZone") + 1.0), 0.0)]
                #m.constraints += [(0.0,  self.var_states_zone("T_Air") + self.var_states_zone("TAir_lb_slack")- (self.var_inputs_zone("TSetOneZone") - 1.0),  ca.inf)]

                m.constraints += [(-ca.inf, self.var_states_zone("T_Air") - self.var_states_zone("TAir_ub_slack") - (self.var_inputs_zone("TSetOneZone") + 2.0), 0.0)]
                m.constraints += [(0.0, self.var_states_zone("T_Air") + self.var_states_zone("TAir_lb_slack") - (self.var_inputs_zone("TSetOneZone")), ca.inf)]

                # m.constraints += [(-ca.inf, self.var_states_zone("T_Air") - self.var_states_zone("TAir_ub_slack") - self.var_parameter("T_Air_ub"), 0.0)]
                # m.constraints += [(0.0, self.var_states_zone("T_Air") + self.var_states_zone("TAir_lb_slack") - (self.var_inputs_zone("TSetOneZone")), ca.inf)]



            self.var_outputs_zone("TSetOneZone").alg = self.var_inputs_zone("TSetOneZone")
        else:
            m.constraints += [
                # heater
                (self.var_states_zone("T_Air"), m.TTraSup + self.var_states("TTraSup_slack"), ca.inf),
                (-ca.inf, self.var_states_zone("T_Air") - self.var_states_zone("TAir_ub_slack"), self.var_parameter("T_Air_ub")),
                (self.var_parameter("T_Air_lb"), self.var_states_zone("T_Air") + self.var_states_zone("TAir_lb_slack"), ca.inf),
            ]

    def set_outputs(self):
        m = self.m

        if self.m.test_case:
            self.var_outputs_zone("Qdot_Air_int").alg = (self.var_outputs_zone("T_IntWall_sur_out").alg - self.var_states_zone("T_Air")) * self.k_int_air
            self.var_outputs_zone("Qdot_Air_ext").alg = (self.var_outputs_zone("T_ExtWall_sur_out").alg - self.var_states_zone("T_Air")) * self.k_ext_air
            self.var_outputs_zone("Qdot_Air_roof").alg = (self.var_outputs_zone("T_Roof_sur_out").alg - self.var_states_zone("T_Air")) * self.k_roof_air
            self.var_outputs_zone("Qdot_Air_floor").alg = (self.var_outputs_zone("T_Floor_sur_out").alg - self.var_states_zone("T_Air")) * self.k_floor_air
            self.var_outputs_zone("Qdot_Air_win").alg = (self.var_outputs_zone("T_Win_sur_out").alg - self.var_states_zone("T_Air")) * self.k_win_air
            self.var_outputs_zone("Qdot_Air_gain").alg = self.q_ig_conv

            self.var_outputs_zone("Qdot_Air").alg = (
                    (self.var_outputs_zone("T_IntWall_sur_out").alg - self.var_states_zone("T_Air")) * self.k_int_air +
                    (self.var_outputs_zone("T_ExtWall_sur_out").alg - self.var_states_zone("T_Air")) * self.k_ext_air +
                    (self.var_outputs_zone("T_Roof_sur_out").alg - self.var_states_zone("T_Air")) * self.k_roof_air +
                    (self.var_outputs_zone("T_Floor_sur_out").alg - self.var_states_zone("T_Air")) * self.k_floor_air +
                    (self.var_outputs_zone("T_Win_sur_out").alg - self.var_states_zone("T_Air")) * self.k_win_air +
                    self.q_ig_conv +
                    self.var_outputs_zone("Qdot_Air_sol").alg +
                    self.QTraCon_flow
                    #+self.Qdot_vent
            )

            self.var_outputs_zone("Q_RadSol_or").alg = self.var_inputs("Q_RadSol")





