import sympy
from bes_rules.simulation_based_optimization.agentlib_mpc.get_idf_data import get_idf_data
import pandas as pd



def inv0(x):  # CHANGED
    """1/x, aber 0 wenn x==0 oder None."""  # CHANGED
    return 0.0 if x in (0, None) else 1.0 / x  # CHANGED

def div0(num, den):  # CHANGED
    """num/den, aber 0 wenn den==0 oder None."""  # CHANGED
    return 0.0 if den in (0, None) else num / den  # CHANGED

def series_cond(UA, Rrem):  # CHANGED
    """
    k = 1 / (1/UA + Rrem), aber mit Zero-Guards.  # CHANGED
    Wenn 1/UA eine Division durch 0 wäre (UA==0) oder der gesamte Nenner 0 wird,  # CHANGED
    setzen wir den gesamten Term k auf 0 (gemäß deiner Vorgabe).  # CHANGED
    """  # CHANGED
    if UA in (0, None):  # CHANGED
        return 0.0  # CHANGED
    denom = inv0(UA) + (0.0 if Rrem is None else Rrem)  # CHANGED
    return 0.0 if denom in (0, None) else 1.0 / denom  # CHANGED
# -------------------------------------------------------


def calc_resistances_zone_specific(name: str, HOM_Predictor: bool):

    if HOM_Predictor:
        material: pd.DataFrame
        zone_construction: pd.DataFrame
        windows: pd.DataFrame

        material, zone_construction, windows = get_idf_data()

        ainttot = 0
        aexttot = 0
        arooftot = 0
        afloortot = 0

        zone_parameters: dict[str, float] = {}

        zone_parameters['hRad'] = 5

        # --- ROOF (Dach) -----------------------------------------------------------------
        dach = list(zone_construction.loc[name].filter(regex=r"^Dach").items())
        match = material[material["Construction"].str.contains("Roof", case=False)]
        if not match.empty:
            capacity_Roof = match.iloc[0]["C_total [J/K·m²]"]
            resistance_Roof = match.iloc[0]["R_total [m²K/W]"]
            k_amb_Roof = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
            k_int_Roof = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of Roof invalid")

        for roof in dach:
            if roof[1] > 0:
                RoofName, RoofArea = roof
                arooftot += RoofArea

        # --- GROUND FLOOR (Boden gegen Erdreich) -----------------------------------------
        erdboden = list(zone_construction.loc[name].filter(regex=r"^GroundFloor").items())
        match = material[material["Construction"].str.contains("GroundFloor", case=False)]
        if not match.empty:
            capacity_GroundFloor = match.iloc[0]["C_total [J/K·m²]"]
            resistance_GroundFloor = match.iloc[0]["R_total [m²K/W]"]
            k_amb_GroundFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
            k_int_GroundFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of GroundFloor invalid")

        for groundfloor in erdboden:
            if groundfloor[1] > 0:
                GroundFloorName, GroundFloorArea = groundfloor
                afloortot += GroundFloorArea

        # --- INNER FLOOR (Innenboden) ----------------------------------------------------
        innenboden = list(zone_construction.loc[name].filter(regex=r"^InnerFloor").items())
        match = material[material["Construction"].str.contains("InnerFloor", case=False)]
        if not match.empty:
            capacity_InnerFloor = match.iloc[0]["C_total [J/K·m²]"]
            resistance_InnerFloor = match.iloc[0]["R_total [m²K/W]"]
            k_amb_InnerFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
            k_int_InnerFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of InnerFloor invalid")

        for innerfloor in innenboden:
            if innerfloor[1] > 0:
                InnerFloorName, InnerFloorArea = innerfloor
                afloortot += InnerFloorArea

        # --- DECKE (Ceiling; Zuordnung wie bei dir: Material = InnerFloor) --------------
        decke = list(zone_construction.loc[name].filter(regex=r"^Decke").items())
        match = material[material["Construction"].str.contains("InnerFloor", case=False)]
        if not match.empty:
            capacity_InnerFloor = match.iloc[0]["C_total [J/K·m²]"]
            resistance_InnerFloor = match.iloc[0]["R_total [m²K/W]"]
            k_amb_InnerFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
            k_int_InnerFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of InnerFloor invalid (Ceiling)")

        for ceiling in decke:
            if ceiling[1] > 0:
                InnerFloorName, InnerFloorArea = ceiling
                arooftot += InnerFloorArea

        # --- INNER WALL (Innenwände) -----------------------------------------------------
        innenwand = list(zone_construction.loc[name].filter(regex=r"^InnerWall").items())
        match = material[material["Construction"].str.contains("InnerWall", case=False)]
        if not match.empty:
            capacity_InnerWall = match.iloc[0]["C_total [J/K·m²]"]
            resistance_InnerWall = match.iloc[0]["R_total [m²K/W]"]
            k_int_InnerWall = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of InnerWall invalid")

        for innerwall in innenwand:
            if innerwall[1] > 0:
                InnerWallName, InnerWallArea = innerwall
                ainttot += InnerWallArea

        # --- OUTER WALL (Außenwände, innenseitige Konvektion) ---------------------------
        aussenwand = list(zone_construction.loc[name].filter(regex=r"^OuterWall").items())
        match = material[material["Construction"].str.contains("OuterWall", case=False)]
        if not match.empty:
            capacity_OuterWall = match.iloc[0]["C_total [J/K·m²]"]
            resistance_OuterWall = match.iloc[0]["R_total [m²K/W]"]
            k_amb_OuterWall = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
            k_int_OuterWall = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of OuterWall invalid")

        for outerwall in aussenwand:
            if outerwall[1] > 0:
                OuterWallName, OuterWallArea = outerwall
                aexttot += OuterWallArea

        # building envelope

        area_tot = ainttot + aexttot + arooftot + afloortot

        # split factor for internal radiation (internal gains, radiator)
        split_rad_int = {
            'int': div0(ainttot, area_tot),   # CHANGED
            'ext': div0(aexttot, area_tot),   # CHANGED
            'roof': div0(arooftot, area_tot), # CHANGED
            'floor': div0(afloortot, area_tot),  # CHANGED
        }

        # split factors for solar radiation
        split_rad_sol = {
            'int': 0.125,
            'ext': 0.125,
            'roof': 0.25,
            'floor': 0.5,
        }

        """
        Calculates the coefficients for the algebraic equations of the surface temperatures.
        """
        # inputs
        QTraRad_flow = sympy.symbols('QTraRad_flow')
        q_ig_rad = sympy.symbols('q_ig_rad')
        Q_RadSol = sympy.symbols('Q_RadSol')
        # states
        T_Air = sympy.symbols('T_Air')
        T_int = sympy.symbols('T_int')
        T_ext = sympy.symbols('T_ext')
        T_roof = sympy.symbols('T_roof')
        T_floor = sympy.symbols('T_floor')
        # surface temps
        T_int_sur = sympy.symbols('T_int_sur')
        T_ext_sur = sympy.symbols('T_ext_sur')
        T_roof_sur = sympy.symbols('T_roof_sur')
        T_floor_sur = sympy.symbols('T_floor_sur')

        # parameters
        # split factors
        split_rad_int_int, split_rad_sol_int = split_rad_int['int'], split_rad_sol['int']
        split_rad_int_ext, split_rad_sol_ext = split_rad_int['ext'], split_rad_sol['ext']
        split_rad_int_roof, split_rad_sol_roof = split_rad_int['roof'], split_rad_sol['roof']
        split_rad_int_floor, split_rad_sol_floor = split_rad_int['floor'], split_rad_sol['floor']

        # thermal transmittance
        # air
        k_int_air = inv0(k_int_InnerWall) * ainttot    # CHANGED
        k_ext_air = inv0(k_int_OuterWall) * aexttot    # CHANGED
        k_roof_air = inv0(k_int_Roof) * arooftot       # CHANGED
        k_floor_air = inv0(k_int_InnerFloor) * afloortot  # CHANGED

        # internal walls
        if ainttot == 0:
            k_int = 0
        else:
            # 1 / (resistance_InnerWall/ainttot)
            k_int = inv0(div0(resistance_InnerWall, ainttot))  # CHANGED

        k_air_int = k_int_air
        k_ext_int = zone_parameters['hRad'] * min(ainttot, aexttot)
        k_roof_int = zone_parameters['hRad'] * min(ainttot, arooftot)
        k_floor_int = zone_parameters['hRad'] * min(ainttot, afloortot)
        ### hier noch zu ergänzen

        # external walls
        if aexttot == 0:
            k_ext = 0
        else:
            # 1 / (resistance_OuterWall/aexttot)
            k_ext = inv0(div0(resistance_OuterWall, aexttot))  # CHANGED

        k_air_ext = k_ext_air
        k_int_ext = k_ext_int
        k_roof_ext = zone_parameters['hRad'] * min(aexttot, arooftot)
        k_floor_ext = zone_parameters['hRad'] * min(aexttot, afloortot)

        # roof
        if arooftot == 0:
            k_roof = 0
        else:
            # 1 / (resistance_Roof/arooftot)
            k_roof = inv0(div0(resistance_Roof, arooftot))  # CHANGED

        k_air_roof = k_roof_air
        k_int_roof = k_roof_int
        k_ext_roof = k_roof_ext
        k_floor_roof = zone_parameters['hRad'] * min(afloortot, arooftot)

        # groundfloor
        if afloortot == 0:
            k_floor = 0
        else:
            # 1 / (resistance_GroundFloor/afloortot)
            k_floor = inv0(div0(resistance_GroundFloor, afloortot))  # CHANGED

        k_roof_floor = k_floor_roof
        k_ext_floor = k_floor_ext
        k_int_floor = k_floor_int


        # equations
        eq_int = sympy.Eq(
            k_int * (T_int - T_int_sur) +
            k_int_air * (T_Air - T_int_sur) +
            k_int_ext * (T_ext_sur - T_int_sur) +
            k_int_roof * (T_roof_sur - T_int_sur) +
            k_int_floor * (T_floor_sur - T_int_sur) +
            split_rad_int_int * QTraRad_flow +
            split_rad_int_int * q_ig_rad +
            split_rad_sol_int * Q_RadSol, 0)

        eq_ext = sympy.Eq(
            k_ext * (T_ext - T_ext_sur) +
            k_ext_air * (T_Air - T_ext_sur) +
            k_ext_int * (T_int_sur - T_ext_sur) +
            k_ext_roof * (T_roof_sur - T_ext_sur) +
            k_ext_floor * (T_floor_sur - T_ext_sur) +
            split_rad_int_ext * QTraRad_flow +
            split_rad_int_ext * q_ig_rad +
            split_rad_sol_ext * Q_RadSol, 0)

        eq_roof = sympy.Eq(
            k_roof * (T_roof - T_roof_sur) +
            k_roof_air * (T_Air - T_roof_sur) +
            k_roof_int * (T_int_sur - T_roof_sur) +
            k_roof_ext * (T_ext_sur - T_roof_sur) +
            k_roof_floor * (T_floor_sur - T_roof_sur) +
            split_rad_int_roof * QTraRad_flow +
            split_rad_int_roof * q_ig_rad +
            split_rad_sol_roof * Q_RadSol, 0)

        eq_floor = sympy.Eq(
            k_floor * (T_floor - T_floor_sur) +
            k_floor_air * (T_Air - T_floor_sur) +
            k_floor_int * (T_int_sur - T_floor_sur) +
            k_floor_ext * (T_ext_sur - T_floor_sur) +
            k_floor_roof * (T_roof_sur - T_floor_sur) +
            split_rad_int_floor * QTraRad_flow +
            split_rad_int_floor * q_ig_rad +
            split_rad_sol_floor * Q_RadSol, 0)


        sol = sympy.solve([eq_int, eq_ext, eq_roof, eq_floor],
                          [T_int_sur, T_ext_sur, T_roof_sur, T_floor_sur])

        # Extract coefficients from the solution
        coefficients = {}

        # Iterate over the equations in the solution
        for var_sur, eq in sol.items():
            # Extract coefficients for each symbolic variable
            coeffs = {}
            for var in [T_Air, T_int, T_ext, T_roof, T_floor, QTraRad_flow, q_ig_rad, Q_RadSol]:
                coeffs[str(var)] = float(eq.coeff(var))

            # Store coefficients for the current equation
            coefficients[str(var_sur)] = coeffs


        print(name)
        s = pd.Series(coefficients).sort_index()
        print(s.to_string(max_rows=None))  # nur für diesen Aufruf
        # oder global:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)  # automatische Breite, weniger Umbruch
        print(s.to_string())

        return coefficients

    else:
        material: pd.DataFrame
        zone_construction: pd.DataFrame
        windows: pd.DataFrame

        material, zone_construction, windows = get_idf_data()

        ainttot = 0
        aexttot = 0
        arooftot = 0
        afloortot = 0
        awintot = 1.08

        zone_parameters: dict[str, float] = {}

        zone_parameters['hConWin'] = 2.7
        zone_parameters['hRad'] = 5
        zone_parameters['hConWallOut'] = 3.8058793490349183
        zone_parameters['hRadWall'] = 5
        zone_parameters['RExtRem'] = 0.031527706378620325
        zone_parameters['hConRoofOut'] = 20.0
        zone_parameters['hRadRoof'] = 5
        zone_parameters['RRoofRem'] = 0.04510842037851488
        zone_parameters['RFloorRem'] = 0.04510842037851488
        zone_parameters['hConWinOut'] = 9.141790447441632
        zone_parameters['RWin'] = 0.029202278432076387

        # --- ROOF (Dach) -----------------------------------------------------------------
        dach = list(zone_construction.loc[name].filter(regex=r"^Dach").items())
        match = material[material["Construction"].str.contains("Roof", case=False)]
        if not match.empty:
            capacity_Roof = match.iloc[0]["C_total [J/K·m²]"]
            resistance_Roof = match.iloc[0]["R_total [m²K/W]"]
            k_amb_Roof = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
            k_int_Roof = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of Roof invalid")

        for roof in dach:
            if roof[1] > 0:
                RoofName, RoofArea = roof
                arooftot += RoofArea

        # --- GROUND FLOOR (Boden gegen Erdreich) -----------------------------------------
        erdboden = list(zone_construction.loc[name].filter(regex=r"^GroundFloor").items())
        match = material[material["Construction"].str.contains("GroundFloor", case=False)]
        if not match.empty:
            capacity_GroundFloor = match.iloc[0]["C_total [J/K·m²]"]
            resistance_GroundFloor = match.iloc[0]["R_total [m²K/W]"]
            k_amb_GroundFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
            k_int_GroundFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of GroundFloor invalid")

        for groundfloor in erdboden:
            if groundfloor[1] > 0:
                GroundFloorName, GroundFloorArea = groundfloor
                afloortot += GroundFloorArea

        # --- INNER FLOOR (Innenboden) ----------------------------------------------------
        innenboden = list(zone_construction.loc[name].filter(regex=r"^InnerFloor").items())
        match = material[material["Construction"].str.contains("InnerFloor", case=False)]
        if not match.empty:
            capacity_InnerFloor = match.iloc[0]["C_total [J/K·m²]"]
            resistance_InnerFloor = match.iloc[0]["R_total [m²K/W]"]
            k_amb_InnerFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
            k_int_InnerFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of InnerFloor invalid")

        for innerfloor in innenboden:
            if innerfloor[1] > 0:
                InnerFloorName, InnerFloorArea = innerfloor
                afloortot += InnerFloorArea

        # --- DECKE (Ceiling; Zuordnung wie bei dir: Material = InnerFloor) --------------
        decke = list(zone_construction.loc[name].filter(regex=r"^Decke").items())
        match = material[material["Construction"].str.contains("InnerFloor", case=False)]
        if not match.empty:
            capacity_InnerFloor = match.iloc[0]["C_total [J/K·m²]"]
            resistance_InnerFloor = match.iloc[0]["R_total [m²K/W]"]
            k_amb_InnerFloor = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
            k_int_InnerFloor = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of InnerFloor invalid (Ceiling)")

        for ceiling in decke:
            if ceiling[1] > 0:
                InnerFloorName, InnerFloorArea = ceiling
                arooftot += InnerFloorArea

        # --- INNER WALL (Innenwände) -----------------------------------------------------
        innenwand = list(zone_construction.loc[name].filter(regex=r"^InnerWall").items())
        match = material[material["Construction"].str.contains("InnerWall", case=False)]
        if not match.empty:
            capacity_InnerWall = match.iloc[0]["C_total [J/K·m²]"]
            resistance_InnerWall = match.iloc[0]["R_total [m²K/W]"]
            k_int_InnerWall = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of InnerWall invalid")

        for innerwall in innenwand:
            if innerwall[1] > 0:
                InnerWallName, InnerWallArea = innerwall
                ainttot += InnerWallArea

        # --- OUTER WALL (Außenwände, innenseitige Konvektion) ---------------------------
        aussenwand = list(zone_construction.loc[name].filter(regex=r"^OuterWall").items())
        match = material[material["Construction"].str.contains("OuterWall", case=False)]
        if not match.empty:
            capacity_OuterWall = match.iloc[0]["C_total [J/K·m²]"]
            resistance_OuterWall = match.iloc[0]["R_total [m²K/W]"]
            k_amb_OuterWall = match.iloc[0]["Aussenwaermeuebergangswiderstand [m²K/W]"]
            k_int_OuterWall = match.iloc[0]["Innenwaermeuebergangswiderstand [m²K/W]"]
        else:
            raise Exception("IDF Data of OuterWall invalid")

        for outerwall in aussenwand:
            if outerwall[1] > 0:
                OuterWallName, OuterWallArea = outerwall
                aexttot += OuterWallArea

        # building envelope

        area_tot = ainttot + aexttot + arooftot + afloortot + awintot

        # split factor for internal radiation (internal gains, radiator)
        split_rad_int = {
            'int': div0(ainttot, area_tot),  # CHANGED
            'ext': div0(aexttot, area_tot),  # CHANGED
            'roof': div0(arooftot, area_tot),  # CHANGED
            'floor': div0(afloortot, area_tot),  # CHANGED
            'win': 0}

        # split factors for solar radiation
        split_rad_sol = {
            'int': 0.125,
            'ext': 0.125,
            'roof': 0.25,
            'floor': 0.5,
            'win': 0}

        """
        Calculates the coefficients for the algebraic equations of the surface temperatures.
        """
        # inputs
        QTraRad_flow = sympy.symbols('QTraRad_flow')
        q_ig_rad = sympy.symbols('q_ig_rad')
        Q_RadSol = sympy.symbols('Q_RadSol')
        T_preTemWin = sympy.symbols('T_preTemWin')
        # states
        T_Air = sympy.symbols('T_Air')
        T_int = sympy.symbols('T_int')
        T_ext = sympy.symbols('T_ext')
        T_roof = sympy.symbols('T_roof')
        T_floor = sympy.symbols('T_floor')
        # surface temps
        T_int_sur = sympy.symbols('T_int_sur')
        T_ext_sur = sympy.symbols('T_ext_sur')
        T_roof_sur = sympy.symbols('T_roof_sur')
        T_floor_sur = sympy.symbols('T_floor_sur')
        T_win_sur = sympy.symbols('T_win_sur')

        # parameters
        # split factors
        split_rad_int_int, split_rad_sol_int = split_rad_int['int'], split_rad_sol['int']
        split_rad_int_ext, split_rad_sol_ext = split_rad_int['ext'], split_rad_sol['ext']
        split_rad_int_roof, split_rad_sol_roof = split_rad_int['roof'], split_rad_sol['roof']
        split_rad_int_floor, split_rad_sol_floor = split_rad_int['floor'], split_rad_sol['floor']
        split_rad_int_win, split_rad_sol_win = split_rad_int['win'], split_rad_sol['win']

        # thermal transmittance
        # air
        k_int_air = inv0(k_int_InnerWall) * ainttot  # CHANGED
        k_ext_air = inv0(k_int_OuterWall) * aexttot  # CHANGED
        k_roof_air = inv0(k_int_Roof) * arooftot  # CHANGED
        k_floor_air = inv0(k_int_InnerFloor) * afloortot  # CHANGED
        k_win_air = zone_parameters['hConWin'] * awintot

        # internal walls
        if ainttot == 0:
            k_int = 0
        else:
            # 1 / (resistance_InnerWall/ainttot)
            k_int = inv0(div0(resistance_InnerWall, ainttot))  # CHANGED

        k_air_int = k_int_air
        k_ext_int = zone_parameters['hRad'] * min(ainttot, aexttot)
        k_roof_int = zone_parameters['hRad'] * min(ainttot, arooftot)
        k_floor_int = zone_parameters['hRad'] * min(ainttot, afloortot)
        k_win_int = zone_parameters['hRad'] * min(ainttot, awintot)
        ### hier noch zu ergänzen

        # external walls
        if aexttot == 0:
            k_ext = 0
        else:
            # 1 / (resistance_OuterWall/aexttot)
            k_ext = inv0(div0(resistance_OuterWall, aexttot))  # CHANGED

        k_air_ext = k_ext_air
        k_int_ext = k_ext_int
        k_roof_ext = zone_parameters['hRad'] * min(aexttot, arooftot)
        k_floor_ext = zone_parameters['hRad'] * min(aexttot, afloortot)
        k_win_ext = zone_parameters['hRad'] * min(aexttot, awintot)
        k_amb_ext = series_cond((zone_parameters['hConWallOut'] + zone_parameters['hRadWall']) * aexttot,
                                zone_parameters['RExtRem'])  # CHANGED

        # roof
        if arooftot == 0:
            k_roof = 0
        else:
            # 1 / (resistance_Roof/arooftot)
            k_roof = inv0(div0(resistance_Roof, arooftot))  # CHANGED

        k_air_roof = k_roof_air
        k_int_roof = k_roof_int
        k_ext_roof = k_roof_ext
        k_floor_roof = zone_parameters['hRad'] * min(afloortot, arooftot)
        k_win_roof = zone_parameters['hRad'] * min(awintot, arooftot)
        k_amb_roof = series_cond((zone_parameters['hConRoofOut'] + zone_parameters['hRadRoof']) * arooftot,
                                 zone_parameters['RRoofRem'])  # CHANGED

        # groundfloor
        if afloortot == 0:
            k_floor = 0
        else:
            # 1 / (resistance_GroundFloor/afloortot)
            k_floor = inv0(div0(resistance_GroundFloor, afloortot))  # CHANGED

        k_air_floor = k_floor_air
        k_roof_floor = k_floor_roof
        k_ext_floor = k_floor_ext
        k_int_floor = k_floor_int
        k_win_floor = zone_parameters['hRad'] * min(awintot, afloortot)
        k_amb_floor = inv0(zone_parameters['RFloorRem'])  # CHANGED

        # windows
        k_air_win = k_win_air
        k_int_win = k_win_int
        k_ext_win = k_win_ext
        k_roof_win = k_win_roof
        k_floor_win = k_win_floor
        k_amb_win = series_cond((zone_parameters['hConWinOut'] + zone_parameters['hRadWall']) * awintot,
                                zone_parameters['RWin'])  # CHANGED
        k_win_amb = k_amb_win

        # equations
        eq_int = sympy.Eq(
            k_int * (T_int - T_int_sur) +
            k_int_air * (T_Air - T_int_sur) +
            k_int_ext * (T_ext_sur - T_int_sur) +
            k_int_roof * (T_roof_sur - T_int_sur) +
            k_int_floor * (T_floor_sur - T_int_sur) +
            k_int_win * (T_win_sur - T_int_sur) +
            split_rad_int_int * QTraRad_flow +
            split_rad_int_int * q_ig_rad +
            split_rad_sol_int * Q_RadSol, 0)

        eq_ext = sympy.Eq(
            k_ext * (T_ext - T_ext_sur) +
            k_ext_air * (T_Air - T_ext_sur) +
            k_ext_int * (T_int_sur - T_ext_sur) +
            k_ext_roof * (T_roof_sur - T_ext_sur) +
            k_ext_floor * (T_floor_sur - T_ext_sur) +
            k_ext_win * (T_win_sur - T_ext_sur) +
            split_rad_int_ext * QTraRad_flow +
            split_rad_int_ext * q_ig_rad +
            split_rad_sol_ext * Q_RadSol, 0)

        eq_roof = sympy.Eq(
            k_roof * (T_roof - T_roof_sur) +
            k_roof_air * (T_Air - T_roof_sur) +
            k_roof_int * (T_int_sur - T_roof_sur) +
            k_roof_ext * (T_ext_sur - T_roof_sur) +
            k_roof_floor * (T_floor_sur - T_roof_sur) +
            k_roof_win * (T_win_sur - T_roof_sur) +
            split_rad_int_roof * QTraRad_flow +
            split_rad_int_roof * q_ig_rad +
            split_rad_sol_roof * Q_RadSol, 0)

        eq_floor = sympy.Eq(
            k_floor * (T_floor - T_floor_sur) +
            k_floor_air * (T_Air - T_floor_sur) +
            k_floor_int * (T_int_sur - T_floor_sur) +
            k_floor_ext * (T_ext_sur - T_floor_sur) +
            k_floor_roof * (T_roof_sur - T_floor_sur) +
            k_floor_win * (T_win_sur - T_floor_sur) +
            split_rad_int_floor * QTraRad_flow +
            split_rad_int_floor * q_ig_rad +
            split_rad_sol_floor * Q_RadSol, 0)

        eq_win = sympy.Eq(
            k_win_amb * (T_preTemWin - T_win_sur) +
            k_win_air * (T_Air - T_win_sur) +
            k_win_int * (T_int_sur - T_win_sur) +
            k_win_ext * (T_ext_sur - T_win_sur) +
            k_win_roof * (T_roof_sur - T_win_sur) +
            k_win_floor * (T_floor_sur - T_win_sur) +
            split_rad_int_win * QTraRad_flow +
            split_rad_int_win * q_ig_rad +
            split_rad_sol_win * Q_RadSol, 0)

        sol = sympy.solve([eq_int, eq_ext, eq_roof, eq_floor, eq_win],
                          [T_int_sur, T_ext_sur, T_roof_sur, T_floor_sur, T_win_sur])

        # Extract coefficients from the solution
        coefficients = {}

        # Iterate over the equations in the solution
        for var_sur, eq in sol.items():
            # Extract coefficients for each symbolic variable
            coeffs = {}
            for var in [T_Air, T_int, T_ext, T_roof, T_floor, T_preTemWin, QTraRad_flow, q_ig_rad, Q_RadSol]:
                coeffs[str(var)] = float(eq.coeff(var))

            # Store coefficients for the current equation
            coefficients[str(var_sur)] = coeffs

        print(name)
        s = pd.Series(coefficients).sort_index()
        print(s.to_string(max_rows=None))  # nur für diesen Aufruf
        # oder global:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)  # automatische Breite, weniger Umbruch
        print(s.to_string())

        return coefficients
