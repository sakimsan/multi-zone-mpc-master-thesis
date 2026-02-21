from bes_rules.simulation_based_optimization.agentlib_mpc.get_idf_data import get_idf_data
# -*- coding: utf-8 -*-
"""
Zone-Surface-Coefficient Pipeline (ohne Fenster, nur RC-Knoten)
EN ISO 52016-1 – Innen-Langwelle via MRT/Sternknoten
---------------------------------------------------------------

Erzeugt pro Zone:
1) einzelne Flächen-Instanzen aus zone_construction (Dach*, GroundFloor, InnerFloor*, Decke,
   InnerWall*, OuterWall*);
2) löst das lineare Oberflächen-Energiebilanzsystem (Konvektion + Leitung zum 1. RC-Knoten
   + linearisiert LW nach EN ISO 52016-1 über MRT/Sternknoten T_r und Eliminierung);
3) extrahiert lineare Koeffizienten (coeff_dict) für jede Oberflächentemperatur T_s_<Instanz>;
4) mappt die Ausdrücke direkt auf dein CasadiModel (var_outputs_zone(...).alg).

Voraussetzungen:
- material-DataFrame mit mind. Spalte: "Construction", "R_total [m²K/W]"
- zone_construction-DataFrame mit deinen Spalten:
    Dach, Dach2, Dach3, GroundFloor, InnerFloor, InnerFloor2..5, Decke,
    InnerWall, InnerWall2..5, OuterWall, OuterWall2, "Volume [m3]" (ignoriert)
- Modell: model._states, model._inputs, model._outputs (dicts)
"""

from typing import Dict, List, Tuple
import re
import sympy as sp
import pandas as pd

SIGMA = 5.670374419e-8


# --------------------------- Helpers ---------------------------

def inv0(x: float) -> float:
    """1/x mit Zero-Guard."""
    return 0.0 if (x is None or x == 0) else 1.0/x


# --------------------------- ISO 52016-1 Sternknoten-Solver ---------------------------

def solve_zone_surfaces_iso_52016(
    surfaces: List[Dict],
    T_air: sp.Symbol,
    T_ref: float = 295.0,
    extra_symbols: List[sp.Symbol] = None,
    sigma: float = SIGMA
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    EN ISO 52016-1-konforme Innen-Langwelle über MRT/Sternknoten T_r.

    Oberflächenbilanz je i:
        h_c,i A_i (T_air - T_s,i) + h_r,i A_i (T_r - T_s,i) + (A_i/R'_i)(T_back,i - T_s,i) + q_i = 0
    mit linearisiertem h_r,i = 4 σ ε_i T_ref^3  (ε_i als Flächen-Emissivität).
    Schließbedingung MRT:
        Σ_i (h_r,i A_i) (T_s,i - T_r) = 0  ⇒  T_r = Σ_i (H_i T_s,i) / Σ_i H_i,  H_i = h_r,i A_i

    Wir setzen die n+1 Gleichungen für (T_s, T_r) auf, eliminieren T_r symbolisch
    und extrahieren lineare Koeffizienten wie in deinem Original.
    """
    n = len(surfaces)
    Ts_syms = [sp.symbols(f"T_s_{s['id']}") for s in surfaces]
    Ts_keys = [f"T_s_{s['id']}" for s in surfaces]
    T_r = sp.symbols("T_r")

    A   = [float(s["A"]) for s in surfaces]
    h_c = [float(s["h_c"]) for s in surfaces]
    eps = [float(s.get("eps", 0.9)) for s in surfaces]
    R   = [float(s.get("R_cond_areal", 0.0)) for s in surfaces]
    T_b = [s["T_back"] for s in surfaces]
    q   = [s.get("q_src", 0.0) for s in surfaces]

    # Leitwerte
    G_conv = [h_c[i]*A[i] for i in range(n)]
    G_cond = [0.0 if R[i]==0 else (A[i]/R[i]) for i in range(n)]

    # ISO 52016-1: linearisiertes h_r,i (gegen MRT)
    h_r_i = [4.0*sigma*eps[i]*(T_ref**3) for i in range(n)]
    H_i   = [h_r_i[i]*A[i] for i in range(n)]
    H_tot = sum(H_i) if sum(H_i) > 0 else 1.0

    # Gleichungssystem (n Oberflächen + 1 MRT-Schließung)
    # Unbekannte: [T_s_1..T_s_n, T_r]
    unknowns = Ts_syms + [T_r]
    A_mat = sp.zeros(n+1, n+1)
    b_vec = sp.zeros(n+1, 1)

    # 1..n: Oberflächenbilanzen
    for i in range(n):
        # Koef. für T_s,i (auf linke Seite)
        #   -G_conv_i - H_i - G_cond_i
        A_mat[i, i] = sp.Float(-(G_conv[i] + H_i[i] + G_cond[i]))
        # Koef. für T_r: + H_i
        A_mat[i, n] = sp.Float(H_i[i])

        # RHS: - (G_conv_i*T_air + G_cond_i*T_back_i + q_i)
        rhs = -(G_conv[i]*T_air + G_cond[i]*T_b[i] + q[i])
        b_vec[i, 0] = rhs

    # n+1: MRT-Schließung Σ H_i (T_s,i - T_r) = 0  ->  Σ H_i T_s,i  - H_tot T_r = 0
    for i in range(n):
        A_mat[n, i] = sp.Float(H_i[i])
    A_mat[n, n] = sp.Float(-H_tot)
    b_vec[n, 0] = 0.0

    # Lösen
    sol = sp.solve(A_mat*sp.Matrix(unknowns) - b_vec, unknowns, dict=True)[0]

    # Koeffizienten extrahieren (wie bei dir: "const" + Ableitungen wrt Treiber)
    wanted = set()
    if isinstance(T_air, sp.Expr): wanted.add(T_air)
    for x in T_b:
        if isinstance(x, sp.Expr): wanted.add(x)
    if extra_symbols:
        for x in extra_symbols:
            if isinstance(x, sp.Expr): wanted.add(x)

    coeffs = {}
    for key, sym in zip(Ts_keys, Ts_syms):
        expr = sol[sym]
        cd = {'const': float(expr.subs({w:0 for w in wanted})) if wanted else float(expr)}
        for w in wanted:
            cd[str(w)] = float(sp.diff(expr, w))
        coeffs[key] = cd

    return coeffs, Ts_keys


# --------------------------- Parser (ohne Fenster) ---------------------------

def calc_resistances_zone_element_specific(
    zone: str,
    material: pd.DataFrame,
    zone_construction: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Baut pro Zone die Surface-Instanzen (nur RC-Kopplung) und liefert coeff_dict:
      "T_s_<Instanz>" -> {"const":..., "<Symbol>": coeff, ...}
    """
    params = default_params_rc_only()

    # --- Splits auf vorhandene Typen normalisieren ---
    row = zone_construction.loc[zone]
    present = {
      "Roof":        row.filter(regex=r"^Dach(\d+)?$").sum() > 0,
      "GroundFloor": float(row.get("GroundFloor", 0)) > 0,
      "InnerFloor":  row.filter(regex=r"^InnerFloor(\d+)?$").sum() > 0,
      "Ceiling":     float(row.get("Decke", 0)) > 0,
      "InnerWall":   row.filter(regex=r"^InnerWall(\d+)?$").sum() > 0,
      "OuterWall":   row.filter(regex=r"^OuterWall(\d+)?$").sum() > 0,
    }
    for key in ("split_int", "split_sol"):
        s = sum(params[key][t] for t,p in present.items() if p)
        if s > 0:
            for t,p in present.items():
                params[key][t] = params[key][t]/s if p else 0.0

    sym = params["symbols"]
    h_c = params["h_c"]; eps = params["eps"]
    split_int = params["split_int"]; split_sol = params["split_sol"]
    Rfrac = params["R_surface_fraction"]

    # Flächen-Instanzen einsammeln (strenge Anker!)
    def _collect(regex: str) -> List[Tuple[str, float]]:
        s = row.filter(regex=regex)
        return [(name, float(A)) for name, A in s.items()
                if name != "Volume [m3]" and float(A) > 0.0]

    groups = {
        "Roof":        _collect(r"^Dach(\d+)?$"),
        "GroundFloor": _collect(r"^GroundFloor$"),
        "InnerFloor":  _collect(r"^InnerFloor(\d+)?$"),
        "Ceiling":     _collect(r"^Decke$"),
        "InnerWall":   _collect(r"^InnerWall(\d+)?$"),
        "OuterWall":   _collect(r"^OuterWall(\d+)?$"),
    }

    # Typflächen für Quellenverteilung
    A_sum_by_type = {k: sum(a for _, a in v) for k, v in groups.items()}

    # Materialdaten je Typ (areal)
    def _mat_row(typ: str) -> pd.Series:
        key = "InnerFloor" if typ == "Ceiling" else typ
        m = material[material["Construction"].str.contains(key, case=False)]
        if m.empty:
            raise ValueError(f"Materialdaten fehlen für {typ}")
        return m.iloc[0]

    def _R_cond_surface_areal(typ: str) -> float:
        Rtot = float(_mat_row(typ)["R_total [m²K/W]"])
        # Anteil des Gesamt-R', der zwischen Oberfläche und erstem RC-Knoten liegt:
        return max(1e-9, Rtot * Rfrac[typ])

    # Quellen pro Fläche (flächen- & typgewichtet)
    Qtr, Qig, Qsol = sym["QTraRad_flow"], sym["q_ig_rad"], sym["Q_RadSol"]

    def _q_src_for(type_name: str, A_i: float) -> sp.Expr:
        A_type = A_sum_by_type.get(type_name, 0.0) or 1.0
        return (split_int[type_name]*(Qtr+Qig) + split_sol[type_name]*Qsol) * (A_i / A_type)

    surfaces: List[Dict] = []

    # Roof (Innenoberfläche -> erster RC-Knoten des Daches)
    for inst, A in groups["Roof"]:
        surfaces.append(dict(
            id=inst, type="Roof", A=A, eps=eps["Roof"], h_c=h_c["Roof"],
            R_cond_areal=_R_cond_surface_areal("Roof"),
            T_back=sp.symbols(f"T_Roof_{inst}_1"),
            q_src=_q_src_for("Roof", A),
        ))

    # GroundFloor (Innenoberfläche -> erster RC-Knoten des Erdreich-Bodens)
    for inst, A in groups["GroundFloor"]:
        surfaces.append(dict(
            id=inst, type="GroundFloor", A=A, eps=eps["GroundFloor"], h_c=h_c["GroundFloor"],
            R_cond_areal=_R_cond_surface_areal("GroundFloor"),
            T_back=sp.symbols(f"T_Floor_{inst}_1"),
            q_src=_q_src_for("GroundFloor", A),
        ))

    # InnerFloor (Innenoberfläche -> Plattenknoten / Nachbarseite via deiner ODE)
    for inst, A in groups["InnerFloor"]:
        surfaces.append(dict(
            id=inst, type="InnerFloor", A=A, eps=eps["InnerFloor"], h_c=h_c["InnerFloor"],
            R_cond_areal=_R_cond_surface_areal("InnerFloor"),
            T_back=sp.symbols(f"T_Floor_{inst}"),
            q_src=_q_src_for("InnerFloor", A),
        ))

    # Ceiling (Material = InnerFloor)
    for inst, A in groups["Ceiling"]:
        surfaces.append(dict(
            id=inst, type="Ceiling", A=A, eps=eps["Ceiling"], h_c=h_c["Ceiling"],
            R_cond_areal=_R_cond_surface_areal("Ceiling"),
            T_back=sp.symbols(f"T_Floor_{inst}"),
            q_src=_q_src_for("Ceiling", A),
        ))

    # InnerWall (Innenoberfläche -> Innenwandknoten)
    for inst, A in groups["InnerWall"]:
        surfaces.append(dict(
            id=inst, type="InnerWall", A=A, eps=eps["InnerWall"], h_c=h_c["InnerWall"],
            R_cond_areal=_R_cond_surface_areal("InnerWall"),
            T_back=sp.symbols(f"T_IntWall_{inst}"),
            q_src=_q_src_for("InnerWall", A),
        ))

    # OuterWall (Innenoberfläche -> erster RC-Knoten der Außenwand)
    for inst, A in groups["OuterWall"]:
        surfaces.append(dict(
            id=inst, type="OuterWall", A=A, eps=eps["OuterWall"], h_c=h_c["OuterWall"],
            R_cond_areal=_R_cond_surface_areal("OuterWall"),
            T_back=sp.symbols(f"T_ExtWall_{inst}_1"),
            q_src=_q_src_for("OuterWall", A),
        ))

    extra = [Qtr, Qig, Qsol]
    coeffs, _ = solve_zone_surfaces_iso_52016(
        surfaces, T_air=sym["T_air"], T_ref=params.get("T_ref", 295.0), extra_symbols=extra
    )
    return coeffs


# --------------------------- Mapping ins Modell ---------------------------

def _get_state_zone(m, zone: str, base: str):
    """Hole State <base>_<zone>, sonst 0."""
    name = f"{base}_{zone}"
    v = m._states.get(name)
    return 0.0 if v is None else v

def _get_input(m, name: str):
    """Hole globales Input, sonst 0."""
    v = m._inputs.get(name)
    return 0.0 if v is None else v

def _get_input_zone(m, zone: str, base: str):
    """Hole zonenspezifisches Input <base>_<zone>, fallback global <base>, sonst 0."""
    v = m._inputs.get(f"{base}_{zone}")
    if v is not None:
        return v
    return _get_input(m, base)

def _get_output_zone_alg(m, zone: str, base: str):
    """Hole zonenspezifisches Output <base>_<zone> und gib .alg; fallback 0."""
    v = m._outputs.get(f"{base}_{zone}")
    return 0.0 if v is None else v.alg

def _infer_type_from_inst(inst: str):
    if re.match(r"^OuterWall(\d+)?$", inst):   return "OuterWall"
    if re.match(r"^InnerWall(\d+)?$", inst):   return "InnerWall"
    if re.match(r"^Dach(\d+)?$", inst):        return "Roof"
    if inst == "Decke":                        return "Ceiling"
    if inst == "GroundFloor":                  return "GroundFloor"
    if re.match(r"^InnerFloor(\d+)?$", inst):  return "InnerFloor"
    raise ValueError(f"Unbekannter Instanzname/Typ: {inst}")

def _output_basename_for_type(surf_type: str, inst: str):
    if surf_type == "OuterWall":   return f"T_ExtWall_sur_out_{inst}"
    if surf_type == "InnerWall":   return f"T_IntWall_sur_out_{inst}"
    if surf_type == "Roof":        return f"T_Roof_sur_out_{inst}"
    if surf_type in ("GroundFloor", "InnerFloor", "Ceiling"):
        return f"T_Floor_sur_out_{inst}"
    raise ValueError(f"Kein Output-Basename für Typ {surf_type}")

def apply_coeff_dict_to_model_outputs(model, zone: str, coeff_dict: dict, verbose: bool=False):
    """
    Schreibt die linearen Ausdrücke aus coeff_dict auf var_outputs_zone(...).alg.

    Mapping der Symbolnamen:
      - T_Air                -> State "T_Air_<zone>"
      - QTraRad_flow         -> Output "QTraRad_flow_out_<zone>.alg"
      - q_ig_rad             -> Output "Q_IntGains_rad_<zone>.alg"
      - Q_RadSol             -> Input  "Q_RadSol_<zone>" (oder "Q_RadSol")
      - alle anderen Symbole (z. B. T_ExtWall_OuterWall2_1, T_Roof_Dach3_1,
        T_Floor_InnerFloor, T_IntWall_InnerWall4) -> States "<symbol>_<zone>"
    """
    for key, cd in coeff_dict.items():
        if not key.startswith("T_s_"):
            if verbose: print(f"Skip unknown key: {key}")
            continue

        inst = key[len("T_s_"):]  # 'OuterWall2', 'Dach3', 'Decke', ...
        surf_type = _infer_type_from_inst(inst)
        out_base  = _output_basename_for_type(surf_type, inst)

        expr = cd.get("const", 0.0)

        for simb, coeff in cd.items():
            if simb == "const" or coeff == 0.0:
                continue

            if   simb == "T_Air":
                expr = expr + coeff * _get_state_zone(model, zone, "T_Air")
            elif simb == "QTraRad_flow":
                expr = expr + coeff * _get_output_zone_alg(model, zone, "QTraRad_flow_out")
            elif simb == "q_ig_rad":
                expr = expr + coeff * _get_output_zone_alg(model, zone, "Q_IntGains_rad")
            elif simb == "Q_RadSol":
                expr = expr + coeff * _get_input_zone(model, zone, "Q_RadSol")
            else:
                # alle übrigen Variablen sind Bauteil-Knotentemperaturen als States der Zone
                expr = expr + coeff * _get_state_zone(model, zone, simb)

        out_name = f"{out_base}_{zone}"
        out_var = model._outputs.get(out_name)
        if out_var is None:
            raise KeyError(f"Output '{out_name}' ist nicht im Modell registriert.")
        out_var.alg = expr

        if verbose:
            print(f"Set {out_name}")

    if verbose:
        print(f"[Zone {zone}] Mapping fertig.")


# --------------------------- Defaults & Beispiel ---------------------------

def default_params_rc_only() -> Dict:
    """
    Defaults (ohne Außen-/Erd-Rand im Surface-Solver).
    Passe Werte gern an (h_c/eps, R_surface_fraction, Verteilungen).
    """
    return {
        "h_c": {
            "InnerWall": 2.5,
            "OuterWall": 2.5,
            "Roof": 1.8,
            "GroundFloor": 1.8,
            "InnerFloor": 1.8,
            "Ceiling": 2.0
        },
        "eps": {"InnerWall":0.9,"OuterWall":0.9,"Roof":0.9,"GroundFloor":0.9,"InnerFloor":0.9,"Ceiling":0.9},
        "split_int":{"InnerWall":0.125,"OuterWall":0.125,"Roof":0.25,"GroundFloor":0.5,"InnerFloor":0.5,"Ceiling":0.25},
        "split_sol":{"InnerWall":0.125,"OuterWall":0.125,"Roof":0.25,"GroundFloor":0.5,"InnerFloor":0.5,"Ceiling":0.25},
        # Anteil des Gesamt-R', der zwischen Oberfläche und erstem RC-Knoten liegt
        "R_surface_fraction":{"InnerWall":1,"OuterWall":1,"Roof":1,"GroundFloor":1,"InnerFloor":1,"Ceiling":1},
        "symbols":{
           "T_air": sp.symbols("T_Air"),
           "QTraRad_flow": sp.symbols("QTraRad_flow"),
           "q_ig_rad": sp.symbols("q_ig_rad"),
           "Q_RadSol": sp.symbols("Q_RadSol"),
        },
        "T_ref": 295.0
    }


# --- Beispielnutzung ---
if __name__ == "__main__":
    material, zone_construction, _ = get_idf_data()

    coeff_dict: dict = {}
    coeff_dict["attic"] = calc_resistances_zone_element_specific("attic", material, zone_construction)
    print(coeff_dict["attic"])
