#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import re
from datetime import datetime

import pandas as pd
import numpy as np

# --- Defaults (wie im Plot-Skript) ---
DEFAULT_SIM_CSV = Path(r"D:\fwu-ssr\res\Studies_not_coupled\studies_parallel\DesignOptimizationResults\TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGaiAbwesend_\Design_0_sim_agent.csv")
DEFAULT_OUT_DIR = Path(r"D:\fwu-ssr\res\plots\study2")

# Mapping & Ziel-Reihenfolge (wie bei dir)
ZONE_NAME_MAP = {
    "livingroom": "Wohnen",
    "hobby": "Hobby",
    "corridor": "Flur",
    "wcstorage": "WC",
    "kitchen": "Küche",
    "bedroom": "Schlafen",
    "children": "Kind",
    "corridor2": "Flur 2",
    "corrdior2": "Flur 2",  # Tippfehler-Variante
    "bath": "Bad",
    "children2": "Kind 2",
    "attic": "Dachgeschoss",
}
ZONE_ORDER_CANONICAL = [
    "livingroom", "hobby", "corridor", "wcstorage",
    "kitchen", "bedroom", "children", "corridor2",
    "bath", "children2", "attic"
]

# --- Komfortgewichte (deine Vorgaben) ---
W_COMF_LB = 100.0     # Untere Verletzung (zu kalt)
W_COMF_UB = 10000.0   # Obere Verletzung (zu warm)

# --- Helper wie im Plot-Skript ---
def _load_sim_keep_causality_and_signal(file: Path) -> pd.DataFrame:
    """CSV laden, Header (causality, signal, dtype) -> dtype droppen, Index numerisch."""
    df = pd.read_csv(file, header=[0, 1, 2], index_col=0)
    df = df.droplevel(level=2, axis=1)  # dtype weg
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.sort_index()
    return df

def _find_columns(df: pd.DataFrame, causality_names: Iterable[str], pattern: str) -> dict[str, tuple[str, str]]:
    """Suche Spalten unter causality_names, deren Signalname regex 'pattern' erfüllt.
    'pattern' muss (?P<zone>...) enthalten. Rückgabe: {zone: (causality, signal)}"""
    caus_set = set(causality_names)
    out: dict[str, tuple[str, str]] = {}
    rx = re.compile(pattern)
    for c0, c1 in df.columns:
        if c0 not in caus_set:
            continue
        m = rx.fullmatch(c1)
        if m:
            out[m.group("zone")] = (c0, c1)
    return out

def _canonical(zone: str) -> str:
    return zone.strip().lower()

def _display_name(zone: str) -> str:
    z = _canonical(zone)
    return ZONE_NAME_MAP.get(z, zone)

def _sanitize_filename(part: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", part)

def _to_celsius(arr: np.ndarray, in_kelvin: bool = True) -> np.ndarray:
    return arr - 273.15 if in_kelvin else arr

# --- Hauptfunktion: identische Signatur wie dein Plot-Aufruf ---
def compute_comfort_cost_from_sim_dates(
    sim_csv_path: Path = DEFAULT_SIM_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    base_year: int = 2015,            # Sekunden sind "seit Jahresbeginn base_year"
    skip_first_days: int = 3,          # erste N Tage komplett nicht berücksichtigen
    remove_first_point: bool = True,   # ersten verbleibenden Punkt danach droppen
    t_end_days: Optional[float] = None,# optional: Länge ab Cutoff
    set_band: float = 2.0,             # Obergrenze = T_set + set_band
    show_sammelplot: bool = True,      # ignoriert (nur zur Signatur-Kompatibilität)
) -> dict:
    """
    Berechnet:
        J_comf = sum_t Δt[h] * ( W_COMF_LB*|s_lb(t)| + W_COMF_UB*|s_ub(t)| )

    mit
        T_ub = T_set + set_band
        T_lb = T_set
        s_ub = max(0, T_Air - T_ub)
        s_lb = max(0, T_lb - T_Air)

    Δt wird aus dem Zeitraster (Sekunden seit base_year-01-01) ermittelt
    und in **Stunden** umgerechnet (z. B. 900 s -> 0.25 h).

    Rückgabe:
        {
          "per_zone": {
              "<zone>": {
                 "J_comf": float,
                 "J_lb": float, "J_ub": float,
                 "hours_lb": float, "hours_ub": float,
                 "mean_violation_lb_K": float, "mean_violation_ub_K": float,
                 "dt_hours": float
              }, ...
          },
          "total": float,
          "csv": Path
        }
    """
    # Pfade prüfen / anlegen
    if not sim_csv_path.exists():
        raise FileNotFoundError(f"SIM-Datei nicht gefunden: {sim_csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Daten laden
    df = _load_sim_keep_causality_and_signal(sim_csv_path)

    # Zeitfilter
    secs_full = df.index.values.astype(float)
    if np.isnan(secs_full).all():
        raise ValueError("Zeitindex enthält nur NaNs.")
    start_sec = np.nanmin(secs_full)
    cutoff_sec = start_sec + skip_first_days * 86400.0

    mask = secs_full >= cutoff_sec
    if t_end_days is not None:
        end_sec = cutoff_sec + float(t_end_days) * 86400.0
        mask &= (secs_full <= end_sec)

    df = df.loc[mask]
    secs = secs_full[mask]

    # ersten verbleibenden Punkt entfernen
    if remove_first_point and len(df) >= 1:
        df = df.iloc[1:]
        secs = secs[1:]

    if df.empty:
        raise ValueError("Gefilterter Datensatz ist leer. Prüfe skip_first_days / t_end_days.")

    # Zeitraster -> Δt in Stunden (robust gegen kleine Unregelmäßigkeiten)
    if len(secs) < 2:
        raise ValueError("Zu wenige Zeitpunkte für Δt-Bestimmung.")
    dt_seconds = np.median(np.diff(secs))
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("Ungültiges Δt aus Zeitindex ermittelt.")
    dt_hours = float(dt_seconds) / 3600.0  # z. B. 900 s -> 0.25 h

    # Spalten suchen
    inputs_keys = {"inputs", "input"}
    local_keys = {"local"}

    tair_cols = _find_columns(df, local_keys,  r"T_Air_(?P<zone>.+)")
    tset_cols = _find_columns(df, inputs_keys, r"TSetOneZone_(?P<zone>.+)")

    if not tair_cols:
        sample_local = sorted({c1 for c0, c1 in df.columns if c0 in local_keys})[:20]
        sample_inputs = sorted({c1 for c0, c1 in df.columns if c0 in inputs_keys})[:20]
        raise ValueError(
            "Keine passenden Zonen gefunden (local.T_Air_* / inputs.TSetOneZone_*).\n"
            f"Beispiele (local):  {sample_local}\nBeispiele (inputs): {sample_inputs}"
        )

    # Zonen in gewünschter Reihenfolge (nur vorhandene)
    zones_present = { _canonical(z): z for z in tair_cols.keys() }
    zones_in_order: list[str] = []
    for zc in ZONE_ORDER_CANONICAL:
        if zc in zones_present:
            zones_in_order.append(zones_present[zc])

    per_zone = {}
    rows = []

    for zone in zones_in_order:
        # T_Air
        c0_tair, c1_tair = tair_cols[zone]
        s_tair = pd.to_numeric(df[(c0_tair, c1_tair)], errors="coerce")
        if s_tair.dropna().empty:
            continue
        # Temperaturannahme: CSV in Kelvin (wie bei dir) -> °C
        T_air = _to_celsius(s_tair.values, in_kelvin=True)

        # T_set
        # Finde matching key in tset_cols (trotz evtl. Tippfehler-Variante)
        match_key = next((k for k in tset_cols.keys() if _canonical(k) == _canonical(zone)), None)
        if match_key is None:
            # Kein Setpoint: Zone für Komfortbewertung überspringen (z. B. Dachgeschoss)
            continue

        c0_tset, c1_tset = tset_cols[match_key]
        s_tset = pd.to_numeric(df[(c0_tset, c1_tset)], errors="coerce")
        if s_tset.dropna().empty:
            continue
        T_set = _to_celsius(s_tset.values, in_kelvin=True)

        # Bounds
        T_lb = T_set
        T_ub = T_set + float(set_band)

        # Slacks (nicht-negativ)
        s_ub = np.maximum(0.0, T_air - T_ub)
        s_lb = np.maximum(0.0, T_lb - T_air)

        # Kostenanteile pro Zeitschritt -> Summe * Δt[h]
        J_ub = float(np.nansum(np.abs(s_ub)) * dt_hours * W_COMF_UB)
        J_lb = float(np.nansum(np.abs(s_lb)) * dt_hours * W_COMF_LB)
        J_comf = J_ub + J_lb

        # Zusatzmetriken
        hours_ub = float((s_ub > 0).sum() * dt_hours)
        hours_lb = float((s_lb > 0).sum() * dt_hours)
        mean_violation_ub = float(np.nanmean(s_ub[s_ub > 0])) if np.any(s_ub > 0) else 0.0
        mean_violation_lb = float(np.nanmean(s_lb[s_lb > 0])) if np.any(s_lb > 0) else 0.0

        disp = _display_name(zone)
        per_zone[disp] = dict(
            J_comf=J_comf,
            J_lb=J_lb, J_ub=J_ub,
            hours_lb=hours_lb, hours_ub=hours_ub,
            mean_violation_lb_K=mean_violation_lb, mean_violation_ub_K=mean_violation_ub,
            dt_hours=dt_hours,
        )

        rows.append({
            "zone": disp,
            "J_comf": J_comf,
            "J_lb": J_lb, "J_ub": J_ub,
            "hours_lb": hours_lb, "hours_ub": hours_ub,
            "mean_violation_lb_K": mean_violation_lb, "mean_violation_ub_K": mean_violation_ub,
            "dt_hours": dt_hours,
            "w_lb": W_COMF_LB, "w_ub": W_COMF_UB,
            "set_band_K": float(set_band),
        })

    # Zusammenfassung
    total = float(sum(z["J_comf"] for z in per_zone.values()))

    # CSV schreiben
    summary_df = pd.DataFrame(rows)
    out_csv = out_dir / f"comfort_summary_{_sanitize_filename(sim_csv_path.stem)}.csv"
    summary_df.to_csv(out_csv, index=False)

    # Konsole
    print("Regelgüte (J_comf) pro Zone [Kosten]:")
    for z, m in per_zone.items():
        print(f"  {z:>14s}: J={m['J_comf']:.3f}  (J_lb={m['J_lb']:.3f}, J_ub={m['J_ub']:.3f})   "
              f"hours_lb={m['hours_lb']:.2f}h, hours_ub={m['hours_ub']:.2f}h, "
              f"mean_lb={m['mean_violation_lb_K']:.2f}K, mean_ub={m['mean_violation_ub_K']:.2f}K")
    print(f"\nTOTAL J_comf = {total:.3f}")
    print(f"Δt = {dt_hours:.5f} h, w_lb={W_COMF_LB}, w_ub={W_COMF_UB}, set_band={set_band} K")
    print(f"CSV gespeichert: {out_csv}")

    return {"per_zone": per_zone, "total": total, "csv": out_csv}


if __name__ == "__main__":
    compute_comfort_cost_from_sim_dates(
        sim_csv_path=DEFAULT_SIM_CSV,
        out_dir=DEFAULT_OUT_DIR,
        base_year=2015,
        skip_first_days=3,
        remove_first_point=True,
        t_end_days=None,
        set_band=2.0,          # T_ub = T_set + 2 K
        show_sammelplot=True,  # ignoriert
    )
