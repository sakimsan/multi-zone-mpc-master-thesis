#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Overlay-Plot: Dymola-MAT (building.groundFloor.zoneLiving1.TAir) + SIM-CSV (local.T_Air_livingroom)
- Gemeinsame Zeitbasis (beide relativ zu MAT t0_abs) auf gewünschtes Startdatum gemappt
- X: Kalenderticks (Mitternacht), harter Cut exakt auf das Cutoff-Datum (Mitternacht dieses Tages)
     Schritt in vollen Tagen; + 3 Minor-Ticks gleichmäßig zwischen den Majors
- Y: 5er-Schritte inkl. Rand-Ticks
- Legende: außerhalb unten, zentriert (2 Spalten)
- Größere Schrift
- A5 Landscape (210x100 mm), nur PDF

Hinweis: Die Achsenbegrenzungen verwenden jetzt strikt `start_date` (links) und `cutoff_date` (rechts),
unabhängig davon, wann die Datenreihen beginnen/enden.
"""
import re
import math
import datetime as dt
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from scipy.io import loadmat
import pandas as pd

# ---- LaTeX/Styling (größere Schrift) ----
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "text.usetex": True,
    "axes.unicode_minus": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.xmargin": 0.0,
})

# ---- Defaults ----
MAT_DEFAULT = Path(r"D:\fwu-ssr\res\plots\used studies\MonovalentVitoCal_HOM_MPC_yValSet_zero.mat")
SIM_DEFAULT = Path(r"D:\fwu-ssr\res\plots\used studies\Problem_Studies\mpc_design_Abkuelverhalten\DesignOptimizationResults\TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai_\Design_0_sim_agent")
OUT_DEFAULT = Path(r"D:\fwu-ssr\res\plots\Dymola\Knoten")
START_DATE_DEFAULT = "2015-01-22"
CUTOFF_DATE_DEFAULT = "2015-01-30"  # rechter Rand = Mitternacht dieses Datums
MAT_VAR = "building.groundFloor.zoneLiving1.TAir"
SIM_VAR = ("local", "T_Air_livingroom")

# ------------ Dymola-Loader (absolute Sekunden) ------------
def _reconstruct_names_rows(name_array: np.ndarray) -> Tuple[List[str], int]:
    rows = list(name_array)
    return rows, len(rows[0])

def _col_name(j: int, rows: List[str]) -> str:
    return "".join(r[j] for r in rows).strip()

def extract_single_series_from_mat_abs(mat_path: Path, varname: str):
    d = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    data_2   = np.array(d["data_2"], dtype=float)
    dataInfo = np.array(d["dataInfo"], dtype=np.int64)
    name_arr = d["name"]
    rows, ncols = _reconstruct_names_rows(name_arr)

    target1 = varname
    target2 = f"{varname}.y"
    match_j = None
    for j in range(ncols):
        namej = _col_name(j, rows)
        if namej == target1 or namej == target2:
            match_j = j
            break
    if match_j is None:
        raise RuntimeError(f"Variable nicht gefunden: '{varname}' (auch nicht als '{varname}.y').")

    ds = int(dataInfo[0, match_j])
    krow = int(dataInfo[1, match_j])
    if ds != 2:
        raise RuntimeError(f"Gefundene Variable '{varname}' liegt nicht im data_2-Datensatz (ds={ds}).")

    t_abs_sec = data_2[0, :].astype(float)
    y = data_2[krow, :].astype(float)
    return t_abs_sec, y

# ------------ SIM-CSV Loader (absolute Sekunden) ------------
def _ensure_csv_path(p: Path) -> Path:
    return p if p.suffix.lower() == ".csv" else p.with_suffix(".csv")

def load_sim_tair_livingroom_abs(sim_csv_path: Path):
    p = _ensure_csv_path(sim_csv_path)
    df = pd.read_csv(p, header=[0, 1, 2], index_col=0)
    df = df.droplevel(level=2, axis=1)
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.sort_index()

    if SIM_VAR not in df.columns:
        cols = {(str(c0).strip().lower(), str(c1).strip()): (c0, c1) for c0, c1 in df.columns}
        key = ("local", "t_air_livingroom")
        if key not in cols:
            sample = list(df.columns)[:20]
            raise RuntimeError(f"Spalte ('local','T_Air_livingroom') nicht gefunden. Beispiele: {sample}")
        col_key = cols[key]
    else:
        col_key = SIM_VAR

    s = pd.to_numeric(df[col_key], errors="coerce").dropna()
    secs_abs = s.index.values.astype(float)
    y_c = s.values - 273.15  # Kelvin -> °C
    return secs_abs, y_c

# ------------ Tick-Helfer (Kalender, harter Cut) ------------
def _midnight(d: dt.datetime) -> dt.datetime:
    return dt.datetime(d.year, d.month, d.day)

def _set_calendar_ticks_exact_cut(ax: plt.Axes,
                                  start_dt: dt.datetime,
                                  cutoff_dt: dt.datetime,
                                  n_major_target: int = 6,
                                  n_minor_between: int = 1,
                                  fmt: str = "%d.%m",
                                  step_days: Optional[int] = None) -> None:
    """
    Major-Ticks exakt auf Mitternacht; Achse links bei start_dt (Mitternacht),
    rechts harter Cut bei cutoff_dt (Mitternacht). Schritt in vollen Tagen.
    """
    s0 = _midnight(start_dt)
    e0 = _midnight(cutoff_dt)
    if e0 <= s0:
        e0 = s0 + dt.timedelta(days=1)

    total_days = (e0 - s0).days  # ganze Tage

    if step_days is None or step_days < 1:
        divisors = [d for d in range(1, total_days + 1) if total_days % d == 0]
        best = None
        best_err = 1e9
        for d in divisors:
            n = total_days // d + 1
            err = abs(n - n_major_target)
            if err < best_err or (err == best_err and (best is None or d > best)):
                best = d; best_err = err
        step_days = best if best is not None else 1

    majors_dt = []
    cur = s0
    while cur <= e0:
        majors_dt.append(cur)
        cur += dt.timedelta(days=step_days)

    majors = mdates.date2num(majors_dt)

    ax.set_xlim(s0, e0)       # <<< harter Cut
    ax.set_xmargin(0)
    ax.xaxis.set_major_locator(FixedLocator(majors))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

    minors = []
    for u, v in zip(majors[:-1], majors[1:]):
        step = (v - u) / (n_minor_between + 1)
        minors.extend([u + step * k for k in range(1, n_minor_between + 1)])
    ax.xaxis.set_minor_locator(FixedLocator(minors))

# ------------ Tick-Helfer (Y) ------------
def _set_y_ticks_step5(ax: plt.Axes, *arrays: np.ndarray, fmt: str = '%.0f') -> None:
    vals = np.concatenate([np.asarray(a, dtype=float).ravel()
                           for a in arrays if a is not None and np.size(a) > 0]) if arrays else np.array([0.0])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = np.array([0.0])
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if math.isclose(vmin, vmax, abs_tol=1e-12):
        lo = math.floor((vmin - 5) / 5.0) * 5.0
        hi = math.ceil((vmin + 5) / 5.0) * 5.0
    else:
        lo = math.floor(vmin / 5.0) * 5.0
        hi = math.ceil(vmax / 5.0) * 5.0
        if hi <= lo:
            hi = lo + 5.0
    ticks = np.arange(lo, hi + 0.1, 5.0)
    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

# ------------ Figure/Plot ------------
def _a5_figure(orientation: str = "landscape"):
    if orientation.lower().startswith("land"):
        w_mm, h_mm = 170, 50   # A5 quer (angepasst)
    else:
        w_mm, h_mm = 100, 155  # A5 hoch (angepasst)
    fig = plt.figure(figsize=(w_mm/25.4, h_mm/25.4))
    ax = fig.add_subplot(111)
    return fig, ax

def plot_overlay_pdf(
    mat_secs_abs: np.ndarray,
    y_mat: np.ndarray,
    sim_secs_abs: np.ndarray,
    y_sim_c: np.ndarray,
    out_pdf: Path,
    *,
    start_date: str = START_DATE_DEFAULT,
    cutoff_date: str = CUTOFF_DATE_DEFAULT,
    orientation: str = "landscape",
    convert_mat_kelvin_to_c: bool = True,
    n_major_target: int = 6,
    n_minor_between: int = 1,
    step_days: Optional[int] = None,
) -> None:
    fig, ax = _a5_figure(orientation=orientation)

    # Gemeinsamer Start in absoluten Sekunden (MAT t0)
    t0_abs = float(mat_secs_abs[0])

    # Datumsbasis strikt aus Parametern (nicht aus den Daten)
    start_dt = dt.datetime.fromisoformat(start_date)
    cutoff_dt = dt.datetime.fromisoformat(cutoff_date)

    # Beide Zeitreihen auf dasselbe Datumssystem mappen
    x_dt_mat = np.array([start_dt + dt.timedelta(seconds=float(s - t0_abs)) for s in mat_secs_abs])
    x_dt_sim = np.array([start_dt + dt.timedelta(seconds=float(s - t0_abs)) for s in sim_secs_abs])

    # Einheiten angleichen
    y_mat_c = (y_mat - 273.15) if convert_mat_kelvin_to_c else y_mat

    # Daten zeichnen
    ax.plot(x_dt_mat, y_mat_c, linewidth=1.2, label=r"Raumtemperatur \textit{EnergyPlus-Simulationsmodell}")
    ax.plot(x_dt_sim, y_sim_c, linewidth=1.2, label=r"Raumtemperatur \textit{Idealmodell}")

    # Achsen + Grid
    ax.set_xlabel("Datum")
    ax.set_ylabel("Temperatur [°C]")
    ax.grid(True, which="major", linestyle="--", alpha=0.25)
    ax.grid(True, which="minor", linestyle=":",  alpha=0.18)

    # X-Ticks & harte Begrenzung exakt nach Parametern
    # Standard: 2-Tage-Schritt, kann via --step-days überschrieben werden
    eff_step_days = 2 if (step_days is None or step_days < 1) else step_days
    _set_calendar_ticks_exact_cut(
        ax, start_dt, cutoff_dt,
        n_major_target=n_major_target,
        n_minor_between=n_minor_between,
        fmt="%d.%m",
        step_days=eff_step_days,
    )

    # Y-Ticks in 5er-Schritten über beide Datensätze
    _set_y_ticks_step5(ax, y_mat_c, y_sim_c, fmt='%.0f')

    # ---- Legende außerhalb unten, zentriert ----
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.1, left=0.10, right=0.99, top=0.98)

    fig.legend(handles=handles, labels=labels,
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, -0.1))

    # Speichern (nur PDF)
    fig.savefig(out_pdf)
    plt.close(fig)

# ------------ CLI ------------
def _ensure_mat_path(p: Path) -> Path:
    return p if p.suffix.lower() == ".mat" else p.with_suffix(".mat")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mat",  type=Path, default=MAT_DEFAULT, help="Pfad zur Dymola .mat-Datei")
    p.add_argument("--sim",  type=Path, default=SIM_DEFAULT, help="Pfad zur SIM-CSV (ohne/mit .csv)")
    p.add_argument("--out",  type=Path, default=OUT_DEFAULT, help="Ausgabeordner")
    p.add_argument("--start", type=str, default=START_DATE_DEFAULT, help="Startdatum (YYYY-MM-DD) für beide Reihen")
    p.add_argument("--cutoff", type=str, default=CUTOFF_DATE_DEFAULT, help="Cutoff-Datum (YYYY-MM-DD), rechter Rand (Mitternacht)")
    p.add_argument("--orient", type=str, default="landscape", choices=["landscape", "portrait"], help="Ausrichtung")
    p.add_argument("--majors", type=int, default=6, help="Zielanzahl Major-Ticks (nur relevant bei Auto-Schritt)")
    p.add_argument("--minors-between", type=int, default=1, help="Minor-Ticks zwischen Majors")
    p.add_argument("--step-days", type=int, default=None, help="Fester Tages-Schritt (1=täglich, 2=alle 2 Tage, ...)")
    p.add_argument("--no_mat_k2c", action="store_true", help="Kein Kelvin->°C für MAT")
    args = p.parse_args()

    mat_path = _ensure_mat_path(args.mat)
    sim_path = _ensure_csv_path(args.sim)
    args.out.mkdir(parents=True, exist_ok=True)

    # Daten laden (absolute Sekunden)
    mat_secs_abs, y_mat = extract_single_series_from_mat_abs(mat_path, MAT_VAR)
    sim_secs_abs, y_sim_c = load_sim_tair_livingroom_abs(sim_path)

    out_pdf = args.out / "TAir_livingroom_overlay.pdf"

    plot_overlay_pdf(
        mat_secs_abs, y_mat, sim_secs_abs, y_sim_c, out_pdf,
        start_date=args.start,
        cutoff_date=args.cutoff,
        orientation=args.orient,
        convert_mat_kelvin_to_c=(not args.no_mat_k2c),
        n_major_target=args.majors,
        n_minor_between=args.minors_between,
        step_days=args.step_days,
    )
    print(f"Gespeichert: {out_pdf}")

if __name__ == "__main__":
    main()
