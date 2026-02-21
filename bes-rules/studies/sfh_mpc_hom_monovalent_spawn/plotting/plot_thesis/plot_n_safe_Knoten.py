#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Plot building.TempExtWallLivingroomInC[i].y aus Dymola-.mat
- X-Achse: Kalenderticks auf Mitternacht, harter Cut am gewünschten Datum (Default 2015-02-03)
  + je 2 Minor-Ticks gleichmäßig zwischen den Majors
- Y-Achse: 5er-Schritte, untere & obere Grenze sind Ticks
- Keine Legende, nur PDF, A5 (Landscape)
- DPI und Schriftgröße einstellbar (--dpi, --fontsize)
"""
import re
import math
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from scipy.io import loadmat

# ---- Defaults (dein Pfad) ----
MAT_DEFAULT = Path(r"D:\fwu-ssr\res\plots\used studies\MonovalentVitoCal_HOM_Nachtabsenkung_Knotentemp.mat")
OUT_DEFAULT = Path(r"D:\fwu-ssr\res\plots\Dymola\Knoten")
START_DATE_DEFAULT = "2015-01-22"   # 22.01.
CUTOFF_DATE_DEFAULT = "2015-02-03"  # harter Cut am 03.02

# ---- Style-Defaults (einheitlich steuerbar) ----
DPI_DEFAULT = 300
FONT_DEFAULT = 10.95       # Basisschriftgröße
LINEWIDTH_DEFAULT = 1.15

def apply_style(base_font: int = FONT_DEFAULT, dpi: int = DPI_DEFAULT) -> None:
    """Einheitliche Matplotlib-Settings inkl. DPI & Schriftgrößen."""
    # Schutz gegen zu kleine Fonts
    f0 = max(int(base_font), 6)
    # LaTeX 11pt ≈ 10.95 pt
    LATEX_BASE_PT = 10.95
    DPI_DEFAULT = 300

    mpl.rcParams.update({
        "figure.dpi": DPI_DEFAULT,
        "savefig.dpi": DPI_DEFAULT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",

        # LaTeX/Fonts
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],  # wird von LaTeX gerendert
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",

        # --- einheitlich 11pt (LaTeX) ---
        "font.size": LATEX_BASE_PT,
        "axes.titlesize": LATEX_BASE_PT,
        "axes.labelsize": LATEX_BASE_PT,
        "legend.fontsize": LATEX_BASE_PT,
        "xtick.labelsize": LATEX_BASE_PT,
        "ytick.labelsize": LATEX_BASE_PT,

        # sonstiges
        "axes.xmargin": 0.0,
    })

# ------------ Dymola-Name-Rekonstruktion ------------
def _reconstruct_names_rows(name_array: np.ndarray) -> Tuple[List[str], int]:
    rows = list(name_array)
    return rows, len(rows[0])

def _col_name(j: int, rows: List[str]) -> str:
    return "".join(r[j] for r in rows).strip()

def extract_series(mat_path: Path, base: str) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    d = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    data_2 = np.array(d["data_2"], dtype=float)
    dataInfo = np.array(d["dataInfo"], dtype=np.int64)
    name_array = d["name"]
    rows, _ = _reconstruct_names_rows(name_array)

    prefix = f"{base}["
    cand = [j for j, ch in enumerate(rows[0]) if ch == prefix[0]]
    for k, ch in enumerate(prefix[1:], start=1):
        row_k = rows[k]
        cand = [j for j in cand if row_k[j] == ch]
        if not cand:
            break

    pat = re.compile(rf"^{re.escape(base)}\[(\d+)\]\.y$")
    matches: List[Tuple[int, int]] = []
    for j in cand:
        name = _col_name(j, rows)
        m = pat.match(name)
        if not m:
            continue
        i_idx = int(m.group(1))
        ds = int(dataInfo[0, j])
        krow = int(dataInfo[1, j])
        if ds != 2:
            continue
        matches.append((i_idx, krow))

    if not matches:
        raise RuntimeError(f"Keine Variablen gefunden für Muster {base}[i].y")

    matches.sort(key=lambda x: x[0])

    # Zeit: erste Zeile data_2 (Sekunden) -> Stunden ab Start
    t_sec = data_2[0, :].astype(float)
    t_h = (t_sec - t_sec[0]) / 3600.0

    series: Dict[int, np.ndarray] = {i_idx: data_2[krow, :].astype(float) for (i_idx, krow) in matches}
    return t_h, series

# ------------ Tick-Helfer ------------
def _set_calendar_ticks_cut(ax: plt.Axes,
                            start_dt: dt.datetime,
                            cutoff_dt: dt.datetime,
                            n_major_target: int = 6,
                            n_minor_between: int = 2,
                            fmt: str = "%d.%m",
                            step_days: Optional[int] = None) -> None:
    """
    Major-Ticks auf Mitternacht im Kalenderraster, Achse hart bei cutoff_dt.
    - Falls step_days=None, wird ein ganzzahliger Tages-Schritt gewählt, der cutoff erreicht
      und die Major-Anzahl nahe n_major_target hält.
    """
    # Start/Ende auf Mitternacht
    s0 = dt.datetime(start_dt.year, start_dt.month, start_dt.day)
    e0 = dt.datetime(cutoff_dt.year, cutoff_dt.month, cutoff_dt.day)

    if e0 <= s0:
        e0 = s0 + dt.timedelta(days=1)

    day_diff = (e0 - s0).days

    if step_days is None or step_days < 1:
        # Kandidaten: alle Teiler von day_diff (mind. 1)
        divs = [d for d in range(1, day_diff + 1) if day_diff % d == 0]
        # Wähle den Schritt, dessen Tickanzahl dem Ziel am nächsten kommt;
        # bei Gleichstand bevorzugt größeren Schritt (weniger Ticks).
        best = None
        best_err = 10**9
        for d in divs:
            n = day_diff // d + 1
            err = abs(n - n_major_target)
            if (err < best_err) or (err == best_err and (best is None or d > best)):
                best = d
                best_err = err
        step_days = best if best is not None else 1

    # Major-Ticks erzeugen
    majors_dt = []
    cur = s0
    while cur <= e0:
        majors_dt.append(cur)
        cur += dt.timedelta(days=step_days)

    majors = mdates.date2num(majors_dt)

    # Achse hart am Cutoff
    ax.set_xlim(majors_dt[0], e0)
    ax.set_xmargin(0)
    ax.xaxis.set_major_locator(FixedLocator(majors))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

    # Minor-Ticks gleichmäßig zwischen den Majors
    minors = []
    for u, v in zip(majors[:-1], majors[1:]):
        step = (v - u) / (n_minor_between + 1)
        minors.extend([u + step * k for k in range(1, n_minor_between + 1)])
    ax.xaxis.set_minor_locator(FixedLocator(minors))

def _set_y_ticks_step5(ax: plt.Axes, *arrays: np.ndarray, fmt: str = '%.0f') -> None:
    """Y-Ticks in 5er-Schritten; untere & obere Grenze sind Ticks."""
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
        w_mm, h_mm = 170, 80   # A5 quer (angepasst)
    else:
        w_mm, h_mm = 100, 155  # A5 hoch (angepasst)
    fig = plt.figure(figsize=(w_mm/25.4, h_mm/25.4))
    ax = fig.add_subplot(111)
    return fig, ax

def plot_all_pdf(t_h: np.ndarray, series: Dict[int, np.ndarray], out_pdf: Path,
                 *, start_date: str = START_DATE_DEFAULT, cutoff_date: Optional[str] = CUTOFF_DATE_DEFAULT,
                 orientation: str = "landscape", line_width: float = LINEWIDTH_DEFAULT,
                 n_major_target: int = 6, n_minor_between: int = 2, step_days: Optional[int] = None) -> None:
    fig, ax = _a5_figure(orientation=orientation)

    # Zeit als Datum ab Startdatum
    base = dt.datetime.fromisoformat(start_date)
    x_dt = np.array([base + dt.timedelta(hours=float(h)) for h in t_h])

    # Linien (ohne Legende)
    for i_idx in sorted(series.keys()):
        ax.plot(x_dt, series[i_idx], linewidth=line_width)

    # Achsen + Ränder
    ax.set_xlabel("Datum")
    ax.set_ylabel("Temperatur [°C]")
    ax.grid(True, which="major", linestyle="--", alpha=0.25)
    ax.grid(True, which="minor", linestyle=":",  alpha=0.18)

    # X-Ticks (Kalender, Cut am cutoff_date)
    if cutoff_date:
        cutoff_dt = dt.datetime.fromisoformat(cutoff_date)
        _set_calendar_ticks_cut(ax, x_dt[0], cutoff_dt,
                                n_major_target=n_major_target,
                                n_minor_between=n_minor_between,
                                fmt="%d.%m",
                                step_days=step_days)
    else:
        # Fallback: Spannendecke (ohne Cut) – gleichmäßig zwischen Start/Ende
        a = mdates.date2num(x_dt[0]); b = mdates.date2num(x_dt[-1])
        majors_num = np.linspace(a, b, n_major_target)
        ax.set_xlim(x_dt[0], x_dt[-1]); ax.set_xmargin(0)
        ax.xaxis.set_major_locator(FixedLocator(majors_num))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        minors = []
        for u, v in zip(majors_num[:-1], majors_num[1:]):
            step = (v - u) / (n_minor_between + 1)
            minors.extend([u + step * k for k in range(1, n_minor_between + 1)])
        ax.xaxis.set_minor_locator(FixedLocator(minors))

    # Y-Ticks
    _set_y_ticks_step5(ax, *series.values(), fmt='%.0f')

    plt.tight_layout()
    plt.savefig(out_pdf)  # PDF (Vektor); DPI wirkt nur bei Raster-Elementen
    plt.close(fig)

# ------------ CLI ------------
def _ensure_mat_path(p: Path) -> Path:
    return p if p.suffix.lower() == ".mat" else p.with_suffix(".mat")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mat", type=Path, default=MAT_DEFAULT, help="Pfad zur Dymola .mat-Datei")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT, help="Ausgabeordner")
    p.add_argument("--base", type=str, default="building.TempExtWallLivingroomInC", help="Basename der Variable")
    p.add_argument("--start", type=str, default=START_DATE_DEFAULT, help="Startdatum (YYYY-MM-DD), z.B. 2015-01-22")
    p.add_argument("--cutoff", type=str, default=CUTOFF_DATE_DEFAULT, help="Cutoff-Datum (YYYY-MM-DD), z.B. 2015-02-03; leer lassen für keinen Cut")
    p.add_argument("--orientation", type=str, default="landscape", choices=["landscape", "portrait"],
                   help="A5-Ausrichtung")
    p.add_argument("--dpi", type=int, default=DPI_DEFAULT, help="DPI für Figure/Savefig")
    p.add_argument("--fontsize", type=int, default=FONT_DEFAULT, help="Basisschriftgröße")
    p.add_argument("--lw", type=float, default=LINEWIDTH_DEFAULT, help="Linienbreite")
    p.add_argument("--majors", type=int, default=6, help="Zielanzahl Major-Ticks (nur für Auto-Schritt)")
    p.add_argument("--minors-between", type=int, default=2, help="Minor-Ticks zwischen Majors")
    p.add_argument("--step-days", type=int, default=None, help="Fester Tages-Schritt (überschreibt Auto-Bestimmung)")
    args = p.parse_args()

    # Style anwenden (einheitlich)
    apply_style(base_font=args.fontsize, dpi=args.dpi)

    mat_path = _ensure_mat_path(args.mat)
    args.out.mkdir(parents=True, exist_ok=True)

    t_h, series = extract_series(mat_path, args.base)
    stem = args.base.replace(".", "_")
    out_pdf = args.out / f"{stem}_all.pdf"

    plot_all_pdf(t_h, series, out_pdf,
                 start_date=args.start,
                 cutoff_date=(args.cutoff if args.cutoff else None),
                 orientation=args.orientation,
                 line_width=args.lw,
                 n_major_target=args.majors,
                 n_minor_between=args.minors_between,
                 step_days=args.step_days)
    print(f"Gespeichert: {out_pdf}")

if __name__ == "__main__":
    main()
