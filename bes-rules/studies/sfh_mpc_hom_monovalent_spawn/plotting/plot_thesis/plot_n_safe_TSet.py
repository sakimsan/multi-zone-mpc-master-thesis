#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ein Plot mit allen Zonen (Step), Legende UNTEN in ZWEI Reihen,
x-Achse Mo..So, DIN A4-Breite x ~1/3 Höhe (8.27 x 3.90 in).
Unterstützt 'second_of_week' und 'datetime'.
Kelvin wird automatisch nach °C umgerechnet.
"""
from __future__ import annotations

from pathlib import Path
import re
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, MultipleLocator, FormatStrFormatter  # ← erweitert

# --- Zonen-Mapping ---
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

# Plot-Reihenfolge (falls vorhanden)
PLOT_ORDER = ["livingroom","hobby","corridor","wcstorage","kitchen",
              "bedroom","children","corridor2","bath","children2","attic"]

# --- LaTeX/Styling ---
def _setup_mpl():
    params = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",        # Hinweis: für exakt physische Seitengröße unten bbox_inches=None setzen
        "axes.unicode_minus": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
    }
    mpl.rcParams.update(params)
    # Fallback ohne TeX
    try:
        fig = plt.figure()
        fig.text(0.5, 0.5, r"TeX Test: $T_{\mathrm{set}}$", ha="center")
        plt.close(fig)
    except Exception:
        params["text.usetex"] = False
        params.pop("text.latex.preamble", None)
        mpl.rcParams.update(params)

def _to_celsius(arr: np.ndarray) -> np.ndarray:
    return arr - 273.15 if np.nanmax(arr) > 200.0 else arr

def _sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-ÄÖÜäöüß]", "", s)

# --- X-Achsen-Styling für Wochenplots ---
def style_week_axis(ax,
                    end_week_sec=7*24*3600,
                    day_labels=("Mo","Di","Mi","Do","Fr","Sa","So"),
                    minor_every_hours=6,
                    shade_zebra=True,
                    shade_weekend=True):
    """Formatiert die x-Achse für Wochenplots (Sekunden 0..604800)."""
    # 1) Major-Ticks an Tagesgrenzen
    day_edges = np.arange(0, end_week_sec+1, 86400, dtype=int)  # 0, 86400, ...
    ax.set_xlim(0, end_week_sec)
    ax.xaxis.set_major_locator(FixedLocator(day_edges[:-1]))
    ax.set_xticklabels(list(day_labels))

    # 2) Minor-Ticks alle X Stunden
    step = int(minor_every_hours * 3600)
    minor_ticks = np.arange(0, end_week_sec+1, step, dtype=int)
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.6)

    # 3) Schattierung (zuerst Zebra, dann Wochenende etwas dunkler)
    if shade_zebra:
        for d in range(7):
            if d % 2 == 0:
                ax.axvspan(d*86400, (d+1)*86400, color="0.97", zorder=0)
    if shade_weekend:
        ax.axvspan(5*86400, 6*86400, color="0.93", zorder=0)
        ax.axvspan(6*86400, 7*86400, color="0.93", zorder=0)

# --- Y-Achsen-Logik: 0,5er-Raster, Rand-Ticks, kein Randkontakt ---
# --- Y-Achsen-Logik: Ganzzahl- (1.0) Raster, Rand-Ticks, kein Randkontakt ---
def style_y_axis_with_half_steps(ax, Y: np.ndarray, major_step: float = 1.0, minor_step: float = 0.5):
    """
    Setzt:
      - ylimits auf 1.0er-Raster, erweitert um 1.0 falls Daten das Limit berühren
      - Minor-Ticks alle 0,5
      - Major-Ticks (standardmäßig 1,0) + garantiert Ticks exakt an y-Min/Max
    """
    def _round_down(x, step): return math.floor(x / step) * step
    def _round_up(x, step):   return math.ceil(x  / step) * step

    y_min_data = float(np.nanmin(Y))
    y_max_data = float(np.nanmax(Y))

    # Auf 1.0er-Raster runden
    y_lo = _round_down(y_min_data, 1.0)
    y_hi = _round_up( y_max_data, 1.0)

    # Wenn Daten das Limit berühren -> um weitere 1.0 erweitern
    if math.isclose(y_lo, y_min_data, rel_tol=0.0, abs_tol=1e-12):
        y_lo -= 1.0
    if math.isclose(y_hi, y_max_data, rel_tol=0.0, abs_tol=1e-12):
        y_hi += 1.0

    # Spezieller Fall: konstante Daten
    if not (y_hi > y_lo):
        y_lo -= 1.0
        y_hi += 1.0

    ax.set_ylim(y_lo, y_hi)

    # Major-Ticks (1.0er Raster) + Grenzen sicherstellen
    start_major = _round_up(y_lo,  major_step)
    end_major   = _round_down(y_hi, major_step)
    majors_core = list(np.arange(start_major, end_major + 0.001*major_step, major_step))
    majors = sorted(set([round(x, 2) for x in (majors_core + [y_lo, y_hi])]))
    ax.yaxis.set_major_locator(FixedLocator(majors))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Minor-Ticks 0.5
    ax.yaxis.set_minor_locator(MultipleLocator(minor_step))

    # dezente Gitterlinien
    ax.grid(True, which="major", axis="y", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", axis="y", alpha=0.15, linewidth=0.6)


def plot_all_zones_one_figure(csv_path: str | Path, out_path: str | Path):
    """Ein Plot mit allen vorhandenen Zonen (Step). Legende unten in zwei Reihen. X-Achse: Mo..So."""
    _setup_mpl()

    csv_path = Path(csv_path)
    out_path = Path(out_path)
    df = pd.read_csv(csv_path)

    # Tippfehler-Korrektur
    if "corrdior2" in df.columns and "corridor2" not in df.columns:
        df = df.rename(columns={"corrdior2": "corridor2"})

    # Zeit in Sekunden innerhalb der Woche
    if "second_of_week" in df.columns:
        sec = df["second_of_week"].to_numpy(dtype=np.int64)
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        sec = (dt.dt.dayofweek.to_numpy(dtype=np.int64) * 86400
               + (dt.dt.hour*3600 + dt.dt.minute*60 + dt.dt.second).to_numpy(dtype=np.int64))
    else:
        raise ValueError("CSV benötigt Spalte 'second_of_week' oder 'datetime'.")

    # Zonen-Spalten
    candidate_cols = list(ZONE_NAME_MAP.keys())
    zone_cols = [c for c in df.columns if c in candidate_cols]
    if not zone_cols:
        raise ValueError(f"Keine Zonen-Spalten gefunden. Vorhanden: {list(df.columns)}")

    zone_cols_sorted = [z for z in PLOT_ORDER if z in zone_cols] + [z for z in zone_cols if z not in PLOT_ORDER]

    # Change-only bis Wochenende ziehen
    END_WEEK = 7*24*3600  # 604800
    if sec.max() < END_WEEK:
        tail_row = {c: df[c].iloc[-1] for c in zone_cols_sorted}
        df_tail = pd.DataFrame([tail_row])
        sec = np.concatenate([sec, np.array([END_WEEK], dtype=np.int64)])
        df = pd.concat([df, df_tail], ignore_index=True)

    # sortieren
    order = np.argsort(sec)
    sec = sec[order]
    Y = df[zone_cols_sorted].to_numpy(dtype=float)[order, :]
    Y = _to_celsius(Y)

    # --- Figure: A4-Breite x ~1/3 Höhe (Quer) ---
    fig, ax = plt.subplots(figsize=(8.27, 3.0), constrained_layout=False)

    # Linien
    for j, z in enumerate(zone_cols_sorted):
        label = ZONE_NAME_MAP.get(z, z)
        ax.step(sec, Y[:, j], where="post", linewidth=1.0, label=label)

    # Achsenformat + X-Achse stylen
    ax.set_ylabel(r"$T\;[^\circ\mathrm{C}]$")
    style_week_axis(ax, minor_every_hours=6, shade_zebra=True, shade_weekend=False)

    # Y-Achse nach Wunsch stylen (siehe Anforderungen)
    style_y_axis_with_half_steps(ax, Y, major_step=1.0, minor_step=0.5)

    # --- Legende unten, zwei Reihen ---
    # Platz unten reservieren (je nach Anzahl Labels evtl. anpassen)
    fig.subplots_adjust(bottom=0.21)
    handles, labels = ax.get_legend_handles_labels()
    ncols = max(1, math.ceil(len(labels) / 2))  # zwei Reihen -> Spalten = ceil(N/2)
    fig.legend(handles, labels,
               loc="lower center",
               ncol=ncols,
               frameon=False,
               handlelength=1.8,
               columnspacing=1.2,
               fontsize=8.5)

    # --- Speichern ---
    if out_path.suffix.lower() in {".pdf", ".png"}:
        out_pdf = out_path.with_suffix(".pdf")
        out_png = out_path.with_suffix(".png")
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        base = _sanitize_filename("Tset_Alle_Zonen")
        out_pdf = out_path / f"{base}.pdf"
        out_png = out_path / f"{base}.png"

    # Für exakt physische Seitengröße ggf. bbox überschreiben:
    fig.savefig(out_pdf, bbox_inches=None)
    fig.savefig(out_png, dpi=300, bbox_inches=None)
    plt.close(fig)
    print(f"[OK] Plot gespeichert: {out_pdf}  (und {out_png})")

# --- Beispiel ---
if __name__ == "__main__":
    IN_CSV = r"D:\fwu-ssr\bes-rules\student_theses\sakimsan\case_study_TSet\Arbeitswoche_NormalerWinter_changeonly_K.csv"
    OUT    = r"D:\fwu-ssr\res\plots"
    plot_all_zones_one_figure(IN_CSV, OUT)
