#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from ast import literal_eval
from typing import Optional, Dict, List, Tuple
import math
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox

# -------------------- LaTeX/Styling --------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "text.usetex": True,
    "axes.unicode_minus": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.xmargin": 0.0,
})

# -------------------- Defaults/Pfade --------------------
DEFAULT_MPC_CSV = Path(
    r"D:\fwu-ssr\res\plots\used studies\Studies_coupled\extreme_Nachtabsenkung\Design_0_mpc_agent.csv"
)
DEFAULT_OUT_DIR = Path(r"D:\fwu-ssr\res\plots\test_plotNsafeMPC")

# -------------------- Farben & Stile --------------------
TSET_COLOR   = "deepskyblue"
TAIR_STYLE   = dict(color="black", linestyle="-",  linewidth=1.15)
TSET_STYLE   = dict(color=TSET_COLOR, linestyle="--", linewidth=1.0)
BAND_STYLE   = dict(color="red",   linestyle="-",  linewidth=1.15)  # nur obere Grenze
VALVE_COLOR  = "navy"

def orange_gradient(n: int, vmin: float = 0.3, vmax: float = 0.9):
    if n <= 0:
        return []
    cmap = cm.get_cmap("Oranges")
    return [cmap(v) for v in np.linspace(vmin, vmax, n)]

SET_BAND_K = 2.0

# -------------------- Papiergrößen --------------------
MM = 25.4
A6_PORTRAIT  = (105 / MM, 148 / MM)  # 4.13" x 5.83"
A6_LANDSCAPE = (148 / MM, 105 / MM)  # 5.83" x 4.13"
# LaTeX-Textfeld (Breite × Höhe) = 155 mm × 222 mm
TEXTBLOCK_IN = (155 / MM, 222 / MM)

# -------------------- Einheit-Label Platzierung --------------------
UNIT_LABEL_X = -0.13  # Default: links neben der Achse; für Einzelplots enger setzen

def _unit_label(ax: plt.Axes, text: str, x_frac: float = UNIT_LABEL_X):
    """Platziert ein vertikales Einheiten-Label links neben der Achse in konstantem Abstand."""
    ax.set_ylabel("")
    ax.yaxis.labelpad = 0
    ax.text(
        x_frac, 0.5, text,
        transform=ax.transAxes,
        rotation=90, va="center", ha="right",
        fontsize=mpl.rcParams.get("axes.labelsize", 9),
        clip_on=False,
    )

# -------------------- Zonen --------------------
ZONE_MAP_DE: Dict[str, str] = {
    "livingroom": "Wohnen", "hobby": "Hobby", "corridor": "Flur",
    "wcstorage": "WC", "kitchen": "Küche", "bedroom": "Schlafen",
    "children": "Kind", "corridor2": "Flur 2", "corrdior2": "Flur 2",
    "bath": "Bad", "children2": "Kind 2", "attic": "Dachgeschoss",
}
ZONE_ORDER_CANONICAL = [
    "livingroom","hobby","corridor","wcstorage",
    "kitchen","bedroom","children","corridor2",
    "bath","children2","attic"
]
def _canonical(s: str) -> str: return s.strip().lower()
def _display_name(zone: str) -> str: return ZONE_MAP_DE.get(_canonical(zone), zone)

# -------------------- MPC laden & Hilfen --------------------
def load_mpc_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    outer = [literal_eval(i) for i in df.index]
    df.index = pd.MultiIndex.from_tuples(outer)
    return df

def filter_mpc_time(df: pd.DataFrame, *, skip_first_days: int = 3, t_end_days: Optional[float] = None) -> pd.DataFrame:
    outer = df.index.get_level_values(0).astype(float)
    if len(outer) == 0:
        return df
    start_sec = float(np.min(outer))
    cutoff = start_sec + skip_first_days * 86400.0
    mask = outer >= cutoff
    if t_end_days is not None:
        end_sec = cutoff + float(t_end_days) * 86400.0
        mask &= (outer <= end_sec)
    return df[mask]

def sec_to_datetime(seconds: np.ndarray, year: int = 2015) -> np.ndarray:
    base = pd.Timestamp(f"{year}-01-01 00:00:00")
    return base + pd.to_timedelta(seconds, unit="s")

def first_values_per_step(series: pd.Series) -> pd.Series:
    return series.groupby(level=0).first()

def prediction_paths(series: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
    paths: List[Tuple[np.ndarray, np.ndarray]] = []
    for t0, grp in series.groupby(level=0):
        g = grp.droplevel(0)
        x = g.index.values.astype(float) + float(t0)
        y = g.values.astype(float)
        paths.append((x, y))
    return paths

def to_celsius(arr: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(arr, dtype=float) - 273.15

def sanitize_filename(part: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", part)

def get_mpc_series(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    if "variable" in df.columns.get_level_values(0) and name in df["variable"].columns:
        return df["variable"][name]
    if "parameter" in df.columns.get_level_values(0) and name in df["parameter"].columns:
        return df["parameter"][name]
    return None

# ---- 4 Ticks (.0/.5) Basis ----
def _n_half_ticks_from_values(values: np.ndarray, n_ticks: int) -> tuple[list[float], tuple[float, float]]:
    assert n_ticks >= 2
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        step = 0.5
        lo = 0.0
        hi = lo + step * (n_ticks - 1)
        ticks = [lo + i * step for i in range(n_ticks)]
        return ticks, (lo, hi)

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    def round_down_half(x: float) -> float: return np.floor(x * 2.0) / 2.0
    def round_up_half(x: float) -> float:   return np.ceil(x * 2.0) / 2.0

    if vmin == vmax:
        center = round_down_half(vmin)
        step = 1.0 if n_ticks == 4 else 0.5
        lo = round_down_half(center - step * ((n_ticks - 1) / 2.0))
        hi = lo + step * (n_ticks - 1)
        ticks = [lo + i * step for i in range(n_ticks)]
        return ticks, (lo, hi)

    raw_step = (vmax - vmin) / (n_ticks - 1)
    step = max(0.5, np.ceil(raw_step / 0.5) * 0.5)

    lo = round_down_half(vmin)
    hi = lo + step * (n_ticks - 1)
    if hi < vmax - 1e-12:
        hi = round_up_half(vmax)
        lo = round_down_half(hi - step * (n_ticks - 1))

    ticks = [lo + i * step for i in range(n_ticks)]
    return ticks, (lo, hi)

# ---- Erweiterte Temp-Ticks (immer äquidistant) ----
def _apply_temp_ticks_4_extend(ax: plt.Axes, *value_arrays: np.ndarray, touch_eps: float = 1e-6):
    """4 Ticks, äquidistant und auf .0/.5. Schrittweite >= 0.5 K."""
    if len(value_arrays) == 0:
        vals = np.array([0.0])
    else:
        vals = np.concatenate([
            np.asarray(v, dtype=float).ravel()
            for v in value_arrays if v is not None and np.size(v) > 0
        ])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = np.array([0.0])

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    half_floor = lambda x: np.floor(x * 2.0) / 2.0
    half_ceil  = lambda x: np.ceil(x * 2.0) / 2.0

    N = 4
    step = max(0.5, half_ceil((vmax - vmin) / (N - 1)))

    lo = half_floor(vmin)
    hi = lo + step * (N - 1)
    if hi < vmax - touch_eps:
        step = max(step, half_ceil((vmax - lo) / (N - 1)))
        hi = lo + step * (N - 1)

    ticks = [lo + i * step for i in range(N)]
    ax.set_yticks(ticks)
    ax.set_ylim(ticks[0], ticks[-1])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# ---- Ventil-Achse fix 0..1 ----
def _apply_valve_ticks_fixed01(ax):
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 1.0])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.minorticks_off()

# ---- Abstand Temp↔Ventil verringern ----
def reduce_vertical_gap(ax_top, ax_bottom, gap: float = 0.02):
    p_top = ax_top.get_position()
    p_bot = ax_bottom.get_position()
    new_top_y1 = max(p_bot.y0 + 0.005, p_top.y0 - gap)
    new_bot = Bbox.from_extents(p_bot.x0, p_bot.y0, p_bot.x1, new_top_y1)
    ax_bottom.set_position(new_bot)

# ---- Legenden (Einzelplots) ----
def _center_legend(legend):
    try:
        legend._legend_box.align = "center"
        for pack in getattr(legend, "_legend_box").get_children():
            try:
                pack.align = "center"
            except Exception:
                pass
    except Exception:
        pass

def _add_centered_multiline_legend(fig: plt.Figure, handles: List[Line2D],
                                   y_top: float = 0.06, y_bottom: float = 0.02):
    """Zwei übereinander zentrierte Legendenzeilen (Einzelplots)."""
    labels = [h.get_label() for h in handles]
    n = len(handles)
    split = int(np.ceil(n / 2))
    row1_h, row1_l = handles[:split], labels[:split]
    row2_h, row2_l = handles[split:], labels[split:]

    leg1 = fig.legend(row1_h, row1_l, loc="lower center",
        ncol=len(row1_h), frameon=False, bbox_to_anchor=(0.5, y_top))
    _center_legend(leg1)
    fig.add_artist(leg1)

    if row2_h:
        leg2 = fig.legend(row2_h, row2_l, loc="lower center",
            ncol=len(row2_h), frameon=False, bbox_to_anchor=(0.5, y_bottom))
        _center_legend(leg2)
        fig.add_artist(leg2)

# ---- Q_RadSol Summen-Hilfe + "Nice" Ticks ----
def sum_first_values_over_zones(df: pd.DataFrame, prefix: str, zones: List[str],
                                idx_ref: pd.Index, remove_first_point: bool) -> Optional[pd.Series]:
    """Summiert first_values_per_step(prefix_<zone>) über Zonen und reindext auf idx_ref."""
    acc: Optional[pd.Series] = None
    any_found = False
    for z in zones:
        s = get_mpc_series(df, f"{prefix}_{z}")
        if s is None:
            continue
        s_act = first_values_per_step(s)
        if remove_first_point and len(s_act) > 0:
            s_act = s_act.iloc[1:]
        s_act = s_act.reindex(idx_ref, fill_value=0.0)  # auf Referenz-Seconds
        acc = (s_act if acc is None else acc.add(s_act, fill_value=0.0))
        any_found = True
    return acc if any_found else None

def _nice_step(data_range: float, n_intervals: int) -> float:
    """1-2-5 Schrittweite (x10^k) passend zur Range/n_intervals."""
    if n_intervals <= 0 or data_range <= 0:
        return 1.0
    raw = data_range / n_intervals
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    if norm <= 1:
        step = 1 * mag
    elif norm <= 2:
        step = 2 * mag
    elif norm <= 5:
        step = 5 * mag
    else:
        step = 10 * mag
    return step

def _apply_numeric_ticks_n(ax: plt.Axes, values: np.ndarray, n_ticks: int = 4, fmt: str = '%.0f', eps: float = 1e-12):
    """Setzt genau n_ticks äquidistante Y-Ticks auf "nice" Werte für beliebige Daten."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = np.array([0.0])
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax == vmin:
        step = _nice_step(max(1.0, abs(vmin)), n_ticks - 1)
        lo = vmin - step * ((n_ticks - 1) / 2.0)
        ticks = [lo + i * step for i in range(n_ticks)]
    else:
        step = _nice_step(vmax - vmin, n_ticks - 1)
        lo = math.floor(vmin / step) * step
        hi = lo + step * (n_ticks - 1)
        if hi < vmax - eps:
            lo = math.floor((vmax - step * (n_ticks - 1)) / step) * step
        ticks = [lo + i * step for i in range(n_ticks)]
    ax.set_yticks(ticks)
    ax.set_ylim(ticks[0], ticks[-1])
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

# ---- Status-Farben & Loader ----
STATUS_COLOR_MAXCPU = "deeppink"         # dunkelpink
STATUS_COLOR_OK     = "darkgoldenrod"    # gelb/dunkel
STATUS_COLOR_SOLVED = "green"            # grün für Solve_Succeeded

def _find_stats_file(mpc_csv_path: Path) -> Optional[Path]:
    folder = mpc_csv_path.parent
    cand = sorted(folder.glob("stats_Design_0_mpc_agent*"))
    return cand[0] if cand else None

def _load_status_series(stats_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(stats_path)
        if "return_status" not in df.columns:
            return None
        time_col = df.columns[0]
        out = pd.DataFrame({
            "t_sec": pd.to_numeric(df[time_col], errors="coerce"),
            "status": df["return_status"].astype(str).str.strip()
        }).dropna(subset=["t_sec"])
        return out
    except Exception:
        return None

def _plot_status(ax: plt.Axes,
                 status_df: Optional[pd.DataFrame],
                 base_year: int,
                 time_dt_full: np.ndarray,
                 show_xlabel: bool):
    """Vertikale Linien je nach return_status. Y-Achse ausgeblendet."""
    if status_df is None or len(status_df) == 0:
        ax.text(0.5, 0.5, "stats_Design_0_mpc_agent* nicht gefunden",
                ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, linestyle="--", alpha=0.15, axis="x")
        if show_xlabel:
            ax.set_xlabel("Datum")
        ax.set_yticks([]); ax.set_ylim(0, 1)
        return

    t_lines = sec_to_datetime(np.asarray(status_df["t_sec"].values, dtype=float), base_year)
    for t, s in zip(t_lines, status_df["status"]):
        if s == "Maximum_CpuTime_Exceeded":
            ax.axvline(t, color=STATUS_COLOR_MAXCPU, linewidth=1.5, alpha=0.95)
        elif s == "Solved_To_Acceptable_Level":
            ax.axvline(t, color=STATUS_COLOR_OK, linewidth=1.5, alpha=0.95)
        elif s == "Solve_Succeeded":
            ax.axvline(t, color=STATUS_COLOR_SOLVED, linewidth=1.5, alpha=0.95)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(True, linestyle="--", alpha=0.15, axis="x")
    ax.margins(x=0)
    if len(time_dt_full) > 1:
        ax.set_xlim(time_dt_full[0], time_dt_full[-1])
    if show_xlabel:
        ax.set_xlabel("Datum")
    ax.set_title("Optimierungs-Status", pad=4)

# -------------------- Plot-Funktion --------------------
def plot_three_panel_mpc_with_yval_tbuf(
    mpc_csv_path: Path = DEFAULT_MPC_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    base_year: int = 2015,
    skip_first_days: int = 3,
    remove_first_point: bool = True,
    t_end_days: Optional[float] = None,
    set_band: float = SET_BAND_K,
    show_plot: bool = True,
    out_name: str = "MPC_Dashboard.pdf",
    # Einzelplots:
    save_individual: bool = True,
    individual_dir: Optional[Path] = None,
    individual_format: str = "pdf",
    individual_dpi: int = 300,
    individual_a6_landscape: bool = True,   # DIN A6 Querformat als Default
) -> Path:
    if not mpc_csv_path.exists():
        raise FileNotFoundError(f"MPC-Datei nicht gefunden: {mpc_csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    zones_dir = (individual_dir or (out_dir / "zones"))
    if save_individual:
        zones_dir.mkdir(parents=True, exist_ok=True)

    df_full = load_mpc_results(mpc_csv_path)
    df = filter_mpc_time(df_full, skip_first_days=skip_first_days, t_end_days=t_end_days)

    zones_available = [z for z in ZONE_ORDER_CANONICAL if get_mpc_series(df, f"T_Air_{z}") is not None]
    if not zones_available:
        raise ValueError("Keine T_Air_<zone> im MPC gefunden.")

    # Referenzzeitachse aus Actual-Werten
    s_ref = get_mpc_series(df, f"T_Air_{zones_available[0]}")
    s_act_ref = first_values_per_step(s_ref)
    if remove_first_point and len(s_act_ref) > 0:
        s_act_ref = s_act_ref.iloc[1:]
    time_idx = s_act_ref.index            # Sekunden (outer)
    time_dt  = sec_to_datetime(time_idx.values, base_year)

    left_zones  = zones_available[:5]
    right_zones = zones_available[5:10]
    attic_zone  = next((z for z in zones_available if _canonical(z) == "attic"), None)
    has_attic   = attic_zone is not None

    # Index für Dateinamen (Einzelplots)
    zone_index_map = {z: i+1 for i, z in enumerate(zones_available)}

    # -------------------- Figure/Gridspec auf LaTeX-Textfeldgröße --------------------
    height_ratios = [3, 1]*5 + [3, 3]   # 5 Zonenpaare + 2 zusätzliche "Temp-Höhen" Reihen
    fig = plt.figure(figsize=TEXTBLOCK_IN, constrained_layout=False)
    gs = GridSpec(nrows=len(height_ratios), ncols=2, figure=fig, height_ratios=height_ratios)

    # Legendeneinträge (inkl. Status-Linien, umbenannt)
    legend_handles = [
        Line2D([0], [0], **TAIR_STYLE, label=r"Raumtemperatur "),
        Line2D([0], [0], **TSET_STYLE, label=r"Solltemperatur "),
        Line2D([0], [0], **BAND_STYLE, label=r"obere Grenze"),
        Line2D([0], [0], color=orange_gradient(2)[-1], linewidth=1.2, label="Prädiktionen"),
        Line2D([0], [0], color=VALVE_COLOR, linewidth=1.2, label="Ventilstellung"),
        Line2D([0], [0], color=STATUS_COLOR_MAXCPU, linewidth=1.3, label="max. CPU-Zeitlimit"),
        Line2D([0], [0], color=STATUS_COLOR_OK,     linewidth=1.3, label="gute Näherung"),
        Line2D([0], [0], color=STATUS_COLOR_SOLVED, linewidth=1.3, label="optimal gelöst"),
    ]

    def _pred_values_c(series: Optional[pd.Series]) -> np.ndarray:
        if series is None:
            return np.array([])
        paths = prediction_paths(series)
        if not paths:
            return np.array([])
        return np.concatenate([to_celsius(y) for _, y in paths])

    # --- MOD: _plot_zone erweitert: show_title + unit_label_x ---
    def _plot_zone(ax_t, ax_y, zone: str, *, show_title: bool = True, unit_label_x: Optional[float] = None):
        zone_de = _display_name(zone)
        s_tair_full = get_mpc_series(df, f"T_Air_{zone}")
        if s_tair_full is None:
            if show_title:
                ax_t.set_title(zone_de + " (keine Daten)", pad=4)
            ax_t.set_visible(True); ax_t.grid(True, linestyle="--", alpha=0.3)
            if ax_y is not None: ax_y.set_visible(False)
            _unit_label(ax_t, r"[$^\circ$C]", x_frac=(unit_label_x if unit_label_x is not None else UNIT_LABEL_X))
            return

        s_tair_act = first_values_per_step(s_tair_full)
        if remove_first_point and len(s_tair_act) > 0:
            s_tair_act = s_tair_act.iloc[1:]
        tair_c = to_celsius(s_tair_act.values)

        n_preds_t = len(s_tair_full.index.get_level_values(0).unique())
        colors_tair = orange_gradient(max(1, n_preds_t))
        for (xp, yp), col in zip(prediction_paths(s_tair_full), colors_tair):
            ax_t.plot(sec_to_datetime(xp, base_year), to_celsius(yp), color=col, linewidth=0.6)
        pred_tair_c = _pred_values_c(s_tair_full)

        s_tset_full = get_mpc_series(df, f"TSetOneZone_{zone}") if _canonical(zone) != "attic" else None
        tset_c = None
        pred_tset_c = np.array([])
        if s_tset_full is not None:
            s_tset_act = first_values_per_step(s_tset_full)
            if remove_first_point and len(s_tset_act) > 0:
                s_tset_act = s_tset_act.iloc[1:]
            tset_c = to_celsius(s_tset_act.values)

            n_preds_s = len(s_tset_full.index.get_level_values(0).unique())
            colors_tset = orange_gradient(max(1, n_preds_s))
            for (xs, ys), col in zip(prediction_paths(s_tset_full), colors_tset):
                ax_t.plot(sec_to_datetime(xs, base_year), to_celsius(ys), color=col, linewidth=0.6)
            pred_tset_c = _pred_values_c(s_tset_full)

        if tset_c is not None:
            ax_t.plot(time_dt, tset_c + set_band, **BAND_STYLE)
            ax_t.plot(time_dt, tset_c, **TSET_STYLE)
        ax_t.plot(time_dt, tair_c, **TAIR_STYLE)

        if show_title:
            ax_t.set_title(zone_de, loc="center", pad=4)
        ax_t.grid(True, linestyle="--", alpha=0.3)
        ax_t.margins(x=0)
        if len(time_dt) > 1:
            ax_t.set_xlim(time_dt[0], time_dt[-1])

        arrays_for_limits = [tair_c, pred_tair_c]
        if tset_c is not None:
            arrays_for_limits += [tset_c, tset_c + set_band, pred_tset_c]
        _apply_temp_ticks_4_extend(ax_t, *arrays_for_limits)

        _unit_label(ax_t, r"[$^\circ$C]", x_frac=(unit_label_x if unit_label_x is not None else UNIT_LABEL_X))

        if ax_y is not None:
            if _canonical(zone) == "attic":
                ax_y.set_visible(False)
            else:
                s_y_full = get_mpc_series(df, f"yValSet_{zone}")
                if s_y_full is not None:
                    for (xy, yy), _ in zip(prediction_paths(s_y_full), range(10**9)):
                        ax_y.plot(sec_to_datetime(xy, base_year), yy, color=VALVE_COLOR, linewidth=0.6, alpha=0.6)
                    s_y_act = first_values_per_step(s_y_full)
                    if remove_first_point and len(s_y_act) > 0:
                        s_y_act = s_y_act.iloc[1:]
                    ax_y.plot(time_dt, s_y_act.values, color=VALVE_COLOR, linewidth=1.2)

                _apply_valve_ticks_fixed01(ax_y)
                ax_y.grid(True, linestyle="--", alpha=0.2)
                ax_y.margins(x=0)
                if len(time_dt) > 1:
                    ax_y.set_xlim(time_dt[0], time_dt[-1])

                _unit_label(ax_y, "[-]", x_frac=(unit_label_x if unit_label_x is not None else UNIT_LABEL_X))

    # --- Einzelplot-Layout-Parameter (für zwei Varianten) ---
    INDIV_LABEL_X      = -0.06   # Y-Label näher an Plot (Einzelplots)
    LEGEND_EXTRA_IN    = 0.55    # zusätzliche Höhe (Zoll) für die 2-zeilige Legende
    LEGEND_BOTTOM_PAD_IN = 0.08  # Abstand Unterkante Figure -> untere Legendenzeile (Zoll)
    LEGEND_ROW_GAP_IN  = 0.18    # vertikaler Abstand zwischen beiden Legendenzeilen (Zoll)

    # ---------- Einzelplots speichern: zwei Varianten ----------
    def _save_individual(zone: str):
        if not save_individual:
            return

        # Basisgröße (ohne Legende) – definiert die Plotfläche
        fig_size = (155 / 25.4, (222 / 25.4) / 4.5)   # (Breite, Höhe) in Zoll
        is_attic = (_canonical(zone) == "attic")

        # ---------- A) Variante OHNE Legende (Basisgröße, Plotfläche definierend) ----------
        figA = plt.figure(figsize=fig_size, constrained_layout=False)

        if is_attic:
            gsA  = GridSpec(nrows=1, ncols=1, figure=figA)
            axT_A = figA.add_subplot(gsA[0, 0])
            _plot_zone(axT_A, None, zone, show_title=False, unit_label_x=INDIV_LABEL_X)

            axT_A.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            axT_A.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
            axT_A.set_xlabel("Datum")

            figA.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.16)

            # Geometrie in Zoll merken
            H0 = fig_size[1]
            posT = axT_A.get_position()
            x0T, x1T = posT.x0, posT.x1
            y0T_in   = posT.y0 * H0
            hT_in    = (posT.y1 - posT.y0) * H0

        else:
            gsA  = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], figure=figA)
            axT_A = figA.add_subplot(gsA[0, 0])
            axY_A = figA.add_subplot(gsA[1, 0], sharex=axT_A)
            _plot_zone(axT_A, axY_A, zone, show_title=False, unit_label_x=INDIV_LABEL_X)

            for ax in (axT_A, axY_A):
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
            axT_A.tick_params(axis='x', which='both', labelbottom=False)
            axT_A.set_xlabel("")

            reduce_vertical_gap(axT_A, axY_A, gap=0.01)

            figA.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.16, hspace=0.35)

            # Geometrie in Zoll merken
            H0   = fig_size[1]
            posT = axT_A.get_position()
            posY = axY_A.get_position()

            x0, x1 = posT.x0, posT.x1
            hT_in = (posT.y1 - posT.y0) * H0
            hY_in = (posY.y1 - posY.y0) * H0
            gap_in = (posT.y0 - posY.y1) * H0
            y0Y_in = posY.y0 * H0
            y0T_in = y0Y_in + hY_in + gap_in

        zone_de = _display_name(zone)
        out_no_legend = zones_dir / (f"{zone_index_map[zone]:02d}_{sanitize_filename(zone_de)}_noLegend.{individual_format}")
        figA.savefig(out_no_legend, dpi=individual_dpi, bbox_inches=None)
        plt.close(figA)

        # ---------- B) Variante MIT Legende (Figur höher, Plotfläche identisch) ----------
        W, H0 = fig_size
        H1 = H0 + LEGEND_EXTRA_IN
        figB = plt.figure(figsize=(W, H1), constrained_layout=False)

        if is_attic:
            # gleiche Plotbreite, vertikal um LEGEND_EXTRA_IN verschoben
            x0 = x0T
            x1 = x1T
            new_y0 = (y0T_in + LEGEND_EXTRA_IN) / H1
            new_y1 = (y0T_in + LEGEND_EXTRA_IN + hT_in) / H1
            axT_B = figB.add_axes([x0, new_y0, (x1 - x0), (new_y1 - new_y0)])

            _plot_zone(axT_B, None, zone, show_title=False, unit_label_x=INDIV_LABEL_X)
            axT_B.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            axT_B.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
            axT_B.set_xlabel("Datum")

            y_bottom = (LEGEND_BOTTOM_PAD_IN) / H1
            y_top    = (LEGEND_BOTTOM_PAD_IN + LEGEND_ROW_GAP_IN) / H1
            _add_centered_multiline_legend(figB, legend_handles[:5], y_top=y_top, y_bottom=y_bottom)

        else:
            # untere Achse
            new_y0Y = (y0Y_in + LEGEND_EXTRA_IN) / H1
            new_y1Y = (y0Y_in + LEGEND_EXTRA_IN + hY_in) / H1
            axY_B = figB.add_axes([x0, new_y0Y, (x1 - x0), (new_y1Y - new_y0Y)])

            # obere Achse (gleicher Gap in Zoll)
            new_y0T = (y0T_in + LEGEND_EXTRA_IN) / H1
            new_y1T = (y0T_in + LEGEND_EXTRA_IN + hT_in) / H1
            axT_B = figB.add_axes([x0, new_y0T, (x1 - x0), (new_y1T - new_y0T)])

            _plot_zone(axT_B, axY_B, zone, show_title=False, unit_label_x=INDIV_LABEL_X)

            for ax in (axT_B, axY_B):
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
            axT_B.tick_params(axis='x', which='both', labelbottom=False)
            axT_B.set_xlabel("")

            y_bottom = (LEGEND_BOTTOM_PAD_IN) / H1
            y_top    = (LEGEND_BOTTOM_PAD_IN + LEGEND_ROW_GAP_IN) / H1
            _add_centered_multiline_legend(figB, legend_handles[:5], y_top=y_top, y_bottom=y_bottom)

        out_with_legend = zones_dir / (f"{zone_index_map[zone]:02d}_{sanitize_filename(zone_de)}_withLegend.{individual_format}")
        figB.savefig(out_with_legend, dpi=individual_dpi, bbox_inches=None)
        plt.close(figB)

    # ---------- Hauptlayout: 5 Zonenpaare ----------
    pair_axes: List[Tuple[plt.Axes, plt.Axes]] = []
    row = 0
    for zone in left_zones:
        ax_t = fig.add_subplot(gs[row, 0])
        ax_y = fig.add_subplot(gs[row+1, 0], sharex=ax_t)
        _plot_zone(ax_t, ax_y, zone)  # Hauptfigur: Titel bleibt, Standard-Label-Offset
        pair_axes.append((ax_t, ax_y)); row += 2
        _save_individual(zone)

    row = 0
    for zone in right_zones:
        ax_t = fig.add_subplot(gs[row, 1])
        ax_y = fig.add_subplot(gs[row+1, 1], sharex=ax_t)
        _plot_zone(ax_t, ax_y, zone)  # Hauptfigur: Titel bleibt, Standard-Label-Offset
        pair_axes.append((ax_t, ax_y)); row += 2
        _save_individual(zone)

    # ---------- Zusätzliche Zeilen: Attic/TBufSet links, rechts Q_RadSol + Status ----------
    next_row_index = 10  # nach 5*2 Zeilen

    # Stats laden für Status-Plot
    stats_path = _find_stats_file(mpc_csv_path)
    status_df = _load_status_series(stats_path) if stats_path else None

    # --- Zeile 1: Attic links, Qsum rechts ---
    ax_attic_left = fig.add_subplot(gs[next_row_index, 0])
    if has_attic:
        _plot_zone(ax_attic_left, None, attic_zone)
    else:
        ax_attic_left.set_title(_display_name("attic") + " (nicht gefunden)", pad=4)
        ax_attic_left.grid(True, linestyle="--", alpha=0.2)

    n_ticks_from_attic = len(ax_attic_left.get_yticks()) if ax_attic_left.get_yticks() is not None else 4

    ax_qsum_top = fig.add_subplot(gs[next_row_index, 1])
    qsum_top = sum_first_values_over_zones(df, "Q_RadSol", zones_available, time_idx, remove_first_point)
    if qsum_top is not None:
        ax_qsum_top.plot(time_dt, qsum_top.values, color="black", linewidth=1.2)
        _apply_numeric_ticks_n(ax_qsum_top, qsum_top.values, n_ticks=n_ticks_from_attic, fmt='%.0f')
    else:
        ax_qsum_top.text(0.5, 0.5, "Q_RadSol_* nicht gefunden", ha="center", va="center", transform=ax_qsum_top.transAxes)
    ax_qsum_top.grid(True, linestyle="--", alpha=0.2)
    ax_qsum_top.margins(x=0)
    if len(time_dt) > 1:
        ax_qsum_top.set_xlim(time_dt[0], time_dt[-1])
    ax_qsum_top.set_title("abs. Solarstrahlung", pad=4)
    _unit_label(ax_qsum_top, "[W]")

    # --- Zeile 2: TBufSet links, Status rechts ---
    next_row_index += 1
    ax_tbuf_left = fig.add_subplot(gs[next_row_index, 0])
    s_tbuf_full = get_mpc_series(df, "TBufSet")
    if s_tbuf_full is not None:
        n_preds_b = len(s_tbuf_full.index.get_level_values(0).unique())
        for (xb, yb), col in zip(prediction_paths(s_tbuf_full), orange_gradient(max(1, n_preds_b))):
            ax_tbuf_left.plot(sec_to_datetime(xb, base_year), to_celsius(yb), color=col, linewidth=0.7)
        s_tbuf_act = first_values_per_step(s_tbuf_full)
        if remove_first_point and len(s_tbuf_act) > 0:
            s_tbuf_act = s_tbuf_act.iloc[1:]
        ybuf_act = to_celsius(s_tbuf_act.values)
        ax_tbuf_left.plot(time_dt, ybuf_act, color="black", linewidth=1.2)
        ax_tbuf_left.set_title(r"Speichertemperatur ", pad=4)
        ax_tbuf_left.grid(True, linestyle="--", alpha=0.2)
        ax_tbuf_left.margins(x=0)
        if len(time_dt) > 1:
            ax_tbuf_left.set_xlim(time_dt[0], time_dt[-1])
        pred_paths = prediction_paths(s_tbuf_full)
        pred_buf_c = (np.concatenate([to_celsius(y) for _, y in pred_paths]) if len(pred_paths) else np.array([]))
        _apply_temp_ticks_4_extend(ax_tbuf_left, ybuf_act, pred_buf_c)
    else:
        ax_tbuf_left.set_title(r"TBufSet", pad=4)
        ax_tbuf_left.text(0.5, 0.5, "TBufSet nicht gefunden", ha="center", va="center", transform=ax_tbuf_left.transAxes)
        ax_tbuf_left.grid(True, linestyle="--", alpha=0.2)

    _unit_label(ax_tbuf_left, r"[$^\circ$C]")

    ax_status_right = fig.add_subplot(gs[next_row_index, 1], sharex=ax_qsum_top)
    _plot_status(ax_status_right, status_df, base_year, time_dt, show_xlabel=True)

    # X-Achse & Layout (Gesamtfigur)
    all_axes = [ax for ax in fig.axes if ax.get_visible()]
    for ax in all_axes:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    # Nur die untersten Achsen automatisch labeln; Status unten hat bereits Label
    min_y0 = min((ax.get_position().y0 for ax in all_axes), default=0.0)
    for ax in all_axes:
        if abs(ax.get_position().y0 - min_y0) < 1e-3:
            ax.set_xlabel("Datum")

    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.06, right=0.985, top=0.965, bottom=0.16, hspace=0.50)

    for ax_t, ax_y in pair_axes:
        reduce_vertical_gap(ax_t, ax_y, gap=0.02)

    # --- zweizeilige Footer-Legende Hauptfigur ---
    row1 = legend_handles[:5]
    row2 = legend_handles[5:]

    leg1 = fig.legend(row1, [h.get_label() for h in row1],
                      loc="lower center", ncol=len(row1), frameon=False,
                      bbox_to_anchor=(0.5, 0.065))
    _center_legend(leg1)
    fig.add_artist(leg1)

    leg2 = fig.legend(row2, [h.get_label() for h in row2],
                      loc="lower center", ncol=len(row2), frameon=False,
                      bbox_to_anchor=(0.5, 0.035))
    _center_legend(leg2)
    fig.add_artist(leg2)

    out_file = out_dir / out_name
    fig.savefig(out_file, bbox_inches=None)
    if show_plot:
        try:
            fig.canvas.manager.set_window_title("MPC: T/Set/Band + Ventil (0..1) + TBufSet – Textblock 155×222 mm")
        except Exception:
            pass
        plt.show()
    else:
        plt.close(fig)

    print(f"Fertig. Gespeichert: {out_file}")
    if save_individual:
        print(f"Einzelplots unter: {zones_dir}")
    return out_file


if __name__ == "__main__":
    plot_three_panel_mpc_with_yval_tbuf(
        mpc_csv_path=DEFAULT_MPC_CSV,
        out_dir=DEFAULT_OUT_DIR,
        base_year=2015,
        skip_first_days=3,
        remove_first_point=True,
        t_end_days=None,
        set_band=2.0,
        show_plot=True,
        out_name="ThreePanel_MPC_T_Ventil_TBufSet_Textblock_155x222mm.pdf",
        save_individual=True,
        individual_dir=None,          # z.B. Path(r"D:\fwu-ssr\res\plots\...\zones")
        individual_format="pdf",      # oder "png"
        individual_dpi=300,
        individual_a6_landscape=True  # DIN A6 Querformat
    )
