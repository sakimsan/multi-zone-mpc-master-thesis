from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple
from datetime import datetime
import math
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox

# -------------------- LaTeX/Styling --------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",  # wird beim Speichern unten überschrieben

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

# -------------------- Abmessungen des LaTeX-Textfelds --------------------
MM_PER_IN = 25.4
TEXTBLOCK_IN: Tuple[float, float] = (155.0 / MM_PER_IN, 222.0 / MM_PER_IN)  # (Breite, Höhe) in inch

# -------------------- Defaults/Pfade --------------------
DEFAULT_SIM_CSV = Path(
    r"D:\fwu-ssr\res\plots\used studies\Studies_not_coupled\Nachtabsenkung\Design_0_sim_agent.csv"
)
DEFAULT_OUT_DIR = Path(r"D:\fwu-ssr\res\plots\test_plotNsafeMPC")

# -------------------- Farben & Stile --------------------
TSET_COLOR  = "deepskyblue"                          # Soll (gestrichelt)
TAIR_STYLE  = dict(color="black", linestyle="-",  linewidth=1.15)   # Raumtemp
TSET_STYLE  = dict(color=TSET_COLOR, linestyle="--", linewidth=1.00)  # Soll
BAND_STYLE  = dict(color="red",   linestyle="-",  linewidth=1.15)   # NUR obere Grenze (rot)
VALVE_COLOR = "navy"                                 # Ventilstellung

SET_BAND_K = 2.0  # +K (nur obere Grenze)

# -------------------- Status-Farben --------------------
STATUS_COLOR_MAXCPU = "deeppink"       # Maximum_CpuTime_Exceeded
STATUS_COLOR_OK     = "darkgoldenrod"  # Solved_To_Acceptable_Level
STATUS_COLOR_SOLVED = "green"          # Solve_Succeeded

# -------------------- Einheiten-Labels (konstanter Abstand) --------------------
UNIT_LABEL_X = -0.13  # konstanter linker Abstand relativ zur Achse

def _unit_label(ax: plt.Axes, text: str, x_frac: float = UNIT_LABEL_X):
    """Einheiten-Label links neben der Achse, mit konstantem Abstand zur Plotfläche."""
    ax.set_ylabel("")
    ax.yaxis.labelpad = 0
    ax.text(x_frac, 0.5, text, transform=ax.transAxes,
            rotation=90, va="center", ha="right",
            fontsize=mpl.rcParams.get("axes.labelsize", 9),
            clip_on=False)

# -------------------- Zonen-Mapping & Reihenfolge --------------------
ZONE_NAME_MAP: Dict[str, str] = {
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

def _canonical(zone: str) -> str:
    return zone.strip().lower()

def _display_name(zone: str) -> str:
    z = _canonical(zone)
    return ZONE_NAME_MAP.get(z, zone)

# -------------------- Loader & Helpers --------------------
def _load_sim_keep_causality_and_signal(file: Path) -> pd.DataFrame:
    """SIM-CSV laden. Spaltenheader: (causality, signal, dtype) -> dtype entfernen."""
    df = pd.read_csv(file, header=[0, 1, 2], index_col=0)
    df = df.droplevel(level=2, axis=1)  # dtype weg
    df.index = pd.to_numeric(df.index, errors="coerce")
    return df.sort_index()

def _find_columns(df: pd.DataFrame, causality_names: Iterable[str], pattern: str) -> dict[str, tuple[str, str]]:
    """Spalten unter bestimmten causalities finden; pattern mit (?P<zone>...)."""
    caus_set = {c.lower() for c in causality_names}
    out: dict[str, tuple[str, str]] = {}
    rx = re.compile(pattern)
    for (c0, c1) in df.columns:
        if c0.lower() not in caus_set:
            continue
        m = rx.fullmatch(c1)
        if m:
            out[m.group("zone")] = (c0, c1)
    return out

def _find_any_column(df: pd.DataFrame, pattern: str) -> Optional[tuple[str, str]]:
    """Erste Spalte (über alle causalities) deren Signalname pattern erfüllt."""
    rx = re.compile(pattern)
    for (c0, c1) in df.columns:
        if rx.fullmatch(c1):
            return (c0, c1)
    return None

def _find_any_columns_by_pattern(df: pd.DataFrame, pattern: str) -> dict[str, tuple[str, str]]:
    """Über alle causalities suchen; pattern mit (?P<zone>...)."""
    rx = re.compile(pattern)
    out: dict[str, tuple[str, str]] = {}
    for (c0, c1) in df.columns:
        m = rx.fullmatch(c1)
        if m:
            out[m.group("zone")] = (c0, c1)
    return out

def _mask_skip_first_days(secs: np.ndarray, skip_days: int) -> np.ndarray:
    """Maske, die die ersten 'skip_days' vollen Tage ausschließt."""
    start_sec = np.nanmin(secs)
    cutoff = start_sec + skip_days * 86400.0
    return secs >= cutoff

def _to_datetime(secs: np.ndarray, base_year: int) -> pd.DatetimeIndex:
    """Sekunden seit Jahresbeginn -> echte Datumsachse."""
    origin_dt = datetime(base_year, 1, 1)
    return pd.to_datetime(secs, unit="s", origin=origin_dt)

def _maybe_drop_first_point(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[1:] if len(df) > 0 else df

# ---- 4 Ticks (.0/.5) Basis ----
def _n_half_ticks_from_values(values: np.ndarray, n_ticks: int) -> tuple[list[float], tuple[float, float]]:
    """n_ticks Ticks (nur Werte auf .0/.5) + (ymin,ymax), die Datenbereich einschließen."""
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

    def round_down_half(x: float) -> float:
        return np.floor(x * 2.0) / 2.0

    def round_up_half(x: float) -> float:
        return np.ceil(x * 2.0) / 2.0

    if vmin == vmax:
        center = round_down_half(vmin)
        step = 0.5
        lo = center - step * ((n_ticks - 1) / 2.0)
        lo = round_down_half(lo)
        hi = lo + step * (n_ticks - 1)
        ticks = [lo + i * step for i in range(n_ticks)]
        return ticks, (lo, hi)

    raw_step = (vmax - vmin) / (n_ticks - 1)
    step = max(0.5, np.ceil(raw_step / 0.5) * 0.5)

    lo = round_down_half(vmin)
    hi = lo + step * (n_ticks - 1)
    if hi < vmax - 1e-12:
        hi = round_up_half(vmax)
        lo = hi - step * (n_ticks - 1)
        lo = round_down_half(lo)

    ticks = [lo + i * step for i in range(n_ticks)]
    return ticks, (lo, hi)

# ---- Temp-Ticks anwenden + .5-Shift bei Randkontakt ----
def _apply_temp_ticks_4(ax: plt.Axes, *value_arrays: np.ndarray,
                        touch_eps: float = 1e-6, shift: float = 0.5):
    """4 Y-Ticks auf .0/.5; wenn Daten die Ränder berühren, Skala um ±0.5 verschieben."""
    if len(value_arrays) == 0:
        values = np.array([0.0])
    else:
        values = np.concatenate([
            np.asarray(v, dtype=float).ravel()
            for v in value_arrays if v is not None
        ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.array([0.0])

    ticks, (ymin, ymax) = _n_half_ticks_from_values(values, n_ticks=4)
    ax.set_yticks(ticks)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    bottom_tick, top_tick = ticks[0], ticks[-1]

    # oben?
    if vmax >= top_tick - touch_eps:
        ticks = [t + shift for t in ticks]
        ax.set_yticks(ticks)
        ax.set_ylim(ticks[0], ticks[-1])
    # unten?
    if vmin <= bottom_tick + touch_eps:
        ticks = [t - shift for t in ticks]
        ax.set_yticks(ticks)
        ax.set_ylim(ticks[0], ticks[-1])

# ---- Ventil-Achse fix 0..1 ----
def _apply_valve_ticks_fixed01(ax):
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 1.0])                 # nur 0 und 1
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.minorticks_off()

# ---- Abstand Temp↔Ventil verringern ----
def reduce_vertical_gap(ax_top, ax_bottom, gap: float = 0.02):
    """Nach fig.subplots_adjust() aufrufen!"""
    p_top = ax_top.get_position()
    p_bot = ax_bottom.get_position()
    new_top_y1 = max(p_bot.y0 + 0.005, p_top.y0 - gap)  # positiv halten
    new_bot = Bbox.from_extents(p_bot.x0, p_bot.y0, p_bot.x1, new_top_y1)
    ax_bottom.set_position(new_bot)

# ---- "Nice" Ticks für beliebige Daten (für Q_RadSol) ----
def _nice_step(data_range: float, n_intervals: int) -> float:
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

# ---- Q_RadSol-Summe über Zonen ----
def _sum_qradsol_over_zones(df: pd.DataFrame, zones_in_order: List[str]) -> Optional[pd.Series]:
    """Summiert Q_RadSol_<zone> über alle vorhandenen Zonen (über alle causalities)."""
    cols = _find_any_columns_by_pattern(df, r"Q_RadSol_(?P<zone>.+)")
    acc: Optional[pd.Series] = None
    any_found = False
    for z in zones_in_order:
        can = _canonical(z)
        # passende Spalte finden (Case-insensitive Vergleich über Canonical)
        match_key = next((k for k in cols.keys() if _canonical(k) == can), None)
        if match_key is None:
            continue
        c0, c1 = cols[match_key]
        s = pd.to_numeric(df[(c0, c1)], errors="coerce")
        if acc is None:
            acc = s.fillna(0.0)
        else:
            acc = acc.add(s, fill_value=0.0)
        any_found = True
    return acc if any_found else None

# ---- Stats Loader (SIM/ MPC tolerant) ----
def _find_stats_file(csv_path: Path) -> Optional[Path]:
    """Sucht bevorzugt stats_*_sim_agent*, ansonsten stats_*_mpc_agent* im selben Ordner."""
    folder = csv_path.parent
    cands = list(folder.glob("stats_*_sim_agent*"))
    if not cands:
        cands = list(folder.glob("stats_*_mpc_agent*"))
    if not cands:
        cands = list(folder.glob("stats_*_agent*"))
    return sorted(cands)[0] if cands else None

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
        ax.text(0.5, 0.5, "stats_*_agent* nicht gefunden",
                ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, linestyle="--", alpha=0.15, axis="x")
        if show_xlabel:
            ax.set_xlabel("Datum")
        ax.set_yticks([]); ax.set_ylim(0, 1)
        return

    t_lines = _to_datetime(np.asarray(status_df["t_sec"].values, dtype=float), base_year)
    for t, s in zip(t_lines, status_df["status"]):
        if s == "Maximum_CpuTime_Exceeded":
            ax.axvline(t, color=STATUS_COLOR_MAXCPU, linewidth=1.3, alpha=0.95)
        elif s == "Solved_To_Acceptable_Level":
            ax.axvline(t, color=STATUS_COLOR_OK, linewidth=1.3, alpha=0.95)
        elif s == "Solve_Succeeded":
            ax.axvline(t, color=STATUS_COLOR_SOLVED, linewidth=1.3, alpha=0.95)

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
def plot_three_panel_with_yvalset(
    sim_csv_path: Path = DEFAULT_SIM_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    base_year: int = 2015,
    skip_first_days: int = 3,
    remove_first_point: bool = True,
    t_end_days: Optional[float] = None,
    set_band: float = SET_BAND_K,
    show_plot: bool = True,
    out_name: str = "simmpc_plot.pdf",
) -> Path:
    """
    Figure exakt auf LaTeX-Textfeldgröße (155 mm × 222 mm).
    Layout:
      - links: 5 Zonen (Temp + darunter Ventilstellung),
      - rechts: nächste 5 Zonen (Temp + darunter Ventilstellung),
      - unten: links Dachraum-Temperatur, rechts abs. Solarstrahlung,
               darunter links Speichertemperatur, rechts Optimierungs-Status.
    """
    if not sim_csv_path.exists():
        raise FileNotFoundError(f"SIM-Datei nicht gefunden: {sim_csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Daten laden & filtern ---
    df_full = _load_sim_keep_causality_and_signal(sim_csv_path)
    secs_full = df_full.index.values.astype(float)
    if np.isnan(secs_full).all():
        raise ValueError("Zeitindex enthält nur NaNs.")

    # Erste 'skip_first_days' entfernen (+ optionales Ende)
    mask = _mask_skip_first_days(secs_full, skip_first_days)
    if t_end_days is not None:
        start_sec_after_skip = np.nanmin(secs_full[mask]) if mask.any() else np.nanmin(secs_full)
        end_sec = start_sec_after_skip + float(t_end_days) * 86400.0
        mask &= (secs_full <= end_sec)

    df = df_full.loc[mask].copy()
    secs = secs_full[mask]

    if remove_first_point and len(df) >= 1:
        df = _maybe_drop_first_point(df)
        secs = secs[1:]

    if df.empty:
        raise ValueError("Gefilterter Datensatz ist leer. Prüfe skip_first_days / t_end_days.")

    time_dt = _to_datetime(secs, base_year=base_year)

    # --- Spalten finden ---
    inputs_keys = {"inputs", "input"}
    local_keys  = {"local"}

    tair_cols = _find_columns(df, local_keys,  r"T_Air_(?P<zone>.+)")
    tset_cols = _find_columns(df, inputs_keys, r"TSetOneZone_(?P<zone>.+)")

    # yValSet über alle Causalities suchen:
    yval_cols_any: dict[str, tuple[str, str]] = {}
    rx_yval = re.compile(r"yValSet_(?P<zone>.+)")
    for (c0, c1) in df.columns:
        m = rx_yval.fullmatch(c1)
        if m:
            yval_cols_any[m.group("zone")] = (c0, c1)

    # TBufSet (ohne Zonenbezug)
    tbuf_col = _find_any_column(df, r"TBufSet")

    # --- Zonenliste (nur mit vorhandener T_Air) in gewünschter Reihenfolge ---
    zones_present = {_canonical(z): z for z in tair_cols.keys()}
    zones_in_order = [zones_present[z] for z in ZONE_ORDER_CANONICAL if z in zones_present]

    # Split in links/rechts (je max. 5)
    left_zones  = zones_in_order[:5]
    right_zones = zones_in_order[5:10]
    has_attic   = any(_canonical(z) == "attic" for z in zones_in_order)
    attic_zone  = next((z for z in zones_in_order if _canonical(z) == "attic"), None)

    # --- Figure/Gridspec auf Textblockgröße ---
    # oben 5*(Temp 3 + Ventil 1) → [3,1]*5 = 10 Zeilen
    # + zwei Zusatzzeilen (je 3 Höhe) für: (DG | Qsum) und (TBuf | Status)
    height_ratios = [3, 1]*5 + [3, 3]
    total_rows = len(height_ratios)

    fig = plt.figure(figsize=TEXTBLOCK_IN, constrained_layout=False)
    gs = GridSpec(nrows=total_rows, ncols=2, figure=fig, height_ratios=height_ratios)

    # Gesamtlegende (zweizeilig, zentriert)
    legend_core = [
        Line2D([0], [0], **TAIR_STYLE, label=r"Raumtemperatur "),
        Line2D([0], [0], **TSET_STYLE, label=r"Solltemperatur "),
        Line2D([0], [0], **BAND_STYLE, label=r"obere Grenze"),
        Line2D([0], [0], color=VALVE_COLOR, linewidth=1.2, label="Ventilstellung"),
    ]
    legend_status = [
        Line2D([0], [0], color=STATUS_COLOR_MAXCPU, linewidth=1.3, label="max. CPU-Zeitlimit"),
        Line2D([0], [0], color=STATUS_COLOR_OK,     linewidth=1.3, label="gute Näherung"),
        Line2D([0], [0], color=STATUS_COLOR_SOLVED, linewidth=1.3, label="optimal gelöst"),
    ]

    def _plot_zone(ax_t: plt.Axes, ax_y: Optional[plt.Axes], zone: str):
        zone_de = _display_name(zone)

        # --- T_Air ---
        c0_tair, c1_tair = tair_cols[zone]
        s_tair = pd.to_numeric(df[(c0_tair, c1_tair)], errors="coerce")
        if s_tair.dropna().empty:
            ax_t.set_visible(False)
            if ax_y is not None:
                ax_y.set_visible(False)
            return

        tair_c = s_tair.values - 273.15
        is_attic = (_canonical(zone) == "attic")

        # --- TSetOneZone (falls vorhanden & nicht attic) ---
        tset_c = None
        if not is_attic:
            can = _canonical(zone)
            match_key = next((k for k in tset_cols.keys() if _canonical(k) == can), None)
            if match_key is not None:
                c0_tset, c1_tset = tset_cols[match_key]
                s_tset = pd.to_numeric(df[(c0_tset, c1_tset)], errors="coerce")
                if not s_tset.dropna().empty:
                    tset_c = s_tset.values - 273.15

        # --- Temperatur-Plot (Actual) ---
        if tset_c is not None:
            ax_t.plot(time_dt, tset_c + set_band, **BAND_STYLE)   # nur obere Grenze
            ax_t.plot(time_dt, tset_c, **TSET_STYLE)
        ax_t.plot(time_dt, tair_c, **TAIR_STYLE)

        ax_t.set_title(zone_de, loc="center", pad=4)
        ax_t.grid(True, linestyle="--", alpha=0.3)
        ax_t.margins(x=0)
        if len(time_dt) > 1:
            ax_t.set_xlim(time_dt[0], time_dt[-1])

        # Temp-Ticks inkl. Band
        if tset_c is not None:
            _apply_temp_ticks_4(ax_t, tair_c, tset_c, tset_c + set_band)
        else:
            _apply_temp_ticks_4(ax_t, tair_c)

        # Einheiten-Label Temperatur
        _unit_label(ax_t, r"[$^\circ$C]")

        # --- Ventilstellung (navy) ---
        if ax_y is not None:
            if is_attic:
                ax_y.set_visible(False)
            else:
                match_key = next((k for k in yval_cols_any.keys() if _canonical(k) == _canonical(zone)), None)
                if match_key is not None:
                    c0_y, c1_y = yval_cols_any[match_key]
                    s_y = pd.to_numeric(df[(c0_y, c1_y)], errors="coerce")
                    if not s_y.dropna().empty:
                        ax_y.plot(time_dt, s_y.values, color=VALVE_COLOR, linewidth=1.2)

                _apply_valve_ticks_fixed01(ax_y)
                ax_y.grid(True, linestyle="--", alpha=0.2)
                ax_y.margins(x=0)
                if len(time_dt) > 1:
                    ax_y.set_xlim(time_dt[0], time_dt[-1])

                # Einheiten-Label Ventil
                _unit_label(ax_y, "[-]")

    # --- linke & rechte Spalte (5 Zonenpaare) ---
    pair_axes: List[Tuple[plt.Axes, plt.Axes]] = []

    # linke Spalte
    row = 0
    for zone in left_zones:
        ax_t = fig.add_subplot(gs[row, 0])
        ax_y = fig.add_subplot(gs[row+1, 0], sharex=ax_t)
        _plot_zone(ax_t, ax_y, zone)
        pair_axes.append((ax_t, ax_y))
        row += 2

    # rechte Spalte
    row = 0
    for zone in right_zones:
        ax_t = fig.add_subplot(gs[row, 1])
        ax_y = fig.add_subplot(gs[row+1, 1], sharex=ax_t)
        _plot_zone(ax_t, ax_y, zone)
        pair_axes.append((ax_t, ax_y))
        row += 2

    # ---------- Zusätzliche Zeilen: (links DG | rechts Qsum) und (links TBuf | rechts Status) ----------
    next_row_index = 10

    # --- Zeile 1: Dachraum links ---
    ax_attic_left = fig.add_subplot(gs[next_row_index, 0])
    if has_attic and attic_zone is not None:
        _plot_zone(ax_attic_left, None, attic_zone)
    else:
        ax_attic_left.set_title(_display_name("attic") + " (nicht gefunden)", pad=4)
        ax_attic_left.grid(True, linestyle="--", alpha=0.2)

    n_ticks_from_attic = len(ax_attic_left.get_yticks()) if ax_attic_left.get_yticks() is not None else 4

    # --- Zeile 1: Q_RadSol-Summe rechts ---
    ax_qsum_right = fig.add_subplot(gs[next_row_index, 1])
    qsum = _sum_qradsol_over_zones(df, zones_in_order)
    if qsum is not None and not qsum.dropna().empty:
        ax_qsum_right.plot(time_dt, qsum.values, color="black", linewidth=1.2)
        _apply_numeric_ticks_n(ax_qsum_right, qsum.values, n_ticks=n_ticks_from_attic, fmt='%.0f')
    else:
        ax_qsum_right.text(0.5, 0.5, "Q_RadSol_* nicht gefunden", ha="center", va="center",
                           transform=ax_qsum_right.transAxes)
    ax_qsum_right.grid(True, linestyle="--", alpha=0.2)
    ax_qsum_right.margins(x=0)
    if len(time_dt) > 1:
        ax_qsum_right.set_xlim(time_dt[0], time_dt[-1])
    ax_qsum_right.set_title("abs. Solarstrahlung", pad=4)
    _unit_label(ax_qsum_right, "[W]")

    # --- Zeile 2: TBufSet links ---
    next_row_index += 1
    ax_buf_left = fig.add_subplot(gs[next_row_index, 0])
    if tbuf_col is not None:
        c0_b, c1_b = tbuf_col
        s_buf = pd.to_numeric(df[(c0_b, c1_b)], errors="coerce")
        if not s_buf.dropna().empty:
            ybuf_act = s_buf.values - 273.15
            ax_buf_left.plot(time_dt, ybuf_act, color="black", linewidth=1.2)
            ax_buf_left.set_title(r"Speichertemperatur ", pad=4)
            ax_buf_left.grid(True, linestyle="--", alpha=0.2)
            ax_buf_left.margins(x=0)
            if len(time_dt) > 1:
                ax_buf_left.set_xlim(time_dt[0], time_dt[-1])
            _apply_temp_ticks_4(ax_buf_left, ybuf_act)
        else:
            ax_buf_left.set_title(r"TBufSet", pad=4)
            ax_buf_left.text(0.5, 0.5, "TBufSet leer", ha="center", va="center", transform=ax_buf_left.transAxes)
            ax_buf_left.grid(True, linestyle="--", alpha=0.2)
    else:
        ax_buf_left.set_title(r"TBufSet", pad=4)
        ax_buf_left.text(0.5, 0.5, "TBufSet nicht gefunden", ha="center", va="center", transform=ax_buf_left.transAxes)
        ax_buf_left.grid(True, linestyle="--", alpha=0.2)

    _unit_label(ax_buf_left, r"[$^\circ$C]")

    # --- Zeile 2: Optimierungs-Status rechts ---
    ax_status_right = fig.add_subplot(gs[next_row_index, 1], sharex=ax_qsum_right)
    stats_path = _find_stats_file(sim_csv_path)
    status_df = _load_status_series(stats_path) if stats_path else None
    _plot_status(ax_status_right, status_df, base_year, time_dt, show_xlabel=True)
    # (keine Y-Einheit nötig)

    # --- X-Achsenformat: Datum täglich, Label unten ---
    all_axes = [ax for ax in fig.axes if ax.get_visible()]
    for ax in all_axes:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    positions = [(ax, ax.get_position().y0) for ax in all_axes]
    if positions:
        min_y0 = min(y0 for _, y0 in positions)
        for ax, y0 in positions:
            if abs(y0 - min_y0) < 1e-3:
                ax.set_xlabel("Datum")

    fig.autofmt_xdate()

    # ---------- Globales Layout ----------
    fig.subplots_adjust(left=0.06, right=0.985, top=0.965, bottom=0.16, hspace=0.50)

    # ---------- Lokale Abstände Temp↔Ventil anpassen ----------
    for ax_t, ax_y in pair_axes:
        reduce_vertical_gap(ax_t, ax_y, gap=0.02)

    # ---------- zweizeilige Footer-Legende (zentriert) ----------
    leg1 = fig.legend(legend_core, [h.get_label() for h in legend_core],
                      loc="lower center", ncol=len(legend_core), frameon=False,
                      bbox_to_anchor=(0.5, 0.065))
    # zweite Zeile (Status)
    leg2 = fig.legend(legend_status, [h.get_label() for h in legend_status],
                      loc="lower center", ncol=len(legend_status), frameon=False,
                      bbox_to_anchor=(0.5, 0.035))
    fig.add_artist(leg1)
    fig.add_artist(leg2)

    out_file = out_dir / out_name
    # exakte Größe beibehalten (override von rcParams['savefig.bbox'] = 'tight')
    fig.savefig(out_file, bbox_inches=None)

    if show_plot:
        try:
            fig.canvas.manager.set_window_title("SIM: T/Set + obere Grenze + Ventil (0..1) | DG/Qsum | TBuf/Status – Textblock 155×222 mm")
        except Exception:
            pass
        plt.show()
    else:
        plt.close(fig)

    print(f"Fertig. Gespeichert: {out_file}")
    return out_file


if __name__ == "__main__":
    plot_three_panel_with_yvalset(
        sim_csv_path=DEFAULT_SIM_CSV,
        out_dir=DEFAULT_OUT_DIR,
        base_year=2015,
        skip_first_days=3,
        remove_first_point=True,
        t_end_days=None,
        set_band=2.0,
        show_plot=True,
        out_name="ThreePanel_T_Ventilstellung_Qsum_Status_Textblock_155x222mm.pdf",
    )
