from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime
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

    "axes.xmargin": 0.0,   # keine weißen Ränder links/rechts
})

# -------------------- Defaults/Pfade --------------------
DEFAULT_SIM_CSV = Path(
    r"D:\fwu-ssr\res\SPAWN_Studies\mpc_design_Nachtabsenkung_TARP_Max\DesignOptimizationResults\TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai_\Design_0_sim_agent.csv"
)
DEFAULT_OUT_DIR = Path(r"D:\fwu-ssr\res\plots\test_plotNsafeMPC")

# -------------------- Farben & Stile --------------------
TSET_COLOR  = "deepskyblue"                         # Soll (gestrichelt)
TAIR_STYLE  = dict(color="black", linestyle="-",  linewidth=1.15)   # Raumtemp (Actual)
TSET_STYLE  = dict(color=TSET_COLOR, linestyle="--", linewidth=1.00) # Soll (Actual)
BAND_STYLE  = dict(color="red",    linestyle="-",  linewidth=1.15)   # nur obere Grenze (rot)
VALVE_COLOR = "navy"                                                # Ventilstellung (Actual)

SET_BAND_K = 2.0  # +K (nur obere Komfortgrenze)

# -------------------- Zonen-Mapping & Reihenfolge --------------------
ZONE_MAP_DE: Dict[str, str] = {
    "livingroom": "Wohnen",
    "hobby": "Hobby",
    "corridor": "Flur",
    "wcstorage": "WC",
    "kitchen": "Küche",
    "bedroom": "Schlafen",
    "children": "Kind",
    "corridor2": "Flur 2",
    "corrdior2": "Flur 2",
    "bath": "Bad",
    "children2": "Kind 2",
    "attic": "Dachgeschoss",
}
ZONE_ORDER_CANONICAL = [
    "livingroom", "hobby", "corridor", "wcstorage",
    "kitchen", "bedroom", "children", "corridor2",
    "bath", "children2", "attic"
]

# Inverse Map: DE-Anzeige -> EN-Key (für Spaltennamen mit deutschem Suffix)
ZONE_DE_TO_EN: Dict[str, str] = {
    str(v).strip().lower(): k for k, v in ZONE_MAP_DE.items()
}

def _canonical(s: str) -> str:
    return s.strip().lower()

def _display_name(zone: str) -> str:
    return ZONE_MAP_DE.get(_canonical(zone), zone)

# -------------------- Laden (3-Level Header) --------------------
def _load_sim_keep_causality_and_signal(file: Path) -> pd.DataFrame:
    """
    SIM-CSV laden. Spaltenheader: (causality, signal, dtype) -> 'dtype' entfernen.
    Index: Sekunden seit Jahresbeginn.
    """
    df = pd.read_csv(file, header=[0, 1, 2], index_col=0)
    df = df.droplevel(level=2, axis=1)  # dtype weg
    df.index = pd.to_numeric(df.index, errors="coerce")
    return df.sort_index()

# -------- Robuste Spalten-Suche in SIM-Datei --------
def build_sim_column_maps(df: pd.DataFrame):
    """
    Liefert:
      - tair_cols: {zone_en: (c0, signal)}
      - tset_cols: {zone_en: (c0, signal)}   (Attic meist nicht vorhanden)
      - yval_cols: {zone_en: (c0, signal)}
      - tbuf_col:  (c0, signal) oder None

    Erkennung tolerant gegen Präfixe (inputs./outputs./local.) und case.
    Erkennt u. a.:
      * building.groundFloor.TZoneMea[i], building.upperFloor.TZoneMea[i], building.attic.TZoneMea
      * user.TZoneSet[i], TSetOneZone[i], TSetOneZone_<zone>  (+ generisch: ...T...Set...[i] / ...T...Set..._<zone>)
      * hydraulic.traCtrl.opening[i]  / yValSet_<zone>
      * TBufSet
    """
    tair_cols: Dict[str, Tuple[str, str]] = {}
    tset_cols: Dict[str, Tuple[str, str]] = {}
    yval_cols: Dict[str, Tuple[str, str]] = {}
    tbuf_col: Optional[Tuple[str, str]] = None

    # Indextabellen (1-basiert)
    idx_to_zone_1_5  = dict(zip(range(1, 6),  ZONE_ORDER_CANONICAL[:5]))    # EG 1..5
    idx_to_zone_6_10 = dict(zip(range(1, 6),  ZONE_ORDER_CANONICAL[5:10]))  # OG 1..5 -> Zonen 6..10
    idx_to_zone_1_11 = dict(zip(range(1, 12), ZONE_ORDER_CANONICAL))        # Ventile 1..11

    # Priorität: inputs > outputs; Bonus wenn "inputs." im Signal
    def _priority(causality: str, signal: str) -> int:
        c = str(causality).strip().lower()
        s = str(signal)
        p = 0
        if c in ("input", "inputs"): p += 2
        if "inputs." in s:           p += 1
        if c in ("output", "outputs"): p += 1
        return p

    def _set_best(d: Dict[str, Tuple[str, str]], key: str, cand: Tuple[str, str]):
        if key not in d:
            d[key] = cand
        else:
            c0_old, sig_old = d[key]
            if _priority(cand[0], cand[1]) > _priority(c0_old, sig_old):
                d[key] = cand

    # Regex-Helfer (case-insensitiv, beliebig viele Präfixebenen)
    def rx(pattern: str) -> re.Pattern:
        return re.compile(pattern, re.IGNORECASE)

    # T_Air
    rx_tair_gf  = rx(r"(?:\w+\.)*building\.groundfloor\.tzonemea\[(\d+)\]")
    rx_tair_uf  = rx(r"(?:\w+\.)*building\.upperfloor\.tzonemea\[(\d+)\]")
    rx_tair_att = rx(r"(?:\w+\.)*building\.attic\.tzonemea$")
    rx_tair_alt = rx(r"(?:\w+\.)*t_air_(.+)")

    # TSet – konkrete Varianten
    rx_tset_user_idx = rx(r"(?:\w+\.)*user\.tzoneset\[(\d+)\]")
    rx_tset_one_idx  = rx(r"(?:\w+\.)*tsetonezone\[(\d+)\]")
    rx_tset_one_zone = rx(r"(?:\w+\.)*tsetonezone_(.+)")

    # TSet – sehr generische Fallbacks
    rx_tset_generic_idx  = rx(r"(?:\w+\.)*t[^.\[]*set[^.\[]*\[(\d+)\]")   # ...T...Set...[i]
    rx_tset_generic_zone = rx(r"(?:\w+\.)*t[^.\[]*set[^.\[]*_(.+)")       # ...T...Set..._<zone>

    # Valve
    rx_valve_idx = rx(r"(?:\w+\.)*hydraulic\.tractrl\.opening\[(\d+)\]")
    rx_valve_alt = rx(r"(?:\w+\.)*yvalset_(.+)")

    # TBufSet
    rx_tbuf = rx(r"(?:\w+\.)*tbufset$")

    # komplette Schleife über (causality, signal)
    for (c0, sig_raw) in df.columns:
        sig = str(sig_raw)

        # ---------- T_Air ----------
        m = rx_tair_gf.fullmatch(sig)
        if m:
            i = int(m.group(1))
            z = idx_to_zone_1_5.get(i)
            if z: _set_best(tair_cols, z, (c0, sig))
            continue

        m = rx_tair_uf.fullmatch(sig)
        if m:
            i = int(m.group(1))
            z = idx_to_zone_6_10.get(i)
            if z: _set_best(tair_cols, z, (c0, sig))
            continue

        if rx_tair_att.fullmatch(sig):
            _set_best(tair_cols, "attic", (c0, sig))
            # kein continue, evtl. taucht auch T_Air_attic als Altform auf – dann gewinnt Priorität

        m = rx_tair_alt.fullmatch(sig)
        if m:
            name = _canonical(m.group(1))
            # english direkt?
            for can in ZONE_ORDER_CANONICAL:
                if _canonical(can) == name:
                    _set_best(tair_cols, can, (c0, sig))
                    break
            else:
                # deutsches Suffix?
                if name in ZONE_DE_TO_EN:
                    _set_best(tair_cols, ZONE_DE_TO_EN[name], (c0, sig))

        # ---------- TSet (konkrete Varianten) ----------
        done_tset = False

        for patt in (rx_tset_user_idx, rx_tset_one_idx):
            m = patt.fullmatch(sig)
            if m:
                i = int(m.group(1))
                if 1 <= i <= 10:
                    z = ZONE_ORDER_CANONICAL[i-1]
                    if _canonical(z) != "attic":
                        _set_best(tset_cols, z, (c0, sig))
                        done_tset = True
                break

        if not done_tset:
            m = rx_tset_one_zone.fullmatch(sig)
            if m:
                name = _canonical(m.group(1))
                # englischer Zonenname?
                for can in ZONE_ORDER_CANONICAL:
                    if _canonical(can) == name and _canonical(can) != "attic":
                        _set_best(tset_cols, can, (c0, sig))
                        done_tset = True
                        break
                # deutscher Zonenname?
                if not done_tset and name in ZONE_DE_TO_EN:
                    can = ZONE_DE_TO_EN[name]
                    if _canonical(can) != "attic":
                        _set_best(tset_cols, can, (c0, sig))
                        done_tset = True

        # ---------- TSet (generische Fallbacks) ----------
        if not done_tset:
            m = rx_tset_generic_idx.fullmatch(sig)
            if m:
                i = int(m.group(1))
                if 1 <= i <= 10:
                    z = ZONE_ORDER_CANONICAL[i-1]
                    if _canonical(z) != "attic":
                        _set_best(tset_cols, z, (c0, sig))
                        done_tset = True

        if not done_tset:
            m = rx_tset_generic_zone.fullmatch(sig)
            if m:
                name = _canonical(m.group(1))
                for can in ZONE_ORDER_CANONICAL:
                    if _canonical(can) == name and _canonical(can) != "attic":
                        _set_best(tset_cols, can, (c0, sig))
                        done_tset = True
                        break
                if not done_tset and name in ZONE_DE_TO_EN:
                    can = ZONE_DE_TO_EN[name]
                    if _canonical(can) != "attic":
                        _set_best(tset_cols, can, (c0, sig))
                        done_tset = True

        # ---------- Valve ----------
        m = rx_valve_idx.fullmatch(sig)
        if m:
            i = int(m.group(1))
            z = idx_to_zone_1_11.get(i)
            if z:
                _set_best(yval_cols, z, (c0, sig))
            continue

        m = rx_valve_alt.fullmatch(sig)
        if m:
            name = _canonical(m.group(1))
            for can in ZONE_ORDER_CANONICAL:
                if _canonical(can) == name:
                    _set_best(yval_cols, can, (c0, sig))
                    break

        # ---------- TBufSet ----------
        if tbuf_col is None and rx_tbuf.fullmatch(sig):
            tbuf_col = (c0, sig)

    return tair_cols, tset_cols, yval_cols, tbuf_col

# -------------------- Zeit/Hilfen --------------------
def _mask_skip_first_days(secs: np.ndarray, skip_days: int) -> np.ndarray:
    start_sec = np.nanmin(secs)
    cutoff = start_sec + skip_days * 86400.0
    return secs >= cutoff

def _to_datetime(secs: np.ndarray, base_year: int) -> pd.DatetimeIndex:
    origin_dt = datetime(base_year, 1, 1)
    return pd.to_datetime(secs, unit="s", origin=origin_dt)

def _maybe_drop_first_point(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[1:] if len(df) > 0 else df

def to_celsius(arr: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(arr, dtype=float) - 273.15

# ---- 0.5-Schritt-Ticks (n Ticks, .0/.5) ----
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

    def rd(x: float) -> float:  # round down to .0/.5
        return np.floor(x * 2.0) / 2.0
    def ru(x: float) -> float:  # round up to .0/.5
        return np.ceil(x * 2.0) / 2.0

    if vmin == vmax:
        center = rd(vmin)
        step = 1.0 if n_ticks == 4 else 0.5
        lo = rd(center - step * ((n_ticks - 1) / 2.0))
        hi = lo + step * (n_ticks - 1)
        ticks = [lo + i * step for i in range(n_ticks)]
        return ticks, (lo, hi)

    raw_step = (vmax - vmin) / (n_ticks - 1)
    step = max(0.5, np.ceil(raw_step / 0.5) * 0.5)

    lo = rd(vmin)
    hi = lo + step * (n_ticks - 1)
    if hi < vmax - 1e-12:
        hi = ru(vmax)
        lo = rd(hi - step * (n_ticks - 1))

    ticks = [lo + i * step for i in range(n_ticks)]
    return ticks, (lo, hi)

# ---- Lokaler Abstand-Reducer (nur Temp↔Ventil) ----
def reduce_vertical_gap(ax_top, ax_bottom, gap: float = 0.02):
    """
    Zieht den Ventil-Achsenrahmen (ax_bottom) nach oben, so dass der Spalt
    zum Temperatur-Plot (ax_top) genau 'gap' (Figure-Normalized) beträgt.
    Nach fig.subplots_adjust() aufrufen!
    """
    p_top = ax_top.get_position()
    p_bot = ax_bottom.get_position()
    new_top_y1 = max(p_bot.y0 + 0.005, p_top.y0 - gap)  # Höhe positiv lassen
    new_bot = Bbox.from_extents(p_bot.x0, p_bot.y0, p_bot.x1, new_top_y1)
    ax_bottom.set_position(new_bot)

# -------------------- Plot-Funktion --------------------
def plot_three_panel_sim_without_predictions(
    sim_csv_path: Path = DEFAULT_SIM_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    base_year: int = 2015,
    skip_first_days: int = 3,
    remove_first_point: bool = True,
    t_end_days: Optional[float] = None,
    set_band: float = SET_BAND_K,
    show_plot: bool = True,
    out_name: str = "ThreePanel_SIM_T_Ventil_TBufSet_A4.pdf",
) -> Path:
    if not sim_csv_path.exists():
        raise FileNotFoundError(f"SIM-Datei nicht gefunden: {sim_csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Daten laden & filtern ---
    df_full = _load_sim_keep_causality_and_signal(sim_csv_path)
    secs_full = df_full.index.values.astype(float)
    if np.isnan(secs_full).all():
        raise ValueError("Zeitindex enthält nur NaNs.")

    mask = _mask_skip_first_days(secs_full, skip_first_days)
    if t_end_days is not None:
        start_after = np.nanmin(secs_full[mask]) if mask.any() else np.nanmin(secs_full)
        end_sec = start_after + float(t_end_days) * 86400.0
        mask &= (secs_full <= end_sec)

    df = df_full.loc[mask].copy()
    secs = secs_full[mask]

    if remove_first_point and len(df) >= 1:
        df = _maybe_drop_first_point(df)
        secs = secs[1:]

    if df.empty:
        raise ValueError("Gefilterter Datensatz ist leer. Prüfe skip_first_days / t_end_days.")

    time_dt = _to_datetime(secs, base_year=base_year)

    # --- Spalten-Mapping für diese SIM-Datei (robust) ---
    tair_cols, tset_cols, yval_cols, tbuf_col = build_sim_column_maps(df)

    # Zonen (nur vorhandene T_Air), gewünschte Reihenfolge
    zones_present = {_canonical(z): z for z in tair_cols.keys()}
    zones_in_order = [zones_present[z] for z in ZONE_ORDER_CANONICAL if z in zones_present]

    left_zones  = zones_in_order[:5]
    right_zones = zones_in_order[5:10]
    has_attic = any(_canonical(z) == "attic" for z in zones_in_order)
    attic_zone = next((z for z in zones_in_order if _canonical(z) == "attic"), None)

    # --- Figure/Gridspec (A4) ---
    a4_size = (8.27, 11.69)
    # oben 5×(Temp3 + Ventil1) = 10 Zeilen; Dachgeschoss (3); TBufSet gleich groß (3)
    height_ratios = [3, 1]*5 + ([3] if has_attic else []) + [3]
    total_rows = len(height_ratios)

    fig = plt.figure(figsize=a4_size, constrained_layout=False)
    gs = GridSpec(nrows=total_rows, ncols=2, figure=fig, height_ratios=height_ratios)

    # Legende unten (Ventilstellung navy)
    legend_handles = [
        Line2D([0], [0], **TAIR_STYLE, label=r"Raumtemperatur [$^\circ$C]"),
        Line2D([0], [0], **TSET_STYLE, label=r"Solltemperatur [$^\circ$C]"),
        Line2D([0], [0], **BAND_STYLE, label=fr"oberes Komfortband (+{set_band:g} K)"),
        Line2D([0], [0], color=VALVE_COLOR, linewidth=1.2, label="Ventilstellung"),
    ]

    # Y-Ticks-Helfer
    def _apply_temp_ticks_4(ax, actual_vals_c: np.ndarray):
        ticks, (ymin, ymax) = _n_half_ticks_from_values(actual_vals_c, n_ticks=4)
        ax.set_yticks(ticks)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    def _apply_valve_ticks_fixed(ax):
        ax.set_yticks([0.0, 1.0])
        ax.set_ylim(0.0, 1.0)

    # Zeichnen einer Zone
    def _plot_zone(ax_t, ax_y, zone: str):
        zone_de = _display_name(zone)

        # --- T_Air (Actual) ---
        if zone not in tair_cols:
            ax_t.set_visible(False)
            if ax_y is not None: ax_y.set_visible(False)
            return

        c0_tair, c1_tair = tair_cols[zone]
        s_tair = pd.to_numeric(df[(c0_tair, c1_tair)], errors="coerce")
        if s_tair.dropna().empty:
            ax_t.set_visible(False)
            if ax_y is not None: ax_y.set_visible(False)
            return
        tair_c = to_celsius(s_tair.values)

        # --- TSet (Actual, kein attic) ---
        tset_c = None
        if _canonical(zone) != "attic" and zone in tset_cols:
            c0_tset, c1_tset = tset_cols[zone]
            s_tset = pd.to_numeric(df[(c0_tset, c1_tset)], errors="coerce")
            if not s_tset.dropna().empty:
                tset_c = to_celsius(s_tset.values)

        # --- Linien: Tair, Tset, nur oberes Band ---
        if tset_c is not None:
            ax_t.plot(time_dt, tset_c, **TSET_STYLE)                # Soll
            ax_t.plot(time_dt, tset_c + SET_BAND_K, **BAND_STYLE)   # nur obere Grenze
        ax_t.plot(time_dt, tair_c, **TAIR_STYLE)                    # Raumtemp

        ax_t.set_title(zone_de, loc="center", pad=4)
        ax_t.grid(True, linestyle="--", alpha=0.3)
        ax_t.margins(x=0)
        if len(time_dt) > 1:
            ax_t.set_xlim(time_dt[0], time_dt[-1])
        _apply_temp_ticks_4(ax_t, tair_c)

        # --- Ventilstellung (0..1, nur wenn vorhanden & nicht attic) ---
        if ax_y is not None:
            if _canonical(zone) == "attic":
                ax_y.set_visible(False)
            else:
                if zone in yval_cols:
                    c0_y, c1_y = yval_cols[zone]
                    s_y = pd.to_numeric(df[(c0_y, c1_y)], errors="coerce")
                    if not s_y.dropna().empty:
                        yval = s_y.values
                        ax_y.plot(time_dt, yval, color=VALVE_COLOR, linewidth=1.2)
                _apply_valve_ticks_fixed(ax_y)
                ax_y.grid(True, linestyle="--", alpha=0.2)
                ax_y.margins(x=0)
                if len(time_dt) > 1:
                    ax_y.set_xlim(time_dt[0], time_dt[-1])

    # --- linke & rechte Spalte ---
    pair_axes: List[tuple[plt.Axes, plt.Axes]] = []

    row = 0
    for zone in left_zones:
        ax_t = fig.add_subplot(gs[row, 0])
        ax_y = fig.add_subplot(gs[row+1, 0], sharex=ax_t)
        _plot_zone(ax_t, ax_y, zone)
        pair_axes.append((ax_t, ax_y))
        row += 2

    row = 0
    for zone in right_zones:
        ax_t = fig.add_subplot(gs[row, 1])
        ax_y = fig.add_subplot(gs[row+1, 1], sharex=ax_t)
        _plot_zone(ax_t, ax_y, zone)
        pair_axes.append((ax_t, ax_y))
        row += 2

    # --- Dachgeschoss (falls vorhanden) ---
    next_row_index = 10
    if has_attic and attic_zone is not None:
        ax_t_bottom = fig.add_subplot(gs[next_row_index, :])
        _plot_zone(ax_t_bottom, None, attic_zone)
        next_row_index += 1

    # --- TBufSet (gleich groß wie Dachgeschoss) ---
    ax_buf = fig.add_subplot(gs[next_row_index, :])
    if tbuf_col is not None:
        c0_b, c1_b = tbuf_col
        s_buf = pd.to_numeric(df[(c0_b, c1_b)], errors="coerce")
        if not s_buf.dropna().empty:
            ybuf_act = to_celsius(s_buf.values)
            ax_buf.plot(time_dt, ybuf_act, color="black", linewidth=1.2)

            ax_buf.set_title(r"TBufSet [$^\circ$C]", loc="center", pad=4)
            ax_buf.grid(True, linestyle="--", alpha=0.2)
            ax_buf.margins(x=0)
            if len(time_dt) > 1:
                ax_buf.set_xlim(time_dt[0], time_dt[-1])

            ticks, (ymin, ymax) = _n_half_ticks_from_values(ybuf_act, n_ticks=4)
            ax_buf.set_yticks(ticks)
            ax_buf.set_ylim(ymin, ymax)
            ax_buf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        else:
            ax_buf.set_title(r"TBufSet [$^\circ$C]", loc="center", pad=4)
            ax_buf.text(0.5, 0.5, "TBufSet leer", ha="center", va="center", transform=ax_buf.transAxes)
            ax_buf.grid(True, linestyle="--", alpha=0.2)
    else:
        ax_buf.set_title(r"TBufSet [$^\circ$C]", loc="center", pad=4)
        ax_buf.text(0.5, 0.5, "TBufSet nicht gefunden", ha="center", va="center", transform=ax_buf.transAxes)
        ax_buf.grid(True, linestyle="--", alpha=0.2)

    # --- X-Achsenformat: Datum täglich, Label ganz unten ---
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

    # ---------- Globales Layout ----------
    fig.subplots_adjust(left=0.04, right=0.985, top=0.965, bottom=0.12, hspace=0.50)

    # Spalt Temp ↔ Ventil lokal verkleinern
    for ax_t, ax_y in pair_axes:
        reduce_vertical_gap(ax_t, ax_y, gap=0.02)

    # ---------- Legende unten ----------
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.04)
    )

    out_file = out_dir / out_name
    fig.savefig(out_file)
    if show_plot:
        try:
            fig.canvas.manager.set_window_title("SIM: T/Set + oberes Band (rot) + Ventil (navy) + TBufSet – A4")
        except Exception:
            pass
        plt.show()
    else:
        plt.close(fig)

    print(f"Fertig. Gespeichert: {out_file}")
    return out_file


if __name__ == "__main__":
    plot_three_panel_sim_without_predictions(
        sim_csv_path=DEFAULT_SIM_CSV,
        out_dir=DEFAULT_OUT_DIR,
        base_year=2015,
        skip_first_days=3,
        remove_first_point=True,
        t_end_days=None,
        set_band=2.0,
        show_plot=True,
        out_name="ThreePanel_SIM_T_Ventil_TBufSet_A4.pdf",
    )
