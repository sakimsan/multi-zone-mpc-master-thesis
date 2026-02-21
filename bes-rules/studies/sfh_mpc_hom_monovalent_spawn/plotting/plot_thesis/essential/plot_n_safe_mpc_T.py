from __future__ import annotations
from pathlib import Path
from ast import literal_eval
from typing import Optional, Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

# =========================
# Matplotlib / LaTeX Setup
# =========================
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",

    # LaTeX Look
    "text.usetex": True,
    "axes.unicode_minus": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",

    # Größen (A4-Plot gut lesbar)
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    # keine x-Ränder links/rechts
    "axes.xmargin": 0.0,
})

# --- Defaults ---
DEFAULT_MPC_CSV = Path(
    r"D:\fwu-ssr\res\SPAWN_Studies\mpc_design_Nachtabsenkung_TARP_Max\DesignOptimizationResults\TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai_\Design_0_mpc_agent.csv"
)
DEFAULT_OUT_DIR = Path(r"D:\fwu-ssr\res\plots\test_plotNsafeMPC")

# Farben / Stil
COLOR_TAIR = "black"            # Actual Raumtemperatur
COLOR_SET  = "deepskyblue"      # Actual Solltemperatur (gestrichelt)
BAND_COLOR = "purple"           # NUR obere Grenze (Tset)

TAIR_STYLE = dict(color=COLOR_TAIR, linestyle="-",  linewidth=1.6)
TSET_STYLE = dict(color=COLOR_SET,  linestyle="--", linewidth=1.6)
BAND_STYLE = dict(color=BAND_COLOR, linestyle="-",  linewidth=1.0)

# nur obere Grenze (anstatt ±K)
SET_BAND_K = 2.0  # +K

# Zonenreihenfolge & Mapping (en -> de)
ZONE_ORDER_EN = [
    "livingroom", "hobby", "corridor", "wcstorage",
    "kitchen", "bedroom", "children", "corridor2",
    "bath", "children2", "attic",
]
ZONE_MAP_DE: Dict[str, str] = {
    "livingroom": "Wohnen",
    "hobby": "Hobby",
    "corridor": "Flur",
    "wcstorage": "WC",
    "kitchen": "Küche",
    "bedroom": "Schlafen",
    "children": "Kind",
    "corridor2": "Flur 2",
    "corrdior2": "Flur 2",  # falls Tippfehler im CSV
    "bath": "Bad",
    "children2": "Kind 2",
    "attic": "Dachgeschoss",
}

TITLE_PAD = 6  # Abstand Titel

# =========================
# Hilfsfunktionen
# =========================
def load_mpc_results(csv_path: Path) -> pd.DataFrame:
    """MPC-Resultate laden (Spalten-MultiIndex, Index=(time, horizon_idx))."""
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    outer = [literal_eval(i) for i in df.index]
    df.index = pd.MultiIndex.from_tuples(outer)
    return df

def filter_mpc_time(
    df: pd.DataFrame, *, skip_first_days: int = 3, t_end_days: Optional[float] = None
) -> pd.DataFrame:
    """Filtert MPC-Ergebnis nach Zeit (Level 0 = Sekunden seit Jahresanfang)."""
    outer = df.index.get_level_values(0).astype(float)
    if len(outer) == 0:
        return df
    start_sec = np.min(outer)
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
    """Je outer_time der erste inner-horizon-Wert (closed-loop 'actual')."""
    return series.groupby(level=0).first()

def prediction_paths(series: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(outer_time, inner_horizon) → Liste Pfade: x = outer_time + inner_horizon (s)."""
    paths: List[Tuple[np.ndarray, np.ndarray]] = []
    for t0, grp in series.groupby(level=0):
        g = grp.droplevel(0)
        x = g.index.values.astype(float) + float(t0)
        y = g.values.astype(float)
        paths.append((x, y))
    return paths

def ensure_columns(df: pd.DataFrame, level: str, names: List[str]) -> List[str]:
    """Filtert 'names' auf existierende Spalten unter 'level'."""
    if level not in df.columns.get_level_values(0):
        return []
    avail = set(df[level].columns)
    return [n for n in names if n in avail]

def zone_label_de(zone_en: str) -> str:
    return ZONE_MAP_DE.get(zone_en, zone_en)

def to_celsius(arr: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(arr, dtype=float) - 273.15

def sanitize_filename(part: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", part)

def orange_gradient(n: int, vmin: float = 0.3, vmax: float = 0.9):
    """n RGBA-Farben aus 'Oranges' von hell→dunkel."""
    if n <= 0:
        return []
    cmap = cm.get_cmap("Oranges")
    return [cmap(v) for v in np.linspace(vmin, vmax, n)]

# ---- Drei Y-Ticks (.0 oder .5) berechnen ----
def _round_down_to_step(x: float, step: float = 0.5) -> float:
    return np.floor(x / step) * step

def _round_up_to_step(x: float, step: float = 0.5) -> float:
    return np.ceil(x / step) * step

def _round_to_step(x: float, step: float = 0.5) -> float:
    return np.round(x / step) * step

def set_three_y_ticks_from_arrays(ax, arrays: List[Optional[np.ndarray]], step: float = 0.5):
    """
    Setzt exakt drei Y-Ticks (min, mid, max), alle auf .0/.5.
    arrays: Liste von Arrays, aus denen min/max bestimmt werden (z.B. T_air, T_set, T_set+K).
    """
    vals_list = []
    for a in arrays:
        if a is None:
            continue
        arr = np.asarray(a, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals_list.append(arr)
    if not vals_list:
        return  # nichts zu tun

    vals = np.concatenate(vals_list)
    vmin_raw = float(np.min(vals))
    vmax_raw = float(np.max(vals))

    y_min = _round_down_to_step(vmin_raw, step)
    y_max = _round_up_to_step(vmax_raw, step)

    # Mindestens 2*step Spanne, damit 3 Ticks möglich sind
    if (y_max - y_min) < 2 * step:
        center = _round_to_step((y_min + y_max) / 2.0, step)
        y_min = center - step
        y_max = center + step

    mid = _round_to_step((y_min + y_max) / 2.0, step)

    ticks = [y_min, mid, y_max]
    ax.set_yticks(ticks)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.minorticks_off()

# =========================
# Plotter
# =========================
def plot_zone_single_pdf(
    out_dir: Path,
    zone_en: str,
    time_act: np.ndarray,          # datetime Index (actuals)
    tair_act_c: np.ndarray,        # °C
    tset_act_c: Optional[np.ndarray],  # °C
    pred_tair: List[Tuple[np.ndarray, np.ndarray]],   # (sec, K)
    pred_tset: Optional[List[Tuple[np.ndarray, np.ndarray]]],  # (sec, K)
    year: int = 2015,
) -> None:
    """Speichert EINEN Zonenplot als PDF ("T_Plot_<ZoneDE>.pdf") in out_dir."""
    zone_de = zone_label_de(zone_en)

    fig, ax = plt.subplots(figsize=(6.6, 2.1), constrained_layout=False)

    # Predictions (halbe Linienstärke)
    colors_air = orange_gradient(len(pred_tair))
    for (x, y), col in zip(pred_tair, colors_air):
        ax.plot(sec_to_datetime(x, year), to_celsius(y), color=col, linewidth=0.45)
    if (pred_tset is not None) and (zone_en != "attic"):
        colors_set = orange_gradient(len(pred_tset))
        for (x, y), col in zip(pred_tset, colors_set):
            ax.plot(sec_to_datetime(x, year), to_celsius(y), color=col, linewidth=0.45)

    # Actual: obere Grenze + Soll + Raumtemperatur
    if (tset_act_c is not None) and (zone_en != "attic"):
        ax.plot(time_act, tset_act_c + SET_BAND_K, **BAND_STYLE, label=r"obere Grenze")
        ax.plot(time_act, tset_act_c, **TSET_STYLE, label=r"Solltemperatur")
    ax.plot(time_act, tair_act_c, **TAIR_STYLE, label=r"Raumtemperatur")

    # Zonenname als TITEL, MITTIG
    ax.set_title(zone_de, loc="center", pad=TITLE_PAD)

    # Genau drei Y-Ticks (.0/.5)
    arrays_for_ticks = [tair_act_c]
    if (tset_act_c is not None) and (zone_en != "attic"):
        arrays_for_ticks += [tset_act_c, tset_act_c + SET_BAND_K]
    set_three_y_ticks_from_arrays(ax, arrays_for_ticks, step=0.5)

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    ax.margins(x=0)
    if len(time_act) > 1:
        ax.set_xlim(time_act[0], time_act[-1])

    fig.tight_layout(rect=(0.02, 0.10, 0.995, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"T_Plot_{sanitize_filename(zone_de)}.pdf"
    fig.savefig(out_dir / safe_name)
    plt.close(fig)

def plot_sammelplot_pdf(
    out_path: Path,
    zones_blocks: List[
        Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray],
              List[Tuple[np.ndarray, np.ndarray]], Optional[List[Tuple[np.ndarray, np.ndarray]]]]
    ],
    year: int = 2015,
    set_band: float = SET_BAND_K,
    suptitle: Optional[str] = None,
    show: bool = True,
) -> None:
    """Sammelplot A4: alle Zonen untereinander. Zonenname **mittig** als Achsentitel."""
    n = len(zones_blocks)
    fig_h, fig_w = 11.69, 8.27
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), sharex=True, constrained_layout=False)
    if n == 1:
        axes = [axes]

    all_starts, all_ends = [], []

    for ax, (zone_en, time_act, tair_act_c, tset_act_c, pred_tair, pred_tset) in zip(axes, zones_blocks):
        zone_de = zone_label_de(zone_en)

        # Predictions — halbierte Linienstärken
        colors_air = orange_gradient(len(pred_tair))
        for (x, y), col in zip(pred_tair, colors_air):
            ax.plot(sec_to_datetime(x, year), to_celsius(y), color=col, linewidth=0.35)
        if (pred_tset is not None) and (zone_en != "attic"):
            colors_set = orange_gradient(len(pred_tset))
            for (x, y), col in zip(pred_tset, colors_set):
                ax.plot(sec_to_datetime(x, year), to_celsius(y), color=col, linewidth=0.40)

        # Actuals
        if (tset_act_c is not None) and (zone_en != "attic"):
            ax.plot(time_act, tset_act_c + set_band, **BAND_STYLE)
            ax.plot(time_act, tset_act_c, **TSET_STYLE)
        ax.plot(time_act, tair_act_c, **TAIR_STYLE)

        # Zonenname OBEN **mittig**
        ax.set_title(zone_de, loc="center", pad=TITLE_PAD)

        # Genau drei Y-Ticks (.0/.5)
        arrays_for_ticks = [tair_act_c]
        if (tset_act_c is not None) and (zone_en != "attic"):
            arrays_for_ticks += [tset_act_c, tset_act_c + set_band]
        set_three_y_ticks_from_arrays(ax, arrays_for_ticks, step=0.5)

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.margins(x=0)
        if len(time_act) > 1:
            all_starts.append(time_act[0])
            all_ends.append(time_act[-1])

    # Gemeinsame x-Limits
    if all_starts and all_ends:
        xmin, xmax = min(all_starts), max(all_ends)
        for ax in axes:
            ax.set_xlim(xmin, xmax)

    # X-Achse unten
    axes[-1].set_xlabel("Datum")
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    # Y-Gesamtlabel ohne Layout-Reservierung (minimale linke Marge)
    fig.text(0.012, 0.5, r"Temperatur [$^\circ$C]", rotation=90,
             va="center", ha="left")

    # Legende unten
    legend_handles = [
        Line2D([0], [0], **TAIR_STYLE, label=r"Raumtemperatur"),
        Line2D([0], [0], **TSET_STYLE, label=r"Solltemperatur"),
        Line2D([0], [0], **BAND_STYLE, label=fr"obere Grenze"),
        Line2D([0], [0], color=orange_gradient(2)[-1], linewidth=1.2, label="Predictions"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.01), borderaxespad=0.0)

    # Mini-Ränder
    fig.tight_layout(rect=(0.02, 0.06, 0.995, 0.93))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        try:
            fig.canvas.manager.set_window_title("Alle Zonen – A4")
        except Exception:
            pass
        plt.show()
    else:
        plt.close(fig)

# =========================
# Hauptfunktion
# =========================
def plot_mpc_tair_tset(
    mpc_result_path: str | Path = DEFAULT_MPC_CSV,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    year: int = 2015,
    skip_first_days: int = 3,
    remove_first_point: bool = True,
    t_end_days: Optional[float] = None,
    set_band: float = SET_BAND_K,
    show_sammelplot: bool = True,
) -> Path:
    mpc_result_path = Path(mpc_result_path)
    out_dir = Path(out_dir)
    if not mpc_result_path.exists():
        raise FileNotFoundError(f"MPC-Datei nicht gefunden: {mpc_result_path}")

    df_raw = load_mpc_results(mpc_result_path)
    df = filter_mpc_time(df_raw, skip_first_days=skip_first_days, t_end_days=t_end_days)

    tair_cols_all = [f"T_Air_{z}" for z in ZONE_ORDER_EN] + \
                    [f"T_Air_{z}" for z in ZONE_MAP_DE.keys() if z not in ZONE_ORDER_EN]
    tset_cols_all = [f"TSetOneZone_{z}" for z in ZONE_ORDER_EN] + \
                    [f"TSetOneZone_{z}" for z in ZONE_MAP_DE.keys() if z not in ZONE_ORDER_EN]

    tair_cols = ensure_columns(df, "variable",  tair_cols_all)
    tset_cols = ensure_columns(df, "parameter", tset_cols_all)

    zones_present: List[str] = [z for z in ZONE_ORDER_EN if f"T_Air_{z}" in tair_cols]
    if not zones_present:
        raise ValueError("Keine passenden Zonen im MPC-Ergebnis gefunden (T_Air_* unter 'variable').")

    zones_blocks = []

    for z in zones_present:
        tair_name = f"T_Air_{z}"
        tset_name = f"TSetOneZone_{z}"

        s_tair_full = df["variable"][tair_name]
        s_tair_act = first_values_per_step(s_tair_full)
        if remove_first_point and len(s_tair_act) > 0:
            s_tair_act = s_tair_act.iloc[1:]
        t_act_dt = sec_to_datetime(s_tair_act.index.values, year)
        t_air_c = to_celsius(s_tair_act.values)

        pred_tair = prediction_paths(s_tair_full)

        tset_act_c = None
        pred_tset = None
        if (z != "attic") and (tset_name in tset_cols):
            s_tset_full = df["parameter"][tset_name]
            s_tset_act = first_values_per_step(s_tset_full)
            if remove_first_point and len(s_tset_act) > 0:
                s_tset_act = s_tset_act.iloc[1:]
            tset_act_c = to_celsius(s_tset_act.values)
            pred_tset = prediction_paths(s_tset_full)

        plot_zone_single_pdf(
            out_dir=out_dir, zone_en=z,
            time_act=t_act_dt, tair_act_c=t_air_c, tset_act_c=tset_act_c,
            pred_tair=pred_tair, pred_tset=pred_tset, year=year,
        )

        zones_blocks.append((z, t_act_dt, t_air_c, tset_act_c, pred_tair, pred_tset))

    all_pdf = out_dir / "T_Plot_ALL.pdf"
    plot_sammelplot_pdf(
        out_path=all_pdf, zones_blocks=zones_blocks,
        year=year, set_band=set_band, suptitle=None, show=show_sammelplot,
    )
    print(f"Fertig. Einzelplots + Sammelplot gespeichert in: {out_dir}")
    return all_pdf

# =========
# __main__
# =========
if __name__ == "__main__":
    plot_mpc_tair_tset(
        mpc_result_path=DEFAULT_MPC_CSV,
        out_dir=DEFAULT_OUT_DIR,
        year=2015,
        skip_first_days=3,
        remove_first_point=True,
        t_end_days=None,
        set_band=2.0,
        show_sammelplot=True,
    )
