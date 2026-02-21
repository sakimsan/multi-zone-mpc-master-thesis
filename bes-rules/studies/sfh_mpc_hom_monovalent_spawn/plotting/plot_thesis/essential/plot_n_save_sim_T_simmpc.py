from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import re
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# --- LaTeX/Styling ---
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",

    # LaTeX look
    "text.usetex": True,
    "axes.unicode_minus": False,  # saubere Minuszeichen
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    # kleines, robustes Präambel-Setup (ohne inputenc/babel):
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
})

# --- Defaults ---
DEFAULT_SIM_CSV = Path(r"D:\fwu-ssr\res\Studies_not_coupled\studies_parallel\DesignOptimizationResults\TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGaiAbwesend_\Design_0_sim_agent.csv")   # SIM bleibt hier
DEFAULT_OUT_DIR = Path(r"D:\fwu-ssr\res\plots\study2")                    # Output hier

# Farben/Stile
TSET_COLOR = "deepskyblue"  # Sollwert
TAIR_STYLE = dict(color="black", linestyle="-", linewidth=1.6)
TSET_STYLE = dict(color=TSET_COLOR, linestyle="--", linewidth=1.6)
BAND_STYLE = dict(color="red", linestyle="-", linewidth=1.0)

# Mapping & Ziel-Reihenfolge
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

def _load_sim_keep_causality_and_signal(file: Path) -> pd.DataFrame:
    """SIM-CSV laden, Header (causality, signal, dtype) -> dtype entfernen => MultiIndex (causality, signal)."""
    df = pd.read_csv(file, header=[0, 1, 2], index_col=0)
    df = df.droplevel(level=2, axis=1)  # dtype weg
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.sort_index()
    return df

def _find_columns(df: pd.DataFrame, causality_names: Iterable[str], pattern: str) -> dict[str, tuple[str, str]]:
    """
    Suche Spalten unter bestimmten causality-Namen (z.B. 'local' / 'inputs'),
    deren Signalname auf 'pattern' passt. 'pattern' muss eine Regex mit
    Gruppe (?P<zone>...) enthalten, die die Zonenbezeichnung einfängt.
    Rückgabe: {zone: (causality, signal)}
    """
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

def plot_tair_tset_from_sim_dates(
    sim_csv_path: Path = DEFAULT_SIM_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    base_year: int = 2015,            # Sekunden sind "seit Jahresbeginn base_year"
    skip_first_days: int = 3,          # erste N Tage komplett nicht plotten
    remove_first_point: bool = True,   # ersten verbleibenden Punkt danach droppen
    t_end_days: Optional[float] = None,# optional: Länge ab Cutoff
    set_band: float = 2.0,
    show_sammelplot: bool = True,      # nur Sammelplot anzeigen
) -> list[Path]:
    """
    - Einzelplots je Zone: PDF speichern (in out_dir), aber NICHT anzeigen.
    - Sammelplot (A4, alle Zonen in definierter Reihenfolge): PDF speichern und anzeigen (optional).
    - T_Air (local.T_Air_<zone>) und TSetOneZone (inputs.TSetOneZone_<zone>) in °C.
    - 'attic' (Dachgeschoss) ohne TSet/Komfortband.
    - X-Achse: Datum (tägliche Ticks, 'DD.MM').
    """
    # Pfade prüfen / anlegen
    if not sim_csv_path.exists():
        raise FileNotFoundError(f"SIM-Datei nicht gefunden: {sim_csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Daten laden
    df = _load_sim_keep_causality_and_signal(sim_csv_path)

    # Zeitfilter: erste 'skip_first_days' Tage ausschließen
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

    # Zeit -> Datum
    start_dt = datetime(base_year, 1, 1)
    time_dt = pd.to_datetime(secs, unit="s", origin=start_dt)

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

    # Zonen gemäß gewünschter Reihenfolge (nur vorhandene)
    zones_present = { _canonical(z): z for z in tair_cols.keys() }
    zones_in_order: list[str] = []
    for zc in ZONE_ORDER_CANONICAL:
        if zc in zones_present:
            zones_in_order.append(zones_present[zc])

    saved: list[Path] = []

    # ---------- Einzelplots (PDF) ----------
    for zone in zones_in_order:
        c0_tair, c1_tair = tair_cols[zone]
        s_tair = pd.to_numeric(df[(c0_tair, c1_tair)], errors="coerce")
        if s_tair.dropna().empty:
            print(f"[{zone}] übersprungen: T_Air leer.")
            continue

        tair_c = s_tair.values - 273.15

        is_attic = _canonical(zone) == "attic"
        tset_c = None
        if (not is_attic) and (_canonical(zone) in [_canonical(k) for k in tset_cols.keys()]):
            match_key = next((k for k in tset_cols.keys() if _canonical(k) == _canonical(zone)), None)
            if match_key is not None:
                c0_tset, c1_tset = tset_cols[match_key]
                s_tset = pd.to_numeric(df[(c0_tset, c1_tset)], errors="coerce")
                if not s_tset.dropna().empty:
                    tset_c = s_tset.values - 273.15

        fig, ax = plt.subplots()

        # Raumtemperatur
        ax.plot(time_dt, tair_c, **TAIR_STYLE, label=r"Raumtemperatur [$^\circ$C]")

        # Soll + Komfortband (außer Dachgeschoss)
        if tset_c is not None:
            ax.plot(time_dt, tset_c, **TSET_STYLE, label=r"Solltemperatur [$^\circ$C]")
            ax.plot(time_dt, tset_c + set_band, **BAND_STYLE)
            ax.plot(time_dt, tset_c - set_band, **BAND_STYLE)
            handles, labels = ax.get_legend_handles_labels()
            band_handle = Line2D([0], [0], **BAND_STYLE, label=fr"Komfortband $\pm$ {set_band:g} K")
            handles.append(band_handle)
            ax.legend(handles=handles)
        else:
            ax.legend()

        # X-Achse: täglich, DD.MM
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        fig.autofmt_xdate()

        disp = _display_name(zone)
        ax.set_title(f"{disp} — Raum- und Solltemperatur" if not is_attic else f"{disp} — Raumtemperatur")
        ax.set_xlabel("Datum")
        ax.set_ylabel(r"Temperatur [$^\circ$C]")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        out_file = out_dir / f"T_Plot_{_sanitize_filename(disp)}.pdf"
        fig.savefig(out_file)
        saved.append(out_file)
        plt.close(fig)

    # ---------- Sammelplot (A4) ----------
    if zones_in_order:
        n = len(zones_in_order)
        a4_size = (8.27, 11.69)  # Zoll
        fig, axes = plt.subplots(n, 1, figsize=a4_size, sharex=True)
        if n == 1:
            axes = [axes]

        for ax, zone in zip(axes, zones_in_order):
            c0_tair, c1_tair = tair_cols[zone]
            s_tair = pd.to_numeric(df[(c0_tair, c1_tair)], errors="coerce")
            if s_tair.dropna().empty:
                ax.set_visible(False)
                continue

            tair_c = s_tair.values - 273.15

            is_attic = _canonical(zone) == "attic"
            tset_c = None
            if (not is_attic) and (_canonical(zone) in [_canonical(k) for k in tset_cols.keys()]):
                match_key = next((k for k in tset_cols.keys() if _canonical(k) == _canonical(zone)), None)
                if match_key is not None:
                    c0_tset, c1_tset = tset_cols[match_key]
                    s_tset = pd.to_numeric(df[(c0_tset, c1_tset)], errors="coerce")
                    if not s_tset.dropna().empty:
                        tset_c = s_tset.values - 273.15

            # Raumtemperatur
            ax.plot(time_dt, tair_c, **TAIR_STYLE)

            if tset_c is not None:
                # Soll + Komfortband
                ax.plot(time_dt, tset_c, **TSET_STYLE)
                ax.plot(time_dt, tset_c + set_band, **BAND_STYLE)
                ax.plot(time_dt, tset_c - set_band, **BAND_STYLE)

            ax.set_ylabel(_display_name(zone), rotation=0, ha="right", va="center")
            ax.grid(True, linestyle="--", alpha=0.3)

        axes[-1].set_xlabel("Datum")
        axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        fig.autofmt_xdate()

        # Legende oben
        handles = [
            Line2D([0], [0], **TAIR_STYLE, label=r"Raumtemperatur [$^\circ$C]"),
            Line2D([0], [0], **TSET_STYLE, label=r"Solltemperatur [$^\circ$C]"),
            Line2D([0], [0], **BAND_STYLE, label=fr"Komfortband $\pm$ {set_band:g} K"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))

        fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.95))
        out_all = out_dir / "T_Plot_Sammel_A4.pdf"
        fig.savefig(out_all)
        saved.append(out_all)

        if show_sammelplot:
            try:
                fig.canvas.manager.set_window_title("Alle Zonen – A4")
            except Exception:
                pass
            plt.show()
        else:
            plt.close(fig)

    print(f"Fertig. {len(saved)} PDFs gespeichert in: {out_dir}")
    return saved


if __name__ == "__main__":
    plot_tair_tset_from_sim_dates(
        sim_csv_path=DEFAULT_SIM_CSV,
        out_dir=DEFAULT_OUT_DIR,
        base_year=2015,
        skip_first_days=3,
        remove_first_point=True,
        t_end_days=None,
        set_band=2.0,
        show_sammelplot=True,  # nur Sammelplot anzeigen
    )
