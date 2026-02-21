from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# >>> Pfad zu deiner SIM-CSV
DEFAULT_SIM_CSV = Path(
    r"D:\fwu-ssr\res\Studies_not_coupled\studies_parallel\DesignOptimizationResults\TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGaiAbwesend_\Design_0_sim_agent.csv"
)

# --- NEU (oben bei den Defaults) ---
DEFAULT_OUT_DIR = Path(r"D:\fwu-ssr\res\plots\test_plotNsafeMPC")

# LaTeX-Look & Vektor-Settings
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

ScaleUnit = Literal["W", "kW", "MW"]
SkipMode  = Literal["dataset_start", "year_start"]


def _load_sim_two_levels(file: Path) -> pd.DataFrame:
    """CSV mit 3-stufigem Header laden und auf (group, signal) reduzieren."""
    df = pd.read_csv(file, header=[0, 1, 2], index_col=0)
    return df.droplevel(level=2, axis=1)  # MultiIndex: (group, signal)


def _find_series(df: pd.DataFrame, group_names: list[str], signal_names: list[str]) -> Optional[pd.Series]:
    """Finde Spalte tolerant (MultiIndex (group, signal) oder 'group.signal' als flach)."""
    # MultiIndex-Fall
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels >= 2:
        for g in group_names:
            for s in signal_names:
                key = (g, s)
                if key in df.columns:
                    return pd.to_numeric(df[key], errors="coerce")
    # Fallback: flache Namen wie 'outputs.P_el_hp'
    flat = [str(c) for c in df.columns]
    for g in group_names:
        for s in signal_names:
            name = f"{g}.{s}"
            if name in flat:
                return pd.to_numeric(df.loc[:, name], errors="coerce")
    # Letzter Fallback: nur Signalname
    for s in signal_names:
        if s in df.columns:
            return pd.to_numeric(df[s], errors="coerce")
    return None


def _seconds_to_datetime_index(seconds: np.ndarray, base_year: int = 2015) -> pd.DatetimeIndex:
    """Sekunden seit Jahresanfang -> Zeitstempel im Jahr base_year."""
    base = pd.Timestamp(year=base_year, month=1, day=1)
    return base + pd.to_timedelta(seconds.astype(float), unit="s")


def _nice_ticks_from_zero(ymax: float, n_steps: int = 5) -> tuple[np.ndarray, float]:
    """
    Erzeuge 'schöne' gleichmäßige Ticks von 0 bis >= ymax.
    Gibt (ticks, ytop) zurück, wobei ytop exakt auf einem Tick liegt.
    """
    if ymax <= 0:
        return np.array([0.0, 1.0]), 1.0

    # Kandidaten für Schrittweiten (Skalenfaktor * {1,2,5})
    raw = ymax / n_steps
    exp = int(np.floor(np.log10(raw))) if raw > 0 else 0
    base = raw / (10 ** exp)
    if base <= 1:
        step = 1 * (10 ** exp)
    elif base <= 2:
        step = 2 * (10 ** exp)
    elif base <= 5:
        step = 5 * (10 ** exp)
    else:
        step = 10 * (10 ** exp)

    ytop = step * np.ceil(ymax / step)
    ticks = np.arange(0.0, ytop + 0.5 * step, step)
    return ticks, float(ytop)


def plot_hp_pel_qdot_from_sim(
    sim_csv: Path = DEFAULT_SIM_CSV,
    *,
    base_year: int = 2015,
    scale: ScaleUnit = "kW",
    skip_first_days: int = 0,
    skip_mode: SkipMode = "dataset_start",  # 'dataset_start' oder 'year_start'
    remove_first_point: bool = True,        # <<< WICHTIG: ersten Datenpunkt entfernen
    out_dir: Optional[Path] = None,
    show_plot: bool = True,
) -> Path:
    """
    Plottet outputs.P_el_hp und outputs.Qdot_hp vs. Datum und speichert als PDF.

    - skip_first_days: Anzahl zu entfernender Tage (0 = nichts)
    - skip_mode:
        * 'dataset_start' -> relativ zum ersten Zeitstempel in der Datei
        * 'year_start'    -> relativ zum 01.01.{base_year}
    - remove_first_point: entfernt den ersten verbleibenden Datenpunkt,
      damit kein Tages-Tick am linken Rand erscheint (wie in deinen anderen Plots).
    """
    if not sim_csv.exists():
        raise FileNotFoundError(f"SIM-Datei nicht gefunden: {sim_csv}")

    df = _load_sim_two_levels(sim_csv)

    # Zeit -> Datum
    time_sec = pd.to_numeric(pd.Index(df.index), errors="coerce").to_numpy()
    dt_index = _seconds_to_datetime_index(time_sec, base_year=base_year)

    # Signale tolerant finden ('outputs' vs. 'output')
    pel = _find_series(df, group_names=["outputs", "output"], signal_names=["P_el_hp"])
    qdot = _find_series(df, group_names=["outputs", "output"], signal_names=["Qdot_hp"])

    if pel is None and qdot is None:
        raise ValueError("Weder 'outputs.P_el_hp' noch 'outputs.Qdot_hp' gefunden.")

    # In DataFrame packen für robustes Filtern
    data = pd.DataFrame({"time": dt_index})
    if pel is not None:
        data["pel"] = pel.to_numpy()
    if qdot is not None:
        data["qdot"] = qdot.to_numpy()

    # Erste x Tage entfernen
    if skip_first_days and skip_first_days > 0:
        if skip_mode == "dataset_start":
            cut = data["time"].iloc[0] + pd.Timedelta(days=skip_first_days)
        else:  # "year_start"
            cut = pd.Timestamp(year=base_year, month=1, day=1) + pd.Timedelta(days=skip_first_days)
        data = data.loc[data["time"] >= cut]

    # <<< HIER: ersten verbleibenden Punkt entfernen
    if remove_first_point and len(data) > 0:
        data = data.iloc[1:]

    if data.empty:
        raise ValueError("Nach dem Zuschneiden sind keine Daten mehr übrig.")

    # Skalenfaktor
    factor = {"W": 1.0, "kW": 1e-3, "MW": 1e-6}[scale]

    # Energiesummen (kWh) über den geplotteten Zeitraum
    ts = data["time"].astype("int64").to_numpy() / 1e9  # Sekunden
    lines_for_box = []
    if "pel" in data:
        e_el_kwh = np.trapz(np.nan_to_num(data["pel"].to_numpy()), x=ts) / 3600.0 / 1000.0
        lines_for_box.append(r"$E_{\mathrm{el,HP}}$ = " + f"{e_el_kwh:.2f} kWh")
    if "qdot" in data:
        q_hp_kwh = np.trapz(np.nan_to_num(data["qdot"].to_numpy()), x=ts) / 3600.0 / 1000.0
        lines_for_box.append(r"$Q_{\mathrm{HP}}$ = " + f"{q_hp_kwh:.2f} kWh")

    # Plot
    fig, ax = plt.subplots(figsize=(6.3, 3.54))
    if "pel" in data:
        ax.plot(data["time"], data["pel"] * factor, label=r"$P_{\mathrm{el,HP}}$", linewidth=1.6)
    if "qdot" in data:
        ax.plot(data["time"], data["qdot"] * factor, label=r"$\dot{Q}_{\mathrm{HP}}$", linewidth=1.6)

    ax.set_xlabel(r"Zeit [Datum]")
    ax.set_ylabel(f"Leistung / Wärmestrom [{scale}]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    # Tages-Ticks (jeden Tag); durch remove_first_point liegt linker Rand nach 00:00
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))
    # Achsgrenzen exakt auf Daten beschränken, keine weißen Flächen
    ax.set_xlim(data["time"].iloc[0], data["time"].iloc[-1])
    ax.margins(x=0)

    # y-Achse: unten 0, oben „schöner“ Tick inkl. Tick an den Grenzen
    y_max_series = []
    if "pel" in data:
        y_max_series.append(np.nanmax(np.nan_to_num(data["pel"].to_numpy()) * factor))
    if "qdot" in data:
        y_max_series.append(np.nanmax(np.nan_to_num(data["qdot"].to_numpy()) * factor))
    ymax = max(y_max_series) if y_max_series else 1.0
    ticks, ytop = _nice_ticks_from_zero(ymax)
    ax.set_ylim(0.0, ytop)
    ax.set_yticks(ticks)

    # Energiesummen-Box oben links
    if lines_for_box:
        ax.text(
            0.02, 0.95,
            "\n".join([s for s in lines_for_box]),
            transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9)
        )

    fig.autofmt_xdate()
    fig.tight_layout()

    # Ausgabeort
    if out_dir is None:
        out_dir = sim_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / f"Energieplot.pdf"
    fig.savefig(pdf_path)

    if show_plot:
        plt.show_plot()
    else:
        plt.close(fig)

    print(f"Gespeichert: {pdf_path}")
    return pdf_path


def plot_energy_models_by_scenario_vertical(
    sims_model_a: list[Path],
    sims_model_b: list[Path],
    scenarios: list[str],
    *,
    base_year: int = 2015,
    skip_first_days: int = 0,
    skip_mode: SkipMode = "dataset_start",
    remove_first_point: bool = True,
    positive_only: bool = False,
    out_dir: Optional[Path] = None,
    outfile_name: str = "Energievergleich_HP_2Modelle_vertical.pdf",
    show_plot: bool = True,
    model_labels: tuple[str, str] = ("Modell A", "Modell B"),
    colors: tuple[Optional[str], Optional[str]] | None = None,  # (Farbe A, Farbe B)
) -> Path:
    """
    Zwei vertikale Panels:
      oben  : E_el,HP [kWh]
      unten : Q_HP [kWh]
    x-Achse: Szenarien; pro Szenario zwei Balken (Modell A/B).
    Nutzt _compute_hp_energies_kwh_for_period().
    """
    assert len(sims_model_a) == len(sims_model_b) == len(scenarios), "Listenlängen müssen übereinstimmen."

    # 1) Energien berechnen
    rows = []
    for lab, paths in [(model_labels[0], sims_model_a), (model_labels[1], sims_model_b)]:
        for scen, p in zip(scenarios, paths):
            E_el, Q_th = _compute_hp_energies_kwh_for_period(
                p,
                base_year=base_year,
                skip_first_days=skip_first_days,
                skip_mode=skip_mode,
                remove_first_point=remove_first_point,
                positive_only=positive_only,
            )
            rows.append({"Szenario": scen, "Modell": lab, "E_el_kWh": E_el, "Q_th_kWh": Q_th})
    df = pd.DataFrame(rows)

    # 2) Reihenfolge fixieren
    df["Szenario"] = pd.Categorical(df["Szenario"], categories=scenarios, ordered=True)
    df["Modell"]   = pd.Categorical(df["Modell"],   categories=list(model_labels), ordered=True)
    df = df.sort_values(["Szenario", "Modell"])

    # 3) Plot: vertikal gestapelte Panels
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.6), sharex=True)  # A4-geeignet, LaTeX-Spaltenbreite
    width = 0.38
    x = np.arange(len(scenarios))

    # Farben festlegen
    if colors is None:
        palette = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        color_a = palette[0]
        color_b = palette[1] if len(palette) > 1 else None
    else:
        color_a, color_b = colors

    def _panel(ax, col_values: np.ndarray, col_values_b: np.ndarray, title: str):
        b1 = ax.bar(x - width/2, col_values,   width, label=model_labels[0], color=color_a)
        b2 = ax.bar(x + width/2, col_values_b, width, label=model_labels[1], color=color_b)
        ax.set_title(title)
        ax.set_ylabel("Energie [kWh]")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        # schöne y-Ticks
        ymax = np.nanmax([np.nanmax(col_values), np.nanmax(col_values_b)])
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        ticks, ytop = _nice_ticks_from_zero(ymax)
        ax.set_ylim(0.0, ytop)
        ax.set_yticks(ticks)

        # Werte annotieren
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                if np.isnan(h):
                    continue
                ax.annotate(f"{h:.1f}",
                            xy=(b.get_x()+b.get_width()/2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)
        return b1, b2

    # Daten je Panel
    el_a = df[df["Modell"] == model_labels[0]]["E_el_kWh"].to_numpy(dtype=float)
    el_b = df[df["Modell"] == model_labels[1]]["E_el_kWh"].to_numpy(dtype=float)
    th_a = df[df["Modell"] == model_labels[0]]["Q_th_kWh"].to_numpy(dtype=float)
    th_b = df[df["Modell"] == model_labels[1]]["Q_th_kWh"].to_numpy(dtype=float)

    _panel(axes[0], el_a, el_b, r"$E_{\mathrm{el,HP}}$")
    _panel(axes[1], th_a, th_b, r"$Q_{\mathrm{HP}}$")

    # x-Achse nur unten beschriften
    axes[1].set_xticks(x, scenarios, rotation=0)

    # gemeinsame Legende oben
    handles = [
        mpl.patches.Patch(color=color_a, label=model_labels[0]),
        mpl.patches.Patch(color=color_b, label=model_labels[1]),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.00))

    fig.tight_layout(rect=(0, 0, 1, 0.96))  # Platz für Legende oben

    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR if 'DEFAULT_OUT_DIR' in globals() and DEFAULT_OUT_DIR is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile_name

    fig.savefig(out_path)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return out_path



if __name__ == "__main__":
    szenarien = ["Baseline", "Retrofit A", "Retrofit B", "Extreme Nachtabsenkung"]
    modellA = [Path("...A_baseline.csv"), Path("...A_retroA.csv"), Path("...A_retroB.csv"), Path("...A_night.csv")]
    modellB = [Path("...B_baseline.csv"), Path("...B_retroA.csv"), Path("...B_retroB.csv"), Path("...B_night.csv")]

    plot_energy_models_by_scenario_vertical(
        modellA, modellB, szenarien,
        base_year=2015,
        skip_first_days=3,
        skip_mode="dataset_start",
        remove_first_point=True,
        positive_only=False,
        out_dir=DEFAULT_OUT_DIR,
        outfile_name="Energievergleich_HP_2Modelle_vertical.pdf",
        show_plot=True,
        model_labels=("Single-Zone", "Multi-Zone"),
        # colors=("tab:blue", "tab:orange"),  # optional fest verdrahten
    )

    """
    # Beispiel: erste 3 Tage relativ zum Datei-Beginn ausblenden
    plot_hp_pel_qdot_from_sim(
        sim_csv=DEFAULT_SIM_CSV,
        out_dir= DEFAULT_OUT_DIR,
        base_year=2015,
        scale="kW",
        skip_first_days=3,
        skip_mode="dataset_start",
        remove_first_point=True,   # <<< sorgt dafür, dass kein Tick am ersten Tag erscheint
        show_plot=True,
    )
    """