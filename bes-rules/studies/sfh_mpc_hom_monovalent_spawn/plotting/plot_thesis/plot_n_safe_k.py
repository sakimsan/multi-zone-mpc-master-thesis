#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot EnergyPlus ESO time series for
    "Surface Inside Face Convection Heat Transfer Coefficient [W/m2-K]"
using a HARD-CODED mapping of SurfaceKey -> (zone, component).

Usage:
    python plot_eso_hc_timeseries_embedded.py /path/to/file.eso [--only GroundFloor,InnerWall,...] [--out ./plots]

Notes:
- Each surface gets its own figure (no custom colors/styles).
- Output files are PDFs by default.
- The x-axis uses constructed datetimes from ESO time records (Month, Day, Hour, EndMinute).
"""

import sys
import re
from pathlib import Path
from collections import defaultdict
import csv
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FormatStrFormatter

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

TARGET_VAR = "Surface Inside Face Convection Heat Transfer Coefficient"

# *** maximale Zahl geplotteter Stunden ***
MAX_HOURS = 8760
BASE_YEAR = 2015  # für Umrechnung von "Stunden seit Jahresbeginn" in Datum

# === FULLY EMBEDDED COPY OF YOUR zone_construction.csv ===
EMBEDDED_MAPPING_CSV = r"""zone,Dach,Dach2,Dach3,GroundFloor,InnerFloor,InnerFloor2,InnerFloor3,InnerFloor4,InnerFloor5,Decke,InnerWall,InnerWall2,InnerWall3,InnerWall4,InnerWall5,OuterWall,OuterWall2
wcstorage,,,,"36sqYuGcn0ux0DSg3GINwE,",,,,,,"0WVBIR9ef3DxeehDEWYowe,","2Qn8jmLA9F2gLRScv2IoC1,","3aJ33XtHX9ffO5LCPh3sX4,",,,,"2ze26MqU564uLrWziqfSTf,","3Pd10x$H1DD8DlRrqzeTXp,"
children,,,,,"24oZ1lp11CkB0vWm31Xi6Y,",,,,,"Emw3Vznb1keGM7vzzIWqKw,","2lt$6JiRj8sOoQyYpciN60,","3MgVh8WEf6Ug_1Tz617dV1,",,,,"32RNQZOr55ixX7FqTuOSFU,","36KHFp6fr359oP_h$vhm3z,"
bath,,,,,"0r66SF9ML7TBufkSEIcCc0,",,,,,"YqyY556OyEOZmC50RCSVpQ,","16Mt9vyuX6JAP$WQMiyH9W,","2W4aa10bj1VvxY3AXX9_Lc,",,,,"19_8KRnCL2FOzSIJ7sIkuf,","26iwqPqAL5qB_gBb$TzpFa,"
attic,"1Vxn_8iwH0KROQWIXnsdbx,","2dTAU1f75AUwljRxNpWKse,","3GAUyX46nEqfQx0lL$sT2l,",,"0yLQ_yRAf6vwNzc$5F_Id$,","0zzl03JXfF_hp0sOBcrz6E,","1zkFM_$IX39hxLe5vLkKFd,","3dHBvghDj65xNkPfCpxtiK,","3wxUKM0NL93eLMxuu6E3OC,",,,,,,,"2ManKvuD95MOLQkaPDAosb,","3PiRCvTMDFKfmVWphhRYzZ,"
bedroom,,,,,"3Gyc7Ls2vCYRvo$_bSipxD,",,,,,"U1xgFVZ7qU2jHrfvJ9EyTg,","0yqWqfaG92hhk8yvCfVR$Q,","13yBvK24H4cOvD9vMsh$5i,","37L4eZsf1FQ9nr2M7ezqkQ,","3m0Ii62gfAnBSz2eMqABoY,",,"1r2efRVUXEDhgsMYi31MIb,","234LqWHfP08hs9FIRU2_sa,"
hobby,,,,"2PqXZpa2nBQxWxZb$WGjYs,",,,,,,"1UjGuYddP8s8VffDb_tuw$,","0PPTFjDYDAxhLckZzSfnZ8,","2y0c27v_L7kf1PGcQt_gQ,",,,,"18iVrJVIHFLhccRBCuzoKx,","3UQ8DoRDXE4u3mBgXyKGL2,"
livingroom,,,,"3uo3t74qXBy9WMPYe7ua5D,",,,,,,"1y_uwYpEbFN8k9Jmu1h8TT,","2$Btk$6rD0mevm7Klm2knY,","32so0TBWb9TwFiUTZSLUy3,","3LTwVcRbb5_8eQVYkoiN6e,","3Rudn5yrPFzRVPhNHtX4je,",,"2ZxDery31FGfH8cX9KzOlg,","2qKixY2FXFuBAPWAJa27GV,"
corridor,,,,"0TzI7RGvL0t9lfmxS9ijOr,",,,,,,"0zTZeBpz1CfwRoAFgtxCDl,","1DQaQCTnX1fOyZ05iECd_w,","1fT0syOff8puStpx2ChjAF,","2BfBOb0fDALBPtINh6O8aw,","2vzM61dV5DcPXBySmc_f8u,","3sfvZd2pj7Twi$9yZDBXvi,","1kffuyfyT0cRpp3UO09LYE,"
corridor2,,,,,"1Y_7pbzVb0wfu8JFhPQOJG,",,,,,"jaa06PHWwEKY8GAJZY7zGg,","0XAGB7vXr8ROcN7Nk6DsFY,","0fCjgOHnz1GQDz0NzN_Ugo,","1SmJIV5_nEtB8jTD9MNC0p,","1sxF8LSeX8vfU7EV$mpBaV,","2OIA5f365ErBOeTyuCOeLT,","3pcL$8r0L4k8ZzPs5hkyqo,"
kitchen,,,,"1$vWpw4415hQY_6dYKaCo3,",,,,,,"0LqB1RUsr5DeLg2ySl7Okr,","0OKmwFPoT88Am99V$Ot$ze,","1EcsArQhjBYB6InnGD4fq7,","1rBRMm9aH5jPSWLHTY2lPz,","2G4SCG2Pb6Suz$3LlnyPvU,",,"1E5jm9pwTDcAD1_xSuRiAq,","3p6tvhiwvBQe4$EI3VGjvY,"
children2,,,,,"3FA9_i1Qf9dfNs8MdbnEgA,",,,,,"tHsTmDYVR0miJ70tXUrGOw,","0xqCfxBX92Sfci$syPa18c,","1O7UYlrLD7i8OD7yY0r19v,","21LJ5PAYbCle2KNLSiBVra,","35vb7iscr60BT_AQZhlENO,",,"0oI_MGbB5FbQwdic0zCroI,","23cUKRxGH6ieayWkfRSJbO,"
"""

def parse_embedded_mapping(embedded_csv: str):
    mapping = {}
    try:
        first_line = embedded_csv.splitlines()[0]
        delim = csv.Sniffer().sniff(first_line).delimiter
    except Exception:
        delim = ","
    reader = csv.reader(io.StringIO(embedded_csv), delimiter=delim)
    rows = list(reader)
    if not rows:
        return mapping
    header = rows[0]
    if len(header) < 2:
        return mapping

    zone_col_idx = 0
    components = header[1:]
    for r in rows[1:]:
        if not r:
            continue
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        zone = (r[zone_col_idx] or "").strip()
        if not zone:
            continue
        for j, comp in enumerate(components, start=1):
            comp_name = (comp or "").strip()
            if not comp_name:
                continue
            cell = (r[j] or "").strip()
            if not cell:
                continue
            for token in cell.split(","):
                key = token.strip()
                if not key:
                    continue
                mapping[key.upper()] = {"zone": zone, "component": comp_name}
    return mapping

def parse_eso_index(eso_path: Path, target_var: str):
    id_to_key = {}
    with eso_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if target_var in line and "!" in line:
                parts = [p.strip() for p in line.split(",")]
                try:
                    var_id = int(parts[0])
                    key = parts[2]
                    id_to_key[var_id] = key
                except Exception:
                    continue
    return id_to_key

def iter_eso_times(eso_path: Path):
    times = []
    with eso_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("2,") and "!" not in line and "[" not in line and "Day of" not in line and "Month" not in line:
                parts = [p.strip() for p in line.split(",")]
                try:
                    month = int(float(parts[2]))
                    day = int(float(parts[3]))
                    hour = int(float(parts[5]))
                    end_min = int(float(parts[7]))
                    year = 2001
                    times.append(pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=end_min))
                except Exception:
                    times.append(None)
    if any(t is None for t in times):
        return pd.RangeIndex(len(times))
    return pd.DatetimeIndex(times)

def collect_series(eso_path: Path, ids_of_interest: set):
    time_index = iter_eso_times(eso_path)
    series = defaultdict(list)
    with eso_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                left, right = line.split(",", 1)
                var_id = int(left)
            except Exception:
                continue
            if var_id in ids_of_interest:
                if "!" in right or "[" in right or TARGET_VAR in right:
                    continue
                try:
                    series[var_id].append(float(right.strip()))
                except Exception:
                    continue
    return series, time_index

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())[:150]

def _hours_index_to_datetime(idx_like, base_year: int = BASE_YEAR):
    """Konvertiert numerische Stunden (0..N-1) in Datumswerte ab 1.1.<base_year>."""
    if isinstance(idx_like, pd.DatetimeIndex):
        return idx_like
    try:
        arr = np.asarray(idx_like, dtype=float)
        if np.all(np.isfinite(arr)):
            base = pd.Timestamp(f"{base_year}-01-01 00:00:00")
            return base + pd.to_timedelta(arr, unit="h")
    except Exception:
        pass
    try:
        return pd.to_datetime(idx_like)
    except Exception:
        n = len(idx_like)
        base = pd.Timestamp(f"{base_year}-01-01 00:00:00")
        return base + pd.to_timedelta(np.arange(n), unit="h")

def _set_even_major_and_minor_ticks(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp,
                                    n_major: int = 10, n_minor_between: int = 5,
                                    fmt: str = "%d.%m"):
    """
    X: Major = exakt n_major gleichmäßig (inkl. Start/Ende).
       Minor = pro Major-Intervall genau n_minor_between gleichmäßig.
    """
    if n_major < 2:
        n_major = 2
    start_num = mdates.date2num(start)
    end_num = mdates.date2num(end)

    majors_num = np.linspace(start_num, end_num, n_major)
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(FixedLocator(majors_num))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

    if n_minor_between > 0:
        minors = []
        for a, b in zip(majors_num[:-1], majors_num[1:]):
            step = (b - a) / (n_minor_between + 1)
            minors.extend([a + step * k for k in range(1, n_minor_between + 1)])
        ax.xaxis.set_minor_locator(FixedLocator(minors))

def _set_even_y_ticks(ax: plt.Axes, y_values, n_major: int = 6, fmt: str = '%.2f',
                      pad_ratio: float = 0.05, min_abs_pad: float = 1e-6):
    """
    Y: setzt exakt n_major gleichmäßig verteilte Major-Ticks über einem
    *gepaddeten* Bereich, sodass die Kurve oben/unten nicht anstößt.
    - pad_ratio: Anteil des Datenbereichs als Puffer (z.B. 0.05 = 5 %).
    - min_abs_pad: minimaler absoluter Puffer (falls Range sehr klein).
    """
    y = np.asarray(y_values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            y_min, y_max = 0.0, 1.0
        if y_max == y_min:
            # flache Linie: festen Puffer drumherum
            base = 1.0 if abs(y_min) < 1 else 0.05 * abs(y_min)
            y_min -= base
            y_max += base

    data_range = max(y_max - y_min, 0.0)
    pad = max(pad_ratio * data_range, min_abs_pad)
    y_lo = y_min - pad
    y_hi = y_max + pad

    majors = np.linspace(y_lo, y_hi, max(2, n_major))
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(FixedLocator(majors))
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

def main():
    # Deine Pfade (wie in deinem Beispiel)
    eso_path = Path("D:/fwu-ssr/besgriconop/Modelica/BESGriConOp/Resources/BuildingModels/HOM_CISBAT_CondFD_V2420.eso")
    out_dir = Path("D:/fwu-ssr/res/plots/k")
    only_filter = None

    # Optional args: --out ./plots  --only GroundFloor,InnerWall
    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "--out" and i + 1 < len(args):
            out_dir = Path(args[i+1])
        if arg == "--only" and i + 1 < len(args):
            only_filter = [v.strip() for v in args[i+1].split(",") if v.strip()] or None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping from embedded CSV
    surface_map = parse_embedded_mapping(EMBEDDED_MAPPING_CSV)

    # 1) Index variables in ESO
    id_to_key = parse_eso_index(eso_path, TARGET_VAR)
    if not id_to_key:
        print("No matching variables found in ESO for:", TARGET_VAR)
        sys.exit(2)

    # 2) Filter by mapping and optional component filter
    ids_interest = []
    meta_for_id = {}
    for vid, key in id_to_key.items():
        meta = surface_map.get(key.upper())
        if not meta:
            continue
        comp = meta.get("component", "")
        if only_filter and comp not in only_filter:
            continue
        ids_interest.append(vid)
        meta_for_id[vid] = {"key": key, **meta}

    if not ids_interest:
        print("No variables matched the mapping and filters.")
        sys.exit(3)

    # 3) Collect timeseries
    series, tidx = collect_series(eso_path, set(ids_interest))

    # 4) Plot each series — nur die ersten MAX_HOURS Stunden
    for vid in ids_interest:
        vals = series.get(vid, [])
        if not vals:
            continue
        meta = meta_for_id[vid]
        key = meta["key"]
        zone = meta.get("zone", "")
        comp = meta.get("component", "")

        n = min(MAX_HOURS, len(vals), len(tidx))
        if n <= 0:
            continue

        # X-Achse: Stunden seit Jahresbeginn -> Datum (BASE_YEAR)
        x_raw = tidx[:n]
        x_dt = _hours_index_to_datetime(x_raw, BASE_YEAR)
        y = vals[:n]

        fig = plt.figure(figsize=(155 / 25.4, (222 / 25.4) / 4))  # ≈ (6.102", 2.185")
        ax = plt.gca()
        ax.plot(x_dt, y, linewidth=1.2)

        # X: Major 10 inkl. Ränder, Minor 5 dazwischen
        x_start = pd.to_datetime(x_dt[0])
        x_end   = pd.to_datetime(x_dt[-1])
        _set_even_major_and_minor_ticks(ax, x_start, x_end, n_major=10, n_minor_between=3, fmt="%d.%m")

        # Y: Major-Ticks gleichmäßig über gepaddeten Bereich (z.B. 6 Stück) + 5% Puffer
        _set_even_y_ticks(ax, y_values=y, n_major=6, fmt='%.2f', pad_ratio=0.05, min_abs_pad=1e-6)

        # Grid für Major und Minor
        ax.grid(True, which="major", linestyle="--", alpha=0.35)
        ax.grid(True, which="minor", linestyle=":",  alpha=0.20)

        plt.ylabel(r"$h_\mathrm{c}$ [W/m$^2$K]")
        plt.tight_layout()
        fn = safe_name(f"hc_timeseries__{comp}__{zone}__{key}.pdf")
        plt.savefig(out_dir / fn)
        plt.close()

    print(f"Saved plots to: {out_dir.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
