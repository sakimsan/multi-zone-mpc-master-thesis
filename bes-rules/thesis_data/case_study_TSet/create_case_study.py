# create_case_study_export.py
from pathlib import Path
import pandas as pd
import numpy as np
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt

# === Einstellungen ===
BASE_DIR = Path(r"D:\fwu-ssr\bes-rules\student_theses\sakimsan\case_study_TSet")

# Alle vier change-only Varianten; ggf. zusätzlich die vollen Sekunden-Dateien hinzufügen
FILES = sorted(BASE_DIR.glob("*extreme.csv"))
# Falls du auch die vollen Sekunden-Dateien exportieren willst, nimm zusätzlich:
# FILES += sorted(BASE_DIR.glob("*_NormalerWinter_Sekunden_Setpoints.csv"))

OUT_DIR = BASE_DIR / "modelica_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZONES = [
    "bedroom","livingroom","kitchen","hobby","wcstorage",
    "corridor","children","corridor2","bath","children2"
]

# Start der Referenzwoche (für Rekonstruktion aus second_of_week)
START_OF_WEEK = pd.Timestamp("2025-01-06 00:00:00")

# Nur in °C → K umrechnen (auto):
K_OFFSET = 273.15

def csv_to_modelica_table(csv_path: Path, out_dir: Path):
    print(f"[INFO] Lade: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Zeitindex herstellen
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "second_of_week" in df.columns:
        # Datetime aus Sekunden seit Wochenstart rekonstruieren
        df["datetime"] = START_OF_WEEK + pd.to_timedelta(df["second_of_week"].astype(int), unit="s")
    else:
        raise ValueError(f"Weder 'datetime' noch 'second_of_week' in {csv_path.name} gefunden.")

    # Index setzen und sortieren
    df = df.set_index("datetime").sort_index()

    # Nur Zonen behalten
    cols = [c for c in df.columns if c in ZONES]
    if not cols:
        raise ValueError(f"Keine Zonen-Spalten in {csv_path.name} gefunden. Vorhanden: {list(df.columns)}")
    df = df[cols].astype(float)

    # Auto-Einheiten-Check: Wenn die Max-Werte < 200 sind → °C → in K umrechnen; sonst schon K.
    # (Deine *_changeonly_K.csv sind bereits in K.)
    max_val = float(df.max().max())
    if max_val < 200.0:
        df = df + K_OFFSET

    # gewünschte Spaltenreihenfolge
    desired = ["livingroom", "hobby", "corridor", "wcstorage", "kitchen",
               "bedroom", "children", "corridor2", "bath", "children2"]
    # nur die vorhandenen in dieser Reihenfolge
    desired_present = [c for c in desired if c in df.columns]
    df = df.reindex(columns=desired_present)

    # Dateiname / Tabellenname
    base = csv_path.stem
    table_name = base.replace("-", "_").replace(" ", "_")
    out_path = out_dir / f"{base}.txt"

    # In Modelica-Format schreiben (DatetimeIndex wird von ebcpy in Sekunden umgesetzt)
    convert_tsd_to_modelica_txt(
        tsd=df.tz_localize(None) if df.index.tz is not None else df,  # Sicherheit: tz-naiv
        table_name=table_name,
        save_path_file=str(out_path)
    )
    print(f"[OK] Geschrieben: {out_path}")

if __name__ == "__main__":
    if not FILES:
        raise SystemExit(f"Keine passenden CSVs in {BASE_DIR} gefunden (erwarte *_NormalerWinter_changeonly_K.csv).")
    for f in FILES:
        csv_to_modelica_table(f, OUT_DIR)
    print("\nFertig. Dateien liegen in:", OUT_DIR)
