from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# --- externe Plots ---
import plot_n_safe_k  # (nur importiert, wird hier nicht direkt benutzt)

from plot_n_safe_mpc_T_yValSet_TBufSet import (
    plot_three_panel_mpc_with_yval_tbuf,
)
from plot_n_save_sim_T_yValSet_TBufSet_simmpc import (
    plot_three_panel_with_yvalset,
)
from plot_n_safe_sim_power_simmpc import (
    plot_hp_pel_qdot_from_sim,
)

from calculate_discomfort import compute_comfort_cost_from_sim_dates

# --- Konstanten / Pfade ---
PLOTS_ROOT = Path(r"D:\fwu-ssr\res\plots")

SIM_SEARCH_ROOTS = [
    Path(r"D:\fwu-ssr\res\plots\used studies\Studies_coupled"),
    Path(r"D:\fwu-ssr\res\plots\used studies\Studies_not_coupled"),
]
MPC_SEARCH_ROOTS = [
    Path(r"D:\fwu-ssr\res\plots\used studies\SPAWN_Studies_coupled"),
    Path(r"D:\fwu-ssr\res\plots\used studies\SPAWN_Studies_not_coupled"),
]

# Ausgabe-Basen
SIM_OUT_BASE_COUPLED = PLOTS_ROOT / "Studies_coupled"
SIM_OUT_BASE_NOT_COUPLED = PLOTS_ROOT / "Studies_not_coupled"
MPC_OUT_BASE_COUPLED = PLOTS_ROOT / "SPAWN_Studies_coupled"
MPC_OUT_BASE_NOT_COUPLED = PLOTS_ROOT / "SPAWN_Studies_not_coupled"


# --- Hilfsfunktionen ---
def extract_study_from_parent_intgai(p: Path) -> str:
    """
    Nimmt den Parent-Ordnernamen und liefert alles NACH 'IntGai'.
    Beispiel: '..._0K-Per-IntGai2h_Abwesend_' -> '2h_Abwesend'
    Fallback: kompletter Parent-Ordnername.
    """
    parent_name = p.parent.name
    m = re.search(r"IntGai(.*)$", parent_name, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"IntGai(.*)$", str(p), flags=re.IGNORECASE)
    name = m.group(1) if m else parent_name
    name = name.strip("_ -")
    return name or parent_name


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def iter_files(roots: Iterable[Path], pattern: str) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        yield from root.rglob(pattern)


def _is_relative_to(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _relative_to_any(p: Path, roots: Iterable[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Liefert (relativer_pfad, getroffener_root) oder (None, None), wenn kein Root passt.
    """
    for r in roots:
        try:
            return p.resolve().relative_to(r.resolve()), r
        except ValueError:
            continue
    return None, None


def pick_out_base_for_sim(sim_csv: Path) -> Path:
    """Wählt die Ausgabe-Basis je nach Herkunft der SIM-Datei."""
    if _is_relative_to(sim_csv, SIM_SEARCH_ROOTS[0]):
        return SIM_OUT_BASE_COUPLED
    if _is_relative_to(sim_csv, SIM_SEARCH_ROOTS[1]):
        return SIM_OUT_BASE_NOT_COUPLED
    return PLOTS_ROOT / "SIM_Other"


def pick_out_base_for_mpc(mpc_csv: Path) -> Path:
    """Wählt die Ausgabe-Basis je nach Herkunft der MPC-Datei."""
    if _is_relative_to(mpc_csv, MPC_SEARCH_ROOTS[0]):
        return MPC_OUT_BASE_COUPLED
    if _is_relative_to(mpc_csv, MPC_SEARCH_ROOTS[1]):
        return MPC_OUT_BASE_NOT_COUPLED
    return PLOTS_ROOT / "MPC_Other"


def _extract_energy_from_return(ret: Any) -> Dict[str, Optional[float]]:
    """
    Versucht robuste Extraktion der Energie-Kennzahlen aus dem Rückgabewert
    von plot_hp_pel_qdot_from_sim(...).

    Gibt ein Dict mit 'E_el_kWh' und 'E_th_kWh' zurück (oder NaN).
    """
    def _to_float(x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return np.nan

    E_el = np.nan
    E_th = np.nan

    if isinstance(ret, dict):
        for k in ["E_el_kWh", "E_el", "Elec_kWh", "E_e_kWh", "E_el_total_kWh"]:
            if k in ret:
                E_el = _to_float(ret[k])
                break
        for k in ["E_th_kWh", "Q_th_kWh", "Thermal_kWh", "E_th_total_kWh"]:
            if k in ret:
                E_th = _to_float(ret[k])
                break
    elif isinstance(ret, (tuple, list)) and len(ret) >= 2:
        # heuristisch: (E_el_kWh, E_th_kWh, ...)
        E_el = _to_float(ret[0])
        E_th = _to_float(ret[1])

    return {"E_el_kWh": E_el, "E_th_kWh": E_th}


# --- SIM-Batch: T/Valve/TBuf + HP Power/Qdot + Komfortkosten + Energiebedarf ---
def process_all_sim() -> None:
    sim_csvs = list(iter_files(SIM_SEARCH_ROOTS, "Design_0_sim_agent.csv"))
    if not sim_csvs:
        print("Keine SIM-Dateien gefunden.")
        return

    index_rows = []

    for sim_csv in sim_csvs:
        try:
            study = extract_study_from_parent_intgai(sim_csv)

            out_base = pick_out_base_for_sim(sim_csv)
            out_dir = out_base / study
            ensure_dir(out_dir)

            print(f"[SIM] {study} :: {sim_csv}")
            print(f"      -> out_dir: {out_dir}")

            # 1) Drei-Panel (T / Ventilstellung / TBuf-Set)
            plot_three_panel_with_yvalset(
                sim_csv_path=sim_csv,
                out_dir=out_dir,
                base_year=2015,
                skip_first_days=3,
                remove_first_point=True,
                t_end_days=None,
                set_band=2.0,
                show_plot=False,
                out_name=f"{study}__ThreePanel_T_Ventilstellung_TBufSet_A4.pdf",
            )

            # 2) HP P_el und Qdot (+ Energiebedarf via Rückgabewert, falls bereitgestellt)
            ret_energy = plot_hp_pel_qdot_from_sim(
                sim_csv=sim_csv,
                out_dir=out_dir,
                base_year=2015,
                scale="kW",
                skip_first_days=3,
                skip_mode="dataset_start",
                remove_first_point=True,
                show_plot=False,
            )
            energy = _extract_energy_from_return(ret_energy)

            # 3) Komfortkosten (nur für SIM)
            res_comf = compute_comfort_cost_from_sim_dates(
                sim_csv_path=sim_csv,
                out_dir=out_dir,
                base_year=2015,
                skip_first_days=3,
                remove_first_point=True,
                t_end_days=None,
                set_band=2.0,
                show_sammelplot=False,
            )
            print(f"      Komfort  J_total={res_comf['total']:.3f}  -> {res_comf['csv']}")

            index_rows.append({
                "category": out_base.name,     # Studies_coupled / Studies_not_coupled / SIM_Other
                "study": study,
                "sim_csv": str(sim_csv),
                "out_dir": str(out_dir),
                # Energiebedarf (kWh), falls extrahierbar
                "E_el_kWh": energy.get("E_el_kWh"),
                "E_th_kWh": energy.get("E_th_kWh"),
                # Komfort
                "summary_csv": str(res_comf["csv"]),
                "J_total": res_comf["total"],
            })

        except Exception as e:
            print(f"   -> FEHLER bei {sim_csv}: {e}")

    # Index-CSV schreiben (Energie + Komfort)
    if index_rows:
        df_idx = pd.DataFrame(index_rows).sort_values(by=["category", "study"])
        out_index = PLOTS_ROOT / "comfort_index_all_sim.csv"
        ensure_dir(out_index.parent)
        df_idx.to_csv(out_index, index=False)
        print(f"[SIM] Index geschrieben: {out_index}")


# --- MPC-Batch: Drei-Panel MPC (ohne Komfort, ohne Energiebedarf) ---
def process_all_mpc() -> None:
    mpc_csvs = list(iter_files(MPC_SEARCH_ROOTS, "Design_0_mpc_agent.csv"))
    if not mpc_csvs:
        print("Keine MPC-Dateien gefunden.")
        return

    for mpc_csv in mpc_csvs:
        try:
            rel_rel, matched_root = _relative_to_any(mpc_csv, MPC_SEARCH_ROOTS)
            rel_parts = rel_rel.parts if rel_rel else ()
            first_child = rel_parts[0] if rel_parts else mpc_csv.parent.name

            design_name = re.sub(r"^mpc_design_", "", first_child, flags=re.IGNORECASE)
            design_name = design_name.strip("_ -") or first_child

            out_base = pick_out_base_for_mpc(mpc_csv)
            out_dir = out_base / design_name
            ensure_dir(out_dir)

            print(f"[MPC] {design_name} :: {mpc_csv}")
            print(f"      -> out_dir: {out_dir}")

            plot_three_panel_mpc_with_yval_tbuf(
                mpc_csv_path=mpc_csv,
                out_dir=out_dir,
                base_year=2015,
                skip_first_days=3,
                remove_first_point=True,
                t_end_days=None,
                set_band=2.0,
                show_plot=False,
                out_name=f"{design_name}__ThreePanel_MPC_T_Ventil_TBufSet_A4.pdf",
            )

        except Exception as e:
            print(f"   -> FEHLER bei {mpc_csv}: {e}")


if __name__ == "__main__":
    process_all_sim()
    process_all_mpc()
    print("Fertig.")
