import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

zones = ["livingroom", "kitchen", "hobby", "wcstorage", "corridor", "bedroom", "children", "corridor2", "bath",
         "attic", "children2"]


def load_and_prepare_data(filepath):

    """Loads XLSX data, sets time index, and prepares for analysis."""
    df = pd.read_excel(filepath, engine='openpyxl', index_col=0).iloc[10:] # Skip first steps
    return df


def parse_and_categorize_columns(df):
    """Parses column names to categorize them by type and room."""
    # (Implementation from previous script is sufficient)
    categorized = defaultdict(list)

    for col in df.columns:
        if col == 'T_amb' or col == 'Time': continue

        room = col.split('_')[-1]

        if not any([room.startswith(zone) for zone in zones]): continue

        if col.startswith('Q_RadSol'):
            categorized['solar'].append(col)
        elif 'T_Air' in col:
            categorized['air'].append(col); categorized[f'air_{room}'].append(col)
        elif 'T_ExtWall_sur' in col:
            categorized['ext_wall_sur'].append(col); categorized[f'ext_wall_sur_{room}'].append(col)
        elif 'T_ExtWall' in col:
            categorized['ext_wall'].append(col); categorized[f'ext_wall_{room}'].append(col)
        elif 'T_IntWall_sur_out' in col:
            categorized['int_wall_sur'].append(col); categorized[f'int_wall_sur_{room}'].append(col)
        elif 'T_IntWall' in col:
            categorized['int_wall'].append(col); categorized[f'int_wall_{room}'].append(col)
        elif 'T_Floor_sur_out' in col:
            categorized['floor_ceiling_sur'].append(col); categorized[f'floor_ceiling_sur_{room}'].append(col)
        elif 'T_Floor' in col or 'T_Decke' in col:
            categorized['floor_ceiling'].append(col); categorized[f'floor_ceiling_{room}'].append(col)
        elif 'T_Roof' in col:
            categorized['roof'].append(col); categorized[f'roof_{room}'].append(col)
        else:
            categorized['other'].append(col)

    return categorized


# ==============================================================================
# PLAUSIBILITY CHECKS (UPDATED FOR SOLAR)
# ==============================================================================

def check_1_thermodynamic_boundaries(df, internal_cols, no_sun_mask):
    """Check if temps stay within physical bounds, accounting for solar radiation."""
    print("\n--- CHECK 1: Thermodynamic Boundaries ---")
    print("Verifying that internal temperatures adhere to physical laws.")

    # CHECK 1A: Temperatures must never drop below ambient (unless there's active cooling)
    # This check is ALWAYS valid, with or without sun.
    print("\n  CHECK 1A: No temperature should be below ambient.")
    below_ambient = df[internal_cols].lt(df['T_amb'], axis=0).any().any()
    if below_ambient:
        violation_time = df[df[internal_cols].lt(df['T_amb'], axis=0).any(axis=1)].index[0]
        print(f"    [FAIL] Physical Violation: An internal temperature dropped below ambient.")
        print(f"           First violation occurred around: {violation_time}")
        print("           This is physically impossible without an active cooling source.")
    else:
        print("    [PASS] All internal temperatures remained above or at ambient temperature.")

    # CHECK 1B: During NO-SUN periods, temperatures must not rise.
    # This check is ONLY applied to the filtered "no-sun" data.
    print("\n  CHECK 1B: During periods with NO solar radiation, temperatures must not rise.")
    if no_sun_mask.sum() == 0:
        print("    [SKIP] No periods without solar radiation were found. Cannot perform this check.")
        return

    df_no_sun = df[no_sun_mask]
    temp_diff_no_sun = df_no_sun[internal_cols].diff()

    # Check for any temperature increase (with a small tolerance for numerical noise)
    spontaneous_rise = (temp_diff_no_sun > 0.01).any().any()

    if spontaneous_rise:
        violation_time = temp_diff_no_sun[(temp_diff_no_sun > 0.01).any(axis=1)].index[0]
        print(f"    [FAIL] Physical Violation: An internal temperature rose without any solar input.")
        print(f"           First violation occurred during a no-sun period around: {violation_time}")
        print("           This is impossible without an internal heat source.")
    else:
        print("    [PASS] No temperatures rose during periods without solar radiation.")


def check_2_convergence_trend(df, internal_cols, no_sun_mask):
    """Check if temperatures converge towards ambient during no-sun periods."""
    print("\n--- CHECK 2: Convergence Trend (during no-sun periods) ---")
    if no_sun_mask.sum() < 2:
        print("  [SKIP] Not enough data during no-sun periods to check for convergence.")
        return

    df_no_sun = df[no_sun_mask]
    temp_diff = df_no_sun[internal_cols].sub(df_no_sun['T_amb'], axis=0)

    # The standard deviation of temperatures should decrease over time as they converge
    std_dev_trend = temp_diff.std(axis=1).diff().mean()

    if std_dev_trend < 0:
        print("  [PASS] The spread of internal temperatures correctly decreases during no-sun periods.")
    else:
        print("  [WARN] The spread of internal temperatures does not consistently decrease during no-sun periods.")
        print("         This could indicate an issue or simply a fluctuating T_amb during those times.")


def check_3_room_consistency(df, categories, room):
    """Check if air temperature is bounded by its surrounding surfaces. ALWAYS VALID."""
    # (This function remains the same as it's always physically true)
    print(f"\n--- CHECK 3: Room Consistency for '{room}' ---")
    print("Verifying that room air temperature is bounded by its surface temperatures.")

    air_col = categories.get(f'air_{room}', [None])[0]
    surface_cols = (categories.get(f'ext_wall_{room}', []) +
                    categories.get(f'int_wall_{room}', []) +
                    categories.get(f'floor_ceiling_{room}', []) +
                    categories.get(f'roof_{room}', []))

    if not air_col or not surface_cols:
        print(f"  [SKIP] Could not find sufficient air/surface data for room '{room}'.")
        return

    room_df = df[[air_col] + surface_cols]
    surface_min = room_df[surface_cols].min(axis=1)
    surface_max = room_df[surface_cols].max(axis=1)

    is_below = (room_df[air_col] < surface_min - 0.1).any()
    is_above = (room_df[air_col] > surface_max + 0.1).any()

    if is_below or is_above:
        print(f"  [FAIL] Physical Violation: T_Air in '{room}' is not bounded by its surfaces.")
        if is_below: print("         At some point, T_Air was significantly colder than all surfaces.")
        if is_above: print("         At some point, T_Air was significantly hotter than all surfaces.")
        print("         This is physically impossible, even with solar radiation.")
    else:
        print("  [PASS] T_Air is correctly bounded by its surrounding surface temperatures.")


# ==============================================================================
# PLOTTING FUNCTIONS (UPDATED TO SHOW SUN PERIODS)
# ==============================================================================

def plot_with_sun_periods(ax, sun_mask):
    """Helper function to draw shaded regions on a plot for sun periods."""
    sun_periods = sun_mask.astype(int).diff().fillna(0)
    starts = sun_periods[sun_periods == 1].index
    ends = sun_periods[sun_periods == -1].index

    if isinstance(ax, plt.Axes):
        ax = [ax]

    # Handle cases where sun period starts/ends outside the plot range
    if len(starts) > len(ends): ends = ends.append(pd.Index([sun_mask.index[-1]]))
    if len(ends) > len(starts): starts = pd.Index([sun_mask.index[0]]).append(starts)

    for start, end in zip(starts, ends):
        for ax_ in ax:
            ax_.axvspan(start, end, facecolor='yellow', alpha=0.2, label='_nolegend_')
    # Add one dummy for the legend
    if len(starts) > 0:
        for ax_ in ax:
            ax_.axvspan(starts[0], starts[0], facecolor='yellow', alpha=0.3, label='Solar Radiation > 0')


def plot_1_overall_dynamics(df, categories, sun_mask):
    """Plot average temperatures, indicating sun periods."""
    print("\n--- Plotting 1: Overall Building Dynamics (with Solar Periods) ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    plot_with_sun_periods(ax, sun_mask)

    # Calculate averages
    if categories['air']: ax.plot(df.index, df[categories['air']].mean(axis=1), label='Avg Air Temp', lw=2)
    if categories['ext_wall']: ax.plot(df.index, df[categories['ext_wall']].mean(axis=1), label='Avg Ext Wall Temp',
                                       lw=2)
    if categories['int_wall']: ax.plot(df.index, df[categories['int_wall']].mean(axis=1), label='Avg Int Wall Temp',
                                       lw=1.5, alpha=0.8)
    if categories['floor_ceiling']: ax.plot(df.index, df[categories['floor_ceiling']].mean(axis=1),
                                            label='Avg Floor/Ceiling Temp', lw=2, linestyle=':')
    ax.plot(df.index, df['T_amb'], color='k', linestyle='--', label='Ambient (T_amb)')

    ax.set_title('Overall Building Thermal Dynamics', fontsize=16)
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Time')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    plt.show()


def plot_2_thermal_inertia(df, categories, room, sun_mask, save_path):
    """Plot component types in one room, indicating sun periods."""
    print(f"\n--- Plotting 2: Thermal Inertia in '{room}' (with Solar Periods) ---")
    fig, ax = plt.subplots(3, figsize=(7, 13))
    plot_with_sun_periods(ax, sun_mask)

    air_cols = categories[f'air_{room}']
    ext_wall_cols = categories[f'ext_wall_{room}']
    int_wall_cols = categories[f'int_wall_{room}']
    floor_cols = categories[f'floor_ceiling_{room}']
    ext_wall_sur_cols = categories[f'ext_wall_sur_{room}']
    int_wall_sur_cols = categories[f'int_wall_sur_{room}']
    floor_sur_cols = categories[f'floor_ceiling_sur_{room}']

    def _plot_all_temps(_ax, _df, cols, label, remove="", **kwargs):
        from bes_rules.plotting import EBCColors
        palette = [EBCColors.blue, EBCColors.dark_red, EBCColors.grey, "black", EBCColors.green, "yellow"]
        for idx, col in enumerate(cols):
            name = col.replace(room, "").replace(remove, "").replace("_", "")
            _ax.plot(_df.index, -273.15 + _df[col], label=f'{label} {name}', color=palette[idx], **kwargs)

    ax[0].plot(df.index, -273.15 + df['T_amb'], color='k', linestyle='--', label='Ambient (T_amb)')
    _plot_all_temps(ax[0], df, ext_wall_cols, "Ext Wall", remove="T_ExtWall_OuterWall", )
    _plot_all_temps(ax[0], df, ext_wall_sur_cols, "Ext Wall Sur", remove="T_ExtWall_sur_out_OuterWall", marker="^", )

    _plot_all_temps(ax[1], df, int_wall_cols, "Int Wall", remove="T_IntWall_InnerWall", )
    _plot_all_temps(ax[1], df, int_wall_sur_cols, "Int Wall Sur", remove="T_IntWall_sur_out_InnerWall", marker="^", )

    _plot_all_temps(ax[2], df, floor_cols, "Floor/Ceiling", remove="T_Floor_", )
    _plot_all_temps(ax[2], df, floor_sur_cols, "Floor/Ceiling Sur", remove="T_Floor_sur_out", marker="^", )

    fig.suptitle(f'Thermal Inertia Comparison in "{room}"', fontsize=16)
    ax[-1].set_xlabel('Time')
    for ax_ in ax:
        ax_.plot(df.index, -273.15 + df[air_cols].sum(axis=1) / len(air_cols), label=f'Air ({room})', lw=2.5, color="red")
        ax_.set_ylabel('Temperature (°C)')
        ax_.legend()
        ax_.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path)
    #plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    PATH = pathlib.Path(r"D:\fwu-ssr\MonovalentVitoCal_HOM_2h_Abwesend\DesignOptimizationResults\TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai_\generated_configs_Design_0")

    ACH_CASE = "no_ach"
    #ACH_CASE = "05_ach"

    df_results = load_and_prepare_data(PATH.joinpath(f"sim_test_{ACH_CASE}.xlsx"))

    categorized_cols = parse_and_categorize_columns(df_results)
    all_internal_cols = [col for col in df_results.columns if col.startswith('T_') and col != 'T_amb']

    # --- Identify Solar Radiation Periods ---
    solar_cols = categorized_cols.get('solar', [])
    if solar_cols:
        total_solar_rad = df_results[solar_cols].sum(axis=1)
        sun_mask = total_solar_rad > 0.01  # True when sun is shining
        no_sun_mask = ~sun_mask  # True when there is no sun
        print(f"Found {len(solar_cols)} solar radiation columns.")
        print(f"Identified {sun_mask.sum()} timesteps with solar radiation and {no_sun_mask.sum()} without.")
    else:
        print("No solar radiation columns (Q_RadSol_or_*) found. Assuming no sun for all checks.")
        no_sun_mask = pd.Series(True, index=df_results.index)
        sun_mask = pd.Series(False, index=df_results.index)

    room_for_detailed_analysis = 'livingroom'

    # --- Perform Logical Checks ---
    #check_1_thermodynamic_boundaries(df_results, all_internal_cols, no_sun_mask)
    #check_2_convergence_trend(df_results, all_internal_cols, no_sun_mask)
    #check_3_room_consistency(df_results, categorized_cols, room_for_detailed_analysis)

    # --- Generate Plots ---
    #plot_1_overall_dynamics(df_results, categorized_cols, sun_mask)
    for zone in zones:
        plot_2_thermal_inertia(df_results, categorized_cols, zone, sun_mask, save_path=PATH.joinpath(f"{ACH_CASE}_{zone}.png"))
