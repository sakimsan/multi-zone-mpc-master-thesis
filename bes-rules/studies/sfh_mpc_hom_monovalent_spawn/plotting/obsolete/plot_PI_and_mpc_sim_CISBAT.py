import matplotlib.pyplot as plt
from pathlib import Path
from ebcpy import TimeSeriesData
from bes_rules import RESULTS_FOLDER
from agentlib_mpc.utils.analysis import load_sim, load_mpc_stats
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_sim_stats(mpc_path: str, pi_path: str, start_time_in_days: int = 24, stop_time_in_days: int = 31,
                   save_png: bool = True, save_pdf: bool = False, save_svg: bool = False, save_dir: str = None,
                   label_bounds: bool = False):
    """
    Create publication-quality plots comparing MPC and RBC simulation results.

    Parameters:
    -----------
    mpc_path : str
        Path to the MPC simulation results
    pi_path : str
        Path to the RBC (Rule-Based Control) simulation results
    start_time_in_days : int, default=24
        Start time for plotting in days
    stop_time_in_days : int, default=31
        Stop time for plotting in days
    save_png, save_pdf, save_svg : bool
        Options to save the figure in different formats
    save_dir : str, optional
        Directory to save figures, defaults to current working directory
    label_bounds : bool
        Option to label the comfort bounds

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : list
        List of axes objects
    """
    # Set scientific style with LaTeX if available
    if plt.rcParams.get('text.usetex', False):
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath,amssymb,amsfonts,textcomp,mathptmx}',
        })
    else:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
        })

    # General style settings for publication quality
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.8,  # Thinner axes borders
        'grid.linewidth': 0.6,  # Thinner grid lines
        'lines.linewidth': 1.2,  # Default line width
        'lines.markersize': 4,  # Smaller markers
        'xtick.major.width': 0.8,  # Thinner ticks
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 3.0,  # Shorter ticks
        'ytick.major.size': 3.0,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.direction': 'in',  # Ticks pointing inward
        'ytick.direction': 'in',
    })

    # Create figure with adjusted aspect ratio for better use of space
    n_plots = 4
    fig, axes = plt.subplots(n_plots, 1, figsize=(7, 5.2), sharex=True)

    # Load data
    df_mpc_sim = load_sim(Path(mpc_path))
    tsd_sim = TimeSeriesData(Path(pi_path))

    # Convert seconds to days for time index
    df_mpc_sim.index = df_mpc_sim.index / (24 * 60 * 60)
    tsd_sim.index = tsd_sim.index / (24 * 60 * 60)

    # Filter time range
    df_mpc_sim = df_mpc_sim[(df_mpc_sim.index >= start_time_in_days) & (df_mpc_sim.index <= stop_time_in_days)]
    tsd_sim = tsd_sim[(tsd_sim.index >= start_time_in_days) & (tsd_sim.index <= stop_time_in_days)]

    # Improved color definitions for better contrast in grayscale printing
    mpc_color = '#8B0000'  # dark red
    rbc_color = '#707070'  # darker gray for better contrast with white background
    comfort_bound_color = '#404040'  # dark gray for comfort bounds
    outdoor_temp_color = '#004C99'  # dark blue for outdoor temperature

    # Define bound line properties
    bound_props = {
        'color': comfort_bound_color,
        'linestyle': '--',
        'alpha': 0.7,
        'linewidth': 0.8,
        'zorder': 1  # ensure bounds are behind the data
    }

    # Plot creation function to reduce repetition
    def create_temperature_plot(ax, mpc_data, rbc_data, title, y_label, bounds=None, y_padding=0.5, y_min=None,
                                y_max=None, title_loc='center'):
        ax.plot(df_mpc_sim.index, mpc_data, label='MPC', color=mpc_color, linewidth=1.3, zorder=3)
        ax.plot(tsd_sim.index, rbc_data, label='RBC', color=rbc_color, linewidth=1.3, zorder=2)

        if bounds:
            for bound in bounds:
                ax.axhline(y=bound, **bound_props)

        ax.set_title(title, horizontalalignment=title_loc)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.6)

        # Add minor grid for better readability
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        # Set y-limits with provided values or calculate from data
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        elif bounds:
            data_min = min(min(mpc_data), min(rbc_data))
            data_max = max(max(mpc_data), max(rbc_data))
            lower_bound = min(bounds) if bounds else data_min
            upper_bound = max(bounds) if bounds else data_max

            # Include data and bounds with padding
            y_min = min(data_min, lower_bound) - y_padding
            y_max = max(data_max, upper_bound) + y_padding
            ax.set_ylim(y_min, y_max)

        return ax

    def set_integer_ticks(ax, major_step: int = 5, minor_step: float = 1.0):
        ax.yaxis.set_major_locator(MultipleLocator(major_step))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_step))

    # First subplot: Weighted Mean Air Temperature
    weighted_temp_mpc = df_mpc_sim['TZoneAreaWeighted.y'] - 273.15
    weighted_temp_rbc = tsd_sim.loc[:, ('TZoneAreaWeighted.y', 'raw')] - 273.15
    lower_bound_c = 292.06 - 273.15  # ~18.91°C
    upper_bound_c = 296.06 - 273.15  # ~22.91°C

    create_temperature_plot(
        axes[0],
        weighted_temp_mpc,
        weighted_temp_rbc,
        'Area-weighted mean air temperature',
        'Temperature in °C',
        bounds=[lower_bound_c, upper_bound_c],
        title_loc='center'
    )

    # Second subplot: Temperature in the living room
    living_temp_mpc = df_mpc_sim['building.groundFloor.TZoneMea[1]'] - 273.15
    living_temp_rbc = tsd_sim.loc[:, ('building.groundFloor.TZoneMea[1]', 'raw')] - 273.15

    create_temperature_plot(
        axes[1],
        living_temp_mpc,
        living_temp_rbc,
        'Living room air temperature',
        'Temperature in °C',
        bounds=[20, 24]
    )

    # Third subplot: Temperature of top layer of buffer storage
    buffer_temp_mpc = df_mpc_sim['hydraulic.distribution.stoBuf.layer[4].T'] - 273.15
    buffer_temp_rbc = tsd_sim.loc[:, ('hydraulic.distribution.stoBuf.layer[4].T', 'raw')] - 273.15

    # Use extended y-range for buffer storage temperature (with more padding)
    create_temperature_plot(
        axes[2],
        buffer_temp_mpc,
        buffer_temp_rbc,
        'Buffer storage top layer temperature',
        'Temperature in °C',
        bounds=[20, 70],
        y_min=15,  # Extend lower range
        y_max=75  # Extend upper range
    )

    # Fourth subplot: Outdoor air temperature
    outdoor_temp = df_mpc_sim['outputs.weather.TDryBul'] - 273.15

    axes[3].plot(df_mpc_sim.index, outdoor_temp, color=outdoor_temp_color, linewidth=1.3)
    axes[3].set_title('Outdoor air temperature')
    axes[3].set_ylabel('Temperature in °C')
    axes[3].grid(True, linestyle=':', alpha=0.6, linewidth=0.6)
    axes[3].yaxis.set_minor_locator(AutoMinorLocator(2))
    axes[3].xaxis.set_minor_locator(AutoMinorLocator(2))

    # X-axis formatting for bottom subplot
    axes[3].set_xlabel('Time in days')

    # Set major tick intervals for days
    axes[3].xaxis.set_major_locator(MultipleLocator(1))

    # Ensure x-limits are set to the exact range
    for ax in axes:
        ax.set_xlim(start_time_in_days, stop_time_in_days)

    # Add red shaded areas for grid constraint periods (3 AM to 9 AM each day)
    for day in range(int(start_time_in_days), int(stop_time_in_days) + 1):
        grid_constraint_start = day + 3 / 24  # 3 AM
        grid_constraint_end = day + 9 / 24  # 9 AM

        # Only add shading if the constraint period is within our plot range
        if grid_constraint_start <= stop_time_in_days and grid_constraint_end >= start_time_in_days:
            # Trim to plot boundaries if needed
            actual_start = max(grid_constraint_start, start_time_in_days)
            actual_end = min(grid_constraint_end, stop_time_in_days)

            # Add shading to all subplots
            for ax in axes:
                ax.axvspan(actual_start, actual_end, alpha=0.10, color='red', zorder=0)

    # First apply tight_layout to properly position the subplots
    plt.tight_layout()

    # Get the position of the rightmost edge of the subplots
    # This ensures the legend is aligned with the right edge of the plots
    right_edge = axes[0].get_position().x1

    # Create custom elements for the legend
    mpc_line = Line2D([0], [0], color=mpc_color, linewidth=1.3, label='MPC')
    rbc_line = Line2D([0], [0], color=rbc_color, linewidth=1.3, label='RBC')
    boundary_line = Line2D([0], [0], color=comfort_bound_color, linestyle='--',
                            linewidth=0.8, label='Boundaries')
    deactivation_patch = Patch(facecolor='red', alpha=0.1,
                             label='Deactivations')

    # Combine all legend elements
    legend_elements = [mpc_line, rbc_line, boundary_line, deactivation_patch]

    # Add a single legend at the top, aligned with the right edge of plots
    leg = fig.legend(handles=legend_elements, loc='upper right',
                     bbox_to_anchor=(right_edge + 0.008, 1.01),
                     ncol=2, frameon=True, fancybox=False, edgecolor='black',
                     borderaxespad=0.5, columnspacing=1.0)

    # Improve legend appearance
    leg.get_frame().set_linewidth(0.8)

    # Remove individual legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # set y-axis ticks
    set_integer_ticks(axes[0], major_step=2, minor_step=1)
    set_integer_ticks(axes[1], major_step=2, minor_step=1)
    set_integer_ticks(axes[2], major_step=20, minor_step=10)
    set_integer_ticks(axes[3], major_step=5, minor_step=2.5)

    # Label the comfort bounds directly instead of using an arrow
    if label_bounds:
        # Add small text label beside one of the dashed lines
        axes[0].text(
            x=start_time_in_days + 0.2,  # Position near the start of plot
            y=upper_bound_c + 0.2,  # Just above the upper bound
            s='Comfort bounds',
            fontsize=7,
            ha='left',
            va='bottom'
        )

    # Adjust layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, top=0.915)

    # Save figure in high resolution if requested
    if save_dir is None:
        save_dir = os.getcwd()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename_base = 'sim_plot_CISBAT2025'

    if save_pdf:
        plt.savefig(os.path.join(save_dir, f'{filename_base}.pdf'),
                    format='pdf', dpi=600, bbox_inches='tight')
    if save_png:
        plt.savefig(os.path.join(save_dir, f'{filename_base}.png'),
                    format='png', dpi=600, bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(save_dir, f'{filename_base}.svg'),
                    format='svg', dpi=600, bbox_inches='tight')
    plt.show()
    return fig, axes

if __name__ == "__main__":
    mpc_sim_results_path = RESULTS_FOLDER.joinpath(
        "SFH_MPCRom_monovalent_spawn",
        "mpc_design_CISBAT2025_6hgrid_48hpred_3600s",   # 6 h grid constrain
        "DesignOptimizationResults",
        "TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai_",
        "Design_1_sim_agent.csv"   # 35 L/kW buffer storage
    )
    pi_sim_results_path = RESULTS_FOLDER.joinpath(
        "SFH_PIctrl_monovalent_spawn",
        "PI_design_CISBAT2025_6hgrid",   # 6 h grid constrain
        "VPerQFlow_23.5.mat",   # 35 L/kW buffer storage
    )
    plot_sim_stats(
        mpc_path=mpc_sim_results_path,
        pi_path=pi_sim_results_path,
        start_time_in_days=24,
        stop_time_in_days=31,
        save_dir="./figures",
        save_pdf=False,
        save_png=True,
        save_svg=False,
        label_bounds=False
    )