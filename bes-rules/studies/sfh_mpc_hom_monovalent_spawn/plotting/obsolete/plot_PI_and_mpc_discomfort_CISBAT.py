from bes_rules import RESULTS_FOLDER
from studies_ssr.sfh_PIctrl_monovalent_spawn.plotting.plot_PI_discomfort_CISBAT import load_discomfort_dict_json
from studies_ssr.sfh_mpc_hom_monovalent_spawn.postprocessing.save_mpc_discomfort_results import get_heated_zone_names


def plot_combined_discomfort_values(pi_data, mpc_data,
                                    pi_zone='sum', mpc_zones=['sum', 'sum'],
                                    titles=["PI Controller", "MPC sum", "MPC one zone"],
                                    show_values=True, show_vdi_line=False,
                                    save_dir=None, filename="combined_thermal_discomfort",
                                    save_pdf=True, save_png=True, save_svg=False,
                                    figsize=(18, 7)):
    """
    Plot thermal discomfort datasets side by side with shared colorbar.
    Each column across all diagrams has equal width, making plots with fewer columns narrower.
    Uses correctly inverted y-axis and proper data alignment.

    Parameters:
    -----------
    pi_data : dict
        The PI controller dataset
    mpc_data : dict
        The MPC controller dataset
    pi_zone : str
        Zone to use for PI visualization (default: 'sum')
    mpc_zones : list
        Zones to use for MPC visualizations (default: ['sum', 'sum'])
    titles : list
        Titles for each subplot
    show_values : bool
        Whether to display numerical values in cells
    show_vdi_line : bool
        Whether to show VDI 4645 recommendations (only on PI results)
    save_dir : str
        Directory to save figures (if None, saves in current working directory)
    filename : str
        Base filename for saved figures
    save_pdf, save_png, save_svg : bool
        Flags to determine which formats to save
    figsize : tuple
        Figure size (width, height) in inches
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    import pandas as pd
    import numpy as np
    import os
    import re

    # Define function to extract duration from constraint strings
    def extract_duration(constraint_str):
        """Extracts duration from a constraint string"""
        match = re.search(r'(\d+)h', constraint_str)
        if match:
            return match.group(1)  # Returns just the number
        return None

    # Process each dataset to create the dataframes
    def process_data(data, zone_name='sum'):
        if zone_name not in data:
            print(f"Warning: Zone '{zone_name}' not found in data. Available zones: {list(data.keys())}")
            return None

        zone_data = data[zone_name]

        # Get buffer sizes and grid constraints
        buffer_sizes = list(zone_data.keys())
        grid_constraints = list(zone_data[buffer_sizes[0]].keys())

        # Extract grid time values
        grid_time_values = []
        for constraint in grid_constraints:
            duration = extract_duration(constraint)
            if duration:
                grid_time_values.append(duration)

        # Create DataFrame
        df = pd.DataFrame(index=grid_time_values, columns=buffer_sizes)

        # Fill DataFrame
        for buf_size in buffer_sizes:
            for i, grid_constraint in enumerate(grid_constraints):
                try:
                    df.loc[grid_time_values[i], buf_size] = float(zone_data[buf_size][grid_constraint])
                except (ValueError, TypeError, IndexError, KeyError) as e:
                    print(f"Warning: Error processing value: {e}")
                    df.loc[grid_time_values[i], buf_size] = 0.0

        # Convert to float and sort
        df = df.astype(float)
        df = df.sort_index(axis=1, key=lambda x: x.astype(float))

        # Sort index by numerical value, with higher values first (6h at the top, 2h at the bottom)
        df = df.sort_index(axis=0, key=lambda x: x.astype(int), ascending=True)

        return df

    # Process all datasets
    df_pi = process_data(pi_data, pi_zone)
    df_mpc1 = process_data(mpc_data, mpc_zones[0])
    df_mpc2 = process_data(mpc_data, mpc_zones[1])

    # Verify that we have valid dataframes
    if df_pi is None or df_mpc1 is None or df_mpc2 is None:
        print("Error: One or more datasets could not be processed. Aborting plot.")
        return None, None

    # Set up improved publication-quality parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 24,  # Larger base font
        'axes.labelsize': 24,  # Larger axis labels
        'axes.titlesize': 24,  # Larger title
        'xtick.labelsize': 24,  # Larger tick labels
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 24,
        'axes.linewidth': 1.5,
        'grid.linewidth': 1.0,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

    # Create a custom color palette
    colors = ["#f7f4f0", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#990000"]
    cmap = LinearSegmentedColormap.from_list("publication_YlOrRd", colors, N=256)

    # Use the specified boundaries
    bounds = [0, 2, 5, 10, 20, 50, 100, 200, 400]
    norm = BoundaryNorm(bounds, cmap.N)

    # Count columns in each dataframe to determine width ratios
    pi_cols = len(df_pi.columns)
    mpc1_cols = len(df_mpc1.columns)
    mpc2_cols = len(df_mpc2.columns)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Calculate the proper width ratios
    total_width = pi_cols + mpc1_cols + mpc2_cols
    cbar_width = 0.15  # Further reduced width for the colorbar

    # Create a fixed space between plots (in figure fraction) - INCREASED
    fixed_space = 0.04

    # Calculate the effective widths including the fixed spaces
    pi_width = pi_cols / total_width * (1 - 2 * fixed_space - cbar_width)
    mpc1_width = mpc1_cols / total_width * (1 - 2 * fixed_space - cbar_width)
    mpc2_width = mpc2_cols / total_width * (1 - 2 * fixed_space - cbar_width)

    # Create custom subplot positions
    ax1 = fig.add_axes([0.05, 0.15, pi_width, 0.75])
    ax2 = fig.add_axes([0.05 + pi_width + fixed_space, 0.15, mpc1_width, 0.75])
    ax3 = fig.add_axes([0.05 + pi_width + mpc1_width + 2 * fixed_space, 0.15, mpc2_width, 0.75])
    cax = fig.add_axes([0.05 + pi_width + mpc1_width + mpc2_width + 3 * fixed_space, 0.15, cbar_width / 8, 0.75])

    # Function to create a heatmap manually with consistent spacing
    def create_manual_heatmap(ax, df, show_values=True, value_fontsize=20):
        rows, cols = len(df.index), len(df.columns)

        # Create a mesh grid of coordinates
        x, y = np.meshgrid(np.arange(cols + 1), np.arange(rows + 1))

        # Get the data as a 2D array
        data = df.values

        # Create a mesh for pcolormesh
        pc = ax.pcolormesh(x, y, data, cmap=cmap, norm=norm, edgecolors='black', linewidth=1.5)

        # Add value annotations if requested
        if show_values:
            for i in range(rows):
                for j in range(cols):
                    value = data[i, j]
                    ax.text(j + 0.5, i + 0.5, f"{value:.1f}", ha='center', va='center',
                            fontsize=value_fontsize, color='black')

        # Set the correct ticks
        ax.set_xticks(np.arange(0.5, cols, 1))
        ax.set_yticks(np.arange(0.5, rows, 1))

        # Set tick labels
        ax.set_xticklabels(df.columns, fontsize=20)
        ax.set_yticklabels(df.index, fontsize=20)

        # Set proper limits to show all cells
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)

        return pc

    # Create heatmaps on each subplot
    pc1 = create_manual_heatmap(ax1, df_pi, show_values)
    pc2 = create_manual_heatmap(ax2, df_mpc1, show_values)
    pc3 = create_manual_heatmap(ax3, df_mpc2 * 10, show_values)   # multiply with number of heated zones

    # Add titles to each subplot
    ax1.set_title(titles[0], fontsize=24, pad=15)
    ax2.set_title(titles[1], fontsize=24, pad=15)
    ax3.set_title(titles[2], fontsize=24, pad=15)

    # Add axis labels
    ax1.set_xlabel('Storage size in L·kW$^{-1}$', fontsize=24, labelpad=10)
    ax2.set_xlabel('Storage size in L·kW$^{-1}$', fontsize=24, labelpad=10)
    ax3.set_xlabel('Storage size in L·kW$^{-1}$', fontsize=24, labelpad=10)

    # Set y-label only for the first plot
    ax1.set_ylabel('Grid constrained period in h', fontsize=24, labelpad=10)

    # Add VDI 4645 line if requested
    if show_vdi_line:
        vdi_recommendations = {
            '2': '70.0',  # 2h → 70 L/kW
            '3': '105.0',  # 3h → 105 L/kW
            '4': '140.0',  # 4h → 140 L/kW
            '5': '175.0',  # 5h → 175 L/kW
            '6': '210.0'  # 6h → 210 L/kW
        }

        # Find matching points for VDI line
        vdi_points = []

        for grid_time, storage_size in vdi_recommendations.items():
            if grid_time in df_pi.index:
                # Get row position
                row_idx = df_pi.index.get_loc(grid_time)

                # Find closest storage size column
                closest_storage_size = min(df_pi.columns,
                                           key=lambda x: abs(float(x) - float(storage_size)))
                col_idx = list(df_pi.columns).index(closest_storage_size)

                vdi_points.append((col_idx + 0.5, row_idx + 0.5))

        # Draw the line if we have enough points
        if len(vdi_points) >= 2:
            # Sort by y-coordinate (row index)
            vdi_points.sort(key=lambda p: p[1])
            x_coords, y_coords = zip(*vdi_points)

            # Draw the diagonal line
            ax1.plot(x_coords, y_coords, color='steelblue', linewidth=2.5,
                     linestyle='-', zorder=10)

            # Add marker dots
            ax1.scatter(x_coords, y_coords, color='steelblue', s=40,
                        zorder=11, marker='o')

            # Add label at the top left point (6h point)
            label_point_idx = 4  # 6h point (index 4 if starting from 2h)
            if len(vdi_points) > label_point_idx:
                ax1.annotate('VDI 4645', xy=(x_coords[label_point_idx], y_coords[label_point_idx]),
                             xytext=(45, 17), textcoords='offset points',
                             color='steelblue', fontsize=18, ha='right',
                             bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=3),
                             zorder=12)

    # Create a colorbar
    cbar = fig.colorbar(pc1, cax=cax)
    cbar.set_label('Thermal discomfort in K·h', fontsize=24, labelpad=15)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels([str(int(b)) for b in bounds])
    cbar.ax.tick_params(labelsize=20)

    # Save figures if requested
    if save_dir is None:
        save_dir = os.getcwd()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create full file paths
    value_text = "with_values" if show_values else "without_values"
    vdi_text = "with_vdi" if show_vdi_line else "without_vdi"
    zone_text = f"{pi_zone}_{mpc_zones[0]}_{mpc_zones[1]}"
    filename_base = f"{filename}_{zone_text}_{value_text}_{vdi_text}"

    pdf_path = os.path.join(save_dir, f'{filename_base}.pdf')
    png_path = os.path.join(save_dir, f'{filename_base}.png')
    svg_path = os.path.join(save_dir, f'{filename_base}.svg')

    # Save the figure in various formats
    if save_pdf:
        plt.savefig(pdf_path, dpi=600, bbox_inches='tight', transparent=False)
    if save_png:
        plt.savefig(png_path, dpi=600, bbox_inches='tight', transparent=False)
    if save_svg:
        try:
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"SVG figure saved to {save_dir}")
        except Exception as e:
            print(f"SVG export failed: {e}")

    if save_pdf or save_png:
        print(f"Figures saved to {save_dir}")

    return fig, (ax1, ax2, ax3)


if __name__ == "__main__":
    # load the saved dicts with discomfort results for MPC and PI
    mpc_discomfort_dict = load_discomfort_dict_json(filename_path=RESULTS_FOLDER.joinpath("SFH_MPCRom_monovalent_spawn", "discomfort_data.json"))
    pi_discomfort_dict = load_discomfort_dict_json(filename_path=RESULTS_FOLDER.joinpath("SFH_PIctrl_monovalent_spawn", "discomfort_data.json"))
    # load zone names
    zone_names = get_heated_zone_names(with_sum=True, with_TZoneAreaWeighted=True)
    # create heat map with discomfort values of zone names of PI and MPC
    plot_combined_discomfort_values(
        pi_data=pi_discomfort_dict,  # Dein PI-Controller-Datensatz
        mpc_data=mpc_discomfort_dict,  # Dein MPC-Datensatz
        pi_zone='sum',  # Zone für PI-Controller
        mpc_zones=['sum', 'TZoneAreaWeighted'],  # Verschiedene Zonen für MPC-Plots
        titles=["RBC: Sum zones", "MPC: Sum zones", "MPC: Mean zones"],  # Benutzerdefinierte Titel
        show_values=True,
        show_vdi_line=True,
        save_dir="./figures",
        filename="thermal_discomfort_comparison",
        save_pdf=False,
        save_png=True,
        save_svg=False
    )