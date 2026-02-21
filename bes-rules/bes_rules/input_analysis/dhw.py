import matplotlib.pyplot as plt
import pandas as pd


def create_mean_hourly_dhw_profile(
        number_of_occupants: int,
        hourly_index: pd.DatetimeIndex,
        dhw_daily_per_person: float = 25,
):
    """
    Args:
        dhw_daily_per_person (float):
            Daily DHW tapping volume per person at 60 °C
    """
    dhw_daily = dhw_daily_per_person * 4184 * (60 - 10) / 3600  # 1,45 Wh/d, according to EN 15450
    Q_demand_DHW_daily = dhw_daily * number_of_occupants  # in Wh/d
    Q_demand_DHW = pd.DataFrame(index=hourly_index)
    mask_daytime = (Q_demand_DHW.index.hour >= 7) & (Q_demand_DHW.index.hour < 22)
    Q_demand_DHW.loc[:, 'Q_demand_DHW'] = 0.0
    num_entries_in_one_daytime = 22 - 7
    Q_demand_DHW.loc[mask_daytime, 'Q_demand_DHW'] = Q_demand_DHW_daily / num_entries_in_one_daytime
    return Q_demand_DHW


def dhw_design_EN_15450(
        number_of_occupants: int,
        dhw_daily_per_person: float,
        dhw_storage_design: str
):
    # TODO-Assumption: Mean QCrit
    """
    VPerDay QCrit Ratio
    248	4,445	0,017923387
    123	2,24	0,018211382
    43,5 0,945	0,021724138
                Mean:
                0,019286302
    Mean leads to bad fit, linear regression is better:
    QCrit = 0,0213*VDHWDayAt60_l+0,1421

    V	Qcrit	ErrrorRegression	ErrorMean
    200,756	4,445	-0,0267972	-0,573159089
    100,756	2,24	0,0482028	-0,296789322
    36,17	0,945	-0,032479	-0,247414445

    """
    if dhw_daily_per_person == 0:
        return 0, 0
    dhw_daily = dhw_daily_per_person * number_of_occupants
    QCrit = 0.0213 * dhw_daily + 0.1421
    TDHW_nominal = 273.15 + 50
    TDHWCold_nominal = 273.15 + 10
    rho_cp_kWh_in_l = 4184 / 3600000
    QDHWStoLoss_flow_estimate = calculate_loss_per_day(energy_label="A+", volume_l=dhw_daily)
    VStoDHWLos = (QDHWStoLoss_flow_estimate * 24 / 1000) / rho_cp_kWh_in_l / (TDHW_nominal - TDHWCold_nominal)
    if dhw_storage_design == "part_storage":
        tCrit = 3600
        dhw_storage_size = dhw_daily + VStoDHWLos
        # Label A+, taken from BESMod
        QLosPerDay = calculate_loss_per_day(energy_label="A+", volume_l=dhw_storage_size)
        QDHWStoLoss_flow = QLosPerDay / 24  # kWh
        QHeaPumDHW_flow_nominal = (
                (QCrit - (dhw_storage_size * rho_cp_kWh_in_l) * (TDHW_nominal - 313.15)) +
                QDHWStoLoss_flow * (tCrit / 3600)
        ) / tCrit * 1000 * 3600
    elif dhw_storage_design == "full_storage":
        tCrit = 3600 * 24
        dhw_storage_size = dhw_daily * 2 + VStoDHWLos
        QHeaPumDHW_flow_nominal = (
                (dhw_storage_size * rho_cp_kWh_in_l / tCrit) *
                (TDHW_nominal - TDHWCold_nominal) * 3600 * 1000
        )
    else:
        raise ValueError(f"{dhw_storage_design=} not supported")
    return dhw_storage_size, QHeaPumDHW_flow_nominal


def plot_dhw_insulation():
    import numpy as np
    from bes_rules.configs.plotting import PlotConfig
    from bes_rules.plotting.utils import get_figure_size
    from bes_rules import LATEX_FIGURES_FOLDER

    PlotConfig.load_default()

    # Values taken from DIN EN 15450
    volumes = np.array(
        [30, 50, 80, 100, 120, 150, 200, 300, 400, 500, 600,
         700, 800, 900, 1000, 1100, 1200, 1300, 1500, 2000]
    )
    losses = np.array(
        [0.75, 0.90, 1.1, 1.3, 1.4, 1.6, 2.1, 2.6, 3.1,
         3.5, 3.8, 4.1, 4.3, 4.5, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2]
    )

    fig, ax = plt.subplots(1, 1, figsize=get_figure_size(n_columns=1.5))
    ax.set_xlabel("$V$ in l")
    ax.set_ylabel("$Q_\mathrm{Ver,d}$ in kWh/d")
    labels = [
        'A+',
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
    ]
    for label in labels:
        ax.plot(volumes, calculate_loss_per_day(energy_label=label, volume_l=volumes), label=label)
    ax.plot(volumes, losses, label="EN Soll", color="gray", linestyle="--")
    # In Simulation, 50 °C is assumed, while EN 15450 specifies 45 K temperature difference
    ax.plot(volumes, volumes * 0.005 * 30 / 45, label="EN Min", color="blue", linestyle="--")
    ax.plot(volumes[:15], volumes[:15] * 0.015 * 30 / 45, label="EN Max", color="red", linestyle="--")

    ax.plot(volumes, 0.4 + 0.14 * volumes ** 0.5, label="DIN V 18599", color="black", linestyle="-.")

    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.1), mode="expand", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(LATEX_FIGURES_FOLDER.joinpath("Appendix", "Speicherverluste.png"))
    plt.show()


def calculate_loss_per_day(energy_label: str, volume_l: float) -> float:
    """
    Calculate daily heat loss based on energy label and volume

    Args:
        energy_label: Energy efficiency class ('A+' to 'G')
        volume_l: Volume in l

    Returns:
        Heat loss in kWh/day
    """
    labels = {
        'A+': (5.5, 3.16),
        'A': (7.0, 3.705),
        'B': (10.25, 5.09),
        'C': (14.33, 7.13),
        'D': (18.83, 9.33),
        'E': (23.5, 11.995),
        'F': (28.5, 15.16),
        'G': (31.0, 16.66)
    }

    if energy_label not in labels:
        raise KeyError("Label not supported")

    base, factor = labels[energy_label]
    return (base + factor * (volume_l ** 0.4)) * 0.024


if __name__ == '__main__':
    import logging
    logging.basicConfig(level="INFO")
    print(dhw_design_EN_15450(4, 25, "part_storage"))
    print(dhw_design_EN_15450(4, 25, "full_storage"))
    #plot_dhw_insulation()
