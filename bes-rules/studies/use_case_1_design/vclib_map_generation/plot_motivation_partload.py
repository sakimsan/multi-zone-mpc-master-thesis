import matplotlib.pyplot as plt

from bes_rules.configs.plotting import PlotConfig
import numpy as np
from bes_rules.utils.functions import heating_curve
from bes_rules.plotting.utils import get_figure_size

PART_LOAD_EFFECTS = [
    "const", "inc", "pol"
]


def get_eta_carnot(TOda, TSup, part_load_effect: str):
    assert part_load_effect in PART_LOAD_EFFECTS
    return np.ones(len(TOda)) * 0.3


def plot_motivation_part_load():
    QBui_flow_nominal = 10000
    TOda_nominal = 273.15 - 12
    TRoom_nominal = 273.15 + 20
    TOda = np.linspace(TOda_nominal, TRoom_nominal - 5, 100)
    UA_nominal = QBui_flow_nominal / (TRoom_nominal - TOda_nominal)
    QBui_flow = UA_nominal * (TRoom_nominal - TOda)
    designs = {
        "monovalent": (TOda_nominal, "blue"),
        "bivalent": (273.15 - 2, "gray")
    }
    TSup = heating_curve(
        TOda=TOda, TRoom=TRoom_nominal, TOda_nominal=TOda_nominal,
        TSup_nominal=273.15 + 55, TBase_nominal=273.15 + 25
    )
    n_min = 0.3
    n_max = 1

    fig, ax = plt.subplots(5, 1, sharex=True, figsize=get_figure_size(2, 3))
    ax[-1].set_xlabel("$T_\mathrm{Oda}$\n  in °C")
    ax[0].set_ylabel("$p_\mathrm{Oda}$\n  in %")
    ax[1].set_ylabel("$\dot{Q}$\n  in W")
    ax[2].set_ylabel("$n$\n  in %")
    ax[3].set_ylabel("$COP$\n in W")
    ax[4].set_ylabel("$\eta_\mathrm{Carnot}$\n in %")
    ax[1].plot(TOda - 273.15, QBui_flow, label="Demand", color="red")
    part_load_results = {}
    for part_load_effect in PART_LOAD_EFFECTS:
        eta_carnot = get_eta_carnot(TOda, TSup, part_load_effect=part_load_effect)
        COPCarnot = eta_carnot * TSup / (TSup - TOda)
        part_load_results[part_load_effect] = {"eta": eta_carnot, "COP": COPCarnot}
        ax[4].plot(TOda - 273.15, eta_carnot * 100, label=part_load_effect)
    COPCarnot = part_load_results["const"]["COP"]
    eta_carnot = part_load_results["const"]["eta"]
    ax[3].plot(TOda - 273.15, COPCarnot, color="red")

    for name, args in designs.items():
        T_biv, color = args
        QHeaPum_flow_nominal = UA_nominal * (TRoom_nominal - T_biv)
        TSup_biv = heating_curve(
            TOda=T_biv, TRoom=TRoom_nominal, TOda_nominal=TOda_nominal,
            TSup_nominal=273.15 + 55, TBase_nominal=273.15 + 25
        )
        COPCarnot_biv = eta_carnot * TSup_biv / (TSup_biv - T_biv)
        PEle_biv = QHeaPum_flow_nominal / COPCarnot_biv
        QHeaPum_flow_min = PEle_biv * COPCarnot * n_min
        QHeaPum_flow_max = PEle_biv * COPCarnot * n_max
        n = np.max([
            np.min([QHeaPum_flow_max, QBui_flow], axis=0),
            QHeaPum_flow_min], axis=0
        ) / QHeaPum_flow_max
        ax[1].plot(TOda - 273.15, QHeaPum_flow_min, label=f"{name} min", color=color, linestyle="-.")
        ax[1].plot(TOda - 273.15, QHeaPum_flow_max, label=f"{name} min", color=color, linestyle="-.")
        ax[2].plot(TOda - 273.15, n, label=name, color=color)

    ax[1].set_ylim([0, QBui_flow_nominal * 1.1])
    ax[1].set_yticks([])
    ax[1].legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax[2].legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax[3].legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax[4].legend(bbox_to_anchor=(1, 1), loc="upper left")

    fig.tight_layout()
    fig.align_ylabels()
    plt.show()


if __name__ == '__main__':
    PlotConfig.load_default()
    plot_motivation_part_load()
