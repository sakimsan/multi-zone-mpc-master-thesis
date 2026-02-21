import numpy as np

from bes_rules.plotting import utils
from bes_rules.utils.functions import heating_curve


def plot_m_flow_over_TBiv():
    QDem_nominal = 10
    TOda_nominal = -15 + 273.15
    THeaThr = 15 + 273.15
    TBiv = -15 + 273.15
    TSup_nominal = 70 + 273.15
    TOda = np.linspace(TOda_nominal, THeaThr, 100)
    QDem = QDem_nominal * (TOda - THeaThr) / (TOda_nominal - THeaThr)
    TSupply = heating_curve(
        TOda=TOda,
        TRoom=293.15,
        TOda_nominal=TOda_nominal,
        TSup_nominal=TSup_nominal
    )

    QBiv = QDem_nominal * (TBiv - THeaThr) / (TOda_nominal - THeaThr)
    from bes_rules.input_analysis.heat_pump_system import VitoCal250

    TSupplyAtBiv = heating_curve(
        TOda=TBiv,
        TRoom=293.15,
        TOda_nominal=TOda_nominal,
        TSup_nominal=TSup_nominal
    )
    scaling_factor = QBiv / VitoCal250.QConMax(TCon=TSupplyAtBiv, TAir=TBiv)
    QHeaPum_max = VitoCal250.QConMax(TSupply, TOda) * scaling_factor

    def get_dT_design(TSup: float):
        if TSup >= 273.15 + 55:
            return 10
        if TSup >= 273.15 + 45:
            return 8
        return 5

    m_flow_biv_at_every_oda = QDem / 4.184 / (
        np.array([get_dT_design(TSup) for TSup in TSupply])
    )

    dT_biv = get_dT_design(TSupplyAtBiv)
    dT_nom = get_dT_design(TSup_nominal)
    m_flow_biv = QBiv / dT_biv / 4.184
    m_flow_nom = QDem_nominal / dT_nom / 4.184
    dT_biv_ope_max = QHeaPum_max / m_flow_biv / 4.184
    dT_nom_ope_max = QHeaPum_max / m_flow_nom / 4.184
    dT_biv_ope_min = QHeaPum_max / m_flow_biv / 4.184 * 0.25
    dT_nom_ope_min = QHeaPum_max / m_flow_nom / 4.184 * 0.25
    x_variable = "$T_\mathrm{Auß}$ in °C"
    plot_config = utils.load_plot_config()

    fig, axes = utils.create_plots(
        plot_config=plot_config,
        x_variables=[x_variable],
        y_variables=[
            "$\dot{Q}$ in kW",
            "$\Delta T$ in K",
            "$\dot{m}$ in kg/s",
        ]
    )
    axes[0, 0].plot(TOda - 273.15, QDem, label="$\dot{Q}_\mathrm{Bed}$", color="red")
    axes[0, 0].plot(TOda - 273.15, QHeaPum_max, label="$\dot{Q}_\mathrm{WP,Max}$", color="blue")
    axes[0, 0].axhline(QDem_nominal, linestyle="--", label="$\dot{Q}_\mathrm{Bed,Nom}$")

    axes[1, 0].plot(TOda - 273.15, dT_nom_ope_max, label="Nominal-max", color="blue")
    axes[1, 0].plot(TOda - 273.15, dT_biv_ope_max, label="Bivalent-max", color="red")
    axes[1, 0].plot(TOda - 273.15, dT_nom_ope_min, label="Nominal-min", color="blue", linestyle="--")
    axes[1, 0].plot(TOda - 273.15, dT_biv_ope_min, label="Bivalent-min", color="red", linestyle="--")

    axes[2, 0].axhline(m_flow_nom, label="Nominal", color="blue")
    axes[2, 0].scatter(TOda - 273.15, m_flow_biv_at_every_oda, label="Bivalent", color="red")

    from bes_rules import LATEX_FIGURES_FOLDER
    for ax in axes[:, 0]:
        ax.axvline(TBiv - 273.15, label="$T_\mathrm{Biv}$", color="black")
        ax.legend(loc="upper left")

    utils.save(
        fig=fig, axes=axes,
        save_path=LATEX_FIGURES_FOLDER.joinpath("Appendix", "m_flow_design.png"),
        show=True, with_legend=False, file_endings=["png"]
    )


if __name__ == '__main__':
    plot_m_flow_over_TBiv()
