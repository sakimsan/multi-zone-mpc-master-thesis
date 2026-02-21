import matplotlib.pyplot as plt
import numpy as np

from ebcpy import TimeSeriesData
from agentlib_mpc.models.casadi_model import CasadiModel
import casadi as ca

from bes_rules import STARTUP_BESMOD_MOS
from bes_rules.utils.function_fit import fit_linear_regression


def radiator_no_exponent_outlet_temperature(
        casadi_model: CasadiModel,
        TTraSup: float,
        mTra_flow: float,
        T_Air
):
    # assumption stationary energy balance
    # no delay by volume elements/thermal inertia
    # sim: simple radiator model (EN442-2), no delay between buffer storage and heater
    # from energy balance and heat transfer
    return T_Air + (TTraSup - T_Air) * ca.exp(-casadi_model.UA_heater / (mTra_flow * casadi_model.cp_water))


def radiator_Q_flow_given_exponent_outlet_temperature(
        casadi_model: CasadiModel,
        TTraSup: float,
        mTra_flow: float,
        Q_flow: float,
        T_Air
):
    n = casadi_model.n_heater_exp
    return T_Air + (TTraSup - T_Air) * ca.exp(
        - casadi_model.UA_heater ** (1 / n) * Q_flow ** (1 - 1 / n) /
        (mTra_flow * casadi_model.cp_water)
    )


def create_regression_for_opening():
    """
    Simulate BESMod.Examples.GasBoilerBuildingOnly for e.g. 180 days first
    to generate data.

    Ideas:
    - Q_flow and TSup is given in optimization
    - Using the opening from simulation is not accurate, as TSup may be higher than the one in the simulation
    - if opening is fixed, TRet follows, but then hot water is fed into lower storage layer
    - if opening and TRet is free, various solutions exist --> opening must be a control variable.
    """
    y = "outputs.hydraulic.tra.opening[1]"
    Q_flow = "outputs.building.eneBal[1].traGain.value"
    TSup = "outputs.hydraulic.tra.TSup[1]"
    TRet = "outputs.hydraulic.tra.TRet[1]"
    TRoom = "outputs.building.TZone[1]"
    tsd = TimeSeriesData(STARTUP_BESMOD_MOS.parent.joinpath("working_dir", "GasBoilerBuildingOnly.mat"),
                         variable_names=[Q_flow, TSup, TRet, TRoom, y]).to_df().loc[86400 * 2:]

    delta_T_mean = (tsd.loc[:, TSup] + tsd.loc[:, TRet]) / 2 - tsd.loc[:, TRoom]
    dT_A = tsd.loc[:, TSup] - tsd.loc[:, TRoom]
    dT_B = tsd.loc[:, TRet] - tsd.loc[:, TRoom]
    delta_T_log = (dT_A - dT_B) / np.log(dT_A / dT_B)
    mask_nan = np.isnan(delta_T_log)
    plt.figure()
    plt.scatter(tsd.loc[:, Q_flow], tsd.loc[:, y])
    plt.xlabel("Q_flow")
    plt.ylabel("opening")
    plt.figure()
    plt.scatter(tsd.loc[:, Q_flow], tsd.loc[:, TSup])
    plt.xlabel("Q_flow")
    plt.ylabel("delta_T_log")
    # Theory is Q = opening * m_flow_nominal * cp * dT_log
    # So Q / dT_log should lead to linear regression
    nHeaTra = 1.3
    fit_linear_regression(
        variables=[tsd.loc[~mask_nan, Q_flow] / (delta_T_log.loc[~mask_nan] ** nHeaTra)],
        y=tsd.loc[~mask_nan, y]
    )


def understand_return_temperature_equation():
    """
    Simulate BESMod.Examples.GasBoilerBuildingOnly for e.g. 30 days
    with protected outputs to generate data and set n=1.001
    """
    y = "outputs.hydraulic.tra.opening[1]"
    Q_flow = "outputs.building.eneBal[1].traGain.value"
    TSup = "outputs.hydraulic.tra.TSup[1]"
    TRet = "outputs.hydraulic.tra.TRet[1]"
    TRet = "hydraulic.transfer.rad[1].vol[5].T"
    TRoom = "outputs.building.TZone[1]"
    UA = "hydraulic.transfer.rad[1].UAEle"
    m_flow_nominal = "hydraulic.transfer.m_flow_nominal[1]"
    m_flow = "hydraulic.transfer.portTra_in[1].m_flow"
    variable_names = [Q_flow, TSup, TRet, TRoom, y, UA, m_flow_nominal, m_flow]
    tsd = TimeSeriesData(
        STARTUP_BESMOD_MOS.parent.joinpath("working_dir", "GasBoilerBuildingOnly.mat"),
        variable_names=variable_names
    ).to_df().loc[86400 * 2:]

    y = tsd.loc[:, y].values
    Q_flow = tsd.loc[:, Q_flow].values
    TSup = tsd.loc[:, TSup].values
    TRet = tsd.loc[:, TRet].values
    TRoom = tsd.loc[:, TRoom].values
    UA = tsd.loc[:, UA].values * 5
    m_flow = tsd.loc[:, m_flow].values
    # m_flow = tsd.loc[:, m_flow_nominal].values * y
    cp_water = 4184
    TTraRet_equation = TRoom + (TSup - TRoom) * np.exp(- UA / (m_flow * cp_water))
    TTraRet_chen = get_T_return_chen_and_underwood(
        TAir=TRoom, TSup=TSup, UA_Nom=UA, nHea=1.001, Q_flow=Q_flow
    )
    QTra_flow_equation = m_flow * cp_water * (TSup - TTraRet_equation)
    QTra_flow_chen = m_flow * cp_water * (TSup - TTraRet_chen)

    UA_equ = m_flow * cp_water * np.log(
        (TSup - TRoom) /
        (TRet - TRoom)
    )
    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(tsd.index, TSup, label="Supply")
    ax[0].plot(tsd.index, TRet, label="Ret-Real")
    ax[0].plot(tsd.index, TTraRet_equation, label="Ret-Equ")
    ax[0].plot(tsd.index, TTraRet_chen, label="Ret-chen", linestyle="--")
    ax[0].set_ylabel("T")
    ax[1].plot(tsd.index, y)
    ax[1].set_ylabel("opening")
    ax[2].plot(tsd.index, Q_flow, label="Q-Real")
    ax[2].plot(tsd.index, QTra_flow_equation, label="Q-Equ")
    ax[2].plot(tsd.index, QTra_flow_chen, label="Q-chen", linestyle="--")
    ax[2].legend()
    ax[2].set_ylabel("Q_flow")
    ax[3].plot(tsd.index, UA_equ, label="Equ")
    ax[3].plot(tsd.index, UA, label="Real")
    ax[3].legend()
    ax[3].set_ylabel("UA")
    ax[-1].set_xlabel("Time")
    ax[0].legend()
    plt.show()


def get_T_return_chen_and_underwood(
        TAir: float,
        nHea: float,
        TSup: float,
        Q_flow: float,
        UA_Nom: float
):
    """
    Based on: https://www.sciencedirect.com/science/article/pii/0009250987801288
    """
    n_chen = 0.3275
    dTSup = (TSup - TAir)
    dTRet = (2 * (Q_flow / UA_Nom) ** (n_chen / nHea) - dTSup ** n_chen) ** (1 / n_chen)
    return dTRet + TAir


if __name__ == '__main__':
    understand_return_temperature_equation()
