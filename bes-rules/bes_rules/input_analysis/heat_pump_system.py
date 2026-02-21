import logging
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bes_rules.configs import InputConfig
from bes_rules.utils.functions import heating_curve
from bes_rules.utils.function_fit import create_linear_regression, plot_surface_of_function_fit
from bes_rules.utils.heat_pumps import load_vitocal250_COPs
from bes_rules.utils.functions import get_heating_threshold_temperature_for_building

logger = logging.getLogger(__name__)


def create_heat_pump_regression(df_cop: pd.DataFrame, df_Q: pd.DataFrame):
    cop = df_cop.to_numpy().flatten()
    Q = df_Q.to_numpy().flatten()
    TCon = np.concatenate([df_cop.columns.to_numpy() for _ in df_cop.index])
    TAir = np.array([[idx for _ in df_cop.columns] for idx in df_cop.index]).flatten()
    variables = {
        "TAir": TAir,
        "TCon": TCon,
        "TAir*TCon": TAir * TCon,
        "TAir**2": TAir * TAir,
        "TCon**2": TCon * TCon,
        "TAir**2*TCon": TAir * TAir * TCon,
        "TCon**2*TAir": TCon * TCon * TAir,
        "TAir**3": TAir * TAir * TAir,
        "TCon**3": TCon * TCon * TCon
    }
    create_linear_regression(variables=variables, y=cop, y_name="COP")
    create_linear_regression(variables=variables, y=Q, y_name="Q_HP_max")


class PartialHeatPump:
    THeaPumMax: np.ndarray

    @staticmethod
    def QConMax(TCon, TAir):
        raise NotImplemented

    @staticmethod
    def COP(TCon, TAir):
        raise NotImplemented

    @classmethod
    def get_THeaPumMax(cls, TOda: (np.ndarray, float)) -> (np.ndarray, float):
        if np.any(TOda < cls.THeaPumMax[:, 0].min()):
            raise ValueError("The operational envelope is not designed for these temperatures")
        return np.interp(TOda, cls.THeaPumMax[:, 0], cls.THeaPumMax[:, 1])

    @staticmethod
    def create_regressions():
        raise NotImplementedError


def plot_regressions(heat_pump: Type[PartialHeatPump]):
    plot_surface_of_function_fit(variable="QConMax", version="old", regression=heat_pump.QConMax)
    plot_surface_of_function_fit(variable="COP", version="old", regression=heat_pump.COP)


class VitoCal250(PartialHeatPump):
    THeaPumMax: np.ndarray = np.array([
        [253.15, 333.15],
        [263.15, 343.15],
        [313.15, 343.15],
    ])

    @staticmethod
    def QConMax(TCon, TAir):
        # r2_score was 0.9839573351890547
        return (
                2876607.476163592 +
                -28654.59952844908 * TAir +
                -2378.1437551061795 * TCon +
                -11.280419895419598 * TAir * TCon +
                109.2716100619158 * TAir ** 2 +
                13.118067919983522 * TCon ** 2 +
                -0.016922003609579406 * TAir ** 2 * TCon +
                0.033644674471247527 * TCon ** 2 * TAir +
                -0.12370531548481267 * TAir ** 3 +
                -0.024644808743204294 * TCon ** 3
        )

    @staticmethod
    def COP(TCon, TAir):
        # r2_score was 0.9883064477405125
        return (
                476.4000945531263 +
                -9.081195518584966 * TAir +
                3.1818547284970395 * TCon +
                0.0059883011957250765 * TAir * TCon +
                0.02948169734516407 * TAir ** 2 +
                -0.012002022174265 * TCon ** 2 +
                -1.8993258525723533e-05 * TAir ** 2 * TCon +
                4.264575616419464e-06 * TCon ** 2 * TAir +
                -2.659933891280275e-05 * TAir ** 3 +
                1.1326140565271967e-05 * TCon ** 3
        )

    @staticmethod
    def create_regressions():
        df_cop = load_vitocal250_COPs("cop_extrapolation")
        df_Q = load_vitocal250_COPs("QConMax")
        create_heat_pump_regression(df_cop=df_cop, df_Q=df_Q)


class OptiHorst(PartialHeatPump):
    THeaPumMax: np.ndarray = np.array([
        [253.15, 323.15],
        [263.15, 333.15],
        [303.15, 333.15],
        [308.15, 328.15],
    ])

    @staticmethod
    def QConMax(TCon, TAir):
        # r2_score was 0.9839573351890547
        return (
                2876607.476159788 +
                -28654.59952847836 * TAir +
                -2378.1437550409805 * TCon +
                -11.280419896129132 * TAir * TCon +
                109.27161006224023 * TAir ** 2 +
                13.118067919952246 * TCon ** 2 +
                -0.01692200360872269 * TAir ** 2 * TCon +
                0.03364467447178983 * TCon ** 2 * TAir +
                -0.12370531548478049 * TAir ** 3 +
                -0.02464480874336303 * TCon ** 3
        )

    @staticmethod
    def COP(TCon, TAir):
        # r2_score was 0.9883064477405125
        return (
                476.40009453031337 +
                -9.081195518532432 * TAir +
                3.181854728465573 * TCon +
                0.005988301196612106 * TAir * TCon +
                0.029481697344723643 * TAir ** 2 +
                -0.01200202217436286 * TCon ** 2 +
                -1.8993258526443118e-05 * TAir ** 2 * TCon +
                4.264575616152461e-06 * TCon ** 2 * TAir +
                -2.659933891254294e-05 * TAir ** 3 +
                1.1326140565359479e-05 * TCon ** 3
        )

    @staticmethod
    def create_regressions():
        df_cop = load_vitocal250_COPs("cop_extrapolation")
        df_Q = load_vitocal250_COPs("QConMax")
        create_heat_pump_regression(df_cop=df_cop, df_Q=df_Q)


def estimate_heat_pump_system_demands(
        Q_demand_DHW: pd.Series,
        Q_demand_building: pd.Series,
        TOda_mean: pd.Series,
        input_config: InputConfig,
        use_hybrid: bool,
        objectives: dict,
        heat_pump: Type[PartialHeatPump],
        QHeaPumDHW_flow_nominal: float,
        THyd_nominal: float,
        dTHyd_nominal: float,
        TBiv: float = None,
        TCutOff: float = None,
        etaBiv: float = 0.97,
        save_path_plot: Path = None,
        with_vdi4645: bool = False
):
    T_DHW = 273.15 + 50
    TRoom = input_config.user.room_set_temperature
    TOda_nominal = input_config.weather.TOda_nominal

    TSupply = heating_curve(
        TOda=TOda_mean,
        TRoom=TRoom,
        TOda_nominal=TOda_nominal,
        TSup_nominal=THyd_nominal
    )
    THeaPumMaxAtTOda = heat_pump.get_THeaPumMax(TOda=TOda_mean)
    mask_TOutsideMax = TSupply > THeaPumMaxAtTOda
    TSupplyHeaPum = TSupply.copy()
    TSupplyHeaPum[mask_TOutsideMax] = THeaPumMaxAtTOda

    if TCutOff < TOda_mean[mask_TOutsideMax].max():
        logger.warning(
            "Supply temperatures higher than allowed values, "
            "making operation likely impossible below %s °C. Case %s",
            round(TOda_mean[mask_TOutsideMax].max() - 273.15, 1), input_config.get_name()
        )
    bivalent_parallel_design = not (use_hybrid or TCutOff > -np.inf or np.any(mask_TOutsideMax))

    QDem_flow_nominal = input_config.building.get_heating_load(
        TOda_nominal=TOda_nominal,
        TRoom_nominal=TRoom
    )
    # Heat Pump Design
    QHeaPumBiv_flow = QDem_flow_nominal * (TRoom - TBiv) / (TRoom - TOda_nominal)
    QHeaPumBiv_flow += QHeaPumDHW_flow_nominal
    TSupplyAtBiv = heating_curve(
        TOda=TBiv,
        TRoom=TRoom,
        TOda_nominal=TOda_nominal,
        TSup_nominal=THyd_nominal
    )
    TSupplyAtBiv = min(TSupplyAtBiv, heat_pump.get_THeaPumMax(TOda=TBiv))
    scaling_factor = QHeaPumBiv_flow / heat_pump.QConMax(TCon=TSupplyAtBiv, TAir=TBiv)

    QHeaPumMax_flow = heat_pump.QConMax(TCon=TSupplyHeaPum, TAir=TOda_mean) * scaling_factor
    QHeaPumBui_flow = np.minimum(QHeaPumMax_flow, Q_demand_building)
    QHeaPumBui_flow[TOda_mean < TCutOff] = 0
    zero_series = pd.Series(0, index=TOda_mean.index)
    QHeaPum_flow_remaining = np.maximum(QHeaPumMax_flow - Q_demand_building, zero_series)
    # Operation
    QBivBui_flow = np.maximum(Q_demand_building - QHeaPumBui_flow, zero_series)
    # Further add temperature outside operation envelope
    QBivBui_flow += (TSupply - TSupplyHeaPum) * QDem_flow_nominal / dTHyd_nominal
    PBivBui = QBivBui_flow / etaBiv
    PHeaPumBui = QHeaPumBui_flow / heat_pump.COP(TCon=TSupplyHeaPum, TAir=TOda_mean)

    QHeaPumDHW_flow = np.minimum(QHeaPum_flow_remaining, Q_demand_DHW)
    QHeaPumDHW_flow[TOda_mean < TCutOff] = 0
    # Operation
    QBivDHW_flow = np.maximum(Q_demand_DHW - QHeaPumDHW_flow, zero_series)
    PBivDHW = QBivDHW_flow / etaBiv
    PHeaPumDHW = QHeaPumDHW_flow / heat_pump.COP(TCon=T_DHW, TAir=TOda_mean)

    # Design size
    # TODO-Assumption: Note A-7W55 based on GEG and full load
    heat_pump_size = heat_pump.QConMax(TCon=273.15 + 55, TAir=273.15 - 7) * scaling_factor

    # TODO-Assumption: Note assumption of idealized heating rod size and potential issues in extreme weather
    QHeaPumAtOdaNom = heat_pump.QConMax(
        TCon=min(THyd_nominal, heat_pump.get_THeaPumMax(TOda=TOda_nominal)),
        TAir=TOda_nominal
    ) * scaling_factor
    if bivalent_parallel_design:
        biv_size = max(
            0,
            QDem_flow_nominal + QHeaPumDHW_flow_nominal - QHeaPumAtOdaNom
        )
    else:
        biv_size = QDem_flow_nominal + QHeaPumDHW_flow_nominal
    if biv_size / etaBiv < max(PBivBui + PBivDHW):
        oversize_requirement = (
                (Q_demand_building + Q_demand_DHW).max() /
                (QDem_flow_nominal + QHeaPumDHW_flow_nominal)
        )
        if oversize_requirement <= 1:
            missing_size = max(PBivBui + PBivDHW) - biv_size / etaBiv
            percent_to_small = missing_size / (QDem_flow_nominal + QHeaPumDHW_flow_nominal) * 100
            if percent_to_small < 0.5 or missing_size < 100 or TOda_mean.min() < TOda_nominal:
                logger.info(
                    "Bivalent device is not big enough at TBiv=%s °C, "
                    "Missing size: %s W, %s percent of load. Case %s",
                    round(TBiv - 273.15, 1), round(missing_size, 1),
                    percent_to_small, input_config.get_name()
                )
            else:
                logger.error(
                    '{"TBiv": %s, "Missing Size": %s, "Percent to small": %s}' % (
                        TBiv - 273.15, round(missing_size, 1), percent_to_small
                    )
                )

    if use_hybrid:
        gas_boiler_size = biv_size
        heating_rod_size = 0
        PEle_dhw = PHeaPumDHW
        PEle_bui = PHeaPumBui
        gas_demand = np.sum(PBivBui + PBivDHW) * 3600  # In J
        heating_rod_electricity_demand = 0
    else:
        gas_boiler_size = 0
        heating_rod_size = biv_size
        PEle_dhw = PHeaPumDHW + PBivDHW
        PEle_bui = PHeaPumBui + PBivBui
        heating_rod_electricity_demand = np.sum(PBivBui + PBivDHW) * 3600  # In J
        gas_demand = 0

    if save_path_plot is not None:
        fig, ax = plt.subplots(4, 1, sharex=True)
        ax[0].plot(TOda_mean - 273.15, color="blue")
        ax[0].set_ylabel("$T_\mathrm{Oda}$ in °C")
        ax[1].plot(Q_demand_building / 1000, label="Demand", color="blue")
        ax[1].plot(QHeaPumMax_flow / 1000, label="HP", color="red")
        ax[1].plot(QBivBui_flow / 1000, label="AH", color="red", linestyle="--")
        ax[1].legend()
        ax[1].set_ylabel("$\dot{Q}_\mathrm{Bui}$ in kW")
        ax[2].plot(Q_demand_DHW / 1000, label="Demand", color="blue")
        ax[2].plot(QHeaPum_flow_remaining / 1000, label="HP", color="red")
        ax[2].plot(QBivDHW_flow / 1000, label="AH", color="red", linestyle="--")
        ax[2].set_ylabel("$\dot{Q}_\mathrm{DHW}$ in kW")
        ax[2].legend()
        ax[3].plot((PHeaPumDHW + PHeaPumBui) / 1000, label="HP", color="red")
        ax[3].plot((PBivBui + PBivDHW) / 1000, label="AH", color="red", linestyle="--")
        ax[3].set_ylabel("$P$ in kW")
        ax[3].legend()
        ax[3].set_xlabel("Time")
        fig.tight_layout()
        fig.savefig(save_path_plot.joinpath(f"TBiv={TBiv - 273.15}.png"))

    mapping_scop = objectives["SCOP"].mapping
    mapping_annuity = objectives["Annuity"].mapping

    eps_hp = min(1, QHeaPumAtOdaNom / QDem_flow_nominal)

    sizing_results = {
        mapping_annuity.heat_pump_size: heat_pump_size,
        mapping_annuity.heating_rod_size: heating_rod_size,
        mapping_annuity.gas_boiler_size: gas_boiler_size,
        mapping_annuity.buffer_storage_size: 23.5e-6 * QDem_flow_nominal,
        "eps_hp": eps_hp
    }

    results = {
        **sizing_results,
        mapping_annuity.gas_demand: gas_demand,
        mapping_scop.heat_pump_electricity_demand: np.sum(PHeaPumDHW + PHeaPumBui) * 3600,  # In J
        mapping_scop.heating_rod_electricity_demand: heating_rod_electricity_demand,
        mapping_scop.heat_pump_heat_supplied: np.sum(QHeaPumBui_flow + QHeaPumDHW_flow) * 3600,  # In J
        mapping_scop.heating_rod_heat_supplied: heating_rod_electricity_demand * etaBiv,  # In J
    }

    if not with_vdi4645:
        return PEle_bui, PEle_dhw, results

    total_heat_demand = (np.sum(Q_demand_building) + np.sum(Q_demand_DHW))

    alpha_dhw = np.sum(Q_demand_DHW) / total_heat_demand
    if THyd_nominal > 273.15 + 55:
        dTCon_nominal = 10
    elif THyd_nominal >= 273.15 + 45:
        dTCon_nominal = 8
    else:
        dTCon_nominal = 5
    from bes_rules.input_analysis.vdi_4650 import get_SCOP

    SCOP_WPA, SCOP_H, SCOP_W, alpha_hp = get_SCOP(
        heat_pump=heat_pump,
        TOda_nominal=TOda_nominal,
        TSup_nominal=THyd_nominal,
        THeaThr=get_heating_threshold_temperature_for_building(building=input_config.building),
        dTCon_nominal=dTCon_nominal,
        eps=eps_hp,
        TCutOff=TCutOff,
        dTCon_measured=dTCon_nominal,
        inverter=True,
        alpha_dhw=alpha_dhw,
        hybrid=use_hybrid,
        operation="alternative" if use_hybrid and TCutOff is not None else "parallel"
    )

    heat_pump_heat_supplied = alpha_hp * total_heat_demand * 3600  # In J

    SCOP_WP = 1 / (
            (1 - alpha_dhw) / SCOP_H +
            alpha_dhw / SCOP_W
    )

    heat_pump_electricity_demand = heat_pump_heat_supplied / SCOP_WP
    if use_hybrid:
        heating_rod_electricity_demand = 0
        gas_demand = (1 - alpha_hp) * total_heat_demand * 3600
    else:
        gas_demand = 0
        heating_rod_electricity_demand = (1 - alpha_hp) * total_heat_demand * 3600
    results_vdi = {
        **sizing_results,
        mapping_annuity.gas_demand: gas_demand,
        mapping_scop.heat_pump_electricity_demand: heat_pump_electricity_demand,
        mapping_scop.heating_rod_electricity_demand: heating_rod_electricity_demand,
        mapping_scop.heat_pump_heat_supplied: heat_pump_heat_supplied,
        mapping_scop.heating_rod_heat_supplied: heating_rod_electricity_demand * etaBiv,  # In J
    }
    return PEle_bui, PEle_dhw, results, results_vdi


if __name__ == '__main__':
    OptiHorst.create_regressions()
