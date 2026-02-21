import numpy as np
import logging

from vclibpy.components.heat_exchangers import InternalHeatExchanger, ntu
from vclibpy.components.heat_exchangers.heat_transfer.heat_transfer import HeatTransfer, TwoPhaseHeatTransfer
from vclibpy import Inputs, FlowsheetState

logger = logging.getLogger(__name__)


class IHX_SinglePhase(InternalHeatExchanger):
    """
    Logic for an internal heat exchanger in counter flow arrangement.
    The regime logics are depicted here: docs/source/ihx_logic.svg
    """

    def __init__(
            self,
            alpha_low_side: float,
            alpha_high_side: float,
            dT_min: float = 10,
            **kwargs):
        super().__init__(**kwargs)
        self.dT_min = dT_min
        assert self.flow_type == "counter", "Other types are not implemented"
        self.k = self.calc_k(
            alpha_pri=alpha_high_side,
            alpha_sec=alpha_low_side
        )

    def calc(self, inputs: Inputs, fs_state: FlowsheetState) -> (float, float):
        m_flow_primary_cp = self.m_flow_low * self.med_prop.calc_transport_properties(self.state_inlet_low).cp
        m_flow_secondary_cp = self.m_flow_high * self.med_prop.calc_transport_properties(self.state_inlet_high).cp
        dT_max = self.state_inlet_high.T - self.state_inlet_low.T
        Q_ntu = ntu.calc_Q_ntu(
            dT_max=dT_max,
            k=self.k,
            A=self.A,
            m_flow_primary_cp=m_flow_primary_cp,
            m_flow_secondary_cp=m_flow_secondary_cp,
            flow_type=self.flow_type
        )
        self.set_missing_states(Q=Q_ntu)
        return None, None  # Irrelevant for this heat exchanger for now.

    def set_missing_states(self, Q: float):
        self.state_outlet_low = self.med_prop.calc_state(
            "PH",
            self.state_inlet_low.p,
            self.state_inlet_low.h + Q / self.m_flow_low
        )
        self.state_outlet_high = self.med_prop.calc_state(
            "PH",
            self.state_inlet_high.p,
            self.state_inlet_high.h - Q / self.m_flow_high
        )
