import logging

from vclibpy.flowsheets import BaseCycle
from vclibpy.datamodels import FlowsheetState, Inputs
from vclibpy.components.compressors import Compressor
from vclibpy.components.expansion_valves import ExpansionValve

from studies.use_case_1_design.vclib_map_generation.ihx_ntu_single_phase import IHX_SinglePhase

logger = logging.getLogger(__name__)


class IHXNew(BaseCycle):
    """
    Class for a IHX cycle with internal heat exchanger (ihx).

    For the standard cycle, we have 4 possible states:

    1. Before compressor, after ihx low temperature side
    2. Before condenser, after compressor
    3. Before first EV, after condenser
    4. Before ihx high temperature side, after first EV
    5. Before second EV, after ihx high temperature side
    6. Before evaporator, after second EV
    7. Before idx low temperature side, after evaporator
    """

    flowsheet_name = "IHXNew"

    def __init__(
            self,
            compressor: Compressor,
            expansion_valve_high: ExpansionValve,
            expansion_valve_low: ExpansionValve,
            ihx: IHX_SinglePhase,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.compressor = compressor
        self.expansion_valve_high = expansion_valve_high
        self.expansion_valve_low = expansion_valve_low
        self.ihx = ihx
        self.flowsheet_name = "IHX"

    def get_all_components(self):
        return super().get_all_components() + [
            self.compressor,
            self.expansion_valve_high,
            self.expansion_valve_low,
            self.ihx
        ]

    def get_states_in_order_for_plotting(self):
        first_states = [
            self.compressor.state_inlet,
            self.compressor.state_outlet,
            self.condenser.state_inlet,
            self.med_prop.calc_state("PQ", self.condenser.state_outlet.p, 1),
            self.med_prop.calc_state("PQ", self.condenser.state_outlet.p, 0),
            self.condenser.state_outlet,
            self.expansion_valve_high.state_inlet,
            self.expansion_valve_high.state_outlet,
            self.ihx.state_inlet_high
        ]
        second_part_states = [
            self.ihx.state_outlet_high,
            self.expansion_valve_low.state_inlet,
            self.expansion_valve_low.state_outlet,
            self.evaporator.state_inlet,
            self.evaporator.state_outlet,
            self.ihx.state_inlet_low,
        ]
        third_part_states = [
            self.ihx.state_outlet_low
        ]
        # ihx_high in tp region:
        state_ihx_high_q0 = self.med_prop.calc_state("PQ", self.ihx.state_inlet_high.p, 0)
        if state_ihx_high_q0.h < self.ihx.state_inlet_high.h:
            states_until_low_side = first_states + [state_ihx_high_q0] + second_part_states
        else:
            states_until_low_side = first_states + second_part_states
        state_ihx_low_q1 = self.med_prop.calc_state("PQ", self.ihx.state_inlet_low.p, 1)
        if state_ihx_low_q1.h > self.ihx.state_inlet_low.h:
            return states_until_low_side + [state_ihx_low_q1] + third_part_states
        return states_until_low_side + third_part_states

    def calc_states(self, p_1, p_2, inputs: Inputs, fs_state: FlowsheetState):
        # State 7
        self.set_evaporator_outlet_based_on_superheating(p_eva=p_1, inputs=inputs)
        self.ihx.state_inlet_low = self.evaporator.state_outlet
        # State 3
        self.set_condenser_outlet_based_on_subcooling(p_con=p_2, inputs=inputs)
        self.expansion_valve_high.state_inlet = self.condenser.state_outlet

        # In C10-polynoms, the superheat has no effect on the lambda_h.
        # So the only influence of a wrong superheat value for assumed_m_flow is the density.
        errors = {}  # Build data
        for dT_ihx_assumed in range(0, 30, 2):
            self._calc_for_assumptions(
                p_1=p_1, p_2=p_2, inputs=inputs,
                fs_state=fs_state, dT_ihx_assumed=dT_ihx_assumed
            )
            errors[dT_ihx_assumed] = self.ihx.state_outlet_low.T - self.ihx.state_inlet_low.T - dT_ihx_assumed
        if self.iteration_converged:
            log_func = logger.warning
        else:
            log_func = logger.debug
        dT_ihx_intersection = find_intersection(errors, log_func=log_func)

        self._calc_for_assumptions(
            p_1=p_1, p_2=p_2, inputs=inputs,
            fs_state=fs_state, dT_ihx_assumed=dT_ihx_intersection
        )
        self.expansion_valve_low.state_inlet = self.ihx.state_outlet_high
        # State 6
        self.expansion_valve_low.calc_outlet(p_outlet=p_1)
        self.evaporator.state_inlet = self.expansion_valve_low.state_outlet
        for no, state in enumerate([
            self.compressor.state_inlet,
            self.compressor.state_outlet,
            self.condenser.state_outlet,
            self.expansion_valve_high.state_outlet,
            self.ihx.state_outlet_high,
            self.expansion_valve_low.state_outlet,
            self.evaporator.state_outlet,
        ]):
            fs_state.set(name=f"T_{no+1}", value=state.T, unit="K", description=f"Temperature in state {no+1}")

        fs_state.set(name="p_con", value=p_2 / 1e5, unit="bar", description="Condensation pressure")
        fs_state.set(name="p_eva", value=p_1 / 1e5, unit="bar", description="Evaporation pressure")

    def _calc_for_assumptions(self, p_1, p_2, inputs: Inputs, fs_state: FlowsheetState, dT_ihx_assumed):
        assumed_ihx_outlet_low = self.med_prop.calc_state("PT", p_1, self.evaporator.state_outlet.T + dT_ihx_assumed)
        self.compressor.state_inlet = assumed_ihx_outlet_low
        self.compressor.calc_state_outlet(p_outlet=p_2, inputs=inputs, fs_state=fs_state)
        self.condenser.state_inlet = self.compressor.state_outlet
        # Mass flow rate:
        self.compressor.calc_m_flow(inputs=inputs, fs_state=fs_state)
        self.condenser.m_flow = self.compressor.m_flow
        self.expansion_valve_high.m_flow = self.compressor.m_flow
        self.ihx.m_flow = self.compressor.m_flow
        self.expansion_valve_low.m_flow = self.compressor.m_flow
        self.evaporator.m_flow = self.compressor.m_flow
        self.ihx.m_flow_high = self.compressor.m_flow
        self.ihx.m_flow_low = self.compressor.m_flow
        # State 4
        if "opening" in inputs.control.get_variable_names():
            opening = inputs.control.opening
        else:
            opening = 1
        try:
            self.expansion_valve_high.calc_outlet_pressure_at_m_flow_and_opening(
                m_flow=self.compressor.m_flow,
                opening=opening
            )
        except ValueError as err:
            # During iteration, p_1 and p_2 can get very close together,
            # leading to high m_flows and then p_outlets below zero, depending on the EV model
            T_min = self.ihx.state_inlet_low.T + self.ihx.dT_min
            self.expansion_valve_high.calc_outlet(p_outlet=self.med_prop.calc_state("TQ", T_min, 0).p)
        dp_ev_high = self.expansion_valve_high.state_inlet.p - self.expansion_valve_high.state_outlet.p
        if self.expansion_valve_high.state_outlet.T - self.ihx.dT_min < self.ihx.state_inlet_low.T:
            T_min = self.ihx.state_inlet_low.T + self.ihx.dT_min
            self.expansion_valve_high.calc_outlet(p_outlet=self.med_prop.calc_state("TQ", T_min, 0).p)
            logger.info(
                "Pressure to low to at given opening to match dT_min=%s. "
                "Setting minimal required pressure."
            )
        fs_state.set(name="dp_ev_high", value=dp_ev_high / 1e5, unit="bar", description="Pressure difference due to first EV")

        self.ihx.state_inlet_high = self.expansion_valve_high.state_outlet
        # State 5
        self.ihx.calc(inputs=inputs, fs_state=fs_state)

    def calc_electrical_power(self, inputs: Inputs, fs_state: FlowsheetState):
        """Based on simple energy balance - Adiabatic"""
        return self.compressor.calc_electrical_power(inputs=inputs, fs_state=fs_state)


def find_intersection(data, log_func: callable):
    # Convert to list of (x,y) pairs and sort by x
    points = [(k, v) for k, v in data.items()]
    points.sort()
    # Find where x-y changes sign
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        # Calculate x-y for both points
        diff1 = x1 - y1
        diff2 = x2 - y2
        # If sign changes, we found our interval
        if diff1 * diff2 <= 0:
            # Linear interpolation
            # (x - x1)/(x2 - x1) = (y - y1)/(y2 - y1)
            # At intersection, x = y
            # Solve for x: (x - x1)/(x2 - x1) = (x - y1)/(y2 - y1)
            # x = (y1*x2 - y2*x1)/(y1 - y2 + x2 - x1)
            x = (y1 * x2 - y2 * x1) / (y1 - y2 + x2 - x1)
            return x
    log_func(
        "No intersection found for dT_ihx: %s. Using edge value.",
        y2
    )
    return y2
