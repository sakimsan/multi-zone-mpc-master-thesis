"""Module with typical optimization contraints"""

from bes_rules.configs.optimization import OptimizationConstraint
from bes_rules.configs.inputs.custom_modifiers import HydraulicSeperatorModifier


class BivalenceTemperatureGreaterNominalOutdoorAirTemperature(OptimizationConstraint):
    """
    Constraint to only study bivalence temperatures above
    nominal outdoor air temperatures.
    """
    name: str = "TBiv must be greater TOdaNominal"

    def apply(self, df, input_config):
        mask = df.loc[:, "parameterStudy.TBiv"] >= input_config.weather.TOda_nominal
        return df.loc[mask]


class HydraulicSeperatorConstraint(OptimizationConstraint):
    """
    Constraint to only one storage volume if a hydraulic seperator is
    simulated, as VPerQFlow is not used in that case and, thus, has no influence
    on the system.
    """
    name: str = "Hydraulic seperator comes in one size"

    def apply(self, df, input_config):
        if input_config.modifiers is None:
            return df
        for modifier in input_config.modifiers:
            if isinstance(modifier, HydraulicSeperatorModifier):
                first_v = df.loc[:, "parameterStudy.VPerQFlow"].unique()[0]
                return df.loc[df.loc[:, "parameterStudy.VPerQFlow"] == first_v]
        return df
