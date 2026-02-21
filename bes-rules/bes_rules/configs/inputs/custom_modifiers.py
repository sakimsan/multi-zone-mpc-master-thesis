from typing import TYPE_CHECKING
from bes_rules.configs.inputs.base import BaseInputConfig

if TYPE_CHECKING:
    from bes_rules.configs import InputConfig


class CustomModifierConfig(BaseInputConfig):
    name: str
    modifier: str

    def get_name(self):
        return self.name

    def get_modelica_modifier(self, input_config: "InputConfig"):
        return self.modifier


class HydraulicSeperatorModifier(CustomModifierConfig):
    name: str = "HydraulicSeperator"
    modifier: str = "hydraulic(distribution(redeclare BESRules.RecordsCollection.HydraulicSeperator parStoBuf))"


class OnOffControlModifier(CustomModifierConfig):
    name: str = "OnOff"
    modifier: str = (
        "hydraulic(control("
        "redeclare BESMod.Systems.Hydraulical.Control.Components.RelativeSpeedController.OnOff priGenPIDCtrl))"
    )


class NoMinimalCompressorSpeed(CustomModifierConfig):
    name: str = "100"
    modifier: str = "hydraulic(control(parPIDHeaPum(yMin=0.01)))"


class NoModifier(CustomModifierConfig):
    # Useful to compare a modifier with the default model in a full factor input variation
    name: str = ""
    modifier: str = ""


class StartLossModifier(CustomModifierConfig):
    name: str = "StartLoss"
    modifier: str = "enableInertia=true"


if __name__ == '__main__':
    OnOffControlModifier()
