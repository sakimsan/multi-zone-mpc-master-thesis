from studies_ssr.sfh_mpc_hom_monovalent_spawn import base_design_optimization
from bes_rules import configs
from bes_rules import boundary_conditions
from bes_rules.configs.inputs import custom_modifiers


def get_optimization_config():
    return base_design_optimization.get_optimization_config(
        configs.OptimizationVariable(
            name="parameterStudy.TBiv",
            lower_bound=273.15 - 12,
            upper_bound=273.15 - 12,
            levels=1
        ),
        configs.OptimizationVariable(
            name="parameterStudy.VPerQFlow",
            lower_bound=12,
            upper_bound=12,
            levels=1
        )
    )

def get_inputs_config(
        inverter_uses_storage: bool,
        no_minimal_compressor_speed: bool,
        with_start_losses: bool,
        no_dhw: bool,
        model_hom_names: list = None,
        model_predictions_names: list = None,
        only_inverter: bool = False,
        hom: bool = False,
        existing_zone_record_path: str = None,

):
    weathers = boundary_conditions.weather.get_weather_configs_by_names(region_names=["Potsdam"])
    buildings = boundary_conditions.building.get_building_configs_by_name(
        building_names=["NoRetrofit1983"],
        modify_transfer_system=True
    )
    if inverter_uses_storage:
        inverter_modifier = [custom_modifiers.NoModifier()]
    else:
        inverter_modifier = [custom_modifiers.HydraulicSeperatorModifier()]
    if no_minimal_compressor_speed:
        inverter_modifier.append(custom_modifiers.NoMinimalCompressorSpeed())

    if with_start_losses:
        modifiers = [
            [custom_modifiers.OnOffControlModifier(), custom_modifiers.StartLossModifier()],
            inverter_modifier + [custom_modifiers.StartLossModifier()]
        ]
    else:
        modifiers = [
            [custom_modifiers.OnOffControlModifier()],
            inverter_modifier
        ]
    if model_hom_names is None or len(model_hom_names) == 1:
        return configs.InputsConfig(
            weathers=weathers,
            buildings=buildings,
            users=[configs.inputs.users.UserProfile(use_stochastic_internal_gains=True)],
            dhw_profiles=[{"profile": "NoDHW" if no_dhw else "M"}],
            modifiers=[modifiers[1]] if only_inverter else modifiers,
            hom=hom,
            existing_zone_record_path=existing_zone_record_path
        )
    return configs.InputsConfig(
        weathers=weathers,
        buildings=buildings,
        users=[
            configs.inputs.users.UserProfile(
                use_stochastic_internal_gains=True,
                custom_model=f"BESGriConOp.Studies.SFH.MPCModelROM.BESCISBAT.{model_prediction_name}",
                custom_model_hom=f"BESGriConOp.Studies.SFH.MPCModelROM.BESCISBAT.{model_hom_name}"
            )
            for model_hom_name, model_prediction_name in zip(model_hom_names, model_predictions_names)
        ],
        dhw_profiles=[{"profile": "NoDHW" if no_dhw else "M"}],
        modifiers=[modifiers[1]] if only_inverter else modifiers,
        hom=hom,
        existing_zone_record_path=existing_zone_record_path
    )