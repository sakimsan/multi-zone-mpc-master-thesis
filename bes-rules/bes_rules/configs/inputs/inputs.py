import logging
import os
import pathlib

from bes_rules.configs.inputs.building import BuildingConfig
from bes_rules.configs.inputs.custom_modifiers import CustomModifierConfig
from bes_rules.configs.inputs.dhw import DHWProfile
from bes_rules.configs.inputs.evu import EVUProfile
from bes_rules.configs.inputs.users import UserProfile
from bes_rules.configs.inputs.weather import WeatherConfig
from bes_rules.configs.inputs.prices import Prices
from pydantic import BaseModel, field_validator, ConfigDict
import itertools
from typing import List, Union, Optional

logger = logging.getLogger(__name__)

ModifierConfigList = Union[CustomModifierConfig, List[CustomModifierConfig]]


class InputConfig(BaseModel):
    weather: WeatherConfig
    building: BuildingConfig
    dhw_profile: DHWProfile
    user: UserProfile
    evu_profile: EVUProfile = EVUProfile()
    modifiers: Optional[Union[ModifierConfigList, None]] = None
    prices: Optional[Union[Prices, None]] = None
    hom: Union[bool, List[bool]] = False
    existing_zone_record_path: Union[str, List[str]] = None


    @field_validator("modifiers", "existing_zone_record_path")
    @classmethod
    def convert_none(cls, value):
        if value is None:
            return
        if isinstance(value, list):
            return value
        return [value]

    @field_validator("hom")
    @classmethod
    def convert_bool(cls, value):
        if isinstance(value, list):
            return value
        return [value]

    def get_modelica_modifier(self, with_custom_modifier: bool = True):
        modifiers = [
            self.building.get_modelica_modifier(input_config=self),
            self.dhw_profile.get_modelica_modifier(input_config=self),
            self.weather.get_modelica_modifier(input_config=self),
            self.user.get_modelica_modifier(input_config=self),
            self.evu_profile.get_modelica_modifier(input_config=self)
        ]
        if self.modifiers and with_custom_modifier:
            for modifier in self.modifiers:
                modifier_str = modifier.get_modelica_modifier(input_config=self)
                if modifier_str:  # Case for dummy-no-modifier
                    modifiers.append(modifier_str)
        merged_modifier = ",\n  ".join(
            [modifier for modifier in modifiers if modifier]
        )
        if self.user.custom_model is None:
            return f"({merged_modifier})"
        return self.user.custom_model + f"({merged_modifier})"

    def get_name(self, with_user: bool = True):
        return "_".join(self.get_name_parts(with_user=with_user).values())

    def get_name_parts(self, with_user: bool = True):
        parts = {
            "weather": self.weather.get_name(),
            "building": self.building.get_name(),
            "dhw_profile": self.dhw_profile.get_name(),
            "user": self.user.get_name()
        }
        if with_user:
            parts["user"] = self.user.get_name()
        if self.evu_profile.get_name() != "None":
            parts["evu_profile"] = self.evu_profile.get_name()
        if self.modifiers:
            parts["modifier"] = "_".join(modifier.get_name() for modifier in self.modifiers)
        if self.prices and self.prices.get_name():
            parts["prices"] = self.prices.get_name()
        return parts


class InputsConfig(BaseModel):
    full_factorial: bool = True
    buildings: List[BuildingConfig]
    weathers: List[WeatherConfig]
    dhw_profiles: List[DHWProfile]
    evu_profiles: List[EVUProfile] = [EVUProfile()]
    users: List[UserProfile] = [UserProfile()]
    modifiers: Optional[Union[List[ModifierConfigList], List[None]]] = [None]
    prices: Optional[Union[List[Prices], List[None]]] = [None]
    hom: Union[List[bool], bool] = [False]
    existing_zone_record_path: Union[List[str], str] = [None]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @field_validator("modifiers", "existing_zone_record_path")
    @classmethod
    def return_list_of_none(cls, value):
        if value is None:
            return [None]
        if isinstance(value, list):
            return value
        return [value]

    @field_validator("hom")
    @classmethod
    def convert_bool(cls, value):
        if isinstance(value, list):
            return value
        return [value]

    def get_permutations(self):
        """
        Get all permutations of InputConfigs
        """
        permutations = []
        args = [
            self.weathers, self.buildings,
            self.dhw_profiles, self.users,
            self.evu_profiles, self.modifiers,
            self.prices, self.hom, self.existing_zone_record_path
        ]
        if self.full_factorial:
            func = itertools.product
        else:
            func = zip
            lengths = [len(arg) for arg in args]
            equals = list(set(lengths))
            equals.remove(1)
            assert len(equals) <= 1, f"Inputs have different lengths: {lengths}"
            if len(equals) == 0:
                n_inputs = 1
            else:
                n_inputs = equals[0]
            new_args = []
            for arg in args:
                if len(arg) == 1:
                    new_args.append([arg[0]] * n_inputs)
                else:
                    new_args.append(arg)
            args = new_args

        for wea, bui, dhw, user, evu_profile, modifier, prices, hom, zone_record_path in func(*args):
            permutations.append(InputConfig(
                weather=wea,
                building=bui,
                dhw_profile=dhw,
                user=user,
                evu_profile=evu_profile,
                modifiers=modifier,
                prices=prices,
                hom=hom,
                existing_zone_record_path=zone_record_path
            ))
        return permutations

    def get_additional_packages(self):
        """
        Get new packages like buildings into Dymola
        """
        packages = []
        for bui in self.buildings:
            packages.append(bui.package_path)
        return list(set(packages))

    def generate_files(
            self,
            path: pathlib.Path,
            name: str,
            teaser_export_besmod_kwargs: dict = None
    ):
        """
        Function to generate all files required for the simulation.
        Examples:
        - Weather .mos files
        - TEASER records
        - DHWCalc tables

        :param dict teaser_export_besmod_kwargs:
            Additional kwargs for the export_besmod function in TEASER.
            See its doc for options, aside from `path`.
        """
        from bes_rules.boundary_conditions.weather import create_weather_cases
        from bes_rules.boundary_conditions.building import create_buildings
        from bes_rules.boundary_conditions.dhw import generate_dhw_calc_tapping
        self.buildings = create_buildings(
            path=path,
            name=name,
            building_configs=self.buildings,
            teaser_export_besmod_kwargs=teaser_export_besmod_kwargs
        )
        self.weathers = create_weather_cases(
            path=path,
            weathers=self.weathers
        )

        for dhw in self.dhw_profiles:
            if dhw.profile == "DHWCalc" and dhw.dhw_calc_config.path is None:
                # Export DHWCalc
                dhw_calc_path = path.joinpath("DHW", dhw.dhw_calc_config.get_name() + ".txt")
                os.makedirs(dhw_calc_path.parent, exist_ok=True)
                generate_dhw_calc_tapping(
                    save_path_file=dhw_calc_path,
                    config=dhw.dhw_calc_config,
                    with_plot=False
                )
                dhw.dhw_calc_config.path = dhw_calc_path
