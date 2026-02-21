import pathlib
from typing import Optional, TYPE_CHECKING

from pydantic import field_validator
from aixweather.imports.TRY import load_try_from_file
from ebcpy.preprocessing import convert_index_to_datetime_index
import datetime

from bes_rules.utils.modelica import load_modelica_file_modifier
from bes_rules.configs.inputs.base import BaseInputConfig
from bes_rules import DATA_PATH

if TYPE_CHECKING:
    from bes_rules.configs import InputConfig


class WeatherConfig(BaseInputConfig):
    dat_file: pathlib.Path
    TOda_nominal: float = None
    mos_path: Optional[pathlib.Path] = None

    @field_validator("TOda_nominal")
    @classmethod
    def check_unit(cls, TOda_nominal):
        """Convert to K if < 100"""
        if TOda_nominal is None:
            return TOda_nominal
        if TOda_nominal > 100:
            return TOda_nominal
        return TOda_nominal + 273.15

    @field_validator("dat_file")
    @classmethod
    def ensure_path(cls, _path):
        if isinstance(_path, pathlib.Path):
            return _path
        return pathlib.Path(_path)

    @field_validator("mos_path")
    @classmethod
    def ensure_optional_path(cls, _path):
        if isinstance(_path, pathlib.Path):
            return _path
        if _path is None:
            return None
        return pathlib.Path(_path)

    def get_modelica_modifier(self, input_config: "InputConfig"):
        if input_config.hom:
            return ""
        return f'systemParameters(\n' \
               f'    filNamWea={load_modelica_file_modifier(self.mos_path)},\n' \
               f'    TOda_nominal={self.TOda_nominal})'

    def get_name(self, location_name=False, pretty_print: bool = False):
        if location_name:
            return str(self.dat_file.parents[1].stem).replace(" ", "_") + self.dat_file.stem.split("_")[-1]
        if pretty_print:
            year, _, _typ = self.dat_file.stem.split("_")
            _typ = {"Jahr": "average", "Somm": "warm", "Wint": "cold"}[_typ]
            return f"{self.dat_file.parents[1].stem}-{year}-{_typ}"
        return self.dat_file.stem

    def get_hourly_weather_data(self):
        if len(self.dat_file.parents) >= 4 and self.dat_file.parents[4].name == "bes-rules":
            dat_file_new_pc = DATA_PATH.joinpath(self.dat_file.relative_to(self.dat_file.parents[3]))
        else:
            dat_file_new_pc = self.dat_file

        df = load_try_from_file(path=str(dat_file_new_pc))
        df.index *= 3600
        first_day_of_year = datetime.datetime(2015, 1, 1, 0, 0)
        return convert_index_to_datetime_index(df, origin=first_day_of_year)
