import itertools
import logging
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Union, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
from ebcpy import DymolaAPI, TimeSeriesData
from teaser.logic.buildingobjects.building import Building
from teaser.project import Project

if TYPE_CHECKING:
    from bes_rules.configs import InputConfig, BuildingConfig

logger = logging.getLogger(__name__)


def create_buildings(
        name: str,
        building_configs: List["BuildingConfig"],
        export: bool = True,
        path: Path = None,
        teaser_export_besmod_kwargs: dict = None,
        t_bt: float = None
):
    """
    Generate the buildings using teaser and possibly export it

    :param str name:
        Name of the TEASER project
    :param List["BuildingConfig"] building_configs:
        List of buildings to create
    :param bool export:
        Whether to export the modelica files or not
    :param Path path:
        Path to save the export
    :param dict teaser_export_besmod_kwargs:
        Additional kwargs for the export_besmod function in TEASER.
        See its doc for options, aside from `path`.

    :return: List["BuildingConfig"]
        Modified list of buildings with calculated outputs
    """
    if teaser_export_besmod_kwargs is None:
        teaser_export_besmod_kwargs = {}

    prj = Project()

    prj.name = name

    building_configs_to_create = {}
    for building_config in building_configs:
        if building_config.name in building_configs_to_create:
            config_with_the_same_name = building_configs_to_create[building_config.name]
            fields_to_exclude_in_comparison = {
                # Will be added later:
                "building_parameters", "record_name", "building_model_name", "package_path",
                # Irrelevant for TEASER
                "retrofit_transfer_system_to_at_least", "possibly_use_underfloor_heating", "use_verboseEnergyBalance"
            }
            if (
                    config_with_the_same_name.model_dump(exclude=fields_to_exclude_in_comparison) !=
                    building_config.model_dump(exclude=fields_to_exclude_in_comparison)
            ):
                raise ValueError(
                    "Duplicate building configs which are not equal, "
                    "can't automatically select one over another")
        building_configs_to_create[building_config.name] = building_config

    for building_config in building_configs_to_create.values():
        prj.add_residential(
            name=building_config.name,
            geometry_data=building_config.geometry_data,
            construction_data=building_config.construction_data,
            number_of_floors=building_config.number_of_floors,
            height_of_floors=building_config.height_of_floors,
            with_ahu=building_config.with_ahu,
            year_of_construction=building_config.year_of_construction,
            net_leased_area=building_config.net_leased_area,
        )
        if building_config.element_retrofits is not None:
            component_based_retrofit(
                building=prj.buildings[-1],
                element_retrofit_stats=building_config.element_retrofits
            )

    # export building model (see Teaser/project)
    prj.used_library_calc = 'AixLib'
    prj.number_of_elements_calc = 4  # Default value
    if t_bt is not None:
        for bui in prj.buildings:
            bui.t_bt = t_bt
            bui.t_bt_layer = t_bt
    prj.calc_all_buildings(raise_errors=True)
    for building_config, teaser_bui in zip(building_configs_to_create.values(), prj.buildings):
        for zone in teaser_bui.thermal_zones:
            zone.use_conditions.with_heating = False
            zone.use_conditions.with_cooling = False
            zone.use_conditions.base_infiltration = building_config.base_infiltration
            if building_config.use_normative_infiltration:
                zone.use_conditions.use_constant_infiltration = True
                zone.use_conditions.base_infiltration = 0.5
            if building_config.use_led:
                zone.use_conditions.use_maintained_illuminance = True
                zone.use_conditions.lighting_efficiency_lumen = 180  # LED lamps
            zone.use_conditions.persons = (
                    building_config.number_of_occupants / teaser_bui.thermal_zones[0].area
            )
        teaser_bui.library_attr.besmod_version = "0.7.0"

    from bes_rules.configs.inputs.building import RelevantBuildingParameters

    for teaser_bui, building_config in zip(prj.buildings, building_configs_to_create.values()):
        CEff = sum([
            teaser_bui.thermal_zones[0].model_attr.c1_ow,
            teaser_bui.thermal_zones[0].model_attr.c1_iw,
            teaser_bui.thermal_zones[0].model_attr.c1_gf,
            teaser_bui.thermal_zones[0].model_attr.c1_rt,
        ])
        building_config.building_parameters = RelevantBuildingParameters(
            heat_load_ground_factor=teaser_bui.thermal_zones[0].model_attr.heat_load_ground_factor,
            heat_load_outside_factor=teaser_bui.thermal_zones[0].model_attr.heat_load_outside_factor,
            roof_area=teaser_bui.thermal_zones[0].model_attr.area_rt,
            facade_area=sum(teaser_bui.thermal_zones[0].model_attr.facade_areas),
            window_area=sum(teaser_bui.thermal_zones[0].model_attr.window_areas),
            base_infiltration=teaser_bui.thermal_zones[0].use_conditions.base_infiltration,
            volume_air=teaser_bui.thermal_zones[0].volume,
            CEff=CEff
        )

    if export:
        if path is None:
            raise TypeError("For export, you need to specify a path")
        package_path = prj.export_besmod(
            path=path.joinpath("Buildings"),
            **teaser_export_besmod_kwargs
        )
        for building_config in building_configs_to_create.values():
            bui_name = building_config.get_teaser_name()
            building_config.record_name = ".".join([
                name, bui_name,
                f"{bui_name}_DataBase",
                f"{bui_name}_SingleDwelling"
            ])
            building_config.package_path = Path(package_path).joinpath("package.mo")
            building_config.building_model_name = ".".join([
                name, bui_name, bui_name
            ])
    # Reverse the map to the requested buildings and update TEASER specific parameters
    for building_config in building_configs:
        created_building_config = building_configs_to_create[building_config.name]
        building_config.record_name = created_building_config.record_name
        building_config.package_path = created_building_config.package_path
        building_config.building_model_name = created_building_config.building_model_name
        building_config.building_parameters = created_building_config.building_parameters

    return building_configs


def get_building_configs_by_name(building_names: list, **kwargs) -> List["BuildingConfig"]:
    """
    Return the configs for building_configs according to their names

    :param list building_names:
        List of building names to return

    :return:
        Return list of BuildingConfigs
    """
    from bes_rules.configs import BuildingConfig

    base = {
        "geometry_data": "tabula_de_single_family_house",
        "number_of_occupants": 4,
        "number_of_floors": 2,
        "with_ahu": False,
    }
    buildings_data = {
        "Retrofit1918":
            {
                "name": "Retrofit1918",
                "construction_data": "tabula_de_retrofit",
                "year_of_construction": 1918,
                "net_leased_area": 139.66,
                "height_of_floors": 2.5
            },
        "NoRetrofit1918":
            {
                "name": "NoRetrofit1918",
                "construction_data": "tabula_de_standard",
                "year_of_construction": 1918,
                "net_leased_area": 139.66,
                "height_of_floors": 2.5
            },
        "NoRetrofit1983":
            {
                "name": "NoRetrofit1983",
                "construction_data": "tabula_de_standard",
                "year_of_construction": 1983,
                "net_leased_area": 156.25,
                "height_of_floors": 2.5
            },
        "Retrofit1983":
            {
                "name": "Retrofit1983",
                "construction_data": "tabula_de_retrofit",
                "year_of_construction": 1983,
                "net_leased_area": 156.25,
                "height_of_floors": 2.5
            },
        "RefAachen":
            {
                "name": "RefAachen",
                "construction_type": "tabula_standard",
                "year_of_construction": 1984,
                "net_leased_area": 185.9548,
                "height_of_floors": 2.6
            },
    }
    building_configs = []
    for building_name in building_names:
        if building_name not in buildings_data:
            raise KeyError(f"Given building '{building_name}' not in dataset")
        building_configs.append(
            BuildingConfig(
                **base,
                **buildings_data[building_name],
                **kwargs
            )
        )
    return building_configs


def get_all_tabula_sfh_buildings(
        as_dict: bool = False,
        construction_datas: list = None,
        **kwargs
) -> Union[List["BuildingConfig"], Dict[str, "BuildingConfig"]]:
    """
    Get all tabula single family buildings.

    :param bool as_dict:
        Return the configs as a dict instead of a list.
        Keys are the names, values the configs.
    :param list construction_datas:
        Specify which construction types should be included.
        Default: Standard, retrofit, advanced retrofit.

    Additional settings passed directly to all BuildingConfigs

    :returns
        Either a list or a dict, depending on `as_dict`
    """
    tabula_areas_sfh = {
        2015: 187,
        2009: 147,
        2001: 122,
        1994: 150,
        1980: 216,
        1970: 173,
        1960: 121,
        1950: 111,
        1948: 303,
        1918: 142,
        1859: 219,
    }
    from bes_rules.configs import BuildingConfig
    base = {
        "geometry_data": "tabula_de_single_family_house",
        "number_of_occupants": 4,
        "number_of_floors": 2,
        "height_of_floors": 2.5,
        "with_ahu": False,
    }
    if construction_datas is None:
        construction_datas = ["standard", "retrofit", "adv_retrofit"]
    building_configs = []
    for construction_data in construction_datas:
        for year_of_construction, net_leased_area in tabula_areas_sfh.items():
            building_configs.append(
                BuildingConfig(
                    name=f"{year_of_construction}_{construction_data}",
                    construction_data=f"tabula_de_{construction_data}",
                    year_of_construction=year_of_construction,
                    net_leased_area=net_leased_area,
                    **base,
                    **kwargs
                ))
    if as_dict:
        return {building_config.name: building_config for building_config in building_configs}
    return building_configs


def get_bivalence_temperature_for_geg_goal(
        input_config: "InputConfig", dym_api: DymolaAPI = None,
        geg_goal: float = 65, THeatingThreshold: float = 293.15
):
    """
    Return the design bivalence temperature to ensure a share
    of 65 % renewables share in the heat power generation
    according to the German GEG.
    If dym_api is None, the value is only calculated based
    on weather data.
    """
    df_hourly_weather_data = input_config.weather.get_hourly_weather_data()
    T_air = df_hourly_weather_data["t"].values + 273.15
    T_biv_options = np.arange(input_config.weather.TOda_nominal, THeatingThreshold, 0.1)

    def _get_geg_share_for_temperature(_T_air, _T_threshold, _T_biv):
        """
        \frac{\sum_{i=N_\mathrm{Biv}}^N UA \cdot (T_\mathrm{Raum} - T_\mathrm{Aus,i})}{\sum_{i=0}^N UA \cdot (T_\mathrm{Raum} - T_\mathrm{Aus,i})}
        """
        Q_demand = (_T_threshold - _T_air)
        Q_renewable = Q_demand[_T_air >= _T_biv]
        return np.sum(Q_renewable) / np.sum(Q_demand)

    def _get_geg_share_for_teaser(_Q_house_demand, _T_air, _T_biv):
        """
        65 \% < \frac{Q_\mathrm{WP}}{(Q_\mathrm{Ges}}
        """
        Q_renewable = _Q_house_demand[_T_air >= _T_biv]
        return np.sum(Q_renewable) / np.sum(_Q_house_demand)

    geg_shares_weather = np.array(
        [_get_geg_share_for_temperature(_T_air=T_air, _T_threshold=THeatingThreshold, _T_biv=TBiv) * 100
         for TBiv in T_biv_options]
    )
    T_biv_according_to_weather = T_biv_options[geg_shares_weather >= geg_goal][-1]
    if dym_api is None:
        logger.info("Only calculating bivalence temperature according to weather data.")
        return T_biv_according_to_weather
    old_setup = dym_api.sim_setup
    dym_api.set_sim_setup(dict(stop_time=365 * 86400, output_interval=3600))
    Q_house_demand = get_ideal_heating_demand(
        record_name=input_config.building.record_name,
        mos_file=input_config.weather.mos_path,
        dym_api=dym_api
    ).loc[:86400 * 365 - 1]
    dym_api.set_sim_setup(old_setup)
    geg_shares_teaser = np.array(
        [_get_geg_share_for_teaser(_Q_house_demand=Q_house_demand, _T_air=T_air, _T_biv=TBiv) * 100
         for TBiv in T_biv_options]
    )
    T_biv_according_to_teaser = T_biv_options[geg_shares_teaser >= geg_goal][-1]
    logger.info(f"{T_biv_according_to_teaser=}, {T_biv_according_to_weather=}")
    return T_biv_according_to_teaser


def get_ideal_heating_demand(input_config: "InputConfig", dym_api: DymolaAPI, get_all_results: bool = False):
    old_sim_setup = dym_api.sim_setup
    # always simulate full year + 2 more days for initialization
    dym_api.set_sim_setup(dict(start_time=0, stop_time=367 * 86400, output_interval=900))
    # Add modifier
    simulation_model_name = get_teaser_model_name_with_modifiers(input_config=input_config)

    old_result_names = dym_api.result_names
    old_extract_variables = dym_api.extract_variables
    old_model_name = dym_api.model_name
    dym_api.extract_variables = False
    tsd_path = dym_api.simulate(
        model_names=[simulation_model_name],
        return_option="savepath"
    )
    heater_name = "outputs.electrical.tra.PHea[1].value"
    if get_all_results:
        tsd = TimeSeriesData(tsd_path).to_df()
    else:
        tsd = TimeSeriesData(tsd_path, variable_names=[heater_name]).to_df().loc[:, heater_name]
    # take all days > 365 (2) --> one index less, time_step=900
    last_two_days = tsd.loc[365 * 86400 + 900:]
    last_two_days.index -= last_two_days.index[0]
    # overwrite first two days from last two days (365 bis 367) and adjust index for correct lenght
    tsd = pd.concat([last_two_days, tsd.loc[86400 * 2:365 * 86400 - 900]])
    os.remove(tsd_path)
    dym_api.extract_variables = old_extract_variables
    dym_api.model_name = old_model_name
    dym_api.set_sim_setup(old_sim_setup)
    dym_api.result_names = old_result_names
    return tsd


def get_teaser_model_name_with_modifiers(input_config: "InputConfig"):
    old_input_config_modify_transfer_system = input_config.building.modify_transfer_system
    input_config.building.modify_transfer_system = False
    modifier = input_config.get_modelica_modifier(with_custom_modifier=False)
    input_config.building.modify_transfer_system = old_input_config_modify_transfer_system
    return f"BESMod.Examples.TEASERHeatLoadCalculation.PartialCalculation{modifier}"


def compare_tabula_buildings(
        pickle_path: Path,
        save_path: Path,
        weather_name: str = "TRY2015_523845130645_Jahr",
        TRoom: float = 293.15,
        TOda_nominal: float = 273.15 - 12.1
):
    buildings = get_all_tabula_sfh_buildings()
    create_buildings(name="TabulaComparison", building_configs=buildings, export=False)
    with open(pickle_path, "rb") as file:
        simulated_heat_demand = pickle.load(file)
    df = pd.DataFrame()
    for building in buildings:
        Q_demand_building = simulated_heat_demand[f"{weather_name}____{building.name}"]
        df.loc[building.name, "heat_load"] = building.get_heating_load(
            TOda_nominal=TOda_nominal, TRoom_nominal=TRoom
        )
        df.loc[building.name, "heat_demand"] = Q_demand_building.sum()
        df.loc[building.name, "net_leased_area"] = building.net_leased_area

    df.to_excel(save_path.joinpath(f"TabulaComparison_{weather_name}_raw.xlsx"))


def generate_buildings_for_all_element_combinations(
        building_configs: List["BuildingConfig"],
        elements: list = None,
        retrofit_choices: list = None
) -> List["BuildingConfig"]:
    """
    This function is necessary because I want to get the bes-rules building
    config object and not the Building class from TEASER. Else, the function
    in TEASER could be used.
    """
    # Define mapping for later naming
    retrofit_dict = {'standard': 0, 'retrofit': 1, 'adv_retrofit': 2}
    if elements is None:
        elements = ['outer_walls', 'windows', 'rooftops', "ground_floors"]
    if retrofit_choices is None:
        retrofit_choices = list(retrofit_dict.keys())

    # Generate all possible combinations of retrofit statuses for each element
    combinations = itertools.product(retrofit_choices, repeat=len(elements))

    # Create a list to store the resulting dictionaries
    combinations = [
        {
            element: status
            for element, status in zip(elements, combo)
        }
        for combo in combinations
    ]
    generated_building_configs = []
    for building_config in building_configs:
        for element_retrofit_stats in combinations:
            # Code for retrofit status OiWiRiGi with i from 0 to 2
            retrofit_code = ''.join(
                f"{element[0]}{retrofit_dict[retrofit_option]}"
                for element, retrofit_option in element_retrofit_stats.items()
            )
            new_building_config = building_config.copy()
            new_building_config.element_retrofits = element_retrofit_stats
            new_building_config.name = building_config.name + f"_{retrofit_code}"
            generated_building_configs.append(new_building_config)
    return generated_building_configs


def component_based_retrofit(building: Building, element_retrofit_stats: dict):
    for zone in building.thermal_zones:
        for element, retrofit_option in element_retrofit_stats.items():
            if retrofit_option == "standard":
                continue
            for wall_count in getattr(zone, element):
                wall_count.load_type_element(
                    year=building.year_of_construction,
                    construction=wall_count.construction_data.replace("standard", retrofit_option)
                )


def find_duplicates(lst):
    # Count occurrences of each element
    count = Counter(lst)

    # Find elements with count > 1
    duplicates = [item for item, count in count.items() if count > 1]

    return duplicates


def get_nominal_supply_temperature(year_of_construction):
    """
    Sources:
    https://www.ffe.de/projekte/waermepumpen-fahrplan-finanzielle-kipppunkte-zur-modernisierung-mit-waermepumpen-im-wohngebaeudebestand/

    Recknagel:
    – bei Fußbodenheizungen n = 1,1
    – bei Plattenheizkörpern n = 1,20…1,30
    – bei Rohren n = 1,25
    – bei Rippenrohren n = 1,25
    – bei Radiatoren n = 1,30
    – bei Konvektoren n = 1,25…1,45

    Kapitel 2.7.3:
    Die Auslegungstemperaturen von Heizkreisen haben sich in den letzten Jahrzehnten
    deutlich in Richtung niedrigerer Werte entwickelt. Die bis in die 1980er Jahre übliche 90/
    70 °C Auslegung wurde durch 70/55 °C abgelöst. Seit Einführung der EnEV werden
    i.d.R. Niedertemperaturheizungen mit max. Vorlauftemperaturen von 55 °C eingesetzt
    """
    # TODO-Assumption: Note in thesis
    # n values as in BESMod / IBPSA.
    # Recknagel would use 1.1 for UFH and 1.2 up to 1.3 for residential buildings.
    # Konvektoren are not expected.
    if year_of_construction < 1950:
        return 90 + 273.15, 20, 1.24
    if year_of_construction < 1980:
        return 70 + 273.15, 15, 1.24
    if year_of_construction < 2010:
        return 55 + 273.15, 10, 1.24
    return 35 + 273.15, 5, 1.1


def get_supply_temperature_after_retrofit(
        TRoom1: float,
        TRoom2: float,
        n: float,
        TSup1: float,
        TRet1: float,
        QNom1: float,
        QNom2: float
):
    """
    This helper function uses the exact formulation from
    "Lämmle et al. 2022, Chapter 4.1" to avoid confusion.

    :param TRoom1:
        Old room temperature
    :param TRoom2:
        New room temperature
    :param n:
        Heat transfer exponent
    :param TSup1:
        Old supply temperature
    :param TRet1:
        Old return temperature
    :param QNom1:
        Old heating load
    :param QNom2:
        New heating load

    :return: T2:
        New supply temperature
    :return: T2:
        New return temperature
    """
    dT2 = (TSup1 - TRet1) * QNom2 / QNom1
    dTLog2 = (
            (TSup1 - TRet1) /
            np.log((TSup1 - TRoom1) / (TRet1 - TRoom1)) *
            (QNom2 / QNom1) ** (1 / n)
    )
    TSup2 = TRoom2 + dT2 * (np.exp(dT2 / dTLog2) / (np.exp(dT2 / dTLog2) - 1))
    return max(308.15, TSup2), max(5.0, dT2)


def get_retrofit_temperatures(
        building_config: "BuildingConfig",
        TOda_nominal: float,
        TRoom_nominal: float,
        retrofit_transfer_system_to_at_least: tuple = None
):
    """
    According to Lämmle et al. 2022, Chapter 4.1

    Also, if `retrofit_transfer_system_to_at_least` is specified,
    the maximum supply temperature returned for both retrofit and non-retrofit be the first value of the tuple,
    the new supply temperature. If this value is used, the returned temperature difference between supply and
    return is set to the second value. Required as BESMod sizes the radiators based on the old supply values,
    and the option is to emulate a transfer-system retrofit to lower supply temperatures for heat pump usage.
    If the retrofit or both supply temperature are lower than the given value, they are not used.
    """
    THydNoRet, dTHydNoRet, n_heat_exponent = get_nominal_supply_temperature(
        year_of_construction=building_config.year_of_construction
    )
    building_config_without_retrofit = building_config.copy()
    building_config_without_retrofit.construction_data = "tabula_de_standard"
    building_config_without_retrofit.element_retrofits = None

    from bes_rules.boundary_conditions.building import create_buildings
    building_config_without_retrofit = create_buildings(
        name="Temporary",
        building_configs=[building_config_without_retrofit],
        export=False
    )[0]
    QRet_flow_nominal = building_config.get_heating_load(
        TOda_nominal=TOda_nominal, TRoom_nominal=TRoom_nominal
    )
    QNoRet_flow_nominal = building_config_without_retrofit.get_heating_load(
        TOda_nominal=TOda_nominal, TRoom_nominal=TRoom_nominal
    )

    if QNoRet_flow_nominal < QRet_flow_nominal:
        raise ValueError("Somehow, heat load increased due to retrofit")
    if QRet_flow_nominal == QNoRet_flow_nominal:
        THydRet = THydNoRet
        dTHydRet = dTHydNoRet
    else:
        THydRet, dTHydRet = get_supply_temperature_after_retrofit(
            TRoom1=TRoom_nominal,
            TRoom2=TRoom_nominal,
            n=n_heat_exponent,
            TSup1=THydNoRet,
            TRet1=THydNoRet - dTHydNoRet,
            QNom1=QNoRet_flow_nominal,
            QNom2=QRet_flow_nominal
        )
    if retrofit_transfer_system_to_at_least is not None:
        THydNew, dTHydNew = retrofit_transfer_system_to_at_least
        if THydRet > THydNew:
            THydRet = min(THydNew, THydRet)
            dTHydRet = dTHydNew
        if THydNoRet > THydNew:
            THydNoRet = min(THydNew, THydNoRet)
            dTHydNoRet = dTHydNew

    return THydRet, dTHydRet, THydNoRet, dTHydNoRet, QNoRet_flow_nominal, QRet_flow_nominal


if __name__ == '__main__':
    BLGS = [get_all_tabula_sfh_buildings()[0]]
    BLGS = generate_buildings_for_all_element_combinations(
        building_configs=BLGS,
        elements=[
            "outer_walls",
            "windows",
            # "rooftops", "ground_floors"
        ],
        # retrofit_choices=["standard", "retrofit"]
    )
    create_buildings(name="TestPartialRetrofit", building_configs=BLGS, path=Path(r"D:\00_temp\testzoneTEASER"))
