from bes_rules.configs.plotting import PlotConfig
from bes_rules.plotting.time_over_year import plot_days_in_year_over_hours_in_day
from ebcpy import TimeSeriesData


def plot_air_exchange_rate_teaser():
    variables = [
        "outputs.weather.TDryBul",
        "building.thermalZone[1].airExc.ventRate",
        "outputs.electrical.tra.PHea[1].value",

    ]
    from bes_rules import RESULTS_FOLDER

    df = TimeSeriesData(RESULTS_FOLDER.joinpath("show_air_exchange_problems", "data.hdf"))
    df = df.loc[range(0, 365 * 86400, 900)]
    plot_days_in_year_over_hours_in_day(
        df=df,
        variables=variables,
        plot_config=PlotConfig.load_default(),
        save_path=RESULTS_FOLDER.joinpath("show_air_exchange_problems")
    )
    from bes_rules.boundary_conditions.weather import get_weather_configs_by_names
    from bes_rules.utils.functions import get_heating_degree_days

    weather_config = get_weather_configs_by_names(region_names=["Potsdam"])[0]
    height_rooms = 2.5
    air_exchange = 0.5
    GTZ_Ti_HT = get_heating_degree_days(
        weather_config.get_hourly_weather_data().loc[:, "t"] + 273.15,
        293.15, 273.15 + 15
    ) * 24  # In Kh/a
    UA_per_ANet = height_rooms * air_exchange / 3600 * 1014.54 * 1.2  # W/K/m2
    specific_heat_demand = (
            UA_per_ANet * GTZ_Ti_HT / 1000
    )  # Wh/m2/a / (1/h)
    specific_heat_demand_through_air_exchange = specific_heat_demand / air_exchange  # kWh/m2/a / (1/h)
    print(
        f"Specific heat demand increases by {specific_heat_demand_through_air_exchange} "
        f"kWh/m2/a for each 1/h air exchange rate."
    )


if __name__ == '__main__':
    plot_air_exchange_rate_teaser()
