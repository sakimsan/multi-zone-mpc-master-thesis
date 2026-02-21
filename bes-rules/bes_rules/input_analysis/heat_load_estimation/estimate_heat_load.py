import datetime
import os
import warnings
from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from bes_rules.input_analysis.heat_load_estimation import heater_name, outdoor_air_name, heat_load_din
from bes_rules.plotting import EBCColors
from bes_rules.plotting.utils import get_figure_size
from bes_rules.configs.inputs import InputConfig
from bes_rules.utils.functions import get_heating_degree_days
from bes_rules.plotting.boxplots import plot_box

name_map = {'h': 'Hourly', 'd': 'Daily', 'w': 'Weekly', 'm': 'Monthly', 'GTZ': 'Heating Degree\n Days',
            'BG': 'Load Factor'}


def plot_heat_load_estimation(
        data,
        save_path: Path,
        TOda_nominal: float,
        heat_load: float,
        summer_temperature: float,
        heating_threshold_temperature: float,
        constant_summer_demand: float,
        show_measurement_points: bool,
        show_estimation: bool
):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True,
                           figsize=get_figure_size(n_columns=2, height_factor=2))
    TOda_nominal -= 273.15
    heating_threshold_temperature -= 273.15
    max_load = 0
    TOda_arange = np.arange(TOda_nominal, heating_threshold_temperature)
    for _ax, freq, results in zip(ax.flatten(), data.keys(), data.values()):
        df = results["df"]
        model = results["model"]
        if show_measurement_points:
            _ax.scatter(df[outdoor_air_name] - 273.15, df["Q_flow_total"],
                        marker='o', label='Measurements', color=EBCColors.grey)
        _ax.scatter([TOda_nominal], [heat_load],
                    marker="x", label="EN 12831", color=EBCColors.red)
        if show_estimation:
            _ax.scatter([TOda_nominal], [model.predict([[TOda_nominal]])],
                        marker="x", label="Estimation", color=EBCColors.blue)
        linear_regression = model.predict(TOda_arange.reshape(-1, 1))
        if show_estimation:
            _ax.plot(TOda_arange, linear_regression, color="black")
            _ax.plot(
                [summer_temperature - 273.15, df[outdoor_air_name].max() - 273.15],
                [constant_summer_demand, constant_summer_demand], color="black"
            )
        _ax.set_title(name_map[freq])
        _ax.set_xticks([-20, -10, 0, 10, 20, 30])
        max_load = max(max_load, heat_load, max(linear_regression), max(df["Q_flow_total"]))
    for _ax in ax.flatten():
        _ax.set_ylim([0, max_load * 1.05])
    ax[1, 1].legend(
        ncol=1,  # bbox_to_anchor=(0, 1),
        columnspacing=0.2, labelspacing=0, loc="upper right",
        handletextpad=0.1
    )

    ax[0, 0].set_ylabel("$\dot{Q}$ in kW")
    ax[1, 0].set_ylabel("$\dot{Q}$ in kW")
    ax[1, 0].set_xlabel("$\\bar{T}_\mathrm{Oda}$ in °C")
    ax[1, 1].set_xlabel("$\\bar{T}_\mathrm{Oda}$ in °C")
    fig.subplots_adjust(top=0.2)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close("all")


def heat_load_from_yearly_results(
        df_daily: pd.DataFrame,
        with_dhw: bool,
        TOda_nominal: float,
        room_temperature: float = 293.15,
        heating_threshold_temperature: float = 288.15,
        net_leased_area: float = None
):
    """
    DIN/TS 12831-1 Abschnitt 7: Schätzung der Heizlast aus jährlichem Endenergieverbrauch
    """
    # Sum up degree days
    heating_degree_days = get_heating_degree_days(
        df_daily[outdoor_air_name],
        room_temperature=room_temperature,
        heating_threshold_temperature=heating_threshold_temperature
    )  # in Kd
    b_vf_gtz = heating_degree_days / (heating_threshold_temperature - TOda_nominal)  # in d

    heating_period = ((df_daily.index < datetime.datetime(2015, 4, 1, 0, 0)) |
                      (df_daily.index >= datetime.datetime(2015, 10, 1, 0, 0)))

    mean_outdoor_air_temperature_heating_period = df_daily.loc[heating_period, outdoor_air_name].mean()
    b_vf_bg = len(df_daily.loc[heating_period]) * (
            (heating_threshold_temperature - mean_outdoor_air_temperature_heating_period) /
            (heating_threshold_temperature - TOda_nominal)
    )

    heat_demand = df_daily["Q_flow_total"].sum()  # in kWd
    if with_dhw:
        # DIN/TS 12831-1: Nutzenergie TWE = 16 k Wh/(m^2*a) * net_leased_area
        heat_demand += 16000 * net_leased_area

    return heat_demand / b_vf_gtz, heat_demand / b_vf_bg


def calc_linear_regression(
        df: pd.DataFrame,
        with_dhw: bool,
        threshold: float,
        heating_threshold_temperature: float = None,
        summer_temperature: float = 293.15
):
    # find mean temperature-independent load (dhw) above 20 °C.
    # If no measurements above 20 ° are present (e.g. monthly data), decrease until
    # such is found. If decrease goes into the heating season, raise warning.
    if with_dhw:
        raise NotImplementedError
        df_above_cutoff = df[df[outdoor_air_name] > summer_temperature]
        while len(df_above_cutoff) < 2:
            summer_temperature -= 1
            df_above_cutoff = df[df[outdoor_air_name] > summer_temperature]
        if summer_temperature < heating_threshold_temperature:
            warnings.warn("Not enough points in summer to deduct DHW demand. Using zero")
            p_mean_dhw = 0
        else:
            p_mean_dhw = df_above_cutoff["Q_flow_total"].mean()
    else:
        p_mean_dhw = 0
    # Only use points with heating, i.e. "relevant" points according to EN 12831
    df = df[df["Q_flow_total"] > threshold]

    def _get_sample_weights(_x):
        return np.ones(len(_x[:, 0]))
        return (_x[:, 0].max() - _x[:, 0]) / (_x[:, 0].max() - _x[:, 0].min())

    if heating_threshold_temperature is None:
        # split dataframe at 20 celsius
        df_below_cutoff = df[df[outdoor_air_name] < summer_temperature]
        # First create regression to get heating threshold temperature:
        model = LinearRegression()

        x_values = df_below_cutoff[outdoor_air_name].values - 273.15
        x = np.reshape(x_values, (-1, 1))
        y = df_below_cutoff["Q_flow_total"].values
        model.fit(x, y, sample_weight=_get_sample_weights(x))
        heating_threshold_temperature = -model.intercept_ / model.coef_[0]
        if not 10 <= heating_threshold_temperature <= 20:
            print("Heating threshold temperature outside 10-20 °C: ", heating_threshold_temperature)
        heating_threshold_temperature += 273.15  # To K

    df_below_cutoff = df[df[outdoor_air_name] < heating_threshold_temperature - 1]
    x_values = df_below_cutoff[outdoor_air_name].values - 273.15

    # First create regression to get heating threshold temperature:
    model = LinearRegression()
    x = np.reshape(x_values, (-1, 1))
    y = df_below_cutoff["Q_flow_total"].values
    model.fit(x, y, sample_weight=_get_sample_weights(x))
    r_sq = model.score(x, y)
    return model, r_sq, p_mean_dhw, summer_temperature, heating_threshold_temperature


def load_pickle_results(study_path: Path):
    with open(study_path.joinpath("joined_results.pickle"), "rb") as file:
        return pickle.load(file)


def odr(x, y):
    from scipy import odr

    def f(B, x):
        return B[0] * x + B[1]

    linear = odr.Model(f)
    mydata = odr.RealData(x, y, sx=np.zeros(x.shape), sy=np.zeros(y.shape))
    myodr = odr.ODR(mydata, linear, beta0=[1., 2.])
    myoutput = myodr.run()
    myoutput.pprint()


def run_all_estimations(study_path: Path, with_dhw: bool = False, with_plot: bool = False):
    results = load_pickle_results(study_path)
    all_results = {}
    print(len(results))

    results_special = [
        # "TRY2045_485809134724_Somm_B2009_adv_retrofit_SingleDwelling_NoDHW_3.0K-Per-IntGai_dynVen",
        # "TRY2045_485809134724_Somm_B2015_adv_retrofit_SingleDwelling_NoDHW_3.0K-Per-IntGai_dynVen",
        # "TRY2015_474856110632_Somm_B2009_adv_retrofit_SingleDwelling_NoDHW_3.0K-Per-IntGai_dynVen",
        # "TRY2045_485809134724_Somm_B2001_adv_retrofit_SingleDwelling_NoDHW_3.0K-Per-IntGai_dynVen",
        # "TRY2015_474856110632_Somm_B2015_adv_retrofit_SingleDwelling_NoDHW_3.0K-Per-IntGai_dynVen",
        "TRY2015_474856110632_Somm_B2001_adv_retrofit_SingleDwelling_NoDHW_3.0K-Per-IntGai_dynVen",
        # "TRY2045_517332106070_Somm_B2009_adv_retrofit_SingleDwelling_NoDHW_0.0K-NoPer-NoIntGai_conVen",
        # "TRY2045_535312085881_Somm_B2009_adv_retrofit_SingleDwelling_NoDHW_0.0K-NoPer-NoIntGai_conVen",
        # "TRY2045_535312085881_Somm_B2015_adv_retrofit_SingleDwelling_NoDHW_0.0K-NoPer-NoIntGai_conVen",
        # "TRY2045_535312085881_Somm_B1948_adv_retrofit_SingleDwelling_NoDHW_0.0K-NoPer-NoIntGai_conVen",
        # "TRY2045_517332106070_Somm_B1980_adv_retrofit_SingleDwelling_NoDHW_0.0K-NoPer-NoIntGai_conVen",
        # "TRY2045_535312085881_Somm_B1980_adv_retrofit_SingleDwelling_NoDHW_0.0K-NoPer-NoIntGai_conVen",
        # "TRY2045_477397106963_Wint_B2001_standard_SingleDwelling_NoDHW_0.0K-Per-IntGai_conVen",
        # "TRY2015_485809134724_Somm_B1859_retrofit_SingleDwelling_NoDHW_0.0K-Per-IntGai_conVen",
        # "TRY2045_477397106963_Wint_B1960_retrofit_SingleDwelling_NoDHW_0.0K-Per-IntGai_conVen",
        # "TRY2015_474856110632_Somm_B1859_retrofit_SingleDwelling_NoDHW_0.0K-Per-IntGai_conVen",
        # "TRY2015_523845130645_Jahr_B1950_retrofit_SingleDwelling_NoDHW_0.0K-Per-IntGai_conVen",
        "TRY2015_485809134724_Somm_B1994_standard_SingleDwelling_NoDHW_0.0K-Per-IntGai_conVen",
        "TRY2045_474856110632_Somm_B1948_standard_SingleDwelling_NoDHW_3.0K-Per-IntGai_dynVen"
    ]
    # results_special = []

    for name, results in results.items():
        if results_special and name not in results_special:
            continue
        all_results[name] = calc_single_case(
            study_path=study_path, with_dhw=with_dhw, with_plot=bool(results_special) or with_plot,
            name=name,
            **results)
        # if len(all_results) > 50:
        #     break
    if not results_special:
        df = pd.DataFrame(all_results).transpose()
        df.to_excel(study_path.joinpath("ResultsEstimation.xlsx"))


def set_same_lims_to_all_axes(axes):
    min_x, max_x, max_y, min_y = 0, 0, 0, 0
    for ax in axes:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if xmax > max_x:
            max_x = xmax
        if xmin < min_x:
            min_x = xmin
        if ymax > max_y:
            max_y = ymax
        if ymin < min_y:
            min_y = ymin
    for ax in axes:
        ax.set_ylim([min_y, max_y])
        ax.set_xlim([min_x, max_x])


def generate_boxplots_for_options(save_path: Path):
    df = pd.read_excel(save_path.joinpath("ResultsEstimation.xlsx"), index_col=0)
    _generate_boxplots_for_options(df=df, save_path=save_path, filter_name="")


def _generate_boxplots_for_options(df: pd.DataFrame, save_path: Path, filter_name: str = "",
                                   single_options: bool = True):
    save_path_plot = save_path.joinpath(f"boxplotPlots{filter_name}")
    os.makedirs(save_path_plot, exist_ok=True)
    statistics = {}
    masks = {}
    data_cases = {"all": df}

    if not single_options:
        plot_boxplot(data=data_cases, save_path=save_path_plot, study_name=f"{filter_name}_all", statistics=statistics)

    df.loc[:, "index"] = df.index

    def _in(x, v):
        return v in x

    def endswith(x, v):
        return x.endswith(v)

    def startswith(x, v):
        return x.startswith(v)

    def _custom(x, v):
        return "_retrofit" in x and "adv_retrofit" not in x

    filters = {}
    for building_standard in ["_standard", "_adv_retrofit"]:
        filters[f"building{building_standard}"] = ("building", (_in, building_standard))
    filters["building_retrofit"] = ("building", (_custom, 0))

    for forecast in ["2015", "2045"]:
        filters[f"weather{forecast}"] = ("weather", (_in, forecast))

    for year in ["warm", "cold", "average"]:
        filters[f"weather{year}"] = ("weather", (endswith, year))

    for setback in ["0.0K", "3.0K"]:
        filters[f"setback{setback}"] = ("user", (startswith, setback))

    for int_gains in ["NoPer-NoIntGai", "Per-IntGai"]:
        filters[f"intgains{int_gains}"] = ("user", (endswith, int_gains))

    for ventilation in ["dynVen", "conVen"]:
        filters[f"ventilation{ventilation}"] = ("index", (endswith, ventilation))

    for case_name, filter_column_function in filters.items():
        column, filter_function_args = filter_column_function
        mask = df.loc[:, column].apply(filter_function_args[0], v=filter_function_args[1])
        if not np.any(mask):
            continue
        data_cases[case_name] = df.loc[mask].copy()
        masks[case_name] = mask

    plot_boxplot(data=data_cases, save_path=save_path_plot, study_name=filter_name, statistics=statistics)
    df_stats = pd.DataFrame(statistics).transpose()
    df_stats.to_excel(save_path.joinpath(f"Statistics{filter_name}.xlsx"))
    return masks


def create_boxplots_by_order_of_magnitude(save_path: Path):
    df = pd.read_excel(save_path.joinpath("ResultsEstimation.xlsx"), index_col=0)
    masks = _generate_boxplots_for_options(df=df, save_path=save_path, filter_name="")
    for filter_name, mask in masks.items():
        _generate_boxplots_for_options(
            df=df.loc[mask], save_path=save_path,
            filter_name=filter_name.capitalize(), single_options=True
        )


def create_special_masks(save_path: Path):
    df = pd.read_excel(save_path.joinpath("ResultsEstimation.xlsx"), index_col=0)
    masks = _generate_boxplots_for_options(df=df, save_path=save_path, filter_name="")

    def _join_mask(mask_names, masks):
        mask = masks[mask_names[0]]
        for mask_name in mask_names[1:]:
            mask = mask & masks[mask_name]
        return mask

    options = [
        'building_standard', 'building_adv_retrofit', 'building_retrofit',
        'weather2015', 'weather2045',
        'weatherwarm', 'weathercold', 'weatheraverage',
        'setback0.0K', 'setback3.0K',
        'intgainsNoPer-NoIntGai', 'intgainsPer-IntGai',
        'ventilationdynVen', 'ventilationconVen'
    ]

    save_path_plot = save_path.joinpath(f"boxplotPlotsManualCases")
    special_masks = {
        "all": [],
        "real": ["building_standard", "weathercold", "weather2015", "setback3.0K", "intgainsPer-IntGai",
                 "ventilationdynVen"],
        "nc": ["building_standard", "weathercold", "weather2015", "setback0.0K", "intgainsNoPer-NoIntGai",
               "ventilationconVen"],
        "nc_warm": ["building_standard", "weatherwarm", "weather2045", "setback0.0K", "intgainsNoPer-NoIntGai",
                    "ventilationconVen"],
        "nc_dyn": ["building_standard", "weathercold", "weather2015", "setback0.0K", "intgainsNoPer-NoIntGai",
                   "ventilationdynVen"],
        "nc_users": ["building_standard", "weathercold", "weather2015", "setback0.0K", "intgainsPer-IntGai",
                     "ventilationconVen"],
        "nc_3K": ["building_standard", "weathercold", "weather2015", "setback3.0K", "intgainsPer-IntGai",
                  "ventilationconVen"],
        "nc_ret": ["building_adv_retrofit", "weathercold", "weather2015", "setback0.0K", "intgainsNoPer-NoIntGai",
                   "ventilationconVen"],
        "worst_max": ["building_adv_retrofit", "weatherwarm", "weather2045", "setback0.0K", "intgainsNoPer-NoIntGai",
                      "ventilationconVen"],
        "worst_min": ["building_adv_retrofit", "weatherwarm", "weather2045", "setback3.0K", "intgainsPer-IntGai",
                      "ventilationdynVen"],
    }
    os.makedirs(save_path_plot, exist_ok=True)
    statistics = {}
    data_cases = {}
    for filter_name, mask_names in special_masks.items():
        if mask_names:
            data_cases[filter_name] = df.loc[_join_mask(mask_names, masks)].copy()
        else:
            data_cases[filter_name] = df.copy()
    plot_boxplot(data=data_cases, save_path=save_path_plot, study_name="ManualCases",
                 statistics=statistics)
    df_stats = pd.DataFrame(statistics).transpose()
    df_stats.to_excel(save_path.joinpath(f"StatisticsManualCases.xlsx"))


def plot_boxplot(data: dict, save_path: Path, study_name: str, statistics: dict):
    os.makedirs(save_path, exist_ok=True)

    for metric, unit in zip(["error", "difference"], ["%", "kW"]):
        fig, axes = plt.subplots(1, 6, sharex=True, sharey=True,
                                 figsize=get_figure_size(n_columns=4, height_factor=2), squeeze=False)
        for name, ax in zip(name_map.keys(), axes.flatten()):
            column = f"{name} {metric}"
            data_plot = {}
            idx = 0
            for case_name, df in data.items():
                data_plot[idx] = df.loc[:, column]
                _statistics_case = statistics.get(case_name, {})
                _statistics_case[f"{column} std"] = df.loc[:, column].std()
                _statistics_case[f"{column} mean"] = df.loc[:, column].mean()
                statistics[case_name] = _statistics_case
                idx += 1

            plot_box(df=pd.DataFrame(data_plot), orient='v', axes=ax)
            ax.set_title(name_map[name])

        set_same_lims_to_all_axes(axes.flatten())
        for ax in axes[0, :]:
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        axes[0, 0].set_ylabel(f"Error in {unit}")
        fig.tight_layout()
        fig.savefig(save_path.joinpath(f"boxplots_{study_name}_{metric}.png"))
        plt.close("all")
    return statistics


def get_datetime_index_hour_conversions(first_index, last_index, index):
    conversion_to_hour = []
    for idx in index[:-1]:
        conversion_to_hour.append((idx - first_index).total_seconds() / 3600)
        first_index = idx
    conversion_to_hour.append((last_index - index[-2]).total_seconds() / 3600)
    return np.array(conversion_to_hour)


def calc_single_case(study_path: Path, df: pd.DataFrame, input_config: InputConfig, with_dhw: bool, with_plot: bool,
                     name: str):
    heating_threshold_temperature = None
    heat_load_en12831 = df.iloc[-1][heat_load_din] / 1000

    # Define threshold above which heating points are relevant
    threshold = heat_load_en12831 * 0.05

    TOda_nominal = input_config.weather.TOda_nominal
    data = {}
    estimations = {}
    for freq in ["h", "d", "w", "m"]:
        if freq == "w":
            _freq_pd = "W-FRI"
            df_to_sample = df.iloc[:-25]  # Omit single day
        else:
            _freq_pd = freq
            df_to_sample = df.iloc[:-25]  # single hour
        df_heater = df_to_sample.iloc[:-1][heater_name].resample(_freq_pd).sum() / 1000
        if freq == "d":
            df_heater /= 24
        if freq in ["w", "m"]:
            df_heater /= get_datetime_index_hour_conversions(
                first_index=df_to_sample.index[0],
                last_index=df_to_sample.index[-1],
                index=df_heater.index
            )

        df_toda_mean = df_to_sample.iloc[:-1][outdoor_air_name].resample(_freq_pd).mean()
        df_sampled = pd.DataFrame(
            {heater_name: df_heater.values, outdoor_air_name: df_toda_mean.values},
            index=df_heater.index,
        )
        # ######################### Add DHW to DataFrame ##########################
        if with_dhw:
            raise Exception("Not yet implemented")
        else:
            dhw_df = 0
        df_sampled["Q_flow_total"] = df_sampled[heater_name] + dhw_df

        model, r_sq, p_mean_dhw, summer_temperature, heating_threshold_temperature = calc_linear_regression(
            threshold=threshold,
            df=df_sampled, with_dhw=with_dhw,
            heating_threshold_temperature=heating_threshold_temperature
        )
        data[freq] = {"df": df_sampled, "model": model}
        heat_load_estimated = model.predict([[TOda_nominal - 273.15]])[0]
        estimations[freq] = heat_load_estimated
    plot_path = study_path.joinpath("plots")
    os.makedirs(plot_path, exist_ok=True)
    if with_plot:
        kwargs = dict(
            data=data,
            TOda_nominal=TOda_nominal,
            heat_load=heat_load_en12831,
            constant_summer_demand=p_mean_dhw,
            heating_threshold_temperature=heating_threshold_temperature,
            summer_temperature=summer_temperature
        )
        plot_heat_load_estimation(
            show_estimation=False,
            show_measurement_points=False,
            save_path=plot_path.joinpath(f"{name}_hl_only.png"),
            **kwargs
        )
        plot_heat_load_estimation(
            show_estimation=False,
            show_measurement_points=True,
            save_path=plot_path.joinpath(f"{name}_with_points.png"),
            **kwargs
        )
        plot_heat_load_estimation(
            show_estimation=True,
            show_measurement_points=True,
            save_path=plot_path.joinpath(f"{name}.png"),
            **kwargs
        )
    kwargs = dict(
        df_daily=data["d"]["df"], TOda_nominal=TOda_nominal,
        net_leased_area=input_config.building.net_leased_area, with_dhw=with_dhw
    )
    # heating_threshold_temperature = get_heating_threshold_temperature_for_building(building=input_config.building)
    heating_threshold_temperature_for_annual = max(283.15, min(heating_threshold_temperature, 293.15))
    estimations["GTZ"], estimations["BG"] = heat_load_from_yearly_results(
        room_temperature=heating_threshold_temperature_for_annual,
        heating_threshold_temperature=heating_threshold_temperature_for_annual,
        **kwargs
    )

    data = {"EN12831": heat_load_en12831, **estimations}
    for name, estimation in estimations.items():
        data[f"{name} difference"] = estimation - heat_load_en12831
        data[f"{name} error"] = (estimation - heat_load_en12831) / heat_load_en12831 * 100
    data = {k: round(v, 2) for k, v in data.items()}
    data["weather"] = input_config.weather.get_name(pretty_print=True)
    data["building"] = input_config.building.get_name()
    data["user"] = input_config.user.get_name()
    data["dhw"] = with_dhw
    return data


def merge_results_and_update_name(study_path):
    all_results = {}
    all_files = [
        "ExtractedSimulationResults_conVen",
        "ExtractedSimulationResults_dynVen_extra",
        "ExtractedSimulationResults_dynVen",
        "ExtractedSimulationResults_conVen_extra"
    ]

    for fname in all_files:
        with open(study_path.joinpath(f"{fname}.pickle"), "rb") as file:
            results = pickle.load(file)

        print(f"Single length {fname}:", len(results))
        ven = fname.replace("_extra", "").split("_")[-1]
        for res in results.values():
            new_name = res["input_config"].get_name() + f"_{ven}"
            all_results[new_name] = res
    print("Merged results length", len(all_results))
    with open(study_path.joinpath(f"joined_results.pickle"), "wb") as file:
        pickle.dump(all_results, file)


def plot_b_vf():
    from bes_rules.boundary_conditions.weather import get_weather_configs_by_names, WeatherConfig
    from bes_rules.utils.functions import get_heating_degree_days
    from bes_rules.plotting import utils, EBCColors
    from ebcpy.preprocessing import convert_index_to_datetime_index
    from bes_rules import LATEX_FIGURES_FOLDER

    utils.load_plot_config()

    config = get_weather_configs_by_names(region_names=["Potsdam"])[0]
    #config = WeatherConfig(dat_file=Path(r"D:\TRY2015_476354130073_Jahr.dat"), TOda_nominal=273.15 - 16)
    df = config.get_hourly_weather_data()
    TOda = df.loc[:, "t"].resample("D").mean()
    TRoom = 20
    THeaThr = 15
    TOda_nominal = config.TOda_nominal - 273.15
    TOdaHeatPeriod = np.sort(TOda[TOda < THeaThr].values)
    index_heating = len(TOdaHeatPeriod)
    print("Heating days", index_heating)
    TOdaMean = np.mean(TOdaHeatPeriod)
    print(f"{TOdaMean=}")
    fig, ax = plt.subplots(1, 1, figsize=utils.get_figure_size(1.5))
    ax.plot(np.arange(len(TOda)) * 24, np.sort(TOda.values), label="$T_\mathrm{Auß}$", color=EBCColors.blue)
    ax.axhline(TRoom, label="$T_\mathrm{Raum}$", color="red")
    ax.axhline(THeaThr, label="$T_\mathrm{HG}$", color="black")
    ax.axhline(TOda_nominal, label="$T_\mathrm{Auß,Nom}$", color=EBCColors.dark_red)
    ax.fill_between(
        np.arange(index_heating) * 24, TOdaHeatPeriod, [TRoom] * index_heating,
        label="$GTZ_\mathrm{Raum,HG}$", color="red", alpha=0.2)
    ax.axhline(TOdaMean, label="$\\bar{T}_\mathrm{Auß,Heiz}$", color="gray", linestyle="--")
    ax.fill_between(
        np.arange(index_heating) * 24, TOdaHeatPeriod, [TOda_nominal] * index_heating,
        label="Volllast", color="gray", alpha=0.2
    )
    ax.set_xlabel("Stunden im Jahr")
    ax.set_ylabel("$T$ in °C")
    GTZ = get_heating_degree_days(
        room_temperature=TRoom,
        heating_threshold_temperature=THeaThr,
        outdoor_air_temperature=df.loc[:, "t"]
    )
    b_vf_T = (THeaThr - TOdaMean) / (THeaThr - TOda_nominal) * 8760
    b_vf_GTZ = GTZ * 24 / (THeaThr - TOda_nominal)
    ax.text(50, 24, "$b_\mathrm{VF,GTZ}=%s$ h" % int(round(b_vf_GTZ, 0)))
    ax.text(50, 21, "$b_\mathrm{VF,\\bar{T}}=%s$ h" % int(round(b_vf_T, 0)))
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    ax.set_xlim([0, 8760])
    fig.tight_layout()

    fig.savefig(LATEX_FIGURES_FOLDER.joinpath("02_sotr", "heizlast_GTZ.png"))
    plt.show()


if __name__ == '__main__':
    plt.rcParams.update(
        {"figure.figsize": [6.24 * 0.5, 5.78 * 0.5],
         "font.size": 14,
         "figure.dpi": 250,
         "figure.autolayout": True
         }
    )
    plot_b_vf()
    raise Exception
    # PlotConfig.load_default()
    PATH = Path(r"D:\07_offline_arbeiten\BAUSim")
    # merge_results_and_update_name(PATH)
    run_all_estimations(study_path=PATH, with_plot=False)
    # create_special_masks(PATH)
