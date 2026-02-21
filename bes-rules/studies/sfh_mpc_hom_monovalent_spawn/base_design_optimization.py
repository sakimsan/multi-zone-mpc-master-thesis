import logging
from typing import List

from bes_rules import configs, STARTUP_BESMOD_MOS, BESGRICONOP_PACKAGE_MO, BESRULES_PACKAGE_MO, RESULTS_FOLDER, N_CPU
from bes_rules.input_variations import run_input_variations
from bes_rules.boundary_conditions import weather, building
from bes_rules.simulation_based_optimization.utils import constraints


def get_simulation_config(
        model: str,
        model_hom: str = None,
        start_time: int = 0,
        time_step: int = 600,
        n_days: int = 365,
        equidistant_output: bool = True,
        init_period: int = 86400 * 2,
        **kwargs):
    # Anpassen!
    y_variables = {
        "$T_\mathrm{Oda}$ in °C": "outputs.weather.TDryBul",
        #"$T_\mathrm{Room}$ in °C": ["outputs.building.TZone[1]", "outputs.user.TZoneSet[1]"],
        #"$y_\mathrm{Val}$ in %": "outputs.hydraulic.tra.opening[1]",
        #"$T_\mathrm{DHW}$ in °C": ["outputs.hydraulic.disCtrl.TStoDHWBotMea",
        #                           "outputs.hydraulic.disCtrl.TStoDHWTopMea"],
        #"$T_\mathrm{Buf}$ in °C": ["outputs.hydraulic.disCtrl.TStoBufBotMea",
        #                           "outputs.hydraulic.disCtrl.TStoBufTopMea"],
        #"$T_\mathrm{HeaPum}$ in °C": ["outputs.hydraulic.genCtrl.THeaPumIn",
        #                              "outputs.hydraulic.genCtrl.THeaPumOut"],
        # "$COP$ in -": "outputs.hydraulic.genCtrl.COP",
        #"$y_\mathrm{HeaPum}$ in %": "outputs.hydraulic.genCtrl.yHeaPumSet",
        #"$\dot{Q}_\mathrm{DHW}$ in kW": "outputs.DHW.Q_flow.value",
        #"$\dot{Q}_\mathrm{Bui}$ in kW": "outputs.building.eneBal[1].traGain.value",
        #"$P_\mathrm{el,HeaPum}$": "outputs.hydraulic.gen.PEleHeaPum.value",
        #"$P_\mathrm{el,EleHea}$": "outputs.hydraulic.gen.PEleEleHea.value"
    }
    from bes_rules.configs.plotting import PlotConfig
    plot_settings = dict(
        plot_config=PlotConfig.load_default(),
        y_variables=y_variables
    )

    if model_hom:
        return configs.SimulationConfig(
            startup_mos=STARTUP_BESMOD_MOS,
            model_name=f"BESGriConOp.Studies.SFH.MPCModelROM.BESCISBAT.{model}",
            model_name_hom=f"BESGriConOp.Studies.SFH.MPCModelROM.BESCISBAT.{model_hom}",
            sim_setup=dict(start_time=start_time, stop_time=start_time + 86400 * n_days, output_interval=time_step),
            init_period=init_period,
            packages=[BESGRICONOP_PACKAGE_MO, BESRULES_PACKAGE_MO],
            equidistant_output=equidistant_output,
            plot_settings=plot_settings,
            dymola_api_kwargs={"time_delay_between_starts": 5},
            **kwargs
        )
    else:
        return configs.SimulationConfig(
            startup_mos=STARTUP_BESMOD_MOS,
            model_name=f"BESGriConOp.Studies.SFH.MPCModelROM.BESCISBAT.{model}",
            sim_setup=dict(stop_time=86400 * n_days, output_interval=time_step),
            packages=[BESGRICONOP_PACKAGE_MO, BESRULES_PACKAGE_MO],
            equidistant_output=equidistant_output,
            plot_settings=plot_settings,
            dymola_api_kwargs={"time_delay_between_starts": 5},
            **kwargs
        )


def get_optimization_config(*variables: configs.OptimizationVariable):
    return configs.OptimizationConfig(
        framework="doe",
        method="ffd",
        constraints=[
            constraints.BivalenceTemperatureGreaterNominalOutdoorAirTemperature()
        ],
        variables=variables,
    )

# Copy Paste aus studies.case1
def run_180_cases(test_only=False):
    sim_config = get_simulation_config(model="MonoenergeticVitoCal")
    ## Optimization
    optimization_config = configs.OptimizationConfig(
        framework="doe",
        method="ffd",
        constraints=[constraints.BivalenceTemperatureGreaterNominalOutdoorAirTemperature()],
        variables=[
            configs.OptimizationVariable(
                name="parameterStudy.TBiv",
                lower_bound=273.15 - 16,
                upper_bound=278.15,
                levels=24
            ),
            configs.OptimizationVariable(
                name="parameterStudy.VPerQFlow",
                lower_bound=12,
                upper_bound=12,
                levels=1
            )
        ],
    )
    weathers = weather.get_all_weather_configs()
    weathers = weather.get_weather_configs_by_names(region_names=["Potsdam"])
    buildings = building.get_building_configs_by_name(building_names=["Retrofit1918", "NoRetrofit1983"])
    buildings[0].modify_transfer_system = True
    buildings[1].modify_transfer_system = True
    inputs_config = configs.InputsConfig(
        weathers=weathers,
        buildings=buildings,
        dhw_profiles=[{"profile": "M"}],
    )
    config = configs.StudyConfig(
        base_path=RESULTS_FOLDER.joinpath("UseCase_TBivAndV"),
        n_cpu=N_CPU,
        name="180Cases",
        simulation=sim_config,
        optimization=optimization_config,
        inputs=inputs_config,
        test_only=test_only
    )
    run_input_variations(config=config, run_inputs_in_parallel=False)


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_180_cases(test_only=False)
