import datetime
import json
import logging
from pathlib import Path

from agentlib.utils.multi_agent_system import LocalMASAgency

logger = logging.getLogger(__name__)


def run(
        start_time=22 * 24 * 3600,
        n_days=7,
        dt: int = 1
):
    # configs
    env_config = {"rt": False, "t_sample": 180, "offset": start_time, "clock": True}
    stop_time = start_time + n_days * 86400
    path_failed_run = Path(r"D:\fwu-ssr\MonovalentVitoCal_HOM_2h_Abwesend\DesignOptimizationResults\TRY2015_523845130645_Jahr_NoRetrofit1983_SingleDwelling_NoDHW_0K-Per-IntGai_\generated_configs_Design_0")

    with open(path_failed_run.joinpath("simulator.json"), "r") as file:
        simulator_config = json.load(file)

    simulator_config["modules"][1]["model"]["dt"] = dt

    agent_configs = [
        path_failed_run.joinpath("predictor.json"),
        simulator_config,
    ]

    # run
    logging.basicConfig(level=logging.INFO)
    mas = LocalMASAgency(
        agent_configs=agent_configs,
        env=env_config,
        variable_logging=True,
    )
    sim_start = datetime.datetime.now()
    logger.info("Simulation start: %s", sim_start)
    mas.run(until=stop_time)
    sim_end = datetime.datetime.now()
    logger.info("Simulation End: %s", sim_end)
    results = mas.get_results(cleanup=False)
    df = results["simulator"]["sim"]
    df.to_excel(path_failed_run.joinpath("sim_test.xlsx"))


if __name__ == '__main__':
    run(dt=1)
