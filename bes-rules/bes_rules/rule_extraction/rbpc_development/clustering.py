from __future__ import division

import datetime
import logging
import pathlib
import pickle

import numpy as np
import pandas as pd
import os
from bes_rules.rule_extraction.clustering import clustering_medoid
from bes_rules.rule_extraction.rbpc_development import utils
from bes_rules.rule_extraction.rbpc_development import plotting

logger = logging.getLogger(__name__)


def save_results_as_pickle(results, save_path: pathlib.Path):
    with open(save_path.joinpath("clustering_results.pickle"), "wb") as file:
        pickle.dump(results, file)


def load_results_from_pickle(save_path: pathlib.Path):
    with open(save_path.joinpath("clustering_results.pickle"), "rb") as file:
        return pickle.load(file)


def cluster_filter(
        df: pd.DataFrame,
        with_pv: bool = True,
        with_no_demand: bool = False,
        with_valve: bool = True
):

    if with_pv:
        # If PV=0, set TBuf=THeaCur (mpc optimized the heat curve)
        mask_no_pv = df.loc[:, "P_el_pv"] == 0
        df.loc[mask_no_pv, "TBufSet"] = df.loc[mask_no_pv, "THeaCur"]

    if with_no_demand:
        # If no demand, set T_mean>T_HG
        mask_no_demand = df.loc[:, "T_Mean"] >= 288.15
        df.loc[mask_no_demand, "TBufSet"] = df.loc[mask_no_demand, "THeaCur"]

    if with_valve:
        # If valve closed, set TBuf=THeaCur
        mask_no_mdot = df.loc[:, "yValSet"] <= 0.01
        df.loc[mask_no_mdot, "TBufSet"] = df.loc[mask_no_mdot, "THeaCur"]

    return df


def perform_clustering(save_path: pathlib.Path):
    df = utils.load_results(save_path=save_path)
    n_days = len(utils.split_df_into_days(df))
    df = cluster_filter(df)

    # Timeseries to cluster
    inputs_cluster = df.loc[:, "TBufSet"] - df.loc[:, "THeaCur"]
    yValSet = df.loc[:, "yValSet"]

    results_per_day = {}
    for i in range(1, n_days):

        logger.info("Number of typical days:" + str(i))
        number_clusters = i

        (inputs, nc, z, inputsTransformed, obj) = clustering_medoid.cluster(
            np.array([inputs_cluster]),
            number_clusters,
            n_days=n_days,
            norm=2,
            mip_gap=0
        )

        logger.info("Clustering with %s clusters successful!", i)

        # only 0 and 1
        for m in range(len(z)):
            for n in range(len(z)):
                if z[n, m] < 1:
                    z[n, m] = 0

        # build matrix map_days indicating which day of the year is assigned to which typeday
        map_days = np.zeros([len(z), len(z)])
        times_cluster = np.sum(z, axis=1)
        j = 1
        for t in range(len(times_cluster)):
            if times_cluster[t] > 0:
                map_days[t, :] = z[t, :] * j
                j = j + 1

        clustered_timeseries = []

        for m in range(len(z)):
            for n in range(len(z)):
                if map_days[n, m] > 0:
                    clustered_timeseries = np.append(clustered_timeseries,
                                                     inputs[0][int(map_days[n, m] - 1)])

        # save resulting clusterseries and clusterdays in Excel
        clusterseries = []
        clusterdays = []
        for x in range(len(map_days)):
            for y in range(len(map_days)):  # Anzahl der ausgewählten Tage (Bsp. 744 --> Von 0 - 30)
                if map_days[y, x] != 0:  # Alle Spalten durchgehen --> Immer nur beim aktuellen Tag durch Clustertag ist das !=0
                    clusterdays.append(y)
        results = pd.DataFrame(columns=["Clusterseries", "Clusterdays"])
        results["Clusterdays"] = clusterdays  # save the number of the clusterday for all days
        start_day = inputs_cluster.index[0]
        for day in clusterdays:
            data_values = list(np.array([
                inputs_cluster.loc[
                    start_day + datetime.timedelta(day):
                    start_day + datetime.timedelta(day + 1)
                ],
                yValSet.loc[
                    start_day + datetime.timedelta(day):
                    start_day + datetime.timedelta(day + 1)
                ]]
            ))
            clusterseries.append(data_values)
        results["Clusterseries"] = clusterseries  # save the series (96) of the clusterday for every day
        results_per_day[i] = {"results": results, "nc": nc, "z": z, "obj": obj}
    save_results_as_pickle(results_per_day, save_path)

    return results_per_day


def plot_results(save_path: pathlib.Path, results_per_day: dict = None):
    if results_per_day is None:
        results_per_day = load_results_from_pickle(save_path)
    save_path_plots = save_path.joinpath("Clustering_plots")
    os.makedirs(save_path_plots, exist_ok=True)
    plotting.plot_convergence_of_clustering(save_path=save_path_plots, cluster_results=results_per_day)
    plotting.plot_convergence_of_clustering(save_path=save_path_plots, cluster_results=results_per_day, n_days=10)
    plotting.plot_convergence_of_clustering(save_path=save_path_plots, cluster_results=results_per_day, n_days=20)
    plotting.plot_spread_of_clusters(save_path=save_path_plots, cluster_results=results_per_day, n_days=7)

    for n_clusters, _day_res in results_per_day.items():
        if n_clusters > 7:
            continue
        for with_filter in [True, False]:
            if not with_filter:
                save_path_cluster = save_path.joinpath("Clustering_plots_without_filter")
                os.makedirs(save_path_cluster, exist_ok=True)
            else:
                save_path_cluster = save_path_plots
            plotting.plot_clustered_days(
                save_path_plots=save_path_cluster, with_filter=with_filter, save_path=save_path,
                z=_day_res["z"], n_clusters=n_clusters
            )
        columns = ["T_Air", "T_IntWall", "yValSet", "TBufSet", "P_PV", "T_outdoor_air", "THeaCur"]
        for column in columns:
            plotting.plot_clustered_days_other(
                save_path_plots=save_path_plots, save_path=save_path,
                z=_day_res["z"], n_clusters=n_clusters,
                column=column
            )


if __name__ == '__main__':
    SAVE_PATH = pathlib.Path(r"D:\fwu\02_Paper\zcbe\open_loop").joinpath(f"start=274_stop=120")
    plot_results(SAVE_PATH)

