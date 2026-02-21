from bes_rules.objectives.annuity import Annuity

import os.path
import logging
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from bes_rules.objectives import annuity
from bes_rules.plotting import utils


def plot_compare_hp_to_boiler():
    base_data = {
            "outputs.electrical.dis.PEleLoa.integral": 0,
            "outputs.electrical.dis.PEleGen.integral": 0,
            "outputs.hydraulic.dis.PBoi.integral": 0,
            "hydraulic.generation.eleHea.Q_flow_nominal": 0,
            "PElePVMPP": 0,
            "hydraulic.distribution.parStoDHW.V": 0,
            "hydraulic.distribution.parStoBuf.V": 0,
            "QHeaPum_flow_A2W35": 0,
            "hydraulic.distribution.boi.Q_nom": 0
        }
    cases = [
        ["QHeaPum_flow_A2W35", np.linspace(5000, 30000, 100)],
        ["hydraulic.distribution.boi.Q_nom", np.linspace(5000, 30000, 100)],
    ]
    annuities = {
        "Vering et al.": annuity.AnnuityVeringEtAl(),
        "KWP": annuity.TechnikkatalogAssumptions(),
    }
    plot_kwargs = {
        "Vering et al.": dict(label="Vering et al.", color="blue"),
        "KWP": dict(label="KWP", color="red"),
    }
    plot_config = utils.load_plot_config(language="de")
    fig, axes = plt.subplots(2, 1, figsize=utils.get_figure_size(1, 1.5))
    for ann_name, ann in annuities.items():
        for ax, case in zip(axes, cases):
            var, values = case
            df = pd.DataFrame({var: values})
            for k, v in base_data.items():
                if k != var:
                    df.loc[:, k] = v
            df = ann.calc(df)
            df = plot_config.scale_df(df)
            ax.plot(df.loc[:, var], df.loc[:, "invest_total"], **plot_kwargs[ann_name])
            ax.set_ylabel(plot_config.get_label_and_unit("invest_total"))
            ax.set_xlabel(plot_config.get_label_and_unit(var))
            ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def plot_compare_invests():
    base_data = {
            "outputs.electrical.dis.PEleLoa.integral": 0,
            "outputs.electrical.dis.PEleGen.integral": 0,
            "outputs.hydraulic.dis.PBoi.integral": 0,
            "hydraulic.generation.eleHea.Q_flow_nominal": 0,
            "PElePVMPP": 0,
            "hydraulic.distribution.parStoDHW.V": 0,
            "hydraulic.distribution.parStoBuf.V": 0,
            "QHeaPum_flow_A2W35": 0
        }
    hp = [
        "QHeaPum_flow_A2W35", np.linspace(5000, 30000, 100),
        #"hydraulic.distribution.boi.Q_nom", np.linspace(10, 30, 100),
    ]
    storage_buf = [
        "hydraulic.distribution.parStoBuf.V", np.linspace(0.01, 1, 100)
    ]
    storage_dhw = [
        "hydraulic.distribution.parStoDHW.V", np.linspace(0.01, 1, 100),
    ]
    annuities = {
        "Vering et al.": annuity.AnnuityVeringEtAl(),
        "KWP": annuity.TechnikkatalogAssumptions(),
    }
    plot_kwargs = {
        "Vering et al.": dict(label="Vering et al.", color="blue"),
        "KWP": dict(label="KWP", color="red"),
    }
    cases = {
        "WP": hp,
        "PS": storage_buf,
        "TWWS": storage_dhw
    }
    plot_config = utils.load_plot_config(language="de")
    fig, axes = plt.subplots(3, 1, figsize=utils.get_figure_size(1, 2))
    for ann_name, ann in annuities.items():
        for ax, case in zip(axes, cases.values()):
            var, values = case
            df = pd.DataFrame({var: values})
            for k, v in base_data.items():
                if k != var:
                    df.loc[:, k] = v
            df = ann.calc(df)
            df = plot_config.scale_df(df)
            ax.plot(df.loc[:, var], df.loc[:, "invest_total"], **plot_kwargs[ann_name])
            ax.set_ylabel(plot_config.get_label_and_unit("invest_total"))
            ax.set_xlabel(plot_config.get_label_and_unit(var))
            ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def plot_heating_rod():
    i_aeh_0 = -0.037
    i_aeh_a = 5.73
    i_aeh_exp = 0.494
    QHeaRod = np.linspace(2500, 20000, 100)
    invest_aeh = i_aeh_0 + i_aeh_a * QHeaRod ** i_aeh_exp
    # Linear
    #QTaylor = 12500
    #i_linear = i_aeh_a * i_aeh_exp * QTaylor ** (i_aeh_exp - 1)
    #i_linear_0 = i_aeh_0 + i_aeh_a * QTaylor ** i_aeh_exp - i_linear * QTaylor
    #invest_aeh_linear = i_linear_0 + i_linear * QHeaRod
    i_aeh_0_lin = 256.5707509012719
    i_aeh_a_lin = 0.026753483970960015
    i_aeh_exp_lin = 1
    invest_aeh_lin = i_aeh_0_lin + i_aeh_a_lin * QHeaRod ** i_aeh_exp_lin
    #annuity.create_linear_cost_regression()

    #print(f"{i_linear_0=}, {i_linear=}")
    plt.plot(QHeaRod, invest_aeh, color="blue")
    plt.plot(QHeaRod, invest_aeh_lin, color="red")
    plt.show()


if __name__ == '__main__':
    #plot_compare_hp_to_boiler()
    #plot_compare_invests()
    plot_heating_rod()
