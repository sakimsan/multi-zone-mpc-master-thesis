import os
import locale
import matplotlib.pyplot as plt
from cycler import cycler
from agentlib_mpc.utils.plotting.basic import EBCColors

from studies.use_case_2_pv.peter.evaluation.utils.plot_dictionary import plot_dict


locale.setlocale(locale.LC_ALL, "de_DE")
latex_textwidth = 6.1035    # inches
latex_texthight = 8.74187   # inches


def set_plot_settings():
    plt.rcParams["figure.figsize"] = latex_textwidth, latex_texthight / 3 - 0.4
    plt.rcParams["figure.constrained_layout.use"] = True

    plt.rcParams["axes.prop_cycle"] = cycler(color=EBCColors.ebc_palette_sort_2)

    plt.rcParams["axes.formatter.use_locale"] = True
    plt.rcParams["date.autoformatter.month"] = "%b"
    plt.rcParams["date.autoformatter.day"] = "%d.%m"
    plt.rcParams["date.autoformatter.hour"] = "%H:%M"

    # png
    plt.rcParams["figure.dpi"] = 450
    plt.rcParams["font.family"] = "serif"   # "Arial"
    plt.rcParams["font.size"] = 11      # 16

    # pgf
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pgf.preamble"] = "\n".join([
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath}",
    ])
    plt.rcParams["pgf.rcfonts"] = True
    plt.rcParams["pgf.texsystem"] = "xelatex"


def save_plot(fig: plt.Figure, save_path: str, save_name: str, pgf: bool = False):
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(fname=f"{save_path}/{save_name}.png", format="png")
    print(f"{save_path}/{save_name}.png")
    fig.savefig(fname=f"{save_path}/{save_name}.svg", format="svg")
    if pgf:
        fig.savefig(fname=f"{save_path}/{save_name}.pgf", backend="pgf")
        print(f"{save_path}/{save_name}.pgf")
    plt.close("all")


def get_labels_control(previous_labels: list, controls: list):
    labels = []
    for pre_label in previous_labels:
        if "rbpc_zcbe" in pre_label:
            labels.append(plot_dict["rbpc_zcbe"])
        else:
            for control in controls:
                if control in pre_label:
                    if control in plot_dict:
                        labels.append(plot_dict[control])
                    else:
                        labels.append(control)
    return labels
