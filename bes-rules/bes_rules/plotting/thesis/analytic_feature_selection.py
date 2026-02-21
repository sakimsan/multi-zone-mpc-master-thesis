import json

import matplotlib.pyplot as plt

from bes_rules.objectives.annuity import Annuity, TechnikkatalogAssumptions, get_RBF, StochasticParameter
import numpy as np


def plot_possible_dKInv_dTBiv_values():
    """
    Plot for chapter 3.3.1 feature selection
    """
    kwp_katalog = TechnikkatalogAssumptions()
    TRoom_nominal = 293.15
    vitocal = (17 - 10) * 1000 / 20  # At 55 °C
    optihorst = (15.2 - 10) * 1000 / 20  # at 55°C, 45 °C is similar

    # Values are for all tabula sfh retrofit combis min max
    UA_nominal = np.linspace(96.34358878497272, 1024.5361402162184, 2)
    int_rate = np.linspace(0.02, 0.2, 2)
    f_bet_wp = np.linspace(0, 0.05, 2)
    f_bet_zh = np.linspace(0, 0.05, 2)
    TOda_nominal = np.linspace(-19.2, -6.6, 2)
    THeaThr = np.array([10, 12, 15]) + 273.15
    dQ_WP_dT = np.linspace(optihorst, vitocal, 2)
    k_kap_wp = np.linspace(kwp_katalog.i_hp_a * 0.5, kwp_katalog.i_hp_a * 1.5, 2)
    k_kap_zh = np.linspace(kwp_katalog.i_aeh_a * 0.5, kwp_katalog.i_aeh_a * 1.5, 2)

    meshgrid = np.meshgrid(*[
        UA_nominal,
        int_rate,
        TOda_nominal,
        THeaThr,
        dQ_WP_dT,
        k_kap_wp,
        k_kap_zh,
        f_bet_wp,
        # f_bet_zh,
    ], indexing='ij')

    # Flatten arrays for iteration
    combinations = [arr.flatten() for arr in meshgrid]

    dKInv_dTBivs = []

    for i in range(len(combinations[0])):
        UA_nominal_i = combinations[0][i]
        int_rate_i = combinations[1][i]
        TOda_nominal_i = combinations[2][i]
        THeaThr_i = combinations[3][i]
        dQ_WP_dT_i = combinations[4][i]
        k_kap_wp_i = combinations[5][i]
        k_kap_zh_i = combinations[6][i]
        f_bet_i = combinations[7][i]
        # f_bet_zh_i = combinations[8][i]
        dQ_bui = UA_nominal_i * (TRoom_nominal - TOda_nominal_i) / (THeaThr_i - TOda_nominal_i)
        dKInv_dTBiv = - (
                (
                        (1 / get_RBF(n_years=kwp_katalog.n_years, int_rate=int_rate_i) + f_bet_i) * k_kap_wp_i -
                        (1 / get_RBF(n_years=kwp_katalog.n_years, int_rate=int_rate_i) + f_bet_i) * k_kap_zh_i
                ) * (dQ_bui + dQ_WP_dT_i)
        )
        dKInv_dTBivs.append(dKInv_dTBiv)
    plt.scatter(range(len(dKInv_dTBivs)), dKInv_dTBivs)
    plt.show()


def arg_of_value(arr, value):
    # Calculate the absolute differences between each element and the value
    abs_diff = np.abs(arr - value)
    # Find the index of the element with the smallest absolute difference
    return np.argmin(abs_diff)


def arg_percentile(arr, percentile):
    percentile_value = np.percentile(arr, percentile)
    print(f"{percentile=}, {percentile_value=}")
    return arg_of_value(arr, percentile_value)


def get_f_inv_biv(n_monte_carlo: int = 1e6):
    kwp_katalog = TechnikkatalogAssumptions()

    # Values are for all tabula sfh retrofit combis min max
    int_rate = StochasticParameter(value=0.07, distribution_kwargs=dict(lower_bound=0, upper_bound=0.2))
    f_bet = StochasticParameter(value=0.02, distribution_kwargs=dict(lower_bound=0, upper_bound=0.05))
    i_a_hp = kwp_katalog.i_hp_a
    i_a_aeh = kwp_katalog.i_aeh_a
    k_kap_wp = StochasticParameter(value=i_a_hp,
                                   distribution_kwargs=dict(lower_bound=i_a_hp * 0.5, upper_bound=i_a_hp * 1.5))
    k_kap_zh = StochasticParameter(value=i_a_aeh,
                                   distribution_kwargs=dict(lower_bound=i_a_aeh * 0.5, upper_bound=i_a_aeh * 1.5))
    c_el = StochasticParameter(value=0.35, distribution_kwargs=dict(lower_bound=0.2, upper_bound=0.4))
    f_inv_bivs = []
    all_combinations = []
    for _ in range(int(n_monte_carlo / 1e5)):
        meshgrid = np.meshgrid(*[
            int_rate.get_random_values(10),
            k_kap_wp.get_random_values(10),
            k_kap_zh.get_random_values(10),
            f_bet.get_random_values(10),
            c_el.get_random_values(10),
        ], indexing='ij')

        # Flatten arrays for iteration
        combinations = np.array([arr.flatten() for arr in meshgrid])
        all_combinations.append(combinations)
        int_rate_i = combinations[0, :]
        k_kap_wp_i = combinations[1, :]
        k_kap_zh_i = combinations[2, :]
        f_bet_i = combinations[3, :]
        c_el_i = combinations[4, :]
        f_inv_bivs.append(
            (
                    (1 / get_RBF(n_years=18, int_rate=int_rate_i) + f_bet_i) * k_kap_wp_i -
                    (1 / get_RBF(n_years=18, int_rate=int_rate_i) + f_bet_i) * k_kap_zh_i
            ) / c_el_i)
    f_inv_bivs = np.hstack(f_inv_bivs)
    all_combinations = np.hstack(all_combinations)
    return f_inv_bivs, all_combinations


def create_stochastically_relevant_annuity_configs():
    from bes_rules import DATA_PATH
    f_inv_bivs, all_combinations = get_f_inv_biv()
    percentiles = [0.3, 5, 31.8, 50, 68.2, 95, 99.7]
    data = {}
    for percentile in percentiles:
        arg = arg_percentile(f_inv_bivs, percentile=percentile)
        config = _get_annuity_from_arg(all_combinations[:, arg])
        data[f_inv_bivs[arg]] = config.model_dump()
        print(f"percentile={config}")
    with open(DATA_PATH.joinpath("prices", "f_inv_biv.json"), "w") as file:
        json.dump(data, file, indent=2)


def plot_likely_cost_values():
    """
    Plot for chapter 3.3.1 feature selection
    """
    from bes_rules import LATEX_FIGURES_FOLDER
    from bes_rules import plotting
    f_inv_biv, _ = get_f_inv_biv()
    fig, ax = plt.subplots(1, 1, figsize=plotting.get_figure_size(1, 1))
    print(len(f_inv_biv))
    ax.hist(f_inv_biv, bins=1000, density=True, color=plotting.EBCColors.blue)
    ax.set_ylabel("Wahrscheinlichkeitsdichte")
    ax.set_xlabel("$f_{\mathrm{Inv},x,c_\mathrm{el,Bed}}$ in kWh/kW/a")
    percentiles = [0.3, 5, 31.8, 50, 68.2, 95, 99.7]
    for i, percentile in enumerate(percentiles):
        ax.axvline(
            f_inv_biv[arg_percentile(f_inv_biv, percentile=percentile)],
            color=plotting.EBCColors.red,
            label="Featurewerte" if i == 0 else None
        )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(LATEX_FIGURES_FOLDER.joinpath("Appendix", "4_wp", "f_inv_c.png"))
    plt.show()


def _get_annuity_from_arg(combination):
    return TechnikkatalogAssumptions(
        int_rate=combination[0],
        i_hp_a=combination[1],
        i_aeh_a=combination[2],
        k_op_hp=combination[3],
        k_op_aeh=combination[3],
        k_el=combination[4]
    )


if __name__ == '__main__':
    plot_likely_cost_values()
    # create_stochastically_relevant_annuity_configs()
