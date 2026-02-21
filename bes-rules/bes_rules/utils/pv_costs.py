"""
Source: https://www.verbraucherzentrale.de/aktuelle-meldungen/energie/was-kostet-eine-photovoltaikanlage-49155
"""
import matplotlib.pyplot as plt
import numpy as np

data_per_kwp = {
    4: 1900 * 4,
    6: 1740 * 6,
    8: 1630 * 8,
    10: 1550 * 10,
    12: 1440 * 12,
    14: 1400 * 14,
    16: 1360 * 16,
    18: 1320 * 18,
    20: 1300 * 20
}


def sydlik(P_pv):
    return 2687.7 * P_pv ** 0.7551


if __name__ == '__main__':


    P_PV = np.array(list(data_per_kwp.keys()))
    COSTS = np.array(list(data_per_kwp.values()))

    plt.scatter(P_PV, COSTS)
    plt.plot(P_PV, sydlik(P_PV), label="Sydlik")
    plt.legend()
    plt.show()
