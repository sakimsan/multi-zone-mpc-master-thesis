import numpy as np
import matplotlib.pyplot as plt


def get_radiation_portion(dT, with_surface_temperature_estimate: bool = True, dT_walls: float = 0):
    Q_nominal = 8000
    n = 1.2
    r_nominal = 0.35
    T_supply_nominal = 70 + 273.15
    T_return_nominal = 50 + 273.15
    T_room = 20 + 273.15
    T_mean_nominal = (T_supply_nominal + T_return_nominal) / 2
    dT_nominal = T_mean_nominal - T_room
    T_mean = T_room + dT
    Q = Q_nominal * (dT / dT_nominal) ** n
    Q_nominal_per_Q_per = (dT_nominal / dT) ** n

    T_walls = T_room - dT_walls
    if with_surface_temperature_estimate:
        lambda_steel = 50   # W / mK
        thickness = 5e-3  # m
        sigma = 5.670374419e-8
        A_estimate = r_nominal * Q_nominal / (sigma * (T_mean_nominal ** 4 - T_walls ** 4))
        dT_water_surface = (Q / A_estimate) / (lambda_steel / thickness)
        dT_water_surface_nominal = (Q_nominal / A_estimate) / (lambda_steel / thickness)
    else:
        dT_water_surface = 0
        dT_water_surface_nominal = 0

    T_radiator_surface = T_mean - dT_water_surface
    T_radiator_surface_nominal = T_mean_nominal - dT_water_surface_nominal
    r = r_nominal * (
            (
                    (T_radiator_surface ** 4 - T_walls ** 4) /
                    (T_radiator_surface_nominal ** 4 - T_walls ** 4)
            ) * Q_nominal_per_Q_per
    )
    return r


def plot_r():
    dT = np.linspace(0.01, 40, 1000)
    r_no_res = get_radiation_portion(dT, with_surface_temperature_estimate=False)
    r_res = get_radiation_portion(dT, with_surface_temperature_estimate=True)
    r_cold_walls = get_radiation_portion(dT, with_surface_temperature_estimate=False, dT_walls=1)
    plt.plot(dT, r_no_res, label="Mit Widerstand", color="blue")
    plt.plot(dT, r_res, label="Ohne Widerstand", color="red", linestyle="--")
    plt.plot(dT, r_cold_walls, label="1 K kältere Wände", color="red")
    plt.ylabel("Strahlungsanteil -")
    plt.xlabel("Übertemperatur in K")
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_r()
