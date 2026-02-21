import matplotlib.pyplot as plt
import numpy as np

from vclibpy.components.compressors import TenCoefficientCompressor


def get_eta_mech_variables(pi, n):
    return [
        pi,
        n,
        pi * n,
        pi ** 2,
        n ** 2,
        pi ** 2 * n,
        pi * n ** 2,
        pi ** 2 * n ** 2,
        n ** 3,
        pi * n ** 3,
        n ** 4
    ]


def rotary_eta_mech(pi, n):
    intercept = 0.2199
    variables = get_eta_mech_variables(pi, n)
    coef = [
        -0.0193,
        0.02503,
        8.817e-5,
        -0.001345,
        -0.0003382,
        1.584e-5,
        -1.083e-6,
        -5.321e-8,
        1.976e-6,
        4.053e-9,
        -4.292e-9
    ]
    return intercept + np.matmul(coef, variables)


def scale_min(arr, new_min):
    old_min = np.min(arr)
    old_max = np.max(arr)

    # Scale factor that preserves the maximum
    scale = (old_max - new_min) / (old_max - old_min)

    # Scale and shift
    return scale * (arr - old_min) + new_min


def alter_eta_mech_regression(eta_mech_min: float, N_max: float = 100):
    pi = np.linspace(1, 10, 20)
    n = np.linspace(0.3, 1, 20) * 100
    pi, n = np.meshgrid(pi, n)
    pi = pi.flatten()
    n = n.flatten()
    y = rotary_eta_mech(pi=pi, n=n)
    variables = get_eta_mech_variables(pi, n)
    y_new = scale_min(y, eta_mech_min)
    from bes_rules.utils.function_fit import fit_linear_regression
    fit_linear_regression(variables=variables, y=y_new, show_plot=True)


def rotary_eta_mech_injected(self: TenCoefficientCompressor, p_outlet, inputs):
    n = self.get_n_absolute(inputs.control.n)
    pi = p_outlet / self.state_inlet.p
    return rotary_eta_mech(pi, n)


def rotary_eta_mech_c10_injected(self: TenCoefficientCompressor, p_outlet, inputs):
    n = self.get_n_absolute(inputs.control.n)
    pi = p_outlet / self.state_inlet.p
    variables = get_eta_mech_variables(pi, n)
    intercept = 0.6360339019657154
    coef = [-0.0070635129982756705, 0.009160607787344403, 3.226890888477046e-05, -0.0004922499989581538,
            -0.00012377617074292186, 5.797204448720155e-06, -3.963618952461205e-07, -1.9474068731984864e-08,
            7.231866155766564e-07, 1.483337728969118e-09, -1.5708081751129688e-09]
    return intercept + np.matmul(coef, variables) + 0.1


def get_eta_mech_cases():
    # Mirko makes no sense as isentropic efficiency is way to high
    return {
        "mirko": rotary_eta_mech_injected,
        "new": rotary_eta_mech_c10_injected,
        "const": 0.95
    }


if __name__ == '__main__':
    alter_eta_mech_regression(eta_mech_min=0.72)
