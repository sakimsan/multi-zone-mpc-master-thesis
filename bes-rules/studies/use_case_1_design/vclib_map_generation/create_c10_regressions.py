import numpy as np
from vclibpy.utils.ten_coefficient_compressor_reqression import create_regression_data
from vclibpy.media import RefProp
from bes_rules import RESULTS_FOLDER
from eta_mech import get_eta_mech_cases
from custom_compressor import get_vitocal_compressor


def create_regressions():
    from bes_rules import REF_PROP_PATH

    ref_prop = RefProp(fluid_name="Propane", ref_prop_path=REF_PROP_PATH.as_posix())
    cases = get_eta_mech_cases()
    for c10_name in [
        #"10C_WHP07600_corr",
        "10C_WHP07600"
    ]:
        if c10_name.endswith("_corr"):
            n = np.arange(3000, 6001, 1000) / 7200
        else:
            n = np.arange(2000, 6001, 1000) / 7200
        for name in cases:
            print(c10_name, name)
            compressor = get_vitocal_compressor(
                med_prop=ref_prop, eta_mech_name=name,
                c10_name=c10_name, regression=False)
            create_regression_data(
                compressor=compressor,
                save_path=RESULTS_FOLDER.joinpath(
                    "vitocal", "plots_c10",
                    f"new_regressions_{c10_name}_{name}.csv"
                ),
                n=n,
                T_eva=np.arange(-25, 36, 5) + 273.15,
                T_con=np.arange(35, 76, 10) + 273.15,
                with_plot=True
            )


if __name__ == '__main__':
    create_regressions()
