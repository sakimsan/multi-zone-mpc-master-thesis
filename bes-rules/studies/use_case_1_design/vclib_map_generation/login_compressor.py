from vclibpy.components.compressors import Compressor
from typing import Union
from vclibpy.datamodels import Inputs
import numpy as np
from bes_rules.utils.function_fit import create_variables_n_degree


class LoginCompressor(Compressor):
    """
    Compressor model based on calibration on LOGIN test bench.
    Eta mechanic is not known, only the global efficiency.
    Using an assumed eta_mech, eta_isentropic is calculated
    as eta_glob = eta_isentropic * eta_mech.
    """

    def __init__(
            self,
            N_max,
            V_h,
            assumed_eta_mech: Union[float, callable],
            degree_fit: int = 3,
            **kwargs
    ):
        super().__init__(N_max=N_max, V_h=V_h, **kwargs)
        self.degree_fit = degree_fit
        assert self.degree_fit in [2, 3], "Degree not implemented"
        # Don't use lambda function to cast float as a function,
        # as local functions are not pickable for multiprocessing
        self.assumed_eta_mech = assumed_eta_mech

    def get_lambda_h(self, inputs: Inputs) -> float:
        if self.degree_fit == 2:
            # Second order lambda_h
            intercept = 0.9167132247516594
            coef = [0.0008898143628659307, -0.0012940109700567543, -0.002546368506531788, -9.05779156791151e-06,
                    3.313671746211247e-05, -5.886139096100636e-06, -3.7357391015832094e-05, 8.810020740751692e-05,
                    2.9551303920361886e-06]
        elif self.degree_fit == 3:
            # Third order lambda_h
            intercept = 0.9191351156118762
            coef = [0.0007470167186786726, -2.454195154994333e-06, -0.014149295885128286, -4.074890379948167e-05,
                    0.00010026302317311672, 3.53179831713651e-05, -0.00011902492787926385, 0.0005286034034670722,
                    5.9798453718217625e-05, 3.3797757639546185e-07, -6.70138814037973e-07, 4.8415145329317023e-08,
                    2.540060360735842e-07, -9.57153192244725e-07, -8.477912528605248e-07, 5.305832745305081e-07,
                    -3.7776806409926137e-06, -4.5921118438775046e-07, 1.979261809738387e-07]
        else:
            raise ValueError
        return self._calc_eta(intercept=intercept, coef=coef, inputs=inputs, p_outlet=self.state_outlet.p)

    def get_eta_isentropic(self, p_outlet: float, inputs: Inputs) -> float:
        if self.degree_fit == 2:
            # Second order eta_glob
            intercept = 0.5873778594449023
            coef = [0.0004132049652064935, 0.004811123139165324, 0.0022081497815395834, -2.325176271757677e-05,
                    5.027643325596488e-05, -2.021416604924142e-06, -0.00010322697403688158, 1.7533417775361384e-05,
                    -1.3606421116886521e-05]
        elif self.degree_fit == 3:
            # Third order eta_glob
            intercept = 0.6758294984305762
            coef = [0.00039697672459525186, -1.1181387161012293e-06, -0.00013335017545504046, -6.745330220099768e-05,
                    0.00013590283310495446, 8.698005498548864e-05, -5.4231703392526695e-05, 3.3482125282591256e-05,
                    -0.0001204669486417986, 3.8566804208943575e-07, -5.22938289292895e-07, -3.401137686995684e-07,
                    -1.3010642894545037e-07, -8.786659324995106e-07, -1.0773066407562774e-06, -2.9636312857212423e-07,
                    1.478510686292623e-07, 1.991275588133228e-06, 5.488874730900405e-06]
        else:
            raise ValueError
        eta_glob = self._calc_eta(intercept=intercept, coef=coef, inputs=inputs, p_outlet=p_outlet)
        eta_mech = self.get_eta_mech_for_p(p_outlet=p_outlet, inputs=inputs)
        return eta_glob / eta_mech

    def _calc_eta(self, inputs: Inputs, p_outlet: float, coef, intercept):
        # TODO: Use p once available!
        variables = create_variables_n_degree(
            self.degree_fit,
            [self.get_n_absolute(inputs.control.n)],
            [inputs.condenser.T_out - 273.15],
            [inputs.evaporator.T_in - 273.15]
        )
        #variables = create_variables_n_degree(
        #    self.degree_fit,
        #    [self.get_n_absolute(inputs.control.n)],
        #    [self.state_inlet.p / 1e5],
        #    [p_outlet / 1e5]
        #)
        assert variables.shape[1] == 1, "This function only support scalar inputs"
        return (intercept + np.matmul(coef, variables))[0]

    def get_eta_mech(self, inputs: Inputs) -> float:
        return self.get_eta_mech_for_p(p_outlet=self.state_outlet.p, inputs=inputs)

    def get_eta_mech_for_p(self, p_outlet, inputs: Inputs):
        if callable(self.assumed_eta_mech):
            return self.assumed_eta_mech(self=self, p_outlet=p_outlet, inputs=inputs)
        return self.assumed_eta_mech
