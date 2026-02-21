import abc
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from bes_rules.plotting.utils import PlotConfig


class Surrogate(abc.ABC):

    def __init__(self, df: pd.DataFrame, **kwargs):
        pass

    @abc.abstractmethod
    def predict(
            self,
            design_variables: Dict[str, np.ndarray],
            metrics: List[str],
            save_path_plot: Path,
            plot_config: PlotConfig
    ) -> pd.DataFrame:
        raise NotImplementedError
