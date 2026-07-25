from abc import ABC, abstractmethod

import numpy as np

from ..context import EnergyContext
from .pairwise_costs import PairwiseCosts


class PairwiseEnergyTerm(ABC):
    @abstractmethod
    def build(
        self,
        context: EnergyContext,
        residual: np.ndarray,
    ) -> PairwiseCosts:
        pass
