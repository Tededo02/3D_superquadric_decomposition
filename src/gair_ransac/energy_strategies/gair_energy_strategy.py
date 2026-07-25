from abc import ABC, abstractmethod

from .context import EnergyContext
from .graph_cut_energy import GraphCutEnergy


class GairEnergyStrategy(ABC):
    @abstractmethod
    def build(self, context: EnergyContext) -> GraphCutEnergy:
        pass
