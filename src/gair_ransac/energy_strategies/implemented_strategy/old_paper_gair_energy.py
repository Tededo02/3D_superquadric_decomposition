from .composed_gair_energy import ComposedGairEnergy
from ..pairwise.normal_coherence_pairwise_energy import (
    COHERENCE_MIN,
    OUTLIER_SCALE,
    NormalCoherencePairwiseEnergy,
)
from ..unary.residual_unary_energy import ResidualUnaryEnergy


class OldPaperGairEnergy(ComposedGairEnergy):
    __slots__ = ()

    def __init__(
        self,
        coherence_min: float = COHERENCE_MIN,
        outlier_scale: float = OUTLIER_SCALE,
    ) -> None:
        super().__init__(
            unary=ResidualUnaryEnergy(),
            pairwise=NormalCoherencePairwiseEnergy(
                coherence_min=coherence_min,
                outlier_scale=outlier_scale,
            ),
        )
