from dataclasses import dataclass

import numpy as np

from ..context import EnergyContext
from .pairwise_costs import PairwiseCosts
from .pairwise_energy_term import PairwiseEnergyTerm


COHERENCE_MIN = 0.1
OUTLIER_SCALE = 2.5


@dataclass(frozen=True, slots=True)
class NormalCoherencePairwiseEnergy(PairwiseEnergyTerm):
    coherence_min: float = COHERENCE_MIN
    outlier_scale: float = OUTLIER_SCALE

    def __post_init__(self) -> None:
        if not 0.0 <= self.coherence_min <= 1.0:
            raise ValueError("coherence_min must be between 0 and 1")
        if self.outlier_scale <= 0.0:
            raise ValueError("outlier_scale must be positive")

    def build(
        self,
        context: EnergyContext,
        residual: np.ndarray,
    ) -> PairwiseCosts:
        if context.normals is None:
            raise ValueError(
                "NormalCoherencePairwiseEnergy requires point normals"
            )

        point_count = context.points.shape[0]
        if not context.edges.size:
            return PairwiseCosts.empty(point_count)

        normalized_residual = np.clip(
            residual / (self.outlier_scale * context.eps + 1e-12),
            0.0,
            1.0,
        )
        edge_sources = context.edges[:, 0]
        edge_targets = context.edges[:, 1]
        coherence = 0.5 * (
            1.0
            + np.einsum(
                "ij,ij->i",
                context.normals[edge_sources],
                context.normals[edge_targets],
                optimize=True,
            )
        )

        valid_edges = coherence > self.coherence_min
        edge_sources = edge_sources[valid_edges]
        edge_targets = edge_targets[valid_edges]
        coherence = coherence[valid_edges]
        if not edge_sources.size:
            return PairwiseCosts.empty(point_count)

        outlier_pair_cost = coherence * (
            1.0
            - 0.5
            * (
                normalized_residual[edge_sources]
                + normalized_residual[edge_targets]
            )
        )
        np.maximum(outlier_pair_cost, 0.0, out=outlier_pair_cost)

        edge_weights = coherence - 0.5 * outlier_pair_cost
        np.maximum(edge_weights, 0.0, out=edge_weights)

        outlier_correction = np.zeros(point_count, dtype=np.float64)
        endpoint_correction = 0.5 * outlier_pair_cost
        np.add.at(outlier_correction, edge_sources, endpoint_correction)
        np.add.at(outlier_correction, edge_targets, endpoint_correction)

        return PairwiseCosts(
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            edge_weights=edge_weights,
            outlier_correction=outlier_correction,
        )
