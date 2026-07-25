from dataclasses import dataclass

import numpy as np

from ...consensus import distance_err
from ..context import EnergyContext
from ..gair_energy_strategy import GairEnergyStrategy
from ..graph_cut_energy import GraphCutEnergy
from ..unary.residual_unary_energy import ResidualUnaryEnergy


@dataclass(frozen=True, slots=True)
class GcRansacEnergy(GairEnergyStrategy):
    def build(self, context: EnergyContext) -> GraphCutEnergy:
        residual = distance_err(
            context.model,
            context.points,
            error_metric=context.error_metric,
        )
        unary_costs = ResidualUnaryEnergy().build(context, residual)
        normalized_residual = unary_costs.inlier
        inlier_cost = unary_costs.inlier.copy()
        outlier_cost = unary_costs.outlier.copy()
        if not context.edges.size:
            return GraphCutEnergy(
                inlier_cost=inlier_cost,
                outlier_cost=outlier_cost,
                edge_sources=np.empty(0, dtype=np.int64),
                edge_targets=np.empty(0, dtype=np.int64),
                edge_weights=np.empty(0, dtype=np.float64),
            )

        edge_sources = context.edges[:, 0]
        edge_targets = context.edges[:, 1]
        same_inlier_cost = 0.5 * (
            normalized_residual[edge_sources]
            + normalized_residual[edge_targets]
        )
        same_outlier_cost = 1.0 - same_inlier_cost

        np.add.at(inlier_cost, edge_sources, 0.5 * same_inlier_cost)
        np.add.at(inlier_cost, edge_targets, 0.5 * same_inlier_cost)
        np.add.at(outlier_cost, edge_sources, 0.5 * same_outlier_cost)
        np.add.at(outlier_cost, edge_targets, 0.5 * same_outlier_cost)

        edge_weights = np.full(
            edge_sources.shape[0],
            0.5,
            dtype=np.float64,
        )
        return GraphCutEnergy(
            inlier_cost=inlier_cost,
            outlier_cost=outlier_cost,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            edge_weights=edge_weights,
        )
