"""
Graph-cut energies for labelling a candidate model's points inlier vs outlier.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..superquadrics.superquadric_param import SuperQuadricParams
from .consensus import distance_err, normal_alignment_score

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# full_gair_energy: edges below this normal coherence are dropped entirely, and residuals
# are normalised against OUTLIER_SCALE * eps when scoring a pair as jointly outlying.
COHERENCE_MIN = 0.9
OUTLIER_SCALE = 2.5


@dataclass(frozen=True, slots=True)
class EnergyContext:
    """Everything an energy needs: the points, their neighbourhood graph and the model."""

    points: FloatArray
    edges: IntArray
    normals: FloatArray | None
    model: SuperQuadricParams
    eps: float
    error_metric: str

    @classmethod
    def create(
        cls,
        points: np.ndarray,
        edges: np.ndarray,
        normals: np.ndarray | None,
        model: SuperQuadricParams,
        eps: float,
        error_metric: str,
    ) -> EnergyContext:
        point_array = np.asarray(points, dtype=np.float64)
        edge_array = np.asarray(edges, dtype=np.int64)
        normal_array = None if normals is None else np.asarray(normals, dtype=np.float64)

        if point_array.ndim != 2 or point_array.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {point_array.shape}")
        if normal_array is not None and normal_array.shape != point_array.shape:
            raise ValueError(
                f"normals must have shape {point_array.shape}, got {normal_array.shape}"
            )
        if edge_array.ndim != 2 or edge_array.shape[1] != 2:
            raise ValueError(f"edges must have shape (E, 2), got {edge_array.shape}")
        if edge_array.size and (
            int(edge_array.min()) < 0 or int(edge_array.max()) >= point_array.shape[0]
        ):
            raise ValueError("edges contain point indices outside the valid range")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        return cls(
            points=point_array,
            edges=edge_array,
            normals=normal_array,
            model=model,
            eps=float(eps),
            error_metric=error_metric,
        )


@dataclass(frozen=True, slots=True)
class GraphCutEnergy:
    """A built energy, ready for the min-cut solver."""

    inlier_cost: FloatArray
    outlier_cost: FloatArray
    edge_sources: IntArray
    edge_targets: IntArray
    edge_weights: FloatArray

    def __post_init__(self) -> None:
        if self.inlier_cost.ndim != 1:
            raise ValueError("inlier_cost must be one-dimensional")
        if self.outlier_cost.shape != self.inlier_cost.shape:
            raise ValueError("inlier_cost and outlier_cost must have the same shape")
        if self.edge_sources.ndim != 1:
            raise ValueError("edge_sources must be one-dimensional")

        edge_count = self.edge_sources.shape[0]
        if self.edge_targets.shape != (edge_count,):
            raise ValueError("edge_sources and edge_targets must have the same shape")
        if self.edge_weights.shape != (edge_count,):
            raise ValueError("each pairwise edge must have one weight")


# An energy is any callable turning a context into a built energy.
EnergyFn = Callable[[EnergyContext], GraphCutEnergy]


def _no_edges(point_count: int, inlier_cost: FloatArray,
              outlier_cost: FloatArray) -> GraphCutEnergy:
    return GraphCutEnergy(
        inlier_cost=inlier_cost,
        outlier_cost=outlier_cost,
        edge_sources=np.empty(0, dtype=np.int64),
        edge_targets=np.empty(0, dtype=np.int64),
        edge_weights=np.empty(0, dtype=np.float64),
    )


def gc_ransac_energy(context: EnergyContext) -> GraphCutEnergy:
    """Baseline GC-RANSAC energy, with r_i = clip(d_i / eps, 0, 1).

    U_i(inlier)  = r_i + 1/2 sum_j ((r_i + r_j) / 2)
    U_i(outlier) = 1   + 1/2 sum_j (1 - (r_i + r_j) / 2)
    w_ij         = 1/2
    """
    residual = distance_err(context.model, context.points,
                            error_metric=context.error_metric)
    normalized_residual = np.clip(residual / (context.eps + 1e-12), 0.0, 1.0)

    inlier_cost = normalized_residual.copy()
    outlier_cost = np.ones(context.points.shape[0], dtype=np.float64)

    if not context.edges.size:
        return _no_edges(context.points.shape[0], inlier_cost, outlier_cost)

    edge_sources = context.edges[:, 0]
    edge_targets = context.edges[:, 1]

    same_inlier_cost = 0.5 * (
        normalized_residual[edge_sources] + normalized_residual[edge_targets]
    )
    same_outlier_cost = 1.0 - same_inlier_cost

    np.add.at(inlier_cost, edge_sources, 0.5 * same_inlier_cost)
    np.add.at(inlier_cost, edge_targets, 0.5 * same_inlier_cost)
    np.add.at(outlier_cost, edge_sources, 0.5 * same_outlier_cost)
    np.add.at(outlier_cost, edge_targets, 0.5 * same_outlier_cost)

    return GraphCutEnergy(
        inlier_cost=inlier_cost,
        outlier_cost=outlier_cost,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        edge_weights=np.full(edge_sources.shape[0], 0.5, dtype=np.float64),
    )


def full_gair_energy(
    context: EnergyContext,
    coherence_min: float = COHERENCE_MIN,
    outlier_scale: float = OUTLIER_SCALE,
) -> GraphCutEnergy:
    """Normal-aware GAIR energy. Requires point normals.

    U_i(inlier)  = clip(d_i / eps, 0, 1) + clip((1 - <n_i, n_model_i>) / 2, 0, 1)
    U_i(outlier) = 1 + 1/2 sum_j p_ij
    c_ij = (1 + <n_i, n_j>) / 2,  rho_i = clip(d_i / (outlier_scale eps), 0, 1)
    p_ij = c_ij (1 - (rho_i + rho_j) / 2),  w_ij = c_ij - p_ij / 2

    Only edges with c_ij > coherence_min contribute at all.
    """
    if context.normals is None:
        raise ValueError("full_gair_energy requires point normals")
    if not 0.0 <= coherence_min <= 1.0:
        raise ValueError("coherence_min must be between 0 and 1")
    if outlier_scale <= 0.0:
        raise ValueError("outlier_scale must be positive")

    point_count = context.points.shape[0]
    residual = distance_err(context.model, context.points,
                            error_metric=context.error_metric)

    # ── unary: residual, plus disagreement between the point and model normals ──
    inlier_cost = np.clip(residual / (context.eps + 1e-12), 0.0, 1.0)
    alignment = normal_alignment_score(context.model, context.points, context.normals)
    inlier_cost = inlier_cost + np.clip(0.5 * (1.0 - alignment), 0.0, 1.0)
    outlier_cost = np.ones(point_count, dtype=np.float64)

    if not context.edges.size:
        return _no_edges(point_count, inlier_cost, outlier_cost)

    # ── pairwise: keep only edges whose endpoints have coherent normals ──
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

    valid_edges = coherence > coherence_min
    edge_sources = edge_sources[valid_edges]
    edge_targets = edge_targets[valid_edges]
    coherence = coherence[valid_edges]
    if not edge_sources.size:
        return _no_edges(point_count, inlier_cost, outlier_cost)

    # p_ij for two outliers, 0 for two inliers, c_ij otherwise.
    normalized_residual = np.clip(
        residual / (outlier_scale * context.eps + 1e-12), 0.0, 1.0
    )
    outlier_pair_cost = coherence * (
        1.0
        - 0.5 * (normalized_residual[edge_sources] + normalized_residual[edge_targets])
    )
    np.maximum(outlier_pair_cost, 0.0, out=outlier_pair_cost)

    edge_weights = coherence - 0.5 * outlier_pair_cost
    np.maximum(edge_weights, 0.0, out=edge_weights)

    # half of each edge's outlier cost is charged to each endpoint
    outlier_correction = np.zeros(point_count, dtype=np.float64)
    endpoint_correction = 0.5 * outlier_pair_cost
    np.add.at(outlier_correction, edge_sources, endpoint_correction)
    np.add.at(outlier_correction, edge_targets, endpoint_correction)

    return GraphCutEnergy(
        inlier_cost=inlier_cost,
        outlier_cost=outlier_cost + outlier_correction,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        edge_weights=edge_weights,
    )
