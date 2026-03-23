from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.superquadrics.superquadric_param import SuperQuadricParams
from scipy.optimize import least_squares
from .consensus import compute_consensus
from src.superquadrics.superquadric_residual import superquadric_residual_vector


@dataclass
class InnerRansacResult:
    best_model: SuperQuadricParams
    best_inlier_count: int
    best_inliers_mask: np.ndarray

# initialize a superquadric model from a sample of points using PCA and quantiles
def pca_initialization(points: np.ndarray) -> SuperQuadricParams:
    # compute the mean and covariance of the points
    mean = np.mean(points, axis=0)
    cov = np.cov(points - mean, rowvar=False)
    # compute the eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # use the eigenvectors to estimate the rotation of the superquadric
    R = eigvecs
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    # use quantiles (robust extents) in PCA frame to estimate the size of the superquadric
    Xc = (points - mean) @ R
    q_low = np.percentile(Xc, 5.0, axis=0)
    q_high = np.percentile(Xc, 95.0, axis=0)
    half_ranges = 0.5 * (q_high - q_low)
    margin = 1.05
    a1 = max(margin * half_ranges[0], 1e-3)
    a2 = max(margin * half_ranges[1], 1e-3)
    a3 = max(margin * half_ranges[2], 1e-3)
    # use the eigenvectors to estimate the rotation of the superquadric
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    cp = np.cos(pitch)

    if abs(cp) < 1e-8:
        yaw = 0.0
        roll = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    rot = (yaw, pitch, roll)  # (yaw=z, pitch=y, roll=x)
    # use the mean to estimate the translation of the superquadric
    t = mean
    return SuperQuadricParams(a1=a1, a2=a2, a3=a3, e1=1.0, e2=1.0, rot=np.array(rot), t=np.array(t))

# Fit a superquadric to points using non-linear least squares with loss soft_l1.
# Initialization is based on PCA and robust bounding-box estimates.
def fit_superquadric_ls(
    points: np.ndarray,
    error_metric: str = "mix",
    bounds_reference_points: np.ndarray | None = None,
) -> SuperQuadricParams:
    fit_points = np.asarray(points, dtype=np.float64)
    if fit_points.shape[0] < 11:
        raise ValueError("too few points for a stable superqadric fit")
    theta0: SuperQuadricParams
    theta0 = pca_initialization(fit_points)
    bounds_points = fit_points if bounds_reference_points is None else np.asarray(bounds_reference_points, dtype=np.float64)
    if bounds_points.shape[0] == 0:
        raise ValueError("bounds_reference_points must contain at least one point")
    # bounds for stability 
    mins = bounds_points.min(axis=0)
    maxs = bounds_points.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins) + 1e-12)
    a_min = max(1e-3, 5e-2 * diag)
    # MSS samples are local by design, so allow the fit to grow well beyond
    # the patch extent and reach the full object scale during refinement.
    a_max = max(1e-2, 1.2 * diag)
    e_min, e_max = 0.08,4.0
    ang_min, ang_max = -np.pi, np.pi
    t_margin = 0.25 * diag
    lower = np.array([a_min, a_min, a_min, e_min, e_min,ang_min, ang_min, ang_min,mins[0] - t_margin, mins[1] - t_margin, mins[2] - t_margin], dtype=np.float64)
    upper = np.array([a_max, a_max, a_max, e_max, e_max,ang_max, ang_max, ang_max,maxs[0] + t_margin, maxs[1] + t_margin, maxs[2] + t_margin], dtype=np.float64)
    theta0p= np.array([theta0.a1, theta0.a2, theta0.a3, theta0.e1, theta0.e2, theta0.rot[0], theta0.rot[1], theta0.rot[2], theta0.t[0], theta0.t[1], theta0.t[2]], dtype=np.float64)
    # Some MSS samples are locally thin, so the PCA sizes can fall just outside the box constraints.
    # Project the initial guess inside the feasible region to keep least_squares stable.
    theta0p = np.clip(theta0p, lower, upper)
    res = least_squares(
        fun=lambda x, pts: superquadric_residual_vector(
            SuperQuadricParams(
                a1=x[0],
                a2=x[1],
                a3=x[2],
                e1=x[3],
                e2=x[4],
                rot=np.array(x[5:8], dtype=np.float64),
                t=np.array(x[8:11], dtype=np.float64),
            ),
            pts,
            metric=error_metric,
        ),
        x0=theta0p,
        args=(fit_points,),
        method="trf",
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=250
    )

    if not res.success:
        raise RuntimeError(f"least_squares failed: {res.message}")

    a1, a2, a3, e1, e2, yaw, pitch, roll, px, py, pz = res.x.tolist()
    return SuperQuadricParams(
        a1=a1, a2=a2, a3=a3, e1=e1, e2=e2,
        rot=np.array([yaw, pitch, roll], dtype=np.float64),  # (yaw=z, pitch=y, roll=x)
        t=np.array([px, py, pz], dtype=np.float64),
    )


def inner_ransac(
    point_cloud: np.ndarray,
    refined_set_index: np.ndarray,
    actual_set_index: np.ndarray | None,
    threshold: float,
    normals: np.ndarray | None = None,
    error_metric: str = "mix",
    consensus_metric: str | None = None,
    n_iters: int = 50,
    random_seed: int | None = None,
) -> InnerRansacResult:
    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    refined_set_index = np.asarray(refined_set_index, dtype=np.int64)
    actual_points = point_cloud if actual_set_index is None else point_cloud[np.asarray(actual_set_index, dtype=np.int64)]
    actual_normals = None if normals is None else np.asarray(normals, dtype=np.float64)
    bounds_reference_points = point_cloud[refined_set_index]
    points: np.ndarray
    sample_size: int = 30 # minimum number of points to fit a superquadric (11 parameters)
    result: InnerRansacResult
    rng = np.random.default_rng(random_seed)
    model: SuperQuadricParams
    best_model: Optional[SuperQuadricParams] = None
    best_inliers: np.ndarray = np.empty((0,), dtype=bool)
    best_count: int = -1
    if consensus_metric is None:
        consensus_metric = error_metric
    #sampling from gair set
    size_sample=min(np.size(refined_set_index),sample_size)
    for _ in range(n_iters):
        sample_idx = rng.choice(refined_set_index, size=size_sample, replace=False)
        points = point_cloud[sample_idx]
        # model estimation via pca
        try:
            model = fit_superquadric_ls(
                points,
                error_metric=error_metric,
                bounds_reference_points=bounds_reference_points,
            )
        except Exception:
            continue
        inlier_set_index = compute_consensus(
            model,
            actual_points,
            threshold,
            error_metric=consensus_metric,
            normals=actual_normals,
        )
        count = int(np.count_nonzero(inlier_set_index))
        if count > best_count:
            best_count = count
            best_model = model
            best_inliers = np.asarray(inlier_set_index, dtype=bool)

    # after loop: build result
    if best_count < 0 or best_model is None:
        return InnerRansacResult(best_model=SuperQuadricParams(1,1,1,1,1,[0,0,0],[0,0,0]),best_inlier_count=0,best_inliers_mask=np.empty((0,), dtype=bool))
    # final refit using all inliers for better accuracy
    inlier_points = actual_points[best_inliers]
    if inlier_points.shape[0] >= 11:
        try:
            refit_model = fit_superquadric_ls(
                inlier_points,
                error_metric=error_metric,
                bounds_reference_points=inlier_points,
            )
            refit_inlier_set_index = compute_consensus(
                refit_model,
                actual_points,
                threshold,
                error_metric=consensus_metric,
                normals=actual_normals,
            )
            refit_count = int(np.count_nonzero(refit_inlier_set_index))
            if refit_count >= best_count:
                best_model = refit_model
                best_count = refit_count
                best_inliers = np.asarray(refit_inlier_set_index, dtype=bool)
        except Exception:
            pass
    result = InnerRansacResult(best_model=best_model,best_inlier_count=best_count,best_inliers_mask=best_inliers)
    return result

        
