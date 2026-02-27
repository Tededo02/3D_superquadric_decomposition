from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import numpy as np
from superquadric_param import SuperQuadricParams
from scipy.optimize import least_squares
from .consensus import compute_consensus
from .superquadric_residual import implicit_f_superquadric_residual, superquadric_first_order_residual  


@dataclass
class InnerRansacResult:
    best_model: SuperQuadricParams
    best_inlier_count: int

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

# fit superquadric to points using non linear least squares with loss soft_l1.
# initialization via PCA and bbox
def fit_superquadric_ls(points:np.ndarray)-> SuperQuadricParams:
    if points.shape[0]<11:
        raise ValueError ("too few points for a stable superqadric fit")
    theta0: SuperQuadricParams
    theta0 = pca_initialization(points)
    # bounds for stability
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins) + 1e-12)
    a_min = 1e-4 * diag
    a_max = 2.0 * diag
    e_min, e_max = 0.1,3.0
    ang_min, ang_max = -np.pi, np.pi
    t_margin = 0.25 * diag
    lower = np.array([a_min, a_min, a_min, e_min, e_min,ang_min, ang_min, ang_min,mins[0] - t_margin, mins[1] - t_margin, mins[2] - t_margin], dtype=np.float64)
    upper = np.array([a_max, a_max, a_max, e_max, e_max,ang_max, ang_max, ang_max,maxs[0] + t_margin, maxs[1] + t_margin, maxs[2] + t_margin], dtype=np.float64)
    theta0p= np.array([theta0.a1, theta0.a2, theta0.a3, theta0.e1, theta0.e2, theta0.rot[0], theta0.rot[1], theta0.rot[2], theta0.t[0], theta0.t[1], theta0.t[2]], dtype=np.float64)
    res = least_squares(
        fun=lambda x, pts: superquadric_first_order_residual(
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
        ),
        x0=theta0p,
        args=(points,),
        method="trf",
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=200
    )

    if not res.success:
        raise RuntimeError(f"least_squares failed: {res.message}")

    a1, a2, a3, e1, e2, yaw, pitch, roll, px, py, pz = res.x.tolist()
    return SuperQuadricParams(
        a1=a1, a2=a2, a3=a3, e1=e1, e2=e2,
        rot=np.array([yaw, pitch, roll], dtype=np.float64),  # (yaw=z, pitch=y, roll=x)
        t=np.array([px, py, pz], dtype=np.float64),
    )


def inner_ransac(point_cloud: np.ndarray[np.ndarray],refined_set_index: int,actual_set_index: int,threshold: float 
                 ) -> InnerRansacResult:
    points:np.ndarray
    n_iters: int = 50
    sample_size:int = 30 # minimum number of points to fit a superquadric (11 parameters)
    result: InnerRansacResult
    rng = np.random.default_rng()
    model: SuperQuadricParams
    best_model: Optional[SuperQuadricParams] = None
    best_inliers: np.ndarray = np.empty((0,), dtype=bool)
    best_count: int = -1
    #sampling from gair set
    size_sample=min(np.size(refined_set_index),sample_size)
    for i in range(n_iters):
        sample_idx = rng.choice(refined_set_index,size=size_sample,replace=False)
        points=point_cloud[sample_idx]
        # model estimation via pca
        try:
            model = fit_superquadric_ls(points)
        except Exception:
            continue
        inlier_set_index = compute_consensus(model, point_cloud[actual_set_index], threshold)
        count = int(np.count_nonzero(inlier_set_index))
        if count > best_count:
            best_count = count
            best_model = model
            best_inliers = np.asarray(inlier_set_index, dtype=bool)

    # after loop: build result
    if best_count < 0 or best_model is None:
        return InnerRansacResult(best_model=SuperQuadricParams(1,1,1,1,1,[0,0,0],[0,0,0]),best_inlier_count=0)
    # final refit using all inliers for better accuracy
    actual_points = point_cloud[actual_set_index]
    inlier_points = actual_points[best_inliers]
    if inlier_points.shape[0] >= 10:
        try:
            refit_model = fit_superquadric_ls(inlier_points)
            refit_inlier_set_index = compute_consensus(refit_model, actual_points, threshold)
            refit_count = int(np.count_nonzero(refit_inlier_set_index))
            if refit_count >= best_count:
                best_model = refit_model
                best_count = refit_count
        except Exception:
            pass
    result = InnerRansacResult(best_model=best_model,best_inlier_count=best_count)
    return result

        
