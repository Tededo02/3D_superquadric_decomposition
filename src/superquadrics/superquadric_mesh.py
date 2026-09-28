import numpy as np
import scipy.spatial.transform.rotation
from dataclasses import dataclass
from .superquadric_param import SuperQuadricParams
import trimesh

# to use for not integer exp with negative base
def spow(t, p):
    return np.sign(t) * (np.abs(t) ** p)


def superquadric_mesh(superquadric: SuperQuadricParams):
    # # samples for the two parameters
    n_eta:int = 200
    n_omega:int = 400
    f_points = superquadric_point(superquadric=superquadric,n_eta=n_eta,n_omega=n_omega)
    faces = grid_faces_numba(n_eta,n_omega)
    f_points = apply_pose(f_points,superquadric)
    mesh = trimesh.Trimesh(vertices=f_points,faces=faces,process=False)
    return mesh

# find the function value for some points with the same distance
def superquadric_point (superquadric: SuperQuadricParams,n_eta: int,n_omega: int ):
    eta = np.linspace(-np.pi/2,np.pi/2,n_eta)
    omega = np.linspace(-np.pi,np.pi,n_omega,endpoint=False)
    E,O = np.meshgrid(eta,omega,indexing="ij")

    cos_e = spow(np.cos(E), superquadric.e1)
    sen_e = spow(np.sin(E), superquadric.e1)

    cos_o = spow(np.cos(O), superquadric.e2)
    sen_o = spow(np.sin(O), superquadric.e2)

    # compute xyz for each point
    x = superquadric.a1 * cos_e * cos_o
    y = superquadric.a2 * cos_e * sen_o
    z = superquadric.a3 * sen_e

    # put the coordinates in the same array n_eta*n_omega,3 dimension
    f_points = np.stack([x,y,z], axis=-1).reshape(-1,3)
    return f_points


def triangulate_grid(n_eta,n_omega):
    def vid(i, j): return i*n_omega + j
    faces = []
    for i in range(n_eta - 1):
        for j in range(n_omega):
            j2 = (j + 1) % n_omega
            v00 = vid(i,j)
            v01 = vid(i,j2)
            v10 = vid(i+1,j)
            v11 = vid(i+1,j2)
            faces.append([v00, v11, v10])
            faces.append([v00, v01, v11])
    return np.array(faces)


def apply_pose(points, superquadric:SuperQuadricParams):
    R = superquadric.rotation_matrix()
    t = superquadric.t
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3,)
    return (R @ points.T).T + t[None, :]



#------------------------------------------
from numba import njit
# from the second call, numba compile the code
@njit(cache=True)
def grid_faces_numba(n_eta: int, n_omega: int):
    n_tris = 2 * (n_eta - 1) * n_omega
    F = np.empty((n_tris, 3), dtype=np.int64)

    k = 0
    for i in range(n_eta - 1):
        row0 = i * n_omega
        row1 = (i + 1) * n_omega
        for j in range(n_omega):
            j2 = j + 1
            if j2 == n_omega:
                j2 = 0

            v00 = row0 + j
            v01 = row0 + j2
            v10 = row1 + j
            v11 = row1 + j2

            F[k, 0] = v00
            F[k, 1] = v11
            F[k, 2] = v10
            k += 1

            F[k, 0] = v00
            F[k, 1] = v01
            F[k, 2] = v11
            k += 1

    return F
