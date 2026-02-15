import numpy as np
import scipy.spatial.transform.rotation
from dataclasses import dataclass
from superquadric_param import SuperQuadricParams
import trimesh

# to use for not integer exp with negative base
def spow(t, p):
    return np.sign(t) * (np.abs(t) ** p)


def superquadric_mesh(superquadric: SuperQuadricParams):
    # # samples for the two parameters
    n_eta:int = 200
    n_omega:int = 400
    # compute the points on the superquadric surface
    f_points = superquadric_point(superquadric=superquadric,n_eta=n_eta,n_omega=n_omega)
    # triangulate the grid of points
    faces = triangulate_grid(n_eta,n_omega)
    # apply the rotation and translation to the points
    f_points = apply_pose(f_points,superquadric)
    # create a mesh from the points and faces
    # process = False because the points are already in the correct order and we don't want to change it
    mesh = trimesh.Trimesh(vertices=f_points,faces=faces,process=False)
    return mesh

# find the function value for some points with the same distance
def superquadric_point (superquadric: SuperQuadricParams,n_eta: int,n_omega: int ):

    # create an array of n_eta numbers equidistant between -np.pi/2,np.pi/2
    eta = np.linspace(-np.pi/2,np.pi/2,n_eta)
    # create an array of n_omega numbers equidistant between -np.pi,np.pi
    # endpoint= false because pi and -pi are the same point
    omega = np.linspace(-np.pi,np.pi,n_omega,endpoint=False)
    # create 2 tables where E has the same value repeated for n_eta times in rows
    # and O hase the same value repeated for n_omega times in column
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
    # because my verticis (f_points) are in a list 1D
    def vid(i, j): return i*n_omega + j
    faces = []
    for i in range(n_eta - 1):
        for j in range(n_omega):
            j2 = (j + 1) % n_omega
            v00 = vid(i,   j)
            v01 = vid(i,   j2)
            v10 = vid(i+1, j)
            v11 = vid(i+1, j2)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return np.array(faces, dtype=np.int64)


def apply_pose(points, superquadric:SuperQuadricParams):
    R = superquadric.rotation_matrix()
    t = superquadric.t
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3,)
    return (R @ points.T).T + t[None, :]
