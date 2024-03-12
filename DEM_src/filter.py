import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix, csr_matrix

from DEM_src.domains import Mesh


def create_density_filter(radius: float, mesh: Mesh):
    # we can't use mesh.x_grid as it has shape (Nx+1, Ny+1)
    x_ray = np.linspace(0, mesh.width, mesh.Nx)
    y_ray = np.linspace(0, mesh.height, mesh.Ny)
    x_grid, y_grid = np.meshgrid(x_ray, y_ray)
    X = x_grid.flatten()
    Y = y_grid.flatten()

    elements = mesh.Nx * mesh.Ny

    wi, wj, wv = [], [], []
    for i in range(elements):
        dist = np.sqrt((X - X[i]) ** 2 + (Y - Y[i]) ** 2)
        (neighbours,) = np.where(dist <= radius)
        wi += [i] * len(neighbours)
        wj += list(neighbours)
        wv += list(radius - dist[neighbours])

    W = normalize(
        coo_matrix((wv, (wi, wj)), shape=(elements, elements)), norm="l1", axis=1
    )  # Normalize row-wise
    assert isinstance(W, csr_matrix)

    return W
