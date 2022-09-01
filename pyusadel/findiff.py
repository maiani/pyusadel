"""
Code for Usadel equation solver.
Andrea Maiani, 2022
"""

import numpy as np
from scipy import sparse


class DifferentialOperators:
    """
    Empty class that can be filled with differential operators.
    """

    def __init__(self):
        self._Nsites = None
        self._D_x = None
        self._D_y = None
        self._D_z = None
        self._L = None

    @property
    def Nsites(self):
        return self._Nsites

    @property
    def D_x(self):
        return self._D_x

    @D_x.setter
    def D_x(self, D_x):
        if self._Nsites:
            if (D_x.shape[0] == self._Nsites) and (D_x.shape[1] == self._Nsites):
                self._D_x = D_x
            else:
                raise ("Error.")
        else:
            if D_x.shape[0] == D_x.shape[1]:
                self._D_x = D_x
                self._Nsites = D_x.shape[0]
            else:
                raise ("Error.")

    @property
    def D_y(self):
        return self._D_y

    @D_y.setter
    def D_y(self, D_y):
        if self._Nsites:
            if (D_y.shape[0] == self._Nsites) and (D_y.shape[1] == self._Nsites):
                self._D_y = D_y
            else:
                raise ("Error.")
        else:
            if D_y.shape[0] == D_y.shape[1]:
                self._D_y = D_y
                self._Nsites = D_y.shape[0]
            else:
                raise ("Error.")

    @property
    def D_z(self):
        return self._D_z

    @D_x.setter
    def D_z(self, D_z):
        if self._Nsites:
            if (D_z.shape[0] == self._Nsites) and (D_z.shape[1] == self._Nsites):
                self._D_z = D_z
            else:
                raise ("Error.")
        else:
            if D_z.shape[0] == D_z.shape[1]:
                self._D_z = D_z
                self._Nsites = D_z.shape[0]
            else:
                raise ("Error.")

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, L):
        if self._Nsites:
            if (L.shape[0] == self._Nsites) and (L.shape[1] == self._Nsites):
                self._L = L
            else:
                raise ("Error.")
        else:
            if L.shape[0] == L.shape[1]:
                self._L = L
                self._Nsites = L.shape[0]
            else:
                raise ("Error.")

    def get_diffops(self):
        return self.D_x, self.D_y, self.D_z, self.L


def gradient(Nx, boundary_condition):

    """
    Generate a discretized gradient matrix.

    Parameters:
    Nx : int
        Number of points
    boundary_condition : {'open', 'periodic'}
        Boundary condition at the edges of the system.

    Returns
    -------
    D_x : array_like
        Discrete Laplacian.
    """

    D_x = sparse.diags(
        [-np.ones(Nx - 1), 0, np.ones(Nx - 1)],
        [-1, 0, 1],
        shape=(Nx, Nx),
        format="lil",
    )

    if boundary_condition == "open":
        D_x[0, 0] = -1
        D_x[0, 1] = 1
        D_x[-1, -2] = -1
        D_x[-1, -1] = 1

    elif boundary_condition == "periodic":
        D_x[0, -1] = -1
        D_x[0, 1] = 1
        D_x[-1, -2] = -1
        D_x[-1, 0] = 1

    else:
        raise ("Not supported")

    D_x = D_x / 2
    return D_x.tocsr()


def laplacian(Nx, boundary_condition):
    """
    Generate a discretized Laplacian matrix with boundary conditions.

    Parameters:
    Nx : int
        Number of points
    boundary_condition : {'open', 'periodic'}
        Boundary condition at the edges of the system.

    Returns
    -------
    L_x : array_like
        Discrete Laplacian.
    """

    L_x = sparse.diags(
        [np.ones(Nx - 1), -2 * np.ones(Nx), np.ones(Nx - 1)],
        [-1, 0, 1],
        shape=(Nx, Nx),
        format="lil",
    )

    if boundary_condition == "open":
        L_x[0, 0] = -1
        L_x[-1, -1] = -1

    elif boundary_condition == "periodic":
        L_x[0, -1] = 1
        L_x[-1, 0] = 1

    else:
        raise ("Not supported")

    return L_x.tocsr()
