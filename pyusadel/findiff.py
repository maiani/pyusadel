"""
Finite-difference operators for the Usadel equation.
Andrea Maiani, 2022–2025
"""

from __future__ import annotations
from typing import Tuple, Optional, Literal

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix


class DifferentialOperators:
    """
    Container for differential operators acting on a 1D discretized system.

    Attributes
    ----------
    Nsites : int
        Number of spatial sites (determined automatically when the first
        operator is assigned).
    D_x, D_y, D_z : csr_matrix
        First-derivative operators in x,y,z (y,z usually empty for 1D).
    L : csr_matrix
        Second-derivative (Laplacian) operator.
    dx : float
        Lattice spacing.
    """

    def __init__(self) -> None:
        self._Nsites: Optional[int] = None
        self._D_x: Optional[csr_matrix] = None
        self._D_y: Optional[csr_matrix] = None
        self._D_z: Optional[csr_matrix] = None
        self._L: Optional[csr_matrix] = None
        self._dx: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def Nsites(self) -> Optional[int]:
        return self._Nsites

    @property
    def dx(self) -> Optional[float]:
        return self._dx

    @dx.setter
    def dx(self, dx: float) -> None:
        self._dx = float(dx)

    @property
    def D_x(self) -> Optional[csr_matrix]:
        return self._D_x

    @D_x.setter
    def D_x(self, D_x: csr_matrix) -> None:
        self._assign_matrix("_D_x", D_x)

    @property
    def D_y(self) -> Optional[csr_matrix]:
        return self._D_y

    @D_y.setter
    def D_y(self, D_y: csr_matrix) -> None:
        self._assign_matrix("_D_y", D_y)

    @property
    def D_z(self) -> Optional[csr_matrix]:
        return self._D_z

    @D_z.setter
    def D_z(self, D_z: csr_matrix) -> None:
        self._assign_matrix("_D_z", D_z)

    @property
    def L(self) -> Optional[csr_matrix]:
        return self._L

    @L.setter
    def L(self, L: csr_matrix) -> None:
        self._assign_matrix("_L", L)

    def _assign_matrix(self, attr: str, mat: csr_matrix) -> None:
        """Assign matrix mat to attribute attr, checking shape consistency."""
        mat = sparse.csr_matrix(mat)

        if self._Nsites is None:
            if mat.shape[0] != mat.shape[1]:
                raise ValueError("Operator must be square.")
            self._Nsites = mat.shape[0]
            setattr(self, attr, mat)
        else:
            if mat.shape != (self._Nsites, self._Nsites):
                raise ValueError("Shape mismatch for operator.")
            setattr(self, attr, mat)

    def get_diffops(self) -> Tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix, float]:
        """
        Return all differential operators and the lattice spacing.

        Returns
        -------
        (D_x, D_y, D_z, L, dx)
        """
        if self._dx is None:
            raise ValueError("dx has not been specified in DifferentialOperators.")
        return self.D_x, self.D_y, self.D_z, self.L, self._dx


# ----------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------

def gradient(
    Nx: int,
    dx: float,
    boundary_condition: Literal["open", "periodic"],
) -> csr_matrix:
    """
    Construct a finite-difference gradient operator (first derivative).

    Parameters
    ----------
    Nx : int
        Number of discrete spatial points.
    dx : float
        Lattice spacing.
    boundary_condition : {'open', 'periodic'}
        Boundary condition at the ends.

    Returns
    -------
    csr_matrix
        Sparse Nx×Nx matrix implementing d/dx.
    """

    D_x = sparse.diags(
        [-np.ones(Nx - 1), 0.0, np.ones(Nx - 1)],
        [-1, 0, 1],
        shape=(Nx, Nx),
        format="lil",
    )

    if boundary_condition == "open":
        D_x[0, 0] = -1
        D_x[0, 1] = +1
        D_x[-1, -2] = -1
        D_x[-1, -1] = +1

    elif boundary_condition == "periodic":
        D_x[0, -1] = -1
        D_x[0, 1] = +1
        D_x[-1, -2] = -1
        D_x[-1, 0] = +1

    else:
        raise ValueError("Supported BCs: 'open', 'periodic'.")

    return (D_x / (2 * dx)).tocsr()


def laplacian(
    Nx: int,
    dx: float,
    boundary_condition: Literal["open", "periodic"],
) -> csr_matrix:
    """
    Construct a finite-difference Laplacian operator (second derivative).

    Parameters
    ----------
    Nx : int
        Number of discrete spatial points.
    dx : float
        Lattice spacing.
    boundary_condition : {'open', 'periodic'}
        Boundary condition at the ends.

    Returns
    -------
    csr_matrix
        Sparse Nx×Nx Laplacian matrix.
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
        L_x[0, -1] = +1
        L_x[-1, 0] = +1

    else:
        raise ValueError("Supported BCs: 'open', 'periodic'.")

    return (L_x / dx**2).tocsr()


# ----------------------------------------------------------------------
# Helper constructors
# ----------------------------------------------------------------------

def make_1d_diffops(
    Nx: int,
    dx: float,
    boundary: Literal["open", "periodic"] = "open",
) -> DifferentialOperators:
    """
    Construct a DifferentialOperators object for a 1D grid.

    Parameters
    ----------
    Nx : int
        Number of spatial sites.
    dx : float
        Lattice spacing.
    boundary : {'open', 'periodic'}
        Boundary condition for gradient and Laplacian.

    Returns
    -------
    DifferentialOperators
        Fully configured differential operators container.
    """
    do = DifferentialOperators()
    do.dx = dx

    do.D_x = gradient(Nx, dx, boundary)
    do.D_y = sparse.csr_matrix((Nx, Nx))
    do.D_z = sparse.csr_matrix((Nx, Nx))
    do.L = laplacian(Nx, dx, boundary)

    return do


def trivial_diffops() -> DifferentialOperators:
    """
    Create a trivial DifferentialOperators instance for 0D (single-site).

    All operators are 1×1 zero matrices and dx is irrelevant.

    Returns
    -------
    DifferentialOperators
    """
    do = DifferentialOperators()
    do.dx = 1.0
    do.D_x = sparse.csr_matrix([[0.0]])
    do.D_y = sparse.csr_matrix([[0.0]])
    do.D_z = sparse.csr_matrix([[0.0]])
    do.L = sparse.csr_matrix([[0.0]])
    return do
