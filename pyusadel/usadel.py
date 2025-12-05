"""
Code for Usadel equation solver.
Andrea Maiani, 2022-2025
"""
from __future__ import annotations
from typing import Callable, Dict, Tuple, Optional

import numpy as np
from numpy import linalg as la  
from scipy import sparse
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as sla

from .findiff import DifferentialOperators

def gen_assemble_fns(
    diff_ops: DifferentialOperators,
    D: float,
    h_x: float | np.ndarray = np.array([0.0]),
    h_y: float | np.ndarray = np.array([0.0]),
    h_z: float | np.ndarray = np.array([0.0]),
    tau_so_inv: float | np.ndarray = np.array([0.0]),
    tau_sf_inv: float | np.ndarray = np.array([0.0]),
    tau_ob_inv: float | np.ndarray = np.array([0.0]),
    Gamma: float = 0.0,
    use_dense: bool = False,
    kl_left_gammaB: float | None = None,
    kl_right_gammaB: float | None = None,
) -> Dict[str, Callable]:
    """
    Generate the function dictionary used by the Usadel solver.

    This version supports **optional nonlinear Kupriyanov–Lukichev (KL) boundary
    conditions** at the left and/or right boundary in θ-parametrization.

    Parameters
    ----------
    diff_ops : DifferentialOperators
        Container holding (D_x, D_y, D_z, L, dx).
    D : float
        Diffusion constant.
    h_x, h_y, h_z : float or ndarray
        Components of the Zeeman field.
    tau_so_inv, tau_sf_inv, tau_ob_inv : float or ndarray
        Spin-orbit, spin-flip, and orbital pair-breaking rates.
    Gamma : float
        Dynes broadening parameter.
    use_dense : bool
        If True, the Jacobian for θ uses dense arrays (only recommended for small systems).
    kl_left_gammaB : float or None
        Interface transparency parameter γ_B for a KL boundary at the **left** end.
        If None, no KL boundary is applied.
    kl_right_gammaB : float or None
        Same, but for the **right** boundary.

    Returns
    -------
    dict[str, callable]
        Dictionary mapping strings to functions f0,f1,f2,f3 and Jacobians.
    """

    # ----------------------------------------------------------
    # Retrieve operators and lattice spacing
    # ----------------------------------------------------------
    D_x, D_y, D_z, L, dx = diff_ops.get_diffops() 
    Nsites = L.shape[0]

    # ----------------------------------------------------------
    # Boundary-condition flags
    # ----------------------------------------------------------
    use_kl_left = kl_left_gammaB is not None
    use_kl_right = kl_right_gammaB is not None

    # Convenience: diag wrapper 
    diag = sparse.diags

    # ==========================================================
    #  θ-Laplacian with optional nonlinear KL BC
    # ==========================================================

    def theta_laplacian(theta: np.ndarray) -> np.ndarray:
        """
        Compute L θ in the interior, but apply **exact nonlinear KL BC**
        at boundaries where requested.

        KL BC in θ-param:
            ∂θ/∂x = (1/γ_B) sin θ  at interface.

        The second derivative at site i=0 is:
            (∂²θ)_0 = 2(θ₁ - θ₀)/dx²  - (2 / (γ_B dx)) sin θ₀.

        For i=N-1:
            same form with θ_{N-2}, θ_{N-1}.
        """
        lap = L @ theta  # interior values

        if use_kl_left:
            γ = kl_left_gammaB
            lap[0] = (
                2.0 * (theta[1] - theta[0]) / dx**2
                - 2.0 * np.sin(theta[0]) / (γ * dx)
            )

        if use_kl_right:
            γ = kl_right_gammaB
            lap[-1] = (
                2.0 * (theta[-2] - theta[-1]) / dx**2
                - 2.0 * np.sin(theta[-1]) / (γ * dx)
            )

        return lap

    # ==========================================================
    #  f0  (θ-equation)
    # ==========================================================

    def f0(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        lap_theta = theta_laplacian(theta)
        return (
            D * lap_theta
            + 2 * M_0 * (Delta * np.cos(theta) - (omega_n + Gamma) * np.sin(theta))
            - 2 * (h_x * M_x + h_y * M_y + h_z * M_z) * np.cos(theta)
            - (tau_sf_inv / 4 * (2 * M_0**2 + 1)
               + 2 * tau_ob_inv * (2 * M_0**2 - 1)) * np.sin(2 * theta)
        )

    # ==========================================================
    #  Jacobian df0/dtheta with KL BC in θ
    # ==========================================================

    def df0_dtheta(theta, M_0, M_x, M_y, M_z, Delta, omega_n):

        # Start with bulk operator D*L
        if use_dense:
            L_eff = (D * L).todense()
        else:
            L_eff = (D * L).tolil()

        # -------- Left KL row modification --------
        if use_kl_left:
            γ = kl_left_gammaB

            # overwrite row 0 entirely
            L_eff[0, :] = 0.0
            L_eff[0, 0] = D * (
                -2.0 / dx**2
                - 2.0 * np.cos(theta[0]) / (γ * dx)
            )
            L_eff[0, 1] = D * (2.0 / dx**2)

        # -------- Right KL row modification --------
        if use_kl_right:
            γ = kl_right_gammaB
            i = Nsites - 1

            L_eff[i, :] = 0.0
            L_eff[i, i] = D * (
                -2.0 / dx**2
                - 2.0 * np.cos(theta[i]) / (γ * dx)
            )
            L_eff[i, i - 1] = D * (2.0 / dx**2)

        # Convert to CSR after edits
        if not use_dense:
            L_eff = L_eff.tocsr()

        # Add analytic potential-term derivative (unchanged)
        local_diag = (
            2 * M_0 * (-Delta * np.sin(theta) - (omega_n + Gamma) * np.cos(theta))
            + 2 * (h_x * M_x + h_y * M_y + h_z * M_z) * np.sin(theta)
            - (tau_sf_inv / 2 * (2 * M_0**2 + 1)
               + 4 * tau_ob_inv * (2 * M_0**2 - 1)) * np.cos(2 * theta)
        )

        return L_eff + diag(local_diag)

    def df0_dM_x(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return diag(
            2 * M_x / M_0 * (Delta * np.cos(theta) - (omega_n + Gamma) * np.sin(theta))
            - 2 * h_x * np.cos(theta)
            - (tau_sf_inv + 8 * tau_ob_inv) * M_x * np.sin(2 * theta)
        )

    def df0_dM_y(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return diag(
            2 * M_y / M_0 * (Delta * np.cos(theta) - (omega_n + Gamma) * np.sin(theta))
            - 2 * h_y * np.cos(theta)
            - (tau_sf_inv + 8 * tau_ob_inv) * M_y * np.sin(2 * theta)
        )

    def df0_dM_z(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return diag(
            2 * M_z / M_0 * (Delta * np.cos(theta) - (omega_n + Gamma) * np.sin(theta))
            - 2 * h_z * np.cos(theta)
            - (tau_sf_inv + 8 * tau_ob_inv) * M_z * np.sin(2 * theta)
        )

    # ==========================================================
    #  f1, f2, f3 and their Jacobians — all unchanged
    # ==========================================================

    def f1(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (M_x * (L @ M_0) - M_0 * (L @ M_x)) + (
            +2 * M_x * (Delta * np.sin(theta) + (omega_n + Gamma) * np.cos(theta))
            - 2 * np.sin(theta) * h_x * M_0
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * M_0 * M_x
        )

    def df1_dtheta(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return diag(
            2 * M_x * (Delta * np.cos(theta) - (omega_n + Gamma) * np.sin(theta))
            - 2 * h_x * M_0 * np.cos(theta)
            - (tau_sf_inv + 8 * tau_ob_inv) * np.sin(2 * theta) * M_0 * M_x
        )

    def df1_dM_x(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (
            diag(L @ M_0)
            + diag(M_x * (L @ (M_x / M_0)))
            - diag((M_x / M_0) * (L @ M_x))
            - diag(M_0) @ L
        ) + diag(
            +2 * (Delta * np.sin(theta) + (omega_n + Gamma) * np.cos(theta))
            - 2 * h_x * (M_x / M_0) * np.sin(theta)
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * (M_x**2 / M_0 + M_0)
        )

    def df1_dM_y(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (diag(M_x * (L @ (M_y / M_0))) - diag(M_y / M_0) @ L) + diag(
            -2 * h_x * M_y / M_0 * np.sin(theta)
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * M_y * M_x / M_0
        )

    def df1_dM_z(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (diag(M_x * (L @ (M_z / M_0))) - diag(M_z / M_0) @ L) + diag(
            -2 * h_x * M_z / M_0 * np.sin(theta)
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * M_z * M_x / M_0
        )

    # ------------------------------------------------------------------

    def f2(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (M_y * (L @ M_0) - M_0 * (L @ M_y)) + (
            2 * M_y * (Delta * np.sin(theta) + (omega_n + Gamma) * np.cos(theta))
            - 2 * np.sin(theta) * h_y * M_0
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * M_0 * M_y
        )

    def df2_dtheta(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return diag(
            2 * M_y * (Delta * np.cos(theta) - (omega_n + Gamma) * np.sin(theta))
            - 2 * h_y * M_0 * np.cos(theta)
            - (tau_sf_inv + 8 * tau_ob_inv) * np.sin(2 * theta) * M_0 * M_y
        )

    def df2_dM_x(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (diag(M_y * (L @ (M_x / M_0))) - diag(M_x / M_0) @ L) + diag(
            -2 * h_y * M_x / M_0 * np.sin(theta)
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * M_x * M_y / M_0
        )

    def df2_dM_y(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return D * (
            diag(L @ M_0)
            + diag(M_y * (L @ (M_y / M_0)))
            - diag((M_y / M_0) * (L @ M_y))
            - diag(M_0) @ L
        ) + diag(
            2 * (Delta * np.sin(theta) + (omega_n + Gamma) * np.cos(theta))
            - 2 * h_y * (M_y / M_0) * np.sin(theta)
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv)) * (M_y**2 / M_0 + M_0)
        )

    def df2_dM_z(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (diag(M_y * (L @ (M_z / M_0))) - diag(M_z / M_0) @ L) + diag(
            -2 * h_y * M_z / M_0 * np.sin(theta)
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * M_z * M_y / M_0
        )

    # ------------------------------------------------------------------

    def f3(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (M_z * (L @ M_0) - M_0 * (L @ M_z)) + (
            +2 * M_z * (Delta * np.sin(theta) + (omega_n + Gamma) * np.cos(theta))
            - 2 * np.sin(theta) * h_z * M_0
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * M_0 * M_z
        )

    def df3_dtheta(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return diag(
            2 * M_z * (Delta * np.cos(theta) - (omega_n + Gamma) * np.sin(theta))
            - 2 * h_z * M_0 * np.cos(theta)
            - (tau_sf_inv + 8 * tau_ob_inv) * np.sin(2 * theta) * M_0 * M_z
        )

    def df3_dM_x(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (diag(M_z * (L @ (M_x / M_0))) - diag(M_x / M_0) @ L) + diag(
            -2 * h_z * M_x / M_0 * np.sin(theta)
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * M_x * M_z / M_0
        )

    def df3_dM_y(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (diag(M_z * (L @ (M_y / M_0))) - diag(M_y / M_0) @ L) + diag(
            -2 * h_z * M_y / M_0 * np.sin(theta)
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * M_y * M_z / M_0
        )

    def df3_dM_z(theta, M_0, M_x, M_y, M_z, Delta, omega_n):
        return +D * (
            diag(L @ M_0)
            + diag(M_z * (L @ (M_z / M_0)))
            - diag((M_z / M_0) * (L @ M_z))
            - diag(M_0) @ L
        ) + diag(
            +2 * (Delta * np.sin(theta) + (omega_n + Gamma) * np.cos(theta))
            - 2 * h_z * (M_z / M_0) * np.sin(theta)
            + (tau_so_inv + (tau_sf_inv / 2 + 4 * tau_ob_inv) * np.cos(2 * theta)) * (M_z**2 / M_0 + M_0)
        )

    # ==========================================================
    # Free energy functional (unchanged)
    # ==========================================================

    def F_n(h_x, h_y, h_z, theta, M_x, M_y, M_z, Delta, omega_n, T):
        M_0 = np.sqrt(1 + M_x**2 + M_y**2 + M_z**2)
        return (
            np.pi
            * T
            * np.sum(
                4 * omega_n
                - 2 * M_0 * (2 * omega_n * np.cos(theta) + Delta * np.sin(theta))
                + 4 * (h_x * M_x + h_y * M_y + h_z * M_z) * np.sin(theta)
                + D
                * (
                    +((D_x @ theta) ** 2 + (D_y @ theta) ** 2 + (D_z @ theta) ** 2)
                    + ((D_x @ M_0) ** 2 + (D_y @ M_0) ** 2 + (D_z @ M_0) ** 2)
                    - ((D_x @ M_x) ** 2 + (D_y @ M_x) ** 2 + (D_z @ M_x) ** 2)
                    - ((D_x @ M_y) ** 2 + (D_y @ M_y) ** 2 + (D_z @ M_y) ** 2)
                    - ((D_x @ M_z) ** 2 + (D_y @ M_z) ** 2 + (D_z @ M_z) ** 2)
                )
                + (
                    (
                        3 * (tau_so_inv + tau_sf_inv)
                        - 3 * (tau_so_inv + tau_sf_inv * np.cos(2 * theta))
                    )
                    * M_0**2
                    / 4
                    - (tau_so_inv - tau_sf_inv)
                    * np.cos(2 * theta)
                    * (M_x**2 + M_y**2 + M_z**2)
                )
            )
        )

    # ==========================================================
    # Pack functions
    # ==========================================================

    assemble_fns = dict(
        f0=f0,
        df0_dtheta=df0_dtheta,
        df0_dM_x=df0_dM_x,
        df0_dM_y=df0_dM_y,
        df0_dM_z=df0_dM_z,

        f1=f1,
        df1_dtheta=df1_dtheta,
        df1_dM_x=df1_dM_x,
        df1_dM_y=df1_dM_y,
        df1_dM_z=df1_dM_z,

        f2=f2,
        df2_dtheta=df2_dtheta,
        df2_dM_x=df2_dM_x,
        df2_dM_y=df2_dM_y,
        df2_dM_z=df2_dM_z,

        f3=f3,
        df3_dtheta=df3_dtheta,
        df3_dM_x=df3_dM_x,
        df3_dM_y=df3_dM_y,
        df3_dM_z=df3_dM_z,

        F_n=F_n,
    )

    return assemble_fns

def solve_usadel_xyz(
    assemble_fns: dict,
    theta: np.ndarray,
    M_x: np.ndarray,
    M_y: np.ndarray,
    M_z: np.ndarray,
    Delta: np.ndarray,
    omega_ax: np.ndarray,
    omega_idx: int,
    gamma: float,
    tol: float,
    max_iter: int,
    print_exit_status: bool,
    use_dense: bool,
):

    iter_n = 0

    while True:

        M_0 = np.sqrt(1 + M_x[omega_idx] ** 2 + M_y[omega_idx] ** 2 + M_z[omega_idx] ** 2)

        LHS = sparse.bmat(
            [
                [
                    assemble_fns["df0_dtheta"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df0_dM_x"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df0_dM_y"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df0_dM_z"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                ],
                [
                    assemble_fns["df1_dtheta"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df1_dM_x"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df1_dM_y"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df1_dM_z"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                ],
                [
                    assemble_fns["df2_dtheta"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df2_dM_x"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df2_dM_y"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df2_dM_z"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                ],
                [
                    assemble_fns["df3_dtheta"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df3_dM_x"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df3_dM_y"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df3_dM_z"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=M_z[omega_idx],
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                ]
            ]
        ).tocsr()

        RHS = np.block(
            [
                -assemble_fns["f0"](
                    theta=theta[omega_idx],
                    M_0=M_0,
                    M_x=M_x[omega_idx],
                    M_y=M_y[omega_idx],
                    M_z=M_z[omega_idx],
                    Delta=Delta,
                    omega_n=omega_ax[omega_idx],
                ),
                -assemble_fns["f1"](
                    theta=theta[omega_idx],
                    M_0=M_0,
                    M_x=M_x[omega_idx],
                    M_y=M_y[omega_idx],
                    M_z=M_z[omega_idx],
                    Delta=Delta,
                    omega_n=omega_ax[omega_idx],
                ),
                -assemble_fns["f2"](
                    theta=theta[omega_idx],
                    M_0=M_0,
                    M_x=M_x[omega_idx],
                    M_y=M_y[omega_idx],
                    M_z=M_z[omega_idx],
                    Delta=Delta,
                    omega_n=omega_ax[omega_idx],
                ),
                -assemble_fns["f3"](
                    theta=theta[omega_idx],
                    M_0=M_0,
                    M_x=M_x[omega_idx],
                    M_y=M_y[omega_idx],
                    M_z=M_z[omega_idx],
                    Delta=Delta,
                    omega_n=omega_ax[omega_idx],
                ),
            ]
        )

        if use_dense:
            A = LHS.toarray()
            dd = la.solve(A, RHS)
        else:
            dd = sla.spsolve(LHS, RHS)

        dtheta, dM_x, dM_y, dM_z = np.array_split(dd, 4)

        theta[omega_idx] += gamma * dtheta
        M_x[omega_idx] += gamma * dM_x
        M_y[omega_idx] += gamma * dM_y
        M_z[omega_idx] += gamma * dM_z

        res = np.sum(np.abs(dd)) / (
            np.sum(np.abs(theta)) + np.sum(np.abs(M_x)) + np.sum(np.abs(M_y))+ np.sum(np.abs(M_z))
        )

        if res < tol:
            if print_exit_status:
                print(f"omega={omega_ax[omega_idx].imag:1.2f} : converged.")
            break

        elif iter_n > max_iter:
            if print_exit_status:
                print(f"omega={omega_ax[omega_idx].imag:1.2f} : max iteration reached.")
            break

        else:
            iter_n += 1


def solve_usadel_xy(
    assemble_fns: dict,
    theta: np.ndarray,
    M_x: np.ndarray,
    M_y: np.ndarray,
    Delta: np.ndarray,
    omega_ax: np.ndarray,
    omega_idx: int,
    gamma: float,
    tol: float,
    max_iter: int,
    print_exit_status: bool,
    use_dense: bool,
):

    iter_n = 0

    while True:

        M_0 = np.sqrt(1 + M_x[omega_idx] ** 2 + M_y[omega_idx] ** 2)

        LHS = sparse.bmat(
            [
                [
                    assemble_fns["df0_dtheta"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df0_dM_x"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df0_dM_y"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                ],
                [
                    assemble_fns["df1_dtheta"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df1_dM_x"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df1_dM_y"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                ],
                [
                    assemble_fns["df2_dtheta"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df2_dM_x"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df2_dM_y"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=M_y[omega_idx],
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                ],
            ]
        ).tocsr()

        RHS = np.block(
            [
                -assemble_fns["f0"](
                    theta=theta[omega_idx],
                    M_0=M_0,
                    M_x=M_x[omega_idx],
                    M_y=M_y[omega_idx],
                    M_z=0,
                    Delta=Delta,
                    omega_n=omega_ax[omega_idx],
                ),
                -assemble_fns["f1"](
                    theta=theta[omega_idx],
                    M_0=M_0,
                    M_x=M_x[omega_idx],
                    M_y=M_y[omega_idx],
                    M_z=0,
                    Delta=Delta,
                    omega_n=omega_ax[omega_idx],
                ),
                -assemble_fns["f2"](
                    theta=theta[omega_idx],
                    M_0=M_0,
                    M_x=M_x[omega_idx],
                    M_y=M_y[omega_idx],
                    M_z=0,
                    Delta=Delta,
                    omega_n=omega_ax[omega_idx],
                ),
            ]
        )

        if use_dense:
            A = LHS.toarray()
            dd = la.solve(A, RHS)
        else:
            dd = sla.spsolve(LHS, RHS)

        dtheta, dM_x, dM_y = np.array_split(dd, 3)

        theta[omega_idx] += gamma * dtheta
        M_x[omega_idx] += gamma * dM_x
        M_y[omega_idx] += gamma * dM_y

        res = np.sum(np.abs(dd)) / (
            np.sum(np.abs(theta)) + np.sum(np.abs(M_x)) + np.sum(np.abs(M_y))
        )

        if res < tol:
            if print_exit_status:
                print(f"omega={omega_ax[omega_idx].imag:1.2f} : converged.")
            break

        elif iter_n > max_iter:
            if print_exit_status:
                print(f"omega={omega_ax[omega_idx].imag:1.2f} : max iteration reached.")
            break

        else:
            iter_n += 1


def solve_usadel_x(
    assemble_fns: dict,
    theta: np.ndarray,
    M_x: np.ndarray,
    Delta: np.ndarray,
    omega_ax: np.ndarray,
    omega_idx: int,
    gamma: float,
    tol: float,
    max_iter: int,
    print_exit_status: bool,
    use_dense: bool,
):

    iter_n = 0

    while True:

        M_0 = np.sqrt(1 + M_x[omega_idx] ** 2)

        LHS = sparse.bmat(
            [
                [
                    assemble_fns["df0_dtheta"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=0,
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df0_dM_x"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=0,
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                ],
                [
                    assemble_fns["df1_dtheta"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=0,
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                    assemble_fns["df1_dM_x"](
                        theta=theta[omega_idx],
                        M_0=M_0,
                        M_x=M_x[omega_idx],
                        M_y=0,
                        M_z=0,
                        Delta=Delta,
                        omega_n=omega_ax[omega_idx],
                    ),
                ],
            ]
        ).tocsr()

        RHS = np.block(
            [
                -assemble_fns["f0"](
                    theta=theta[omega_idx],
                    M_0=M_0,
                    M_x=M_x[omega_idx],
                    M_y=0,
                    M_z=0,
                    Delta=Delta,
                    omega_n=omega_ax[omega_idx],
                ),
                -assemble_fns["f1"](
                    theta=theta[omega_idx],
                    M_0=M_0,
                    M_x=M_x[omega_idx],
                    M_y=0,
                    M_z=0,
                    Delta=Delta,
                    omega_n=omega_ax[omega_idx],
                ),
            ]
        )

        if use_dense:
            A = LHS.toarray()
            dd = la.solve(A, RHS)
        else:
            dd = sla.spsolve(LHS, RHS)

        dtheta, dM_x = np.array_split(dd, 2)

        theta[omega_idx] += gamma * dtheta
        M_x[omega_idx] += gamma * dM_x

        res = np.sum(np.abs(dd)) / (np.sum(np.abs(theta)) + np.sum(np.abs(M_x)))

        if res < tol:
            if print_exit_status:
                print(f"omega={omega_ax[omega_idx].imag:1.2f} : converged.")
            break

        elif iter_n > max_iter:
            if print_exit_status:
                print(f"omega={omega_ax[omega_idx].imag:1.2f} : max iteration reached.")
            break

        else:
            iter_n += 1


def solve_usadel(
    assemble_fns: dict,
    h_x: np.ndarray,
    h_y: np.ndarray,
    h_z: np.ndarray,
    theta: np.ndarray,
    M_x: np.ndarray,
    M_y: np.ndarray,
    M_z: np.ndarray,
    Delta: np.ndarray,
    omega_ax: np.ndarray,
    gamma: float = 1,
    tol: float = 1e-6,
    max_iter: int = 1000,
    print_exit_status: bool = False,
    use_dense: bool = False,
):
    """
    Solve the Usadel equation for real energies. This routine solve for
    only one frequency specified by the omega_idx variable.
    The results will overwrite theta, M_x, M_y, M_z. These variables are used as initial guesses.
    If h_z and h_y are null, it calls optimized subrutines.

    Parameters:
    ------
    assemble_fns: dict
        Assembly functions.
    theta: np.ndarray
       theta field.
    M_x: np.ndarray
        M_x field.
    M_y: np.ndarray
        M_y field.
    h_x: np.ndarray
        h_x field.
    h_y: np.ndarray
        h_y field.
    Delta: np.ndarray
        Delta field.
    omega_ax: np.ndarray
        Axis of frequencies.
    omega_idx: int
        Index to be solved.
    gamma: float
        Relaxation parameter (default 1).
    tol: float
        Traget tolerance (default 1e-6).
    max_iter: int
        Maximum number of iteration (default 1000).
    print_exit_status: bool
        Wether to print exit status (default False).
    use_dense: bool
        Wether to use dense algebra, useful for small systems (default False).
    """

    if not np.any(h_z):
        if not np.any(h_y):
            for omega_idx in range(omega_ax.shape[0]):
                solve_usadel_x(
                    assemble_fns=assemble_fns,
                    theta=theta,
                    M_x=M_x,
                    Delta=Delta,
                    omega_ax=omega_ax,
                    omega_idx=omega_idx,
                    gamma=gamma,
                    tol=tol,
                    max_iter=max_iter,
                    print_exit_status=print_exit_status,
                    use_dense=use_dense,
                )
        else:
            for omega_idx in range(omega_ax.shape[0]):
                solve_usadel_xy(
                    assemble_fns=assemble_fns,
                    theta=theta,
                    M_x=M_x,
                    M_y=M_y,
                    Delta=Delta,
                    omega_ax=omega_ax,
                    omega_idx=omega_idx,
                    gamma=gamma,
                    tol=tol,
                    max_iter=max_iter,
                    print_exit_status=print_exit_status,
                    use_dense=use_dense,
                )
    else:
        for omega_idx in range(omega_ax.shape[0]):
            solve_usadel_xyz(
                assemble_fns=assemble_fns,
                theta=theta,
                M_x=M_x,
                M_y=M_y,
                M_z=M_z,
                Delta=Delta,
                omega_ax=omega_ax,
                omega_idx=omega_idx,
                gamma=gamma,
                tol=tol,
                max_iter=max_iter,
                print_exit_status=print_exit_status,
                use_dense=use_dense,
            )


def solve_usadel_self_consistent(
    assemble_fns: dict,
    h_x: np.ndarray,
    h_y: np.ndarray,
    h_z: np.ndarray,
    Delta: np.ndarray,
    T: float,
    T_c0: float = 1.0,
    omega_N: int = 100,
    gamma: float = 1.0,
    tol_g: float = 1e-6,
    max_iter_g: int = 1000,
    tol_Delta: float = 1e-6,
    max_iter_Delta: int = 100,
    verbose: bool = False,
    use_dense: bool = False,
):
    """
    Solve the Usadel equation for Matsubara frequencies self-consistently to determine Delta.


    Parameters:
    ------
    assemble_fns: dict
        Assembly functions.
    h_x: np.ndarray
        h_x field.
    h_y: np.ndarray
        h_y field.
    h_z: np.ndarray
        h_z field.
    Delta: np.ndarray
        Delta field (initial guess).
    T : float
        Temperature
    T_c0 : critical temperature
        Temperature
    omega_N: int
        Number of Matsubara frequencies.
    gamma: float
        Relaxation parameter (default 1).
    tol: float
        Traget tolerance (default 1e-6).
    max_iter: int
        Maximum number of iteration (default 1000).
    max_iter_delta: int
        Maximum number of cycles (default 100).
    """

    omega_ax = (2 * np.arange(0, omega_N) + 1) * np.pi * T

    Nsites = Delta.shape[0]
    theta = np.ones((omega_N, Nsites), dtype=float)
    M_x = np.zeros((omega_N, Nsites), dtype=float)
    M_y = np.zeros((omega_N, Nsites), dtype=float)
    M_z = np.zeros((omega_N, Nsites), dtype=float)

    iter_n = 0

    while True:
        iter_n += 1

        solve_usadel(
            assemble_fns,
            h_x,
            h_y,
            h_z,
            theta,
            M_x,
            M_y,
            M_z,
            Delta,
            omega_ax,
            gamma=gamma,
            tol=tol_g,
            max_iter=max_iter_g,
            print_exit_status=False,
            use_dense=use_dense,
        )

        old_Delta = Delta.copy()

        Delta = (
            (2 * np.pi * T)
            * (np.sum(np.sin(theta) * np.sqrt(1 + M_x**2 + M_y**2 + M_z**2), axis=0))
            / (np.log(T / T_c0) + (2 * np.pi * T) * np.sum(omega_ax ** (-1)))
        )

        res = np.sum(np.abs((Delta - old_Delta))) / np.sum(np.abs(Delta))

        F_sn = 0
        for n in range(omega_N):
            F_sn += assemble_fns["F_n"](
                h_x, h_y, h_z, theta[n], M_x[n], M_y[n], M_z[n], Delta, omega_ax[n], T
            )

        if verbose:
            print(
                f"{iter_n:3d}    Max Delta: {Delta.max():4.3f}    Residual: {res:3.2e}    Free energy: {F_sn:3.2e}"
            )

        if res < tol_Delta:
            if verbose:
                print(f"Converged in {iter_n} iterations")
            break

        elif iter_n > max_iter_Delta:
            if verbose:
                print("Max iteration reached")
            break

    return (theta, M_x, M_y, M_z, Delta, omega_ax, F_sn)
