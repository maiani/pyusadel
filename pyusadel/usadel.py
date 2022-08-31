"""
Code for Usadel equation solver.
Andrea Maiani, 2022
"""

import numpy as np
from numpy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as sla
from numba import jit
from .findiff import DifferentialOperators

def gen_assemble_fun(
    diff_ops : DifferentialOperators, 
    D: float, 
    tau_so_inv: float or np.ndarray, 
    tau_sf_inv: float or np.ndarray,
):
    """
    Define the assembly functions.
    
    Parameters:
    -------
    diff_ops : DifferentialOperators
        Differential operators
    D : float
        Diffusion constant
    tau_so_inv : np.ndarray 
        Spin-orbit relaxation rate time
    tau_sf_inv : np.ndarray
        Spin-flip relaxation rate time

    Returns
    -------
    usadel_af : tuple[callable]
        Assembly functions.
    """

    D_x, D_y, D_z, L = diff_ops.get_diffops()
    
    ############ f0 #############
    def f0(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return (
            (D * (L @ theta))
            + 2 * M_0 * (Delta * np.cos(theta) - omega_n * np.sin(theta))
            - 2 * (h_x * M_x + h_y * M_y) * np.cos(theta)
            - tau_sf_inv * (2 * M_0 ** 2 + 1) / 4 * np.sin(2 * theta)
        )

    def df0_dtheta(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return (D * L) + sparse.diags(
            2 * M_0 * (-Delta * np.sin(theta) - omega_n * np.cos(theta))
            + 2 * (h_x * M_x + h_y * M_y) * np.sin(theta)
            - tau_sf_inv * (2 * M_0 ** 2 + 1) / 2 * np.cos(2 * theta)
        )

    def df0_dM_x(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return sparse.diags(
            2 * M_x / M_0 * (Delta * np.cos(theta) - omega_n * np.sin(theta))
            - 2 * h_x * np.cos(theta)
            - tau_sf_inv * M_x * np.sin(2 * theta)
        )

    def df0_dM_y(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return sparse.diags(
            2 * M_y / M_0 * (Delta * np.cos(theta) - omega_n * np.sin(theta))
            - 2 * h_y * np.cos(theta)
            - tau_sf_inv * M_y * np.sin(2 * theta)
        )

    ############ f1 #############
    def f1(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return +D * (M_x * (L @ M_0) - M_0 * (L @ M_x)) + (
            +2 * M_x * (Delta * np.sin(theta) + omega_n * np.cos(theta))
            - 2 * np.sin(theta) * h_x * M_0
            + (tau_so_inv + tau_sf_inv * np.cos(2 * theta) / 2) * M_0 * M_x
        )

    def df1_dtheta(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return sparse.diags(
            2 * M_x * (Delta * np.cos(theta) - omega_n * np.sin(theta))
            - 2 * h_x * M_0 * np.cos(theta)
            - tau_sf_inv * np.sin(2 * theta) * M_0 * M_x
        )

    def df1_dM_x(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return D * (
            sparse.diags(L @ M_0)
            + sparse.diags(M_x * (L @ (M_x / M_0)))
            - sparse.diags((M_x / M_0) * (L @ M_x))
            - sparse.diags(M_0) @ L
        ) + sparse.diags(
            +2 * (Delta * np.sin(theta) + omega_n * np.cos(theta))
            - 2 * h_x * (M_x / M_0) * np.sin(theta)
            + (tau_so_inv + tau_sf_inv * np.cos(2 * theta) / 2) * M_x ** 2 / M_0
        )

    def df1_dM_y(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return +D * (
            sparse.diags(M_x * (L @ (M_y / M_0))) - sparse.diags(M_y / M_0) @ L
        ) + sparse.diags(
            -2 * h_x * M_y / M_0 * np.sin(theta)
            + (tau_so_inv + tau_sf_inv * np.cos(2 * theta) / 2) * M_y * M_x / M_0
        )

    ############ f2 #############
    def f2(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return +D * (M_y * (L @ M_0) - M_0 * (L @ M_y)) + (
            2 * M_y * (Delta * np.sin(theta) + omega_n * np.cos(theta))
            - 2 * np.sin(theta) * h_y * M_0
            + (tau_so_inv + tau_sf_inv * np.cos(2 * theta) / 2) * M_0 * M_y
        )

    def df2_dtheta(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return sparse.diags(
            2 * M_y * (Delta * np.cos(theta) - omega_n * np.sin(theta))
            - 2 * h_y * M_0 * np.cos(theta)
            - tau_sf_inv * np.sin(2 * theta) * M_0 * M_y
        )

    def df2_dM_y(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return D * (
            sparse.diags(L @ M_0)
            + sparse.diags(M_y * (L @ (M_y / M_0)))
            - sparse.diags((M_y / M_0) * (L @ M_y))
            - sparse.diags(M_0) @ L
        ) + sparse.diags(
            2 * (Delta * np.sin(theta) + omega_n * np.cos(theta))
            - 2 * h_y * (M_y / M_0) * np.sin(theta)
            + (tau_so_inv + tau_sf_inv * np.cos(2 * theta) / 2) * (M_y ** 2 / M_0 + M_0)
        )

    def df2_dM_x(theta, M_0, M_x, M_y, h_x, h_y, Delta, omega_n):
        return +D * (
            sparse.diags(M_y * (L @ (M_x / M_0))) - sparse.diags(M_x / M_0) @ L
        ) + sparse.diags(
            -2 * h_y * M_x / M_0 * np.sin(theta)
            + (tau_so_inv + tau_sf_inv * np.cos(2 * theta) / 2) * M_x * M_y / M_0
        )

    def F_n(theta, M_x, M_y, h_x, h_y, Delta, omega_n, T):
        M_0 = np.sqrt(1 + M_x ** 2 + M_y ** 2)

        return (
            np.pi
            * T
            * np.sum(
                4 * omega_n
                - 2 * M_0 * (2 * omega_n * np.cos(theta) + Delta * np.sin(theta))
                + 4 * (h_y * M_y + h_x * M_x) * np.sin(theta)
                + D
                * (
                    (D_x @ theta) ** 2
                    + (D_x @ M_0) ** 2
                    - (D_x @ M_x) ** 2
                    - (D_x @ M_y) ** 2
                )
                + (
                    (
                        3 * (tau_so_inv + tau_sf_inv)
                        - 3 * (tau_so_inv + tau_sf_inv * np.cos(2 * theta))
                    )
                    * M_0 ** 2
                    / 4
                    - (tau_so_inv - tau_sf_inv)
                    * np.cos(2 * theta)
                    * (M_x ** 2 + M_y ** 2)
                )
            )
        )

    return (
        f0,
        df0_dtheta,
        df0_dM_x,
        df0_dM_y,
        f1,
        df1_dtheta,
        df1_dM_x,
        df1_dM_y,
        f2,
        df2_dtheta,
        df2_dM_x,
        df2_dM_y,
        F_n,
    )


def solve_usadel(
    assemble_fun: tuple,
    theta: np.ndarray,
    M_x: np.ndarray,
    M_y: np.ndarray,
    h_x: np.ndarray,
    h_y: np.ndarray,
    Delta: np.ndarray,
    omega_ax: np.ndarray,
    omega_idx: int,
    gamma: float = 1,
    tol: float = 1e-6,
    max_iter: int = 1000,
    print_exit_status: bool = False,
    use_dense: bool = False,
):

    """
    Solve the Usadel equation for real energies. This routine solve for 
    only one frequency specified by the omega_idx variable.
    The results will overwrite theta, M_x, M_y. 
    These variables are used as initial guesses.
    
    Parameters:
    ------
    assemble_fun: tuple
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

    (
        f0,
        df0_dtheta,
        df0_dM_x,
        df0_dM_y,
        f1,
        df1_dtheta,
        df1_dM_x,
        df1_dM_y,
        f2,
        df2_dtheta,
        df2_dM_x,
        df2_dM_y,
    ) = assemble_fun

    iter_n = 0

    while True:

        M_0 = np.sqrt(1 + M_x[omega_idx] ** 2 + M_y[omega_idx] ** 2)

        LHS = sparse.bmat(
            [
                [
                    df0_dtheta(
                        theta[omega_idx],
                        M_0,
                        M_x[omega_idx],
                        M_y[omega_idx],
                        h_x,
                        h_y,
                        Delta,
                        omega_ax[omega_idx],
                    ),
                    df0_dM_x(
                        theta[omega_idx],
                        M_0,
                        M_x[omega_idx],
                        M_y[omega_idx],
                        h_x,
                        h_y,
                        Delta,
                        omega_ax[omega_idx],
                    ),
                    df0_dM_y(
                        theta[omega_idx],
                        M_0,
                        M_x[omega_idx],
                        M_y[omega_idx],
                        h_x,
                        h_y,
                        Delta,
                        omega_ax[omega_idx],
                    ),
                ],
                [
                    df1_dtheta(
                        theta[omega_idx],
                        M_0,
                        M_x[omega_idx],
                        M_y[omega_idx],
                        h_x,
                        h_y,
                        Delta,
                        omega_ax[omega_idx],
                    ),
                    df1_dM_x(
                        theta[omega_idx],
                        M_0,
                        M_x[omega_idx],
                        M_y[omega_idx],
                        h_x,
                        h_y,
                        Delta,
                        omega_ax[omega_idx],
                    ),
                    df1_dM_y(
                        theta[omega_idx],
                        M_0,
                        M_x[omega_idx],
                        M_y[omega_idx],
                        h_x,
                        h_y,
                        Delta,
                        omega_ax[omega_idx],
                    ),
                ],
                [
                    df2_dtheta(
                        theta[omega_idx],
                        M_0,
                        M_x[omega_idx],
                        M_y[omega_idx],
                        h_x,
                        h_y,
                        Delta,
                        omega_ax[omega_idx],
                    ),
                    df2_dM_x(
                        theta[omega_idx],
                        M_0,
                        M_x[omega_idx],
                        M_y[omega_idx],
                        h_x,
                        h_y,
                        Delta,
                        omega_ax[omega_idx],
                    ),
                    df2_dM_y(
                        theta[omega_idx],
                        M_0,
                        M_x[omega_idx],
                        M_y[omega_idx],
                        h_x,
                        h_y,
                        Delta,
                        omega_ax[omega_idx],
                    ),
                ],
            ]
        ).tocsr()

        RHS = np.block(
            [
                -f0(
                    theta[omega_idx],
                    M_0,
                    M_x[omega_idx],
                    M_y[omega_idx],
                    h_x,
                    h_y,
                    Delta,
                    omega_ax[omega_idx],
                ),
                -f1(
                    theta[omega_idx],
                    M_0,
                    M_x[omega_idx],
                    M_y[omega_idx],
                    h_x,
                    h_y,
                    Delta,
                    omega_ax[omega_idx],
                ),
                -f2(
                    theta[omega_idx],
                    M_0,
                    M_x[omega_idx],
                    M_y[omega_idx],
                    h_x,
                    h_y,
                    Delta,
                    omega_ax[omega_idx],
                ),
            ]
        )

        if use_dense:
            dd = la.solve(LHS.todense(), RHS)
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
                print("Converged")
            break

        elif iter_n > max_iter:
            if print_exit_status:
                print("Max iteration reached")
            break

        else:
            iter_n += 1


def solve_usadel_self_consistent(
    assemble_fun,
    h_x,
    h_y,
    Delta,
    T,
    omega_N=100,
    gamma=1,
    tol=1e-6,
    max_iter=100,
    max_iter_delta=100,
):

    T_c0 = 1

    omega_ax = (2 * np.arange(0, omega_N) + 1) * np.pi * T

    Nsites = h_x.shape[0]
    theta = np.ones((omega_N, Nsites), dtype=float)
    M_x = np.zeros((omega_N, Nsites), dtype=float)
    M_y = np.zeros((omega_N, Nsites), dtype=float)

    iter_n = 0

    while True:
        iter_n += 1

        for omega_idx in range(omega_N):
            solve_usadel(
                assemble_fun[:-1],
                theta,
                M_x,
                M_y,
                h_x,
                h_y,
                Delta,
                omega_ax,
                omega_idx,
                gamma=gamma,
                tol=tol,
                max_iter=max_iter,
                print_exit_status=False,
            )

        old_Delta = Delta.copy()

        Delta = (
            (2 * np.pi * T)
            * (np.sum(np.sin(theta) * np.sqrt(1 + M_x ** 2 + M_y ** 2), axis=0))
            / (np.log(T / T_c0) + (2 * np.pi * T) * np.sum(omega_ax ** (-1)))
        )

        res = np.sum(np.abs((Delta - old_Delta))) / np.sum(np.abs(Delta))

        F_n = assemble_fun[-1]

        F_sn = 0
        for n in range(omega_N):
            F_sn += F_n(theta[n], M_x[n], M_y[n], h_x, h_y, Delta, omega_ax[n], T)

        print(
            f"{iter_n:3d}    Max Delta: {Delta.max():4.3f}    Residual: {res:3.2e}    Free energy: {F_sn:3.2e}"
        )

        if res < tol:
            print(f"Converged in {iter_n} iterations")
            break

        elif iter_n > max_iter_delta:
            print("Max iteration reached")
            break
    return (theta, M_x, M_y, Delta, omega_ax, F_sn)
