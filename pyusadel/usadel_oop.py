"""
Code for Usadel equation solver.
Andrea Maiani, 2022-2025
"""

import numpy as np
from .findiff import DifferentialOperators
from .usadel import gen_assemble_fns, solve_usadel, solve_usadel_self_consistent
from typing import Tuple, Optional, Union


def check_array(
    arr: Union[np.ndarray, None],
    shape: Tuple[int, ...],
    name: str,
    dtype: type = float,
    dvalue: float = 0.0,
) -> np.ndarray:
    """
    Checks if an input array has the correct shape and data type.
    If None is passed as first argument, it creates a default array
    with the right shape and default value.
    """

    if arr is None:
        return np.ones(shape, dtype=dtype) * dvalue

    # Safely coerce to numpy array of the desired dtype
    arr = np.asarray(arr, dtype=dtype)

    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")

    return arr


class UsadelProblem:
    """Simple way to define and solve a Usadel problem."""

    def __init__(
        self,
        Nsites: int,
        diff_ops: DifferentialOperators,
        h_x: Optional[np.ndarray] = None,
        h_y: Optional[np.ndarray] = None,
        h_z: Optional[np.ndarray] = None,
        tau_sf_inv: Optional[np.ndarray] = None,
        tau_so_inv: Optional[np.ndarray] = None,
        tau_ob_inv: Optional[np.ndarray] = None,
        D: float = 0,
        T: float = 0,
        T_c0: float = 1,
        Gamma: float = 1e-3,
        kl_left_gammaB: Optional[float] = None,
        kl_right_gammaB: Optional[float] = None,
        use_dense: bool = False,
    ):
        """
        Basic Usadel problem setup.
        """

        self.Nsites = Nsites
        self.diff_ops = diff_ops

        shape = (Nsites,)

        self.h_x = check_array(h_x, shape, "h_x")
        self.h_y = check_array(h_y, shape, "h_y")
        self.h_z = check_array(h_z, shape, "h_z")
        self.tau_sf_inv = check_array(tau_sf_inv, shape, "tau_sf_inv")
        self.tau_so_inv = check_array(tau_so_inv, shape, "tau_so_inv")
        self.tau_ob_inv = check_array(tau_ob_inv, shape, "tau_ob_inv")

        self.D = D
        self.T = T
        self.T_c0 = T_c0
        self.Gamma = Gamma
        self.use_dense = use_dense

        # KL boundary condition parameters
        self.kl_left_gammaB = kl_left_gammaB
        self.kl_right_gammaB = kl_right_gammaB

        self.Delta = np.ones((Nsites,), dtype=float)
        self.F_sn = None

        # Imaginary frequency axis
        self._omega_ax_i = None
        self.theta_i = None
        self.M_x_i = None
        self.M_y_i = None
        self.M_z_i = None

        # Real frequency axis
        self._omega_ax_r = None
        self.theta_r = None
        self.M_x_r = None
        self.M_y_r = None
        self.M_z_r = None
        self.M_0_r = None

        # Assemble operators
        self.assemble_fns = gen_assemble_fns(
            D=self.D,
            diff_ops=self.diff_ops,
            h_x=self.h_x,
            h_y=self.h_y,
            h_z=self.h_z,
            tau_so_inv=self.tau_so_inv,
            tau_sf_inv=self.tau_sf_inv,
            tau_ob_inv=self.tau_ob_inv,
            Gamma=self.Gamma,
            use_dense=self.use_dense,
            kl_left_gammaB=self.kl_left_gammaB,
            kl_right_gammaB=self.kl_right_gammaB,
        )

        self._calculate_scales()

    def _calculate_scales(self):
        """Calculate basic scales."""
        self.Delta_00 = self.T_c0 * 1.7652
        self.h_c0 = self.Delta_00 / np.sqrt(2)
        self.xi_00 = np.sqrt(self.D / self.Delta_00)

    def update_params(
        self,
        h_x: np.ndarray | None = None,
        h_y: np.ndarray | None = None,
        h_z: np.ndarray | None = None,
        tau_sf_inv: np.ndarray | None = None,
        tau_so_inv: np.ndarray | None = None,
        tau_ob_inv: np.ndarray | None = None,
        D: float | None = None,
        T: float | None = None,
        T_c0: float | None = None,
        Gamma: float | None = None,
        kl_left_gammaB: float | None = None,
        kl_right_gammaB: float | None = None,
    ):
        """Update parameters and regenerate assembly functions."""

        if h_x is not None:
            if h_x.shape != (self.Nsites,):
                raise Exception("Dimensions doesn't match.")
            self.h_x = h_x

        if h_y is not None:
            if h_y.shape != (self.Nsites,):
                raise Exception("Dimensions doesn't match.")
            self.h_y = h_y

        if h_z is not None:
            if h_z.shape != (self.Nsites,):
                raise Exception("Dimensions doesn't match.")
            self.h_z = h_z

        if tau_sf_inv is not None:
            if tau_sf_inv.shape != (self.Nsites,):
                raise Exception("Dimensions doesn't match.")
            self.tau_sf_inv = tau_sf_inv

        if tau_so_inv is not None:
            if tau_so_inv.shape != (self.Nsites,):
                raise Exception("Dimensions doesn't match.")
            self.tau_so_inv = tau_so_inv

        if tau_ob_inv is not None:
            if tau_ob_inv.shape != (self.Nsites,):
                raise Exception("Dimensions doesn't match.")
            self.tau_ob_inv = tau_ob_inv

        if D is not None:
            self.D = D

        if T is not None:
            self.T = T

        if T_c0 is not None:
            self.T_c0 = T_c0

        if Gamma is not None:
            self.Gamma = Gamma

        if kl_left_gammaB is not None:
            self.kl_left_gammaB = kl_left_gammaB

        if kl_right_gammaB is not None:
            self.kl_right_gammaB = kl_right_gammaB

        # Rebuild assemble_fns to reflect the new parameters / KL BC
        self.assemble_fns = gen_assemble_fns(
            D=self.D,
            diff_ops=self.diff_ops,
            h_x=self.h_x,
            h_y=self.h_y,
            h_z=self.h_z,
            tau_so_inv=self.tau_so_inv,
            tau_sf_inv=self.tau_sf_inv,
            tau_ob_inv=self.tau_ob_inv,
            Gamma=self.Gamma,
            use_dense=self.use_dense,
            kl_left_gammaB=self.kl_left_gammaB,
            kl_right_gammaB=self.kl_right_gammaB,
        )

        self._calculate_scales()

    def solve_self_consistent(
        self,
        omega_N: int = None,
        gamma: float = 0.5,
        tol_g: float = 1e-6,
        max_iter_g: int = 1000,
        tol_Delta: float = 1e-6,
        max_iter_Delta: int = 100,
        verbose: bool = False,
    ):
        """Solve the self-consistent Usadel equation."""

        # Regenerate assemble_fns to guarantee consistency with current params/KL BCs
        self.assemble_fns = gen_assemble_fns(
            D=self.D,
            diff_ops=self.diff_ops,
            h_x=self.h_x,
            h_y=self.h_y,
            h_z=self.h_z,
            tau_so_inv=self.tau_so_inv,
            tau_sf_inv=self.tau_sf_inv,
            tau_ob_inv=self.tau_ob_inv,
            Gamma=self.Gamma,
            use_dense=self.use_dense,
            kl_left_gammaB=self.kl_left_gammaB,
            kl_right_gammaB=self.kl_right_gammaB,
        )

        if not omega_N:
            # TODO: implement something smarter here
            omega_N = 100

        (
            self.theta_i,
            self.M_x_i,
            self.M_y_i,
            self.M_z_i,
            self.Delta,
            self._omega_ax_i,
            self.F_sn,
        ) = solve_usadel_self_consistent(
            self.assemble_fns,
            self.h_x,
            self.h_y,
            self.h_z,
            self.Delta,
            self.T,
            T_c0=self.T_c0,
            omega_N=omega_N,
            gamma=gamma,
            tol_g=tol_g,
            max_iter_g=max_iter_g,
            tol_Delta=tol_Delta,
            max_iter_Delta=max_iter_Delta,
            verbose=verbose,
            use_dense=self.use_dense,
        )

    def set_real_omega_ax(self, omega_min, omega_max, omega_N):
        """
        Set the energy axis (real frequencies).

        Parameters
        ----------
        omega_min : float
            Minimum energy.
        omega_max : float
            Maximum energy.
        omega_N : int
            Number of points.
        """
        self._omega_ax_r = -1j * np.linspace(omega_min, omega_max, omega_N)
        self.theta_r = np.ones((omega_N, self.Nsites), dtype=complex)
        self.M_x_r = np.zeros((omega_N, self.Nsites), dtype=complex)
        self.M_y_r = np.zeros((omega_N, self.Nsites), dtype=complex)
        self.M_z_r = np.zeros((omega_N, self.Nsites), dtype=complex)

    def get_omega_ax_r(self):
        """Return the real-energy axis."""
        return np.real(1j * self._omega_ax_r)

    def solve_spectral(
        self,
        gamma: float = 1,
        tol: float = 1e-6,
        max_iter: int = 1000,
        print_exit_status: bool = False,
    ):
        """Solve the Usadel equations for real frequencies."""
        solve_usadel(
            assemble_fns=self.assemble_fns,
            h_x=self.h_x,
            h_y=self.h_y,
            h_z=self.h_z,
            theta=self.theta_r,
            M_x=self.M_x_r,
            M_y=self.M_y_r,
            M_z=self.M_z_r,
            Delta=self.Delta,
            omega_ax=self._omega_ax_r,
            gamma=gamma,
            tol=tol,
            max_iter=max_iter,
            print_exit_status=print_exit_status,
            use_dense=self.use_dense,
        )
        self.M_0_r = np.sqrt(1 + self.M_x_r**2 + self.M_y_r**2 + self.M_z_r**2)

    def get_ldos(self):
        """Return the local density of states."""
        return np.real(self.M_0_r * np.cos(self.theta_r))

    def get_spin_resolved_ldos(self, direction: str):
        """Return spin-resolved LDOS along x, y or z."""
        if direction == "x":
            return (
                np.real(
                    self.M_0_r * np.cos(self.theta_r)
                    + 1j * self.M_x_r * np.sin(self.theta_r)
                )
                / 2,
                np.real(
                    self.M_0_r * np.cos(self.theta_r)
                    - 1j * self.M_x_r * np.sin(self.theta_r)
                )
                / 2,
            )
        elif direction == "y":
            return (
                np.real(
                    self.M_0_r * np.cos(self.theta_r)
                    + 1j * self.M_y_r * np.sin(self.theta_r)
                )
                / 2,
                np.real(
                    self.M_0_r * np.cos(self.theta_r)
                    - 1j * self.M_y_r * np.sin(self.theta_r)
                )
                / 2,
            )
        elif direction == "z":
            return (
                np.real(
                    self.M_0_r * np.cos(self.theta_r)
                    + 1j * self.M_z_r * np.sin(self.theta_r)
                )
                / 2,
                np.real(
                    self.M_0_r * np.cos(self.theta_r)
                    - 1j * self.M_z_r * np.sin(self.theta_r)
                )
                / 2,
            )
        else:
            raise Exception("Error: direction must be 'x', 'y', or 'z'.")

    def get_anomalous_correlator(self):
        """
        Return the anomalous correlator components
        (singlet and triplet vector).
        """
        return (
            np.real(self.M_0_r * np.sin(self.theta_r)),
            np.real(-1j * self.M_x_r * np.cos(self.theta_r)),
            np.real(-1j * self.M_y_r * np.cos(self.theta_r)),
            np.real(-1j * self.M_z_r * np.cos(self.theta_r)),
        )

    def get_spin_polarization(self):
        """
        Return the local spin polarization at finite temperature.

        Returns
        -------
        (S_x, S_y, S_z) : np.ndarray
        """
        return (2 * np.pi * self.T) * np.array(
            (
                np.sum(1j * self.M_x_i * np.sin(self.theta_i), axis=0).real,
                np.sum(1j * self.M_y_i * np.sin(self.theta_i), axis=0).real,
                np.sum(1j * self.M_z_i * np.sin(self.theta_i), axis=0).real,
            )
        )
