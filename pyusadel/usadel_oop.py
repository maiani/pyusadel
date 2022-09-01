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
from .usadel import gen_assemble_fns, solve_usadel, solve_usadel_self_consistent


class UsadelProblem:
    """Simple way to define and solve a Usadel problem.

    This class is is a wrapper for the procedural functions.
    """

    def __init__(
        self,
        Nsites: int,
        diff_ops: DifferentialOperators,
        h_x: np.ndarray,
        h_y: np.ndarray,
        h_z: np.ndarray,
        tau_sf_inv: np.ndarray,
        tau_so_inv: np.ndarray,
        D: float,
        T: float,
        T_c0: float = 1,
        Gamma: float = 0,
    ):
        """
        Parameters
        ----------
        Nsites: int
            Number of sites in the system.
        diff_ops: DifferentialOperators
            Differential operators on the lattice.
        h_x: np.ndarray
            Zeeman field (x component)
        h_y: np.ndarray
            Zeeman field (y component)
        h_z: np.ndarray
            Zeeman field (z component)
        tau_sf_inv: np.ndarray
            Inverse of spin-flip scattering time.
        tau_so_inv: np.ndarray
            Inverse of spin-orbit scattering time.
        D: float
            Diffusion constant.
        T: float
            Temperature.
        T_c0: float (default 1)
            Critical temperature in absence of pair-breaking.
        Gamma: float (default 0)
            Dynes parameter.
        """

        self.Nsites = Nsites
        self.diff_ops = diff_ops

        if (
            h_x.shape != (Nsites,)
            or h_y.shape != (Nsites,)
            or h_z.shape != (Nsites,)
            or tau_sf_inv.shape != (Nsites,)
            or tau_so_inv.shape != (Nsites,)
        ):
            raise Exception("Dimensions doesn't match.")
        else:
            self.h_x = h_x
            self.h_y = h_y
            self.h_z = h_z
            self.tau_sf_inv = tau_sf_inv
            self.tau_so_inv = tau_so_inv

        self.D = D
        self.T = T
        self.T_c0 = T_c0
        self.Gamma = Gamma

        self.assemble_fns = gen_assemble_fns(
            D=self.D,
            diff_ops=self.diff_ops,
            h_x=self.h_x,
            h_y=self.h_y,
            h_z=self.h_z,
            tau_so_inv=self.tau_so_inv,
            tau_sf_inv=self.tau_sf_inv,
            Gamma=self.Gamma,
        )

        self.Delta = np.ones((Nsites), dtype=np.float)
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

        # Calculate useful parameters
        self.Delta_00 = self.T_c0 * 1.7652
        self.h_c0 = self.Delta_00 / np.sqrt(2)
        self.xi_00 = np.sqrt(self.D / self.Delta_00)

    def solve_self_consistent(
        self,
        omega_N: int = None,
        gamma=1,
        tol_g: float = 1e-6,
        max_iter_g: int = 1000,
        tol_Delta: float = 1e-6,
        max_iter_Delta: int = 100,
        verbose: bool = False,
    ):
        if not omega_N:
            # TODO: implement something smartere here
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
        )

    def set_real_omega_ax(self, omega_min, omega_max, omega_N):
        """
        Set the energy axis (real frequencies).
        Parameters:
        -------
        omega_min : float
            Minimum energy
        omega_max : float
            Minimum energy
        omega_N : int
            Number of points
        """

        self._omega_ax_r = -1j * np.linspace(omega_min, omega_max, omega_N)
        self.theta_r = np.ones((omega_N, self.Nsites), dtype=complex)
        self.M_x_r = np.zeros((omega_N, self.Nsites), dtype=complex)
        self.M_y_r = np.zeros((omega_N, self.Nsites), dtype=complex)
        self.M_z_r = np.zeros((omega_N, self.Nsites), dtype=complex)

    def get_omega_ax_r(self):
        """
        Returns the energy axis (real frequencies).
        """
        return np.real(1j * self._omega_ax_r)

    def solve_spectral(
        self,
        gamma: float = 1,
        tol: float = 1e-6,
        max_iter: int = 1000,
        print_exit_status: bool = False,
        use_dense: bool = False,
    ):
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
            use_dense=use_dense,
        )
        self.M_0_r = np.sqrt(1 + self.M_x_r**2 + self.M_y_r**2 + self.M_z_r**2)

    def get_dos(self):
        return np.real(self.M_0_r * np.cos(self.theta_r))

    def get_spin_resolved_dos(self, direction: str):
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
            raise Exception("Error.")

    def get_pairing_amplitudes(self):
        return (
            np.real(self.M_0_r * np.sin(self.theta_r)),
            np.real(-1j * self.M_x_r * np.cos(self.theta_r)),
            np.real(-1j * self.M_y_r * np.cos(self.theta_r)),
            np.real(-1j * self.M_z_r * np.cos(self.theta_r)),
        )

    def generate_self_energy(self):
        pass
