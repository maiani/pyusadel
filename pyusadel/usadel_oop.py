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
from .usadel import gen_assemble_fun, solve_usadel, solve_usadel_self_consistent

class UsadelProblem():
    """
    This class provide a simple way to define and solve a Usadel problem.
    """   
    
    def __init__(self, Nsites : int, 
                       diff_ops : DifferentialOperators,
                       h_x : np.ndarray,
                       h_y : np.ndarray,
                       tau_sf_inv : np.ndarray,
                       tau_so_inv : np.ndarray,
                       D : float, 
                       T : float,
                       T_c0 : float = 1
                        ):
        
        self.Nsites = Nsites
        self.diff_ops = diff_ops
        
        self.h_x = h_x
        self.h_y = h_y
        self.tau_sf_inv = tau_sf_inv
        self.tau_so_inv = tau_so_inv
        self.D = D        
        self.T = T
        self.T_c0 = 1
    
        self.assemble_fun = gen_assemble_fun(
            D=self.D, 
            diff_ops=self.diff_ops, 
            tau_so_inv=self.tau_so_inv,
            tau_sf_inv = self.tau_sf_inv)
        
        self.Delta = np.ones((Nsites), dtype = np.float)
        self.f_sn = None
        
        # Imaginary frequency axis
        self._omega_ax_i = None
        self.theta_i = None
        self.M_x_i = None
        self.M_y_i = None
        
        # Real frequency axis
        self._omega_ax_r = None
        self.theta_r = None
        self.M_x_r = None
        self.M_y_r = None
    
    

    def solve_self_consistent(self, omega_N : int = None):
        if not omega_N:
            omega_N = 100
        
        (self.theta_i, self.M_x_i, self.M_y_i, self.Delta, self._omega_ax_i, self.f_sn) = solve_usadel_self_consistent(
        self.assemble_fun,
        self.h_x,
        self.h_y,
        self.Delta,
        self.T,
        omega_N=omega_N,
        #gamma=1,
        #tol=1e-6,
        #max_iter=100,
        #max_iter_delta=100,
        )
    
    
    def set_real_omega_ax(self, omega_min, omega_max, omega_N):
        self._omega_ax_r =  -1j * np.linspace(omega_min, omega_max, omega_N)
        self.theta_r = np.ones((omega_N, self.Nsites), dtype=complex)
        self.M_x_r = np.zeros((omega_N, self.Nsites), dtype=complex)
        self.M_y_r = np.zeros((omega_N, self.Nsites), dtype=complex)
        
    
    def get_omega_ax_r(self):
        return np.real(1j * self._omega_ax_r)
    
    
    def solve_spectral(self):
        for omega_idx in range(self._omega_ax_r.shape[0]):
            solve_usadel(
            assemble_fun = self.assemble_fun[:-1],
            theta = self.theta_r,
            M_x = self.M_x_r,
            M_y = self.M_y_r,
            h_x = self.h_x,
            h_y = self.h_y,
            Delta = self.Delta,
            omega_ax = self._omega_ax_r,
            omega_idx = omega_idx,
            # gamma: float = 1,
            # tol: float = 1e-6,
            # max_iter: int = 1000,
            # print_exit_status: bool = False,
            # use_dense: bool = False,
        )
    
    
    def get_dos(self):
        M_0 = np.sqrt(1 + self.M_x_r ** 2 + self.M_y_r ** 2)
        return np.real(M_0 * np.cos(self.theta_r)) / 2
    
    def generate_self_energy(self):
        pass