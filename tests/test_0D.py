import matplotlib.pyplot as plt
import numpy as np
from pyusadel import (
    DifferentialOperators,
    UsadelProblem,
    thermal_broadening,
    trivial_diffops,
)


def analytical_DOS(e, Delta, h, Gamma):
    return (
        np.imag(
            (e + h + 1j * Gamma) / np.sqrt(Delta - (e + h + 1j * Gamma) ** 2)
            - (e + h - 1j * Gamma) / np.sqrt(Delta - (e + h - 1j * Gamma) ** 2)
            + (e - h + 1j * Gamma) / np.sqrt(Delta - (e - h + 1j * Gamma) ** 2)
            - (e - h - 1j * Gamma) / np.sqrt(Delta - (e - h - 1j * Gamma) ** 2)
        )
        / 4
    )


def test_0D_model():

    Nsites = 1

    do = trivial_diffops()

    h_x = np.array([0.3])
    h_y = np.array([0.0])
    h_z = np.array([0.0])
    tau_sf_inv = np.array([0.0])
    tau_so_inv = np.array([0.0])
    D = 0
    T = 1e-4
    Gamma = 5e-3
    Delta = 1.0

    up = UsadelProblem(
        Nsites=1,
        diff_ops=do,
        h_x=h_x,
        h_y=h_y,
        h_z=h_z,
        tau_sf_inv=tau_sf_inv,
        tau_so_inv=tau_so_inv,
        D=D,
        T=T,
        Gamma=Gamma,
        use_dense=True,
    )

    up.Delta = np.array([Delta])
    up.set_real_omega_ax(-4, 4, 501)

    # Solve the numerical Usadel equation
    e_ax = up.get_omega_ax_r()
    up.solve_spectral(print_exit_status=False, tol=1e-15, max_iter=2000)

    numerical_dos = up.get_dos()[:, 0]
    analytical_dos = analytical_DOS(e_ax, Delta, h_x, Gamma)

    assert np.max(np.abs(numerical_dos - analytical_dos)) < 1e-12
