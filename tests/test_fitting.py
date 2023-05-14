import numpy as np
from pyusadel import UsadelProblem, thermal_broadening, trivial_diffops, fit_nis


def test_fitting():
    Nsites = 1

    do = trivial_diffops()

    h_x = np.array([0.3])
    h_y = np.array([0.0])
    h_z = np.array([0.0])
    tau_sf_inv = np.array([0.1])
    tau_so_inv = np.array([0.2])
    D = 0
    T = 0.01
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

    numerical_nis = thermal_broadening(e_ax, up.get_ldos()[:, 0], T=T)

    # Add noise
    numerical_nis += np.random.randn(e_ax.shape[0]) * 0.01

    params = dict(
        Delta=(0, 0.75, 1.25),
        h=(0.0, 0.2, 0.5),
        G_N=1,
        T=T,
        tau_sf_inv=(0.0, 0.05, 0.5),
        tau_so_inv=(0.0, 0.05, 0.5),
        Gamma=Gamma,
        x_N=0.0,
    )

    popt, pcov = fit_nis(
        e_ax_exp=e_ax,
        dos_exp=numerical_nis,
        **params,
        verbose=False,
        tol=1e-6,
    )

    print(popt)

    assert abs(np.max(popt / np.array([1, 0.3, 0.1, 0.2]) - 1)) < 1e-2
