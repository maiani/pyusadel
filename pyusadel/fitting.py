import numpy as np
from pyusadel import (
    DifferentialOperators,
    UsadelProblem,
    gen_assemble_fns,
    solve_usadel,
    thermal_broadening,
    trivial_diffops,
)
from scipy.optimize import curve_fit


def fit_nsts(
    e_ax_exp,
    dos_exp,
    initial_guess,
    bounds,
    tol=1e-8,
    verbose: bool = False,
    solution=None,
):
    """Fit the experimental data.

    Parameters:
    -------
    e_ax_exp : np.ndarray
        Energy axis of the experimental data.
    dos_exp : np.ndarray
        Spectroscopy data.
    initial_guess: tuple
        Initial guess for the paramteres
    bounds : tuple(tuple, tuple)
        Bounds on the parameters
    tol : float
        Objective tolerance.
    verbose : bool
        Whether printing the status.
    solution : tuple(np.array, np.array)
        Variables (theta, M_x). If provided, the program will use those
        variable for the optimization. Useful to provide as initial guess
        the results of a previous optimization.
    """
    do = trivial_diffops()

    if solution:
        theta, M_x = solution
    else:
        theta = np.ones((e_ax_exp.shape[0], 1), dtype=complex)
        M_x = np.zeros((e_ax_exp.shape[0], 1), dtype=complex)

    def wrapper(omega_ax_exp, *params):

        N_0, Delta, h, tau_sf_inv, tau_so_inv, Gamma, T = params

        if verbose:
            print(
                f"""N_0 = {N_0:4.3f}, Delta = {Delta:4.3f}, h = {h:4.3f}, tau_sf_inv = {tau_sf_inv:4.3f},  tau_so_inv = {tau_so_inv:4.3f}, Gamma = {Gamma:3.2e}, T = {T:3.2e}"""
            )

        assemble_fns = gen_assemble_fns(
            D=0,
            diff_ops=do,
            h_x=h,
            h_y=0,
            h_z=0,
            tau_so_inv=tau_so_inv,
            tau_sf_inv=tau_sf_inv,
            Gamma=Gamma,
        )

        solve_usadel(
            assemble_fns=assemble_fns,
            h_x=h,
            h_y=0,
            h_z=0,
            theta=theta,
            M_x=M_x,
            M_y=0,
            M_z=0,
            Delta=Delta,
            omega_ax=-1j * omega_ax_exp,
            gamma=1.0,
            tol=1e-6,
            max_iter=1000,
            print_exit_status=False,
            use_dense=True,
        )

        M_0 = np.sqrt(1 + M_x**2)

        return thermal_broadening(
            e_ax_exp, N_0 * np.real(M_0 * np.cos(theta))[:, 0], T=T
        )

    verbosity = 2 if verbose else 0

    fit_results = curve_fit(
        f=wrapper,
        xdata=e_ax_exp,
        ydata=dos_exp,
        p0=initial_guess,
        # sigma=None,
        # absolute_sigma=False,
        check_finite=True,
        bounds=bounds,
        # method=None,
        # jac=None,
        # full_output=True,
        verbose=verbosity,
        ftol=tol,
        xtol=tol,
        gtol=tol,
        x_scale="jac",
    )

    return fit_results
