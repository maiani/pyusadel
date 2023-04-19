from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

from pyusadel import (
    gen_assemble_fns,
    solve_usadel,
    thermal_broadening,
    trivial_diffops,
    resize_linspace,
)


def fit_nis_ts(
    e_ax_exp: np.ndarray,
    dos_exp: np.ndarray,
    Delta: Union[float, Sequence[float], np.ndarray],
    h: Union[float, Sequence[float], np.ndarray] = 0.0,
    G_N: Union[float, Sequence[float], np.ndarray] = 1.0,
    T: Union[float, Sequence[float], np.ndarray] = 0.0,
    tau_sf_inv: Union[float, Sequence[float], np.ndarray] = 0.0,
    tau_so_inv: Union[float, Sequence[float], np.ndarray] = 0.0,
    Gamma: Union[float, Sequence[float], np.ndarray] = 1e-6,
    x_N: Union[float, Sequence[float], np.ndarray] = 0.0,
    tol: float = 1e-8,
    verbose: bool = False,
    solution: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Tuple[float], np.ndarray, np.ndarray]:
    """Fit the experimental data.

    Parameters:
    -------
    e_ax_exp : np.ndarray
        Energy axis of the experimental data.
    dos_exp : np.ndarray
        Spectroscopy data.
    Delta : float or Tuple[float, float, float]
        The superconducting gap. If it's a tuple/list/numpy array of three numbers,
        it's treated as bounds and initial guess for a parameter that will be fitted
        by the function. Otherwise is considered a fixed parameter.
    h : float or Tuple[float, float, float]
        The exchange field. If it's a tuple/list/numpy array of three numbers, it's
        treated as bounds and initial guess for a parameter that will be fitted by
        the function. Otherwise is considered a fixed parameter.
    G_N : float or Tuple[float, float, float]
        The normal-state conductance. If it's a tuple/list/numpy array of three
        numbers, it's treated as bounds and initial guess for a parameter that will
        be fitted by the function. Otherwise is considered a fixed parameter.
    T : float or Tuple[float, float, float]
        The temperature. If it's a tuple/list/numpy array of three numbers, it's
        treated as bounds and initial guess for a parameter that will be fitted by
        the function.  Otherwise is considered a fixed parameter.
    tau_sf_inv : float or Tuple[float, float, float]
        The spin-flip scattering rate. If it's a tuple/list/numpy array of three
        numbers, it's treated as bounds and initial guess for a parameter that will
        be fitted by the function. Otherwise is considered a fixed parameter.
    tau_so_inv : float or Tuple[float, float, float]
        The spin-orbit scattering rate. If it's a tuple/list/numpy array of three
        numbers, it's treated as bounds and initial guess for a parameter that will
        be fitted by the function. Otherwise is considered a fixed parameter.
    Gamma : float or Tuple[float, float, float]
        The Dynes parameter. If it's a tuple/list/numpy array of three numbers,
        it's treated as bounds and initial guess for a parameter that will be fitted
        by the function. Otherwise is considered a fixed parameter.
    x_N : float or Tuple[float, float, float]
        The normal state backround. If it's a tuple/list/numpy array of three numbers,
        it's treated as bounds and initial guess for a parameter that will be fitted
        by the function. Otherwise is considered a fixed parameter.
    tol : float
        Tolerance used in the fitting procedure. Default 1e-8.
    verbose : bool
        Whether printing the status while fitting.
    solution : Tuple[np.ndarray, np.ndarray], optional
        Variables (theta, M_x). If provided, the program will use those
        variable for the optimization. Useful to provide as initial guess
        the results of a previous optimization.
    """

    # Fill the params dict with the arguments
    params = {}
    params["Delta"] = np.array(Delta, ndmin=1)
    params["h"] = np.array(h, ndmin=1)
    params["G_N"] = np.array(G_N, ndmin=1)
    params["T"] = np.array(T, ndmin=1)
    params["tau_sf_inv"] = np.array(tau_sf_inv, ndmin=1)
    params["tau_so_inv"] = np.array(tau_so_inv, ndmin=1)
    params["Gamma"] = np.array(Gamma, ndmin=1)
    params["x_N"] = np.array(x_N, ndmin=1)

    # Separate the free parameters from the fixed by fillinf initial guess and bouns arrays
    free_params_keys = []
    free_params_initial_guess = []
    free_params_lbounds = []
    free_params_ubounds = []

    for key, value in params.items():
        if len(value) == 3:
            free_params_keys.append(key)
            free_params_initial_guess.append(value[1])
            free_params_lbounds.append(value[0])
            free_params_ubounds.append(value[2])

        elif len(value) != 1:
            raise Error("Input not valid")

    # Setup the variable for the Usadel model
    do = trivial_diffops()

    new_length = e_ax_exp.shape[0] * 1.2
    e_ax_model = resize_linspace(e_ax_exp, new_length)

    if solution:
        theta, M_x = solution
    else:
        theta = np.ones((e_ax_model.shape[0], 1), dtype=complex)
        M_x = np.zeros((e_ax_model.shape[0], 1), dtype=complex)

    # Define the wrapper function
    def wrapper(e_ax_exp, *x):

        for i, key in enumerate(free_params_keys):
            params[key] = np.array([x[i]])

        if verbose:
            for i, key in enumerate(free_params_keys):
                print(f"{key} = {x[i]:5.4f}, ", end="")
            print()

        assemble_fns = gen_assemble_fns(
            D=0,
            diff_ops=do,
            h_x=params["h"],
            h_y=0,
            h_z=0,
            tau_so_inv=params["tau_so_inv"],
            tau_sf_inv=params["tau_sf_inv"],
            Gamma=params["Gamma"],
        )

        solve_usadel(
            assemble_fns=assemble_fns,
            h_x=h,
            h_y=np.array([0.0]),
            h_z=np.array([0.0]),
            theta=theta,
            M_x=M_x,
            M_y=np.array([0.0]),
            M_z=np.array([0.0]),
            Delta=params["Delta"],
            omega_ax=-1j * e_ax_model,
            gamma=1.0,
            max_iter=1000,
            print_exit_status=False,
            use_dense=True,
        )

        M_0 = np.sqrt(1 + M_x**2)

        zero_temparature_g = (
            params["G_N"]
            * ((1 - params["x_N"]) * np.real(M_0 * np.cos(theta)) + params["x_N"])
        )[:, 0]

        finite_temperature_g = thermal_broadening(
            e_ax_model,
            zero_temparature_g,
            T=params["T"],
        )

        e_ax_resized, g_model = resize_linspace(
            e_ax_model, y=finite_temperature_g, new_length=e_ax_exp.shape[0]
        )
        return g_model

    verbosity = 2 if verbose else 0

    fit_results = curve_fit(
        f=wrapper,
        xdata=e_ax_exp,
        ydata=dos_exp,
        p0=free_params_initial_guess,
        # sigma=None,
        # absolute_sigma=False,
        check_finite=True,
        bounds=(free_params_lbounds, free_params_ubounds),
        # method="trf",
        # jac=None,
        # full_output=True,
        verbose=verbosity,
        ftol=tol,
        xtol=tol,
        gtol=tol,
        x_scale="jac",
    )

    return fit_results
