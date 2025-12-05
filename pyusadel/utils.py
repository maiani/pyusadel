"""
Utility functions for thermal broadening and grid manipulation
used in the Usadel solver.

Andrea Maiani, 2022–2025
"""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from typing import Optional, Tuple, Union

try:
    import numba
except ImportError:
    numba_available = False
else:
    numba_available = True

if numba_available:
    jit = numba.jit
else:
    def jit(fn):
        return fn


def thermal_broadening(e_ax: np.ndarray, y: np.ndarray, T: float) -> np.ndarray:
    """
    Computes the thermal broadening of a given spectrum at a given temperature.

    Parameters:
    -----------
    e_ax : np.ndarray
        Array of energy axis values.
    y : np.ndarray
        Array of corresponding values.
    T : float
        Temperature at which to compute the thermal broadening.

    Returns:
    --------
    tb : np.ndarray
        Array of thermal broadening values.

    Raises:
    -------
    AssertionError:
        If the temperature is too low.
    """
    if T < 0.0007:
        return y
    else:
        y_f = interp1d(e_ax, y, bounds_error=False, fill_value="extrapolate")

        def integrand(x: np.ndarray, e: float, T: float) -> np.ndarray:
            return y_f(e - x * T) / (2 * (1 + np.cosh(x)))

        tb: np.ndarray = np.zeros_like(e_ax)

        for i, e in enumerate(e_ax):
            x = np.linspace(e_ax.min() / T, e_ax.max() / T, 4001)
            dx = x[1] - x[0]
            tb[i] = np.sum(integrand(x, e, T)) * dx

        return tb


def resize_linspace(
    linspace_arr: np.ndarray,
    new_length: int,
    filling_value: Optional[float] = np.nan,
    y: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Resize a linspace-like array and optionally resize associated data y(x).

    This is symmetric resizing: new points are added or removed equally
    on the left and right of the array.

    Parameters
    ----------
    linspace_arr : np.ndarray
        Original 1D linspace array.
    new_length : int
        Desired length of the resized output.
    filling_value : float or None, optional
        Value to use when padding y during upsizing.
        Ignored when downsizing.
    y : np.ndarray, optional
        Array of values defined on linspace_arr. If provided, it is resized
        in a manner consistent with linspace_arr.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        If y is None:
            resized_linspace_arr
        If y is provided:
            (resized_linspace_arr, resized_y_arr)

    Notes
    -----
    - Upsizing adds equally many points to both ends.
    - Downsizing removes equally many points from both ends.
    """
    orig_length = len(linspace_arr)
    spacing = linspace_arr[1] - linspace_arr[0]

    # -----------------
    # Upsizing
    # -----------------
    if new_length > orig_length:
        num_elems = int((new_length - orig_length) / 2)

        # Extend left and right
        resized_linspace_arr = np.concatenate(
            [
                np.linspace(
                    linspace_arr[0] - num_elems * spacing,
                    linspace_arr[0] - spacing,
                    num_elems,
                ),
                linspace_arr,
                np.linspace(
                    linspace_arr[-1] + spacing,
                    linspace_arr[-1] + num_elems * spacing,
                    num_elems,
                ),
            ]
        )

        if y is not None:
            resized_y_arr = np.full(new_length, filling_value, dtype=y.dtype)
            resized_y_arr[num_elems : num_elems + orig_length] = y
            return resized_linspace_arr, resized_y_arr
        return resized_linspace_arr

    # -----------------
    # Downsizing
    # -----------------
    elif new_length < orig_length:
        num_elems = int((orig_length - new_length) / 2)
        resized_linspace_arr = linspace_arr[num_elems:-num_elems]

        if y is not None:
            resized_y_arr = y[num_elems:-num_elems]
            return resized_linspace_arr, resized_y_arr
        return resized_linspace_arr

    # -----------------
    # No change
    # -----------------
    else:
        if y is not None:
            return linspace_arr, y
        return linspace_arr
