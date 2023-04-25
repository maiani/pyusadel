import numpy as np
from scipy.interpolate import interp1d
from typing import List, Optional, Sequence, Tuple, Union

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
    filling_value: Union[float, None] = np.nan,
    y: np.ndarray = None,
) -> Union[np.ndarray, tuple]:
    """
    Given a numpy array generated using linspace_arr, this function resizes the
    array by a given new length. If a y array is provided, it is also resized
    accordingly.

    Args:
    linspace_arr (numpy.ndarray): A 1D numpy array generated using linspace
    new_length (int): The new length to resize the array to
    y (numpy.ndarray, optional): A 1D numpy array containing values evaluated
        using linspace_arr as x.
    fill_value (float, optional): The value to use for filling the new elements in y.

    Returns:
    numpy.ndarray: A 1D numpy array that is the resized version of linspace_arr
    numpy.ndarray or None: If y is provided, a 1D numpy array that is the resized
        version of y, with the fill_value inserted or removed as appropriate. If y is
        not provided, None is returned.
    """
    # Get the length of the original linspace array
    orig_length = len(linspace_arr)

    # Get the spacing between elements in the original linspace array
    spacing = linspace_arr[1] - linspace_arr[0]

    if new_length > orig_length:
        # Upsize the arrays
        # Calculate the number of elements to add to both ends of the arrays
        num_elems = int((new_length - orig_length) / 2)

        # Extend the arrays in both directions by num_elems elements
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
            resized_y_arr = np.full(new_length, fill_value, dtype=y.dtype)
            resized_y_arr[num_elems : num_elems + orig_length] = y
            return resized_linspace_arr, resized_y_arr
        else:
            return resized_linspace_arr

    elif new_length < orig_length:
        # Downsize the arrays
        # Calculate the number of elements to remove from both ends of the arrays
        num_elems = int((orig_length - new_length) / 2)

        # Remove num_elems elements from both ends of the arrays
        resized_linspace_arr = linspace_arr[num_elems:-num_elems]
        if y is not None:
            resized_y_arr = y[num_elems:-num_elems]
            return resized_linspace_arr, resized_y_arr
        else:
            return resized_linspace_arr

    else:
        # Return the original arrays if new_length is the same as the original length
        if y is not None:
            return linspace_arr, y
        else:
            return linspace_arr
