import numpy as np
from scipy.signal import convolve2d

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


@jit
def f(eps, T):
    """
    Fermi-Dirac distribution.
    """
    return 0.5 * (1 - np.tanh(eps / (2 * T)))


@jit
def ndf_de(eps, T):
    """
    Negative of the derivative of Fermi-Dirac distribution.
    """
    return 1 / (2 * T * (1 + np.cosh(eps / T)))


def thermal_broadening(e_ax, x, T):
    """
    Broaden the data in x by convoluting it with df/de.
    """

    de = e_ax[1] - e_ax[0]
    ndf = ndf_de(e_ax, T) * de
    ndf = ndf[:, np.newaxis]
    out = convolve2d(x, ndf, mode="same")

    for i in range(e_ax.shape[0]):
        out[i] += x[0] * (f(e_ax[i] - e_ax[+0] + de / 2, T))
        out[i] += x[-1] * (f(e_ax[-1] - e_ax[i] + de / 2, T))

    return out
