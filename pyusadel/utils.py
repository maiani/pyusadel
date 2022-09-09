import numpy as np


def thermal_broadening(e_ax, x, T):
    """
    Broaden the data in x by convoluting it with df/de.
    """
    if T < 5e-3:
        return x
    else:

        def f(eps, T):
            """
            Fermi-Dirac distribution.
            """
            return 0.5 * (1 + np.tanh(eps / (2 * T)))

        def df_de(eps, T):
            """
            Derivative of Fermi-Dirac distribution.
            """
            return 1 / (2 * T * (1 + np.cosh(eps / T)))

    de = (e_ax[1] - e_ax[0]).astype(np.float128)
    e_ax = e_ax.astype(np.float128)
    out = np.zeros_like(e_ax, dtype=np.float128)

    for i in range(e_ax.shape[0]):
        for j in range(e_ax.shape[0]):
            out[i] += x[j] * df_de(e_ax[i] - e_ax[j], T) * de

        out[i] += x[0] * (1 - f(e_ax[i] - e_ax[+0] + de / 2, T))
        out[i] += x[-1] * (1 - f(e_ax[-1] - e_ax[i] + de / 2, T))

    return out
