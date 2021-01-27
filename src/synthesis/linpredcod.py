import numpy as np
import scipy as sp
from src.conf import config
import scipy.signal as sig
from src.synthesis import autocorrelation


def lpc(signal, order = config.analysis['order']):
    """Compute the Linear Prediction Coefficients.
    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:
      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]
    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.
    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)
    Notes
    ----
    This is just for reference, as it is using the direct inversion of the
    toeplitz matrix, which is really slow"""
    # if signal.ndim > 1:
    #     raise ValueError("Array of rank > 1 not supported yet")
    # if order > len(signal):
    #     raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p)
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        nx = np.min([p, len(signal)])
        x = np.correlate(signal, signal, 'full')
        #tonal1, delays1, x = autocorrelation.acorr(signal, debug=True)
        r[:nx] = x[len(signal)-1:len(signal)+order]
        phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi))
    else:
        return np.ones(1, dtype = signal.dtype)