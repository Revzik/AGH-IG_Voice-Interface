import numpy as np
import scipy as sp
from src.conf import config
import scipy.signal as sig
from src.synthesis import autocorrelation


def excitement(tonal, t, lpc_coeff, enhancement, window_overlap=config.analysis['window_overlap'], fs=config.analysis['sampling_frequency']):
    where = 0
    if t > 80:
        t = np.round(t/2)
    n = int(441)
    s = np.zeros(n)
    bs = np.zeros(len(lpc_coeff))
    for i in range(0, n-1):
        if tonal:
            #pobudzenie tonalne
            if i == where:
                excitement = 1
                where = where+t
            else:
                excitement = 0
        else:
            #pobudzenie szumowe
            excitement = 2 * (np.random.rand(1)-1/2)
        cz_1 = enhancement * excitement
        cz_2 = np.dot(lpc_coeff, bs)
        x = cz_1-cz_2
        s[i] = x
        bs[1:len(lpc_coeff)] = bs[0:len(lpc_coeff)-1]
        bs[0] = x

    return s




