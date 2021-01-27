import numpy as np
import scipy as sp
from src.conf import config
import scipy.signal as sig
from src.synthesis import autocorrelation


def excitement(tonal, t, lpc_coeff, enhancement, window_overlap=config.analysis['window_overlap'], fs=config.analysis['sampling_frequency']):
    where = 0
    s = []
    bs = np.zeros(len(lpc_coeff))
    for i in range(0, round(fs/window_overlap)-1):
        if tonal:
            #pobudzenie tonalne
            if i == where:
                excitement = 1
                where = where+t
            else:
                excitement = 0
        else:
            #pobudzenie szumowe
            excitement = 2 * (np.random(1, 1)-1/2)
        cz_1 = enhancement * excitement
        cz_2 = np.dot(lpc_coeff, bs)
        s.append(cz_1 - cz_2)
        bs = np.insert(bs, 0, s)
        bs = bs[0: len(lpc_coeff)]

    return s




