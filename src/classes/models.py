import numpy as np


class GMM:
    def __init__(self, k, data=None):
        """
        Initializes a gaussian mixture model for MFCC data.
        If data is None, then initial gaussian parameters will be random.
        If data is specified then initial gaussian parameters will be chosen randomly based on the data specified.
        Gaussian parameters:
        mi - (1-D ndarray) mean values of each gaussian
        sigma - (1-D ndarray) variances of each gaussian
        pi - (1-D ndarray) weights of each gaussian

        :param k: (int) number of gaussians
        :param data: (2-D ndarray) MFCC features in consecutive windows
        """

        self.k = k
        if data is None:
            self.mi = 10 * np.random.randn(k)
            self.sigma = 10 * np.random.randn(k)
            self.pi = GMM.normalize(np.random.randn(k))

    @staticmethod
    def normalize(pi):
        return pi / np.sum(pi)

    def fit(self, data):
        pass


def histogram(data):
    pass
