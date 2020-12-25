import numpy as np


class GaussianMixture:
    def __init__(self, k, X=None):
        """
        Initializes a gaussian mixture model for MFCC data.
        If data is None, then initial gaussian parameters will be random.
        If data is specified then initial gaussian parameters will be chosen randomly based on the data specified.
        Gaussian parameters:
        mi - (1-D ndarray) mean values of each gaussian
        cov - (1-D ndarray) variances of each gaussian
        pi - (1-D ndarray) weights of each gaussian

        :param k: (int) number of gaussians
        :param X: (2-D ndarray) data samples, where:
                rows - consecutive samples
                columns - components of each sample
        """
        self.k = k
        self.clusters = []
        for i in range(self.k):
            self.clusters.append(Cluster(X))
            self.pi = GaussianMixture.normalize(np.random.rand(self.k))

    @staticmethod
    def normalize(pi):
        """
        Normalizes the array specified so sum if its components is 1. Used to compute weights for GMM.

        :param pi: (1-D ndarray) input array
        :return: (1-D ndarray) pi with the sum of components equal to 1
        """

        return pi / np.sum(pi)

    def fit(self, X):
        """
        Trains the model with data using EM algorithm.

        :param X: (2-D ndarray) MFCC features for the model
        """
        pass

    def score(self, X):
        """
        Computes the log-likelihood of input data for trained model.

        :param X: (2-D ndarray) MFCC features of scored sample
        :return: (float) log-likelihood of the data for specific model
        """
        pass

    def log_gaussian(self, X, i):
        """
        Computes the log-likelihood of data to be from a specific gaussian. The points in data are 1-D

        :param X: (1-D ndarray) sample to be evaluated
        :param i: (int) index of the gaussian (0 <= i < k)
        :return: likelihood of x to be from gaussian i
        """
        pass

    def gaussian(self, X, i):
        """
        Computes the likelihood of data to be from a specific gaussian. The points in data are 1-D

        :param X: (1-D ndarray) sample to be evaluated
        :param i: (int) index of the gaussian (0 <= i < k)
        :return: likelihood of x to be from gaussian i
        """
        pass


class Cluster:
    def __init__(self, X):
        dim = X.shape[1]
        min_X = np.min(X, axis=1)
        max_X = np.max(X, axis=1)
        self.mi = np.random.rand(dim) * (max_X - min_X) + min_X
        self.cov = np.identity(dim)
