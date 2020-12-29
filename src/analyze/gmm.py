import numpy as np


class GaussianMixture:
    def __init__(self, k, X=None):
        """
        Initializes a gaussian mixture model for MFCC data.
        If data is None, then initial gaussian parameters will be random.
        If data is specified then initial gaussian parameters will be chosen randomly based on the data specified.
        Gaussian cluster parameters:
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

        self.normalize_clusters()

    def normalize_clusters(self):
        """
        Normalizes the clusters weights so they sum up to 1. Used to compute weights for GMM.
        """
        total_weight = 0.0
        for cluster in self.clusters:
            total_weight += cluster.pi

        for cluster in self.clusters:
            cluster.pi /= total_weight

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
        totals = np.zeros((X.shape[0], 1), dtype=np.float64)
        for cluster in self.clusters:
            totals += cluster.gaussian(X)

        return np.sum(np.log(totals))


class Cluster:
    def __init__(self, X):
        self.dim = X.shape[1]
        min_X = np.min(X, axis=0)
        max_X = np.max(X, axis=0)
        self.mu = np.random.rand(self.dim) * (max_X - min_X) + min_X
        self.cov = np.identity(self.dim, dtype=np.float64)
        self.pi = np.random.rand()

    def gaussian(self, X):
        """
        Computes the likelihoods for data samples X to be from this cluster, multiplied by its weight

        :param X: (2-D ndarray) MFCC features of scored sample
        :return: likelihoods for each sample
        """
        diff = (X - self.mu).T
        fraction = 1 / ((2 * np.pi) ** (self.dim / 2) * np.linalg.det(self.cov) ** 0.5)
        exponential = np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(self.cov)), diff))
        return self.pi * np.diagonal(fraction * exponential).reshape(-1, 1)
