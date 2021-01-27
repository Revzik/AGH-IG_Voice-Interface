import numpy as np

from sklearn.cluster import KMeans
from src.conf import config


class GaussianMixture:
    def __init__(self, X=None):
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
        self.k = config.analysis["n_clusters"]
        kmeans = KMeans(n_clusters=self.k).fit(X)
        cluster_centers = kmeans.cluster_centers_
        self.clusters = []
        for i in range(self.k):
            self.clusters.append(Cluster(cluster_centers[i, :]))

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
        iters = config.analysis["iterations"]
        likelihoods = np.zeros(iters + 1)

        likelihoods[0] = self.score(X)
        for i in range(iters):
            gammas = self.expect(X)
            self.maximize(X, gammas)

            likelihoods[i + 1] = self.score(X)

        return likelihoods

    def expect(self, X):
        gammas = np.zeros((X.shape[0], self.k), dtype=np.float64)
        totals = np.zeros((X.shape[0]), dtype=np.float64)

        for i, cluster in enumerate(self.clusters):
            gammas[:, i] = (cluster.pi * cluster.gaussian(X))[:, 0]

        totals[:] = np.sum(gammas, axis=1)

        for i, cluster in enumerate(self.clusters):
            gammas[:, i] /= totals[:]

        return gammas

    def maximize(self, X, gammas):
        N = X.shape[0]
        dim = X.shape[1]

        for i, cluster in enumerate(self.clusters):
            N_k = np.sum(gammas[:, i])

            cluster.pi = N_k / N

            cluster.mu = np.sum(np.multiply(gammas[:, i].T, X.T).T, axis=0) / N_k

            cluster.cov = np.zeros((dim, dim))
            for j in range(N):
                diff = (X[j, :] - cluster.mu).reshape(-1, 1)
                cluster.cov += gammas[j, i] * np.dot(diff, diff.T)
            cluster.cov /= N_k

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
    def __init__(self, centers):
        self.dim = centers.size
        self.mu = centers
        self.cov = np.identity(self.dim, dtype=np.float64)
        self.pi = 1

    def gaussian(self, X):
        """
        Computes the likelihoods for data samples X to be from this cluster, multiplied by its weight

        :param X: (2-D ndarray) MFCC features of scored sample
        :return: likelihoods for each sample
        """
        diff = (X - self.mu).T
        fraction = 1 / ((2 * np.pi) ** (self.dim / 2) * np.linalg.det(self.cov) ** 0.5)
        if np.linalg.matrix_rank(self.cov) != self.cov.shape[0]:
            print("Covariance matrix is singular! Resetting...")
            self.cov = np.identity(self.dim, dtype=np.float64)
        exponential = np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(self.cov)), diff))
        return self.pi * np.diagonal(fraction * exponential).reshape(-1, 1)
