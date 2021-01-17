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

    def fit(self, X, n_clusters, nr_em):
        """
        Trains the model with data using EM algorithm.

        :param X: (2-D ndarray) MFCC features for the model
        """
        history = []
        likelyhoods = np.zeros((nr_em, ))
        scores = np.zeros((X.shape[0], n_clusters))

        for i in range(nr_em):
            # clusters_snapshot = []
            # history.append(clusters_snapshot)

            self.expectation_step(X)
            self.maximization_step(X)
            likelyhoods[i] = self.score(X)

            print('Nr iteracji: ', i+1, 'Likelyhood: ', likelyhoods)

        for i, cluster in enumerate(self.clusters):
            scores[:, 1] = np.log(cluster.gaussian(X)).reshape(-1)

        return self.clusters, likelyhoods, scores  # history

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

    def expectation_step(self, X):
        totals = np.zeros((X.shape[0], 1), dtype=np.float64)
        for cluster in self.clusters:
            totals += cluster.gaussian(X)
            cluster.gamma_nk = cluster.gaussian(X)

        for cluster in self.clusters:
            cluster.gamma_nk /= totals

    def maximization_step(self, X):
        N = float(X.shape[0])
        for cluster in self.clusters:
            cov_k = np.zeros((X.shape[1], X.shape[1]))

            pi_k = np.sum(cluster.gaussian(X), axis=0) / N
            mu_k = np.sum(cluster.gaussian(X) * X, axis=0) / (pi_k * N)

            for j in range(X.shape[0]):
                diff = (X[j] - mu_k).reshape(-1, 1)
                cov_k += cluster.gaussian(X)[j] * np.dot(diff, diff.T)

            cov_k /= (pi_k * N)

            cluster.pi = pi_k
            cluster.mu = mu_k
            cluster.cov = cov_k


class Cluster:
    def __init__(self, X):
        self.dim = X.shape[1]
        min_X = np.min(X, axis=0)
        max_X = np.max(X, axis=0)
        self.mu = np.random.rand(self.dim) * (max_X - min_X) + min_X
        self.cov = np.identity(X.shape[1], dtype=np.float64)
        self.pi = np.random.rand()

    def gaussian(self, X):
        """
        Computes the likelihoods for data samples X to be from this cluster, multiplied by its weight

        :param X: (2-D ndarray) MFCC features of scored sample
        :return: likelihoods for each sample
        """
        diff = (X - self.mu).T
        fraction = 1 / ((2 * np.pi) ** (X.shape[1] / 2) * np.linalg.det(self.cov) ** 0.5)
        exponential = np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(self.cov)), diff))
        return self.pi * np.diagonal(fraction * exponential).reshape(-1, 1)
