import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.analyze import gmm


class GMMTest(unittest.TestCase):
    def test_gaussian_range(self):
        data_min = np.array([-3, -5])
        data_max = np.array([3, 1])
        data = np.random.rand(10, 2) * (data_max - data_min) + data_min

        model = gmm.GaussianMixture(2, data)

        weights = 0

        for cluster in model.clusters:
            for i, mu in enumerate(cluster.mu):
                self.assertGreaterEqual(mu, data_min[i])
                self.assertLessEqual(mu, data_max[i])

            weights += cluster.pi

        self.assertEqual(1.0, weights)

    def test_gaussian_shape(self):
        data_min = np.array([-3])
        data_max = np.array([3])
        data = np.linspace(data_min, data_max, 30)

        model = gmm.GaussianMixture(2, data)

        score = model.score(data)
        scores = np.zeros((data.shape[0], model.k))
        for i in range(model.k):
            scores[:, i] = model.clusters[i].gaussian(data).reshape(30)

        self.assertLess(score, 0)

        mu_indexes = np.zeros(model.k, dtype=int)
        for i in range(model.k):
            mu = model.clusters[i].mu[0]
            for j in range(data.shape[0]):
                if np.abs(mu - data[j, 0]) < np.abs(mu - data[mu_indexes[i], 0]):
                    mu_indexes[i] = j

        max_scores = np.max(scores, axis=0)
        for i in range(mu_indexes.size):
            self.assertEqual(max_scores[i], scores[mu_indexes[i], i])

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes[0].plot(data, scores[:, 0])
        axes[1].plot(data, scores[:, 1])
        fig.show()

    def test_em(self):
        data = np.array([[5.1, 3.5, 1.4, 0.2],
                         [4.9, 3., 1.4, 0.2],
                         [4.7, 3.2, 1.3, 0.2],
                         [4.6, 3.1, 1.5, 0.2],
                         [5., 3.6, 1.4, 0.2],
                         [5.4, 3.9, 1.7, 0.4],
                         [4.6, 3.4, 1.4, 0.3],
                         [5., 3.4, 1.5, 0.2],
                         [4.4, 2.9, 1.4, 0.2],
                         [4.9, 3.1, 1.5, 0.1],
                         [5.4, 3.7, 1.5, 0.2],
                         [4.8, 3.4, 1.6, 0.2],
                         [4.8, 3., 1.4, 0.1],
                         [4.3, 3., 1.1, 0.1],
                         [5.8, 4., 1.2, 0.2],
                         [5.7, 4.4, 1.5, 0.4],
                         [5.4, 3.9, 1.3, 0.4],
                         [5.1, 3.5, 1.4, 0.3],
                         [5.7, 3.8, 1.7, 0.3],
                         [5.1, 3.8, 1.5, 0.3]])

        model_gmm = gmm.GaussianMixture(4, data)

        n_clusters = 4
        n_epochs = 20

        clusters, likelihoods, scores = model_gmm.fit(data, n_clusters, n_epochs)

        plt.figure(figsize=(10, 10))
        plt.title('Log-Likelihood')
        plt.plot(np.arange(1, n_epochs + 1), likelihoods)
        plt.show()

    pass
