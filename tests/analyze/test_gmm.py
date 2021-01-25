import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as pat

from sklearn import datasets
from src.analyze import gmm
from src.conf import config


class GMMTest(unittest.TestCase):
    def tearDown(self):
        config.reload()

    def test_gaussian_range(self):
        config.analysis['n_clusters'] = 2

        data_min = np.array([-3, -5])
        data_max = np.array([3, 1])
        data = np.random.rand(10, 2) * (data_max - data_min) + data_min

        model = gmm.GaussianMixture(data)

        weights = 0

        for cluster in model.clusters:
            for i, mu in enumerate(cluster.mu):
                self.assertGreaterEqual(mu, data_min[i])
                self.assertLessEqual(mu, data_max[i])

            weights += cluster.pi

        self.assertEqual(1.0, weights)

    def test_gaussian_shape(self):
        config.analysis['n_clusters'] = 2

        data_min = np.array([-3])
        data_max = np.array([3])
        data = np.linspace(data_min, data_max, 30)

        model = gmm.GaussianMixture(data)

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

        # fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # axes[0].plot(data, scores[:, 0])
        # axes[1].plot(data, scores[:, 1])
        # fig.show()

    def test_em(self):
        config.analysis['n_clusters'] = 3
        config.analysis['iterations'] = 20

        data = datasets.load_iris().data
        model = gmm.GaussianMixture(data)

        # Plots - 0 iteration
        plot_iteration(data, model)

        likelihoods = model.fit(data)

        # Plots - last iteration
        plot_iteration(data[:, :2], model)

        # Plots - likelihoods
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(likelihoods)
        ax.set_xlabel("iteration")
        ax.set_ylabel("log-likelihood")
        fig.show()


def plot_iteration(data_2D, model):
    colorset = ['blue', 'red', 'black']
    # Plots - 0 iteration
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for i, cluster in enumerate(model.clusters):
        eig_val, eig_vec = np.linalg.eigh(cluster.cov)
        order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[order], eig_vec[:, order]
        vx, vy = eig_vec[:, 0][0], eig_vec[:, 0][1]
        theta = np.arctan2(vy, vx)

        color = colors.to_rgba(colorset[i])

        ax.scatter(cluster.mu[0], cluster.mu[1], c=colorset[i], marker='+')

        for cov_factor in range(1, 4):
            ell = pat.Ellipse(xy=cluster.mu, width=np.sqrt(eig_val[0]) * cov_factor * 2,
                              height=np.sqrt(eig_val[1]) * cov_factor * 2, angle=np.degrees(theta), linewidth=2)
            ell.set_facecolor((color[0], color[1], color[2], 1.0 / (cov_factor * 4.5)))
            ax.add_artist(ell)

    ax.scatter(data_2D[:, 0], data_2D[:, 1])
    fig.show()
