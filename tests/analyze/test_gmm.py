import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.analyze import gmm


class GMMTest(unittest.TestCase):
    def test_gaussian(self):
        data = np.linspace(-5, 5, 100)
        model = gmm.GaussianMixture(1)
        model.mi = np.array([0])
        model.sigma = np.array([2])

        self.assertEqual(1, model.pi[0])

        probs = []
        for point in data:
            probs.append(model.gaussian(point, 0))

        fig, ax = plt.subplots()
        ax.plot(probs)
        fig.show()

    def test_model(self):
        data1 = np.array([np.concatenate((np.linspace(-3, -1, 20), np.linspace(3, 7, 40)))])
        data2 = np.array([np.linspace(-3, 7, 60)])
        data3 = np.array([np.linspace(0, 2, 60)])

        model = gmm.GaussianMixture(2)
        model.mi = np.array([-2, 5])
        model.sigma = np.array([1, 1])
        model.pi = gmm.GaussianMixture.normalize(np.array([1, 2]))

        log1 = model.score(data1)
        log2 = model.score(data2)
        log3 = model.score(data3)

        # plots
        data = np.linspace(-6, 10, 200)
        probs = np.zeros(data.size)
        for i in range(model.k):
            for j in range(data.size):
                probs[j] += model.gaussian(data[j], i) * model.pi[i]

        fig, ax = plt.subplots()
        ax.plot(data, probs)
        ax.plot(data1[0], np.zeros(data1.size), linestyle='', marker='o', label='data1')
        ax.plot(data2[0], np.ones(data2.size) * 0.01, linestyle='', marker='o', label='data2')
        ax.plot(data3[0], np.ones(data3.size) * 0.02, linestyle='', marker='o', label='data3')
        ax.legend(loc='upper left')
        fig.show()

        print(log1)
        print(log2)
        print(log3)
