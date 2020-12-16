import numpy as np


class GaussianMixture:
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
        :param data: (List of Lists of CepstralFrame) CepstralFrames for each sample
        """

        self.k = k
        if data is None:
            self.mi = 10 * np.random.randn(k)
            self.sigma = 10 * np.random.randn(k)
            self.pi = GaussianMixture.normalize(np.random.randn(k))
        else:
            data = concatenate(data)
            low = np.amin(data)
            high = np.amax(data)

            self.mi = (high - low) * np.random.rand(k) + low
            self.sigma = (high - low) * np.random.randn(k) / k
            self.pi = GaussianMixture.normalize(np.random.randn(k))

    @staticmethod
    def normalize(pi):
        """
        Normalizes the array specified so sum if its components is 1. Used to compute weights for GMM.

        :param pi: (1-D ndarray) input array
        :return: (1-D ndarray) pi with the sum of components equal to 1
        """

        return pi / np.sum(pi)

    def fit(self, data):
        """
        Trains the model with data using EM algorithm.

        :param data: (2-D ndarray) MFCC features for the model
        """
        pass

    def score(self, data):
        """
        Computes the log-likelihood of input data for trained model.

        :param data: (2-D ndarray) MFCC features of scored sample
        :return: (float) log-likelihood of the data for specific model
        """
        # TODO: Investigate the function - log likelihood is not correct
        log_likelihood = 0

        for i in range(data.shape[0]):

            likelihood = 0
            for j in range(self.k):
                likelihood += self.pi[j] * self.gaussian(data[i, :], j)

            log_likelihood += np.log(likelihood)

        return log_likelihood

    def gaussian(self, data, i):
        """
        Computes the probability of data to be from a specific gaussian. The points in data are 1-D

        :param data: (1-D ndarray) sample to be evaluated
        :param i: (int) index of the gaussian (0 <= i < k)
        :return: likelihood of x to be from gaussian i
        """

        return 1 / np.sqrt(2 * np.pi * self.sigma[i] ** 2) * \
               np.exp(np.sum((data - self.mi[i]) ** 2) / (-2 * self.sigma[i]))


def concatenate(data):
    """
    Concatenates data specified in the parameter to fit the input shape of GaussianMixture.

    :param data: (List of Lists of CepstralFrame) MFCC features for each word. Each list entry is MFCC of another sample
    :return: (2-D ndarray) concatenated data:
        rows - MFCC frames
        columns - features
    """

    frame_count = 0
    for sample in data:
        frame_count += len(sample)

    con_data = np.empty((frame_count, data[0].length()))
    i = 0
    for sample in data:
        for frame in sample:
            con_data[i] = frame.samples
            i += 1

    return con_data
