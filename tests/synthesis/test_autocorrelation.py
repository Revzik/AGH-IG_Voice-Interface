import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.synthesis import autocorrelation


class SynthesisTest(unittest.TestCase):
    def test_acorr(self):
        fs = 16000
        t1 = 0.025
        t = np.linspace(0, t1, int(t1 * fs))

        f1 = 200
        f2 = 800
        sin1 = np.sin(2 * np.pi * t * f1)
        sin2 = np.sin(2 * np.pi * t * f2)

        tonal1, delays1, ac1 = autocorrelation.acorr(sin1, fs=fs, fmin=200, fmax=400, debug=True)
        tonal2, delays2, ac2 = autocorrelation.acorr(sin2, fs=fs, fmin=200, fmax=400, debug=True)

        self.assertTrue(tonal1)
        self.assertFalse(tonal2)

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        fig.suptitle(str(f1) + " Hz")
        axes[0].plot(t, sin1)
        axes[0].set_title("sygnał")
        axes[1].plot(delays1, ac1)
        axes[1].set_title("autokorelacja")
        axes[1].set_xlabel("czas [s]")
        fig.show()

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        fig.suptitle(str(f2) + " Hz")
        axes[0].plot(t, sin2)
        axes[0].set_title("sygnał")
        axes[1].plot(delays2, ac2)
        axes[1].set_title("autokorelacja")
        axes[1].set_xlabel("czas [s]")
        fig.show()
    pass