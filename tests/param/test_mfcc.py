import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.param.mfcc import *
from src.classes.containers import Window
from src.classes.containers import MelFrame
from src.param import mfcc


class MfccTest(unittest.TestCase):
    def test_first_power_of_2(self):
        x1 = 4
        x2 = 11
        x3 = 1
        x4 = 0

        self.assertEqual(4, first_power_of_2(x1))
        self.assertEqual(16, first_power_of_2(x2))
        self.assertEqual(1, first_power_of_2(x3))
        self.assertIsNone(first_power_of_2(x4))

    def test_fft(self):
        # transform 1: 20 Hz sine, number of samples is a power of 2
        fs1 = 1024
        l1 = 1
        n1 = int(fs1 * l1)
        t1 = np.linspace(0, l1, n1, False)
        x1 = Window(np.sin(2 * 20 * np.pi * t1), fs1)
        y1 = fft(x1)

        self.assertEqual(1024, y1.length())
        self.assertLess(500, y1.spectrum()[20])
        self.assertAlmostEqual(-np.pi/2, np.angle(y1[20]), 2)
        self.assertFalse(any(np.abs(f) > 1 for f in np.concatenate((y1[0:19], y1[21:1003], y1[1005:1024]))))
        self.assertEqual(1, y1.df)

        # transform 2: 100 Hz cosine, number of samples is not a power of 2
        fs2 = 1000
        l2 = 1.5
        n2 = int(fs2 * l2)
        t2 = np.linspace(0, l2, n2, False)
        x2 = Window(np.sin(2 * 100 * np.pi * t2), fs2)
        y2 = fft(x2)

        self.assertEqual(2048, y2.length())
        self.assertGreater(500, y2[205])
        self.assertFalse(any(np.abs(f) > 50 for f in np.concatenate((y2[0:198], y2[212:1836], y2[1850:2048]))))

        # transform 3: Kronecker delta, number of samples is not a power of 2
        n3 = 101
        fs3 = 101
        x3 = Window(np.zeros(n3), fs3)
        x3[50] = 1
        y3 = fft(x3)

        self.assertEqual(128, y3.length())
        self.assertFalse(any(np.abs(f) > 1.01 or np.abs(f) < 0.99 for f in y3))

        # transform 4: sawtooth wave, number of samples is a power of 2
        fs4 = 128
        l4 = 2
        n4 = int(fs4 * l4)
        t4 = np.linspace(0, l4, n4, False)
        x4 = Window(2 * (t4/0.5 - np.floor(0.5 + t4/0.5)), fs4)
        y4 = fft(x4)

        self.assertEqual(256, y4.length())
        self.assertLess(80, y4.spectrum()[4])
        self.assertLess(y4.spectrum()[8], y4.spectrum()[4])
        self.assertLess(y4.spectrum()[12], y4.spectrum()[8])
        self.assertLess(y4.spectrum()[16], y4.spectrum()[12])
        self.assertLess(y4.spectrum()[20], y4.spectrum()[16])

        # Just in case if spectrum needs to be displayed
        # s1 = y1.spectrum()
        # f1 = np.arange(0, fs1 / 2, y1.df)
        # fig = plt.figure(1, figsize=(8, 6))
        # ax = plt.subplot(2, 1, 1)
        # ax.plot(t1, x1.samples)
        # ax = plt.subplot(2, 1, 2)
        # ax.plot(f1, s1)
        # fig.show()
        #
        # s2 = y2.spectrum()
        # f2 = np.arange(0, fs2 / 2, y2.df)
        # fig = plt.figure(2, figsize=(8, 6))
        # ax = plt.subplot(2, 1, 1)
        # ax.plot(t2, x2.samples)
        # ax = plt.subplot(2, 1, 2)
        # ax.plot(f2, s2)
        # fig.show()
        #
        # s3 = y3.spectrum()
        # f3 = np.arange(0, fs3 / 2, y3.df)
        # fig = plt.figure(3, figsize=(8, 6))
        # ax = plt.subplot(2, 1, 1)
        # ax.plot(x3.samples)
        # ax = plt.subplot(2, 1, 2)
        # ax.plot(f3, s3)
        # fig.show()
        #
        # s4 = y4.spectrum()
        # f4 = np.arange(0, fs4 / 2, y4.df)
        # fig = plt.figure(4, figsize=(8, 6))
        # ax = plt.subplot(2, 1, 1)
        # ax.plot(t4, x4.samples)
        # ax = plt.subplot(2, 1, 2)
        # ax.plot(f4, s4)
        # fig.show()

    def test_frequency_to_mel(self):
        f1 = 1000
        self.assertAlmostEqual(1000, get_mel(f1), 0)

        f2 = 4000
        self.assertAlmostEqual(2146, get_mel(f2), 0)

    def test_mel_to_frequency(self):
        m1 = 1000
        self.assertAlmostEqual(1000, get_frequency(m1), 0)

        m2 = 2000
        self.assertAlmostEqual(3429, get_frequency(m2), 0)

    def test_mel_frequencies(self):
        lf1 = 300
        hf1 = 4000
        n1 = 14
        b1 = 1000
        fs1 = 8000

        freqs = get_filter_frequencies(lf1, hf1, n1)
        self.assertEqual(16, freqs.size)

        bins = get_filter_bins(freqs, b1, fs1)
        self.assertEqual(16, bins.size)

        # # Plots just in case
        # lf_p = 300
        # tf_p = 4000
        # fs_p = 8000
        # n_p = 10
        # b_p = 50
        #
        # filter_freq_p = get_filter_frequencies(lf_p, tf_p, n_p)
        # filter_bins_p = get_filter_bins(filter_freq_p, b_p, fs_p)
        #
        # # Mel filters
        # bottom_freq_p = filter_freq_p[:-2]
        # top_freq_p = filter_freq_p[2:]
        # filter_freq_p = filter_freq_p[1:-1]
        # filter_bins_p = filter_bins_p[1:-1]
        #
        # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        # for i in range(filter_freq_p.size):
        #     ax.plot([bottom_freq_p[i], filter_freq_p[i], top_freq_p[i]], [0, 1, 0])
        # ax.set_title('Mel filters')
        # fig.show()
        #
        # # Filter frequencies and corresponding frequency bins
        # fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # axes[0].stem(filter_freq_p, np.ones(filter_freq_p.size))
        # axes[0].set_xlim([0, tf_p])
        # axes[0].set_title('Central mel filter frequencies')
        # axes[1].stem(filter_bins_p, np.ones(filter_bins_p.size))
        # axes[1].set_xlim([0, b_p // 2])
        # axes[1].set_title('Central mel filter frequencies in corresponding frequency bins')
        # fig.show()

    def test_mel_filterbank(self):
        lf1 = 300
        hf1 = 4000
        np1 = 20
        fs1 = 8000
        l1 = 0.1
        n1 = int(fs1 * l1)
        t1 = np.linspace(0, l1, n1, False)
        x1 = Window(np.sin(2 * 1000 * np.pi * t1), fs1)
        y1 = fft(x1)

        mf1 = apply_mel_filterbank(y1, lf1, hf1, np1)
        self.assertTrue([mf1[6] >= m for m in mf1.samples])

        # Plots just in case
        # Applying filterbank
        # lf_p = 300
        # hf_p = 4000
        # nf_p = 20
        # fs_p = 8000
        # l_p = 0.1
        # n_p = int(fs_p * l_p)
        # t_p = np.linspace(0, l_p, n_p, False)
        # x_p = Window(np.sin(2 * 1000 * np.pi * t_p), fs_p)
        # y_p = fft(x_p)
        #
        # mf_p = get_filter_frequencies(lf_p, hf_p, nf_p)
        # m_p = apply_mel_filterbank(y_p, lf_p, hf_p, nf_p)
        #
        # fig, axes = plt.subplots(3, 1, figsize=(8, 9))
        # f_p = np.arange(0, fs_p, y_p.df)
        # axes[0].plot(f_p, y_p.spectrum())
        # axes[0].set_xlim([0, f_p[f_p.size // 2]])
        # axes[0].set_title('Spectrum')
        # axes[1].plot(mf_p[1:-1], m_p.samples)
        # axes[1].set_xlim([0, f_p[f_p.size // 2]])
        # axes[1].set_title('Mel coefficients in frequency domain')
        # axes[2].plot(np.arange(1, m_p.n_filters + 1), m_p.samples)
        # axes[2].set_xlim([1, m_p.n_filters])
        # axes[2].set_title('Mel coefficients')
        # fig.show()


    def test_logarithm(self):

        mel_filter = MelFrame(np.array([10,100,1000]))
        mel_filter_log = mfcc.logarithm(mel_filter).samples

        self.assertEqual(1, mel_filter_log[0])
        self.assertEqual(2, mel_filter_log[1])
        self.assertEqual(3, mel_filter_log[2])

if __name__ == '__main__':
    unittest.main()
