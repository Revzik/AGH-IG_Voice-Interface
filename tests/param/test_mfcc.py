import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.param.mfcc import first_power_of_2, fft
from src.classes.containers import Window


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

        self.assertEqual(1024, y1.get_length())
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

        self.assertEqual(2048, y2.get_length())
        self.assertGreater(500, y2[205])
        self.assertFalse(any(np.abs(f) > 50 for f in np.concatenate((y2[0:198], y2[212:1836], y2[1850:2048]))))

        # transform 3: Kronecker delta, number of samples is not a power of 2
        n3 = 101
        fs3 = 101
        x3 = Window(np.zeros(n3), fs3)
        x3[50] = 1
        y3 = fft(x3)

        self.assertEqual(128, y3.get_length())
        self.assertFalse(any(np.abs(f) > 1.01 or np.abs(f) < 0.99 for f in y3))

        # transform 4: sawtooth wave, number of samples is a power of 2
        fs4 = 128
        l4 = 2
        n4 = int(fs4 * l4)
        t4 = np.linspace(0, l4, n4, False)
        x4 = Window(2 * (t4/0.5 - np.floor(0.5 + t4/0.5)), fs4)
        y4 = fft(x4)

        self.assertEqual(256, y4.get_length())
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
        # fig = plt.figure(2, figsize=(8, 6))
        # ax = plt.subplot(2, 1, 1)
        # ax.plot(t4, x4.samples)
        # ax = plt.subplot(2, 1, 2)
        # ax.plot(f4, s4)
        # fig.show()


if __name__ == '__main__':
    unittest.main()
