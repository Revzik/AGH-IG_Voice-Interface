import unittest
from src.param.mfcc import Mfcc
import numpy as np
import matplotlib.pyplot as plt


class MfccTest(unittest.TestCase):
    def test_first_power_of_2(self):
        x1 = 4
        x2 = 11
        x3 = 1
        x4 = 0

        self.assertEqual(4, Mfcc.first_power_of_2(x1))
        self.assertEqual(16, Mfcc.first_power_of_2(x2))
        self.assertEqual(1, Mfcc.first_power_of_2(x3))
        self.assertIsNone(Mfcc.first_power_of_2(x4))

    def test_fft(self):
        # transform 1: 20 Hz sine, number of samples is a power of 2
        fs1 = 1024
        l1 = 1
        n1 = int(fs1 * l1)
        t1 = np.linspace(0, l1, n1, False)
        x1 = np.sin(2 * 20 * np.pi * t1)
        y1 = Mfcc.fft(x1)

        self.assertEqual(1024, y1.size)
        self.assertLess(500, np.abs(y1[20]))
        self.assertGreater(0.01, np.abs(np.angle(y1[20]) + np.pi/2))
        self.assertFalse(any(np.abs(f) > 1 for f in np.concatenate((y1[0:19], y1[21:1003], y1[1005:1024]))))

        # transform 2: 100 Hz cosine, number of samples is not a power of 2
        fs2 = 1000
        l2 = 1.5
        n2 = int(fs2 * l2)
        t2 = np.linspace(0, l2, n2, False)
        x2 = np.sin(2 * 100 * np.pi * t2)
        y2 = Mfcc.fft(x2)

        self.assertEqual(2048, y2.size)
        self.assertGreater(500, np.abs(y1[205]))
        self.assertFalse(any(np.abs(f) > 50 for f in np.concatenate((y2[0:198], y2[212:1836], y2[1850:2048]))))

        # transform 3: Kronecker delta, number of samples is not a power of 2
        n3 = 101
        x3 = np.zeros(n3)
        x3[50] = 1
        y3 = Mfcc.fft(x3)

        self.assertEqual(128, y3.size)
        self.assertFalse(any(np.abs(f) > 1.01 or np.abs(f) < 0.99 for f in y3))

        # transform 4: sawtooth wave, number of samples is a power of 2
        fs4 = 128
        l4 = 2
        n4 = int(fs4 * l4)
        t4 = np.linspace(0, l4, n4, False)
        x4 = 2 * (t4/0.5 - np.floor(0.5 + t4/0.5))
        y4 = Mfcc.fft(x4)

        self.assertEqual(256, y4.size)
        self.assertLess(80, np.abs(y4[4]))
        self.assertLess(np.abs(y4[8]), np.abs(y4[4]))
        self.assertLess(np.abs(y4[12]), np.abs(y4[8]))
        self.assertLess(np.abs(y4[16]), np.abs(y4[12]))
        self.assertLess(np.abs(y4[20]), np.abs(y4[16]))

        # Just in case if spectrum needs to be displayed
        # f1 = np.linspace(0, fs1, y1.size, False)
        # fig = plt.figure(1, figsize=(8, 6))
        # ax = plt.subplot(2, 1, 1)
        # ax.plot(t1, x1)
        # ax = plt.subplot(2, 1, 2)
        # ax.plot(f1[0:int(f1.size/2)], np.abs(y1[0:int(y1.size/2)]))
        # fig.show()
        #
        # f2 = np.linspace(0, fs2, y2.size, False)
        # fig = plt.figure(2, figsize=(8, 6))
        # ax = plt.subplot(2, 1, 1)
        # ax.plot(t2, x2)
        # ax = plt.subplot(2, 1, 2)
        # ax.plot(f2[0:int(f2.size/2)], np.abs(y2[0:int(y2.size/2)]))
        # fig.show()
        #
        # fs3 = 101
        # f3 = np.linspace(0, fs3, y3.size, False)
        # fig = plt.figure(3, figsize=(8, 6))
        # ax = plt.subplot(2, 1, 1)
        # ax.plot(x3)
        # ax = plt.subplot(2, 1, 2)
        # ax.plot(f3[0:int(f3.size/2)], np.abs(y3[0:int(y3.size/2)]))
        # fig.show()
        #
        # f4 = np.linspace(0, fs4, y4.size, False)
        # fig = plt.figure(2, figsize=(8, 6))
        # ax = plt.subplot(2, 1, 1)
        # ax.plot(t4, x4)
        # ax = plt.subplot(2, 1, 2)
        # ax.plot(f4[0:int(f4.size/2)], np.abs(y4[0:int(y4.size/2)]))
        # fig.show()


if __name__ == '__main__':
    unittest.main()
