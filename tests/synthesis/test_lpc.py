import unittest
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from src.synthesis import linpredcod
from src.analyze import window
from src.classes.containers import SoundWave
from src.analyze import mfcc


class LPCTest(unittest.TestCase):

    def test_lpc(self):

        wav_data, fs = sf.read("C:\\Users\\Kasia\\Desktop\\naprzod.wav")
        sound = SoundWave(wav_data, fs, "C:\\Users\\Kasia\\Desktop\\naprzod.wav")

        frame = window.window(sound, apply_window=False)

        signal_envelope = linpredcod.lpc(frame[50])
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        #fig.suptitle(str(f1) + " Hz")
        axes[0].plot(frame[50])
        axes[0].set_title("sygnał")
        axes[1].plot(signal_envelope)
        axes[1].set_title("obwiednia sygnału")
        axes[1].set_xlabel("czas [s]")
        fig.show()

        fs = 16000
        t1 = 0.025
        t = np.linspace(0, t1, int(t1 * fs))

        f1 = 200
        sin1 = np.sin(2 * np.pi * t * f1)
        sin_envelope = linpredcod.lpc(sin1)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # fig.suptitle(str(f1) + " Hz")
        axes[0].plot(sin1)
        axes[0].set_title("sygnał")
        axes[1].plot(sin_envelope)
        axes[1].set_title("obwiednia sygnału")
        axes[1].set_xlabel("czas [s]")
        fig.show()

        fft_sin = np.abs(mfcc.fft(frame[50]))
        print(fft_sin)
        print(len(fft_sin))
        sin_fft_lpc = linpredcod.lpc(fft_sin[0:512])
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # fig.suptitle(str(f1) + " Hz")
        axes[0].plot(fft_sin[0:512])
        axes[0].set_title("sygnał")
        axes[1].plot(sin_fft_lpc)
        axes[1].set_title("obwiednia sygnału")
        axes[1].set_xlabel("czas [s]")
        fig.show()

    # self.assertEqual(True, False)
    pass


if __name__ == '__main__':
    unittest.main()
