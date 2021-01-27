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

        lpc_coeff, enhancement, filter_imp_response = linpredcod.lpc(frame[50])
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        #fig.suptitle(str(f1) + " Hz")
        axes[0].plot(frame[50])
        axes[0].set_title("sygnał")
        axes[1].plot(lpc_coeff)
        axes[1].set_title("obwiednia sygnału")
        axes[1].set_xlabel("czas [s]")
        fig.show()

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # fig.suptitle(str(f1) + " Hz")
        axes[0].plot(enhancement)
        axes[0].set_title("Wzmocnienie dla LPC w czasie")
        axes[1].plot(np.abs(filter_imp_response[1]))
        axes[1].set_title("Odp. imp. dla LPC w czasie")
        axes[1].set_xlabel("czas [s]")
        fig.show()

        fs = 16000
        t1 = 0.025
        t = np.linspace(0, t1, int(t1 * fs))

        f1 = 200
        sin1 = np.sin(2 * np.pi * t * f1)
        lpc_coeff_sin, en_sin, f_imp_res_sin = linpredcod.lpc(sin1)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # fig.suptitle(str(f1) + " Hz")
        axes[0].plot(sin1)
        axes[0].set_title("sygnał sinusoidalny")
        axes[1].plot(lpc_coeff_sin)
        axes[1].set_title("obwiednia sygnału")
        axes[1].set_xlabel("czas [s]")
        fig.show()

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # fig.suptitle(str(f1) + " Hz")
        axes[0].plot(en_sin)
        axes[0].set_title("wzmocnienie dla s. sin")
        axes[1].plot(f_imp_res_sin[1])
        axes[1].set_title("odp. imp. dla s. sin")
        axes[1].set_xlabel("czas [s]")
        fig.show()

        fft_sig = np.abs(mfcc.fft(frame[50]))
        print(fft_sig)
        print(len(fft_sig))
        lpc_coeff_fft, en_fft, f_imp_res_fft = linpredcod.lpc(fft_sig[0:512])
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # fig.suptitle(str(f1) + " Hz")
        axes[0].plot(fft_sig[0:512])
        axes[0].set_title("Widmo sygnału")
        axes[1].plot(lpc_coeff_fft)
        axes[1].set_title("Obwiednia sygnału")
        axes[1].set_xlabel("czas [s]")
        fig.show()

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # fig.suptitle(str(f1) + " Hz")
        axes[0].plot(en_fft)
        axes[0].set_title("wzmocnienie filtru dla lpc widmowego")
        axes[1].plot(f_imp_res_fft[1])
        axes[1].set_title("Odpowiedź impulsowa filtru LPC")
        axes[1].set_xlabel("czas [s]")
        fig.show()

    # self.assertEqual(True, False)
    pass


if __name__ == '__main__':
    unittest.main()
