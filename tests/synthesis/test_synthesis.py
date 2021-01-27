import unittest
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from src.synthesis import linpredcod
from src.analyze import window
from src.classes.containers import SoundWave
from src.synthesis import autocorrelation
from src.synthesis import synthesis
from src.analyze import mfcc

class TestSynthesis(unittest.TestCase):
    def test_synthesis(self):
        wav_data, fs = sf.read("C:\\Users\\Kasia\\Desktop\\naprzod.wav")
        sound = SoundWave(wav_data, fs, "C:\\Users\\Kasia\\Desktop\\naprzod.wav")

        frame = window.window(sound, apply_window=False)
        lpc_coeff, enhancement, filter_imp_response = linpredcod.lpc(frame[50])

        ramka = window.window(sound, apply_window=False)

        tonal3, delays3, ac3, t = autocorrelation.acorr(ramka[15], debug=True)

        s = synthesis.excitement(tonal3, 5, lpc_coeff, enhancement)

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # fig.suptitle(str(f1) + " Hz")
        axes[0].plot(frame[50])
        axes[0].set_title("sygnał")
        axes[1].plot(s)
        axes[1].set_title("synteza sygnału")
        axes[1].set_xlabel("czas [s]")
        fig.show()

        # fft_sig = np.abs(mfcc.fft(frame[50]))
        # print(fft_sig)
        # print(len(fft_sig))
        # lpc_coeff_fft, en_fft, f_imp_res_fft = linpredcod.lpc(fft_sig[0:512])

    pass


if __name__ == '__main__':
    unittest.main()
