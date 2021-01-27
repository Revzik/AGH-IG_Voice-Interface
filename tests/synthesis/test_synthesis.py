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

        ramka = window.window(sound, apply_window=False)
        s = []
        e_duze = []
        for i in range(0, len(ramka)-1):
            tonal3, delays3, ac3, t = autocorrelation.acorr(ramka[i], debug=True)
            lpc_coeff, enhancement = linpredcod.lpc(ramka[i])
            x = synthesis.excitement(tonal3, t, lpc_coeff[::-1], enhancement)
            s = np.append(s, x)
            e_duze = np.append(e_duze, enhancement)


        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # fig.suptitle(str(f1) + " Hz")
        axes[0].plot(wav_data)
        axes[0].set_title("sygnał")
        axes[1].plot(e_duze)
        axes[1].set_title("obwiednia sygnału")
        axes[1].set_xlabel("próbki [-]")
        fig.show()
        print(s)
        # fft_sig = np.abs(mfcc.fft(frame[50]))
        # print(fft_sig)
        # print(len(fft_sig))
        # lpc_coeff_fft, en_fft, f_imp_res_fft = linpredcod.lpc(fft_sig[0:512])

    pass


if __name__ == '__main__':
    unittest.main()
