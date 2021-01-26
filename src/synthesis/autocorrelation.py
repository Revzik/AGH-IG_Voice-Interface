import numpy as np
from src.conf import config
from src.analyze import window

from src.classes.containers import SoundWave

def acorr(window, fs = config.analysis['sampling_frequency'], fmin = config.analysis['fmin'], fmax = config.analysis['fmax'], debug=False):

    # two the same signal (autocorrelation)
    x = window
    y = window

    # Size of the autocorrelation vector
    N = x.size + y.size - 1

    # Initialize vector
    ac = np.zeros(N)

   #add zeros
    begin = np.zeros(y.size - 1)
    x = np.append(begin, x)
    x = np.append(x, begin)

    for i in range(N):
        ac[i] += np.sum(x[i: i + y.size] * y)  # autocorrelation

    #find max in autocorrelation vectro
    # + value
    start = len(y)
    ac = ac[start:]

    # find max value
    max_ac_index = np.argmax(ac)
    delays = np.arange(1, ac.size) / fs
    tau = delays[max_ac_index]

    if (1/tau >= fmin) and (1/tau <= fmax):
        tonality = True
    else:
        tonality = False

    if not debug:
        return tonality
    else:
        return tonality, delays, ac

#loaduje plik
from src.utils import sound_loader
#load words "naprzód"
fs = 8000
time = 0.1
wav_data = np.arange(0, time, 1 / fs)
sound = SoundWave(wav_data, fs, "naprzod")

# biore ramke
ramka = window.window(sound)
print(ramka)
tonal1, delays1, ac1 = acorr(ramka[5], debug=True)
print(tonal1)

#fs = 8000
#t1 = 0.025
#t = np.linspace(0, t1, int(t1 * fs))
#f1 = 200
#f2 = 8000
#sin1 = np.sin(2 * np.pi * t * f1)
#sin2 = np.sin(2 * np.pi * t * f2)

#tonal1, delays1, ac1 = acorr(sin1, fs=fs, fmin=200, fmax=400, debug=True)
#tonal2, delays2, ac2 = acorr(sin2, fs=fs, fmin=200, fmax=400, debug=True)
#print(tonal1)
#print(tonal2)