import numpy as np
import scipy.stats.mstats as sig

from src.classes.containers import Window, SoundWave
from src.conf import config
from src.param import parametrizer, mfcc


def normalize(sound_wave):

    sound_wave.samples = sound_wave.samples / rms(sound_wave.samples)

    return sound_wave


def rms(samples):
    return np.sqrt(np.mean(np.power(samples, 2)))


def preemphasis(sound_wave):

    pre = config.analysis['preemphasis']
    sound_wave.samples = np.append(sound_wave.samples[0], sound_wave.samples[1:] - pre * sound_wave.samples[:-1])

    return sound_wave


def detect_speech(sound_wave):
    """
    Cuts the noise out of the sound wave using VAD.
    The algorithm is based on paper:
    "A Simple But Efficient Real-Time Voice Activity Detection Algorithm"
    by M. H. Moattar and M. M. Homayoonpoor
    https://www.researchgate.net/publication/255667085_A_simple_but_efficient_real-time_voice_activity_detection_algorithm

    :param sound_wave: (SoundWave) sound wave to be evaluated
    :return: (SoundWave) input with just speech frames
    """

    init_e_thresh = config.analysis['energy_threshold']
    init_f_thresh = config.analysis['f_threshold']
    init_s_thresh = config.analysis['sf_threshold']

    e_thresh = init_e_thresh

    min_e = init_e_thresh
    min_f = init_f_thresh
    min_s = init_s_thresh

    frames = parametrizer.split(sound_wave, 10, 0)

    flags = [False] * len(frames)
    silence_count = 0

    for i, frame in enumerate(frames):
        fft_frame = mfcc.fft(Window(frame, sound_wave.fs))
        spectrum = fft_frame.spectrum()
        power_spectrum = fft_frame.power_spectrum()

        energy = np.sum(np.power(np.abs(frame), 2))
        f = np.argmax(spectrum) * fft_frame.df
        sf = 10 * np.log10(sig.gmean(power_spectrum) / np.mean(power_spectrum))

        if i < 30:
            if min_e > energy:
                min_e = energy
            if min_f > f:
                min_f = f
            if min_s > sf:
                min_s = sf

        e_thresh = init_e_thresh * np.log(min_e)
        f_thresh = init_f_thresh
        s_thresh = init_s_thresh

        counter = 0
        if energy - min_e >= e_thresh:
            counter += 1
        if f - min_f >= f_thresh:
            counter += 1
        if sf - min_s >= s_thresh:
            counter += 1

        if counter > 1:
            flags[i] = True
        else:
            silence_count += 1
            min_e = ((silence_count * min_e) + energy) / (silence_count + 1)

        e_thresh = init_e_thresh * np.log(min_e)

    # dodać 4 i 5

    new_frames = []
    for i, frame in enumerate(frames):
        if flags[i]:
            new_frames.append(frame)

    return SoundWave(np.array(new_frames), sound_wave.fs, sound_wave.phrase), flags
