from src.conf import config
import os
import glob
import soundfile as sf
import math
import numpy as np
import scipy.signal as sig

class SoundLoader:
    def __init__(self):
        # tak się dobiera do ustalonych parametrów (wszystkie są w configurator.py)
        # print(config.analysis['sampling_frequency']) - zostawiam, żeby mieć wzór
        pass


    def sound_load_file():
        i = 0
        sound_list = []
        print("Podaj ścieżkę pliku z nagraniami: ")
        paths = input()
        for filename in glob.glob(os.path.join(paths, '*.wav')):
            wav_data, fs = sf.read(filename)
            sound_list.append({"nazwa": os.listdir(paths)[i][:-4], "fs": fs, "wav": wav_data})
            i += 1

        return sound_list

    def downsample(self, audio, fs):

        sampling_frequency = config.analysis['sampling_frequency']

        #down and up factor
        last_common_multipe = (fs * sampling_frequency) / math.gcd(fs, sampling_frequency)
        upsample_factor = int(last_common_multipe// fs)
        downsample_factor = int(last_common_multipe// sampling_frequency)

        #upsampling
        audio_up = np.zeros(len(audio) * upsample_factor)
        audio_up[upsample_factor//2::upsample_factor] = audio

        #filtering
        filtr = sig.firwin(301, cutoff=sampling_frequency / 2, fs=fs * upsample_factor)
        audio_up = downsample_factor * sig.filtfilt(filtr, 1, audio_up)

        #downsampling
        audio_down = audio_up[downsample_factor//2::downsample_factor]

        return audio_down

    def remove_dc_offset(self, sound_list):

        audio_without_dc = []
        for i in range(0, len(sound_list)):
            audio_without_dc.append(sound_list[i]['wav'] - np.mean(sound_list[i]['wav']))

        return audio_without_dc
