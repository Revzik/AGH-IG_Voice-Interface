import os
import glob
import soundfile as sf

from src.classes.containers import SoundWave


def load_sound_file(path):
    sound_list = []

    if os.path.isdir(path):
        for i, filename in enumerate(glob.glob(os.path.join(path, '*.wav'))):
            wav_data, fs = sf.read(filename)
            if len(wav_data.shape) > 1:
                wav_data = wav_data[0, :]
            sound_list.append(SoundWave(wav_data, fs, os.listdir(path)[i][:-4]))
    else:
        wav_data, fs = sf.read(path)
        if len(wav_data.shape) > 1:
            wav_data = wav_data[0, :]
        sound_list.append(SoundWave(wav_data, fs, path[:-4]))

    return sound_list
