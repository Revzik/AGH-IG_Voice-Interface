import os
import glob
import soundfile as sf

from src.classes.containers import SoundWave
from src.conf import config


def load_sound_file():
    sound_list = []
    print("Please enter the path for the folder with waves: ")
    paths = input()
    for i, filename in enumerate(glob.glob(os.path.join(paths, '*.wav'))):
        wav_data, fs = sf.read(filename)
        sound_list.append(SoundWave(wav_data, fs, os.listdir(paths)[i][:-4]))

    return sound_list