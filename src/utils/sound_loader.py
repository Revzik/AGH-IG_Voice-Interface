import os
import glob
import soundfile as sf
from src.classes.containers import SoundWave


def load_sound_file():
    i = 0
    sound_list = []
    print("Please enter the path for the folder with waves: ")
    paths = input()
    for filename in glob.glob(os.path.join(paths, '*.wav')):
        wav_data, fs = sf.read(filename)
        sound_list.append(SoundWave(wav_data, fs, os.listdir(paths)[i][:-4]))
        i += 1

    return sound_list


def downsample(sound_list):
    for i in range(len(sound_list)):
        sound_list[i].downsample()

    return sound_list


def remove_dc_offset(sound_list):
    for i in range(len(sound_list)):
        sound_list[i].remove_dc()

    return sound_list
