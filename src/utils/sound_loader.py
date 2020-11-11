from src.conf import config
import os
import glob
from src.conf.configurator import Configurator
import soundfile as sf


class SoundLoader:
    def __init__(self):
        # tak się dobiera do ustalonych parametrów (wszystkie są w configurator.py)
        print(Configurator.parse_analysis()['sampling_frequency'])
        print(Configurator.parse_analysis()['path_waves'])
        #print(config.analysis['path_waves'])

    def sound_load_file():

        file_path = str(config.analysis['path_waves'])[1:len(str(config.analysis['path_waves'])) - 1] #przy konwersji
        i = 0
        sound_list = []

        for filename in glob.glob(os.path.join(file_path, '*.wav')):
        #for filename in glob.glob(os.path.join(folder_path, '*.wav')):
            wav_data, fs = sf.read(filename)
            sound_list.append(wav_data)
            i += 1
        return sound_list
