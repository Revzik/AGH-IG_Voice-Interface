from src.conf import config
import os
import glob
import soundfile as sf


class SoundLoader:
    def __init__(self):
        # tak się dobiera do ustalonych parametrów (wszystkie są w configurator.py)
        print(Configurator.parse_analysis()['sampling_frequency'])


    def sound_load_file():

        i = 0
        sound_list = [{}]

        print("Podaj ścieżkę pliku z nagraniami: ")
        paths = input()
        for filename in glob.glob(os.path.join(paths, '*.wav')):
            wav_data, fs = sf.read(filename)
            sound_list.append({"nazwa": os.listdir(paths)[i],"fs": fs, "wav": wav_data})
            i += 1
        del sound_list[0]

        return sound_list
