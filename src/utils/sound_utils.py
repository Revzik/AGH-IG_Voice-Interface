from src.conf import config
import numpy as np

class SoundUtils:
    def __init__(self):
        pass

    def normalze(self, audio):

        root_mean_square = np.sqrt(np.mean(np.power(audio, 2)))
        # signal normalization
        audio_normalize = audio / root_mean_square

        return audio_normalize


