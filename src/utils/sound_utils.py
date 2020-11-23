from src.conf import config
import numpy as np
from src.param.mfcc import Mfcc

class SoundUtils:
    def __init__(self):
        pass

    def normalize(self, audio):

        root_mean_square = np.sqrt(np.mean(np.power(audio, 2)))
        # signal normalization
        audio_normalize = audio / root_mean_square

        return audio_normalize


