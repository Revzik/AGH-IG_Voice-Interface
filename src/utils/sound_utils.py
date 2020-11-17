from src.conf import config
import math

class SoundUtils:
    def __init__(self):
        pass

def normalziation(self,audio):

    square = 0
    mean =0.0
    root_mean_square = 0.0

    #calculate square
    for i in range(0, len(audio)):
        square += (audio[i]**2)

    #calculate mean
    mean = (square / (float)(len(audio)))

    #calculate root
    root_mean_square = math.sqrt(mean)

    # signal normalization
    audio_normalized = audio / root_mean_square

    return audio_normalized