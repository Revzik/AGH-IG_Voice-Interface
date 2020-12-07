from src.utils import sound_utils
from src.analyze import mfcc


def parametrize(sound_wave):
    """
    Prepares a raw sound wave to parametrize and computes its cepstrum

    :param sound_wave: (SoundWave) input sound wave
    :return: (List of CepstralFrame) cepstrum, (string) phrase
    """

    sound_wave = sound_utils.downsample(sound_wave)
    sound_wave = sound_utils.remove_dc_offset(sound_wave)
    sound_wave = sound_utils.preemphasis(sound_wave)
    sound_wave = sound_utils.normalize(sound_wave)

    return mfcc.mfcc(sound_wave)
