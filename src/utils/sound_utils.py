from src.conf import config


class SoundUtils:
    def __init__(self):
        # tak się dobiera do ustalonych parametrów (wszystkie są w configurator.py)
        print(config.analysis['sampling_frequency'])