from src.conf import config


class Parametrizer:
    def __init__(self):
        # tak się dobiera do ustalonych parametrów (wszystkie są w configurator.py)
        print(config.analysis['sampling_frequency'])