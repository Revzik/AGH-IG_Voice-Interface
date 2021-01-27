import pickle
import os

from src.conf import config
from src.utils import plots
from src.analyze import train


MODELS_PATH = "tmp/models.p"
WAV_PATH = "../BAZA/ZNANI/odstaw/odstaw_4.wav"


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def full_set():
    # load words
    data = {}

    for word, path in config.folders['unknown'].items():
        data[word] = []
        data[word] = [os.path.join(path, p) for p in os.listdir(path)]

    models = load(MODELS_PATH)

    cm = train.score_samples(data, models)

    plots.plot_confusion_matrix(cm, list(data), normalize=True)


def single(path):
    models = load(MODELS_PATH)

    _, label = train.score_sample(path, models)


if __name__ == "__main__":
    full_set()
    single(WAV_PATH)
