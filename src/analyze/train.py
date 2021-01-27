import numpy as np
import os
import pickle

from src.analyze import gmm, mfcc
from src.utils import sound_loader
from src.conf import config
from sklearn.mixture import GaussianMixture


def create_models(train_set):
    print("Creating models...")
    models = {}

    i = 1
    for cls, paths in train_set.items():
        print("Creating model for {} ({} / {})".format(cls, i, len(train_set.keys())))
        models[cls] = create_model(paths)
        i += 1

    return models


def create_model(paths):
    features = None
    n_clusters = config.analysis['n_clusters']
    iters = config.analysis['iterations']

    for i, path in enumerate(paths):
        print("Computing mfcc for {} ({} / {})".format(path, i + 1, len(paths)))
        wave = sound_loader.load_sound_file(path)
        cur_feat = mfcc.mfcc(wave, delta_deltas=True)

        if features is None:
            features = cur_feat
        else:
            features = np.vstack((features, cur_feat))

    # it no worke :CCC
    # model = gmm.GaussianMixture(features)
    print("Computing model")
    model = GaussianMixture(n_components=n_clusters, max_iter=iters)
    model.fit(features)

    print("Complete!")
    print()

    return model


def score_samples(test_set, models):
    print("Scoring samples...")

    cm = np.zeros((len(test_set.keys()), len(test_set.keys())), dtype=np.int16)

    actual_index = 0
    for cls, paths in test_set.items():
        print("Scoring for class {}".format(cls))
        for path in paths:
            assigned_index, assigned_label = score_sample(path, models)
            print("File: {} , assigned label: {}".format(path, assigned_label))
            cm[assigned_index, actual_index] += 1
        actual_index += 1

    return cm


def score_sample(path, models):
    wave = sound_loader.load_sound_file(path)
    features = mfcc.mfcc(wave)

    scores = np.zeros(len(models.keys()))

    for i, cls in enumerate(models.keys()):
        scores[i] = models[cls].score(features)

    max_idx = int(np.argmax(scores))
    return max_idx, list(models)[max_idx]


if __name__ == "__main__":
    # load words
    data = {}

    for word, path in config.folders['known'].items():
        data[word] = []
        data[word] = [os.path.join(path, p) for p in os.listdir(path)]

    models = create_models(data)

    with open("../../tmp/models.p", "wb") as f:
        pickle.dump(models, f)
