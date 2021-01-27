import numpy as np

from src.analyze import gmm, mfcc
from src.utils import sound_loader
from src.conf import config
from sklearn.mixture import GaussianMixture


def create_models(train_set):
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
        cur_feat = mfcc.mfcc(wave)

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
    cm = np.zeros((len(test_set.keys()), len(test_set.keys())), dtype=np.int16)

    for actual_index, paths in enumerate(test_set.values()):
        for path in paths:
            assigned_index, _ = score_sample(path, models)
            cm[assigned_index, actual_index] += 1

    return cm


def score_sample(path, models):
    wave = sound_loader.load_sound_file(path)
    features = mfcc.mfcc(wave)

    scores = np.zeros(len(models.keys()))

    for i in range(scores.size):
        scores[i] = models[i].score(features)

    max_idx = np.argmax(scores)
    return max_idx, models.keys()[max_idx]
