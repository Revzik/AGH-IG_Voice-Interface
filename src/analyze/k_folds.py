import random
import math
import numpy as np
import os
import pickle

from src.conf import config
from src.analyze import train
from src.utils import plots


def configure_k_folds():
    # numbers of splits/groups
    group_count = config.analysis['number_of_groups']

    # load words
    data = {}

    for word, path in config.folders['known'].items():
        data[word] = []
        data[word] = [os.path.join(path, p) for p in os.listdir(path)]
        random.shuffle(data[word])

    # numbers of fold, the same len of each words
    word_count = len(data[list(data.keys())[0]])

    # group size initialization
    sizes = np.zeros(group_count, dtype=np.int16)

    # each group size
    current_group_count = group_count
    for i in range(group_count):
        sizes[i] = math.ceil(word_count / current_group_count)
        word_count -= sizes[i]
        current_group_count -= 1

    # dividing samples into groups
    k_folds_groups = []
    a = 0

    for i in range(group_count):
        group = {}
        for key, value in data.items():
            group[key] = value[a:a + sizes[i]]
        a = a + sizes[i]
        k_folds_groups.append(group)

    return k_folds_groups


def initialize_dict(classes):
    target_set = {}

    for cls in classes:
        target_set[cls] = []

    return target_set


def append_group(target_set, group):
    for cls, paths in group.items():
        for path in paths:
            target_set[cls].append(path)


def do_k_folds():
    groups = configure_k_folds()
    classes = list(groups[0])

    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.int16)

    for i in range(len(groups)):
        print("Processing fold {} / {}".format(i + 1, len(groups)))
        train_set = initialize_dict(classes)
        test_set = initialize_dict(classes)

        for j in range(len(groups)):
            if i == j:
                append_group(test_set, groups[j])
            else:
                append_group(train_set, groups[j])

        models = train.create_models(train_set)
        confusion_matrix = confusion_matrix + train.score_samples(test_set, models)

    plots.plot_confusion_matrix(confusion_matrix, classes, normalize=False)
    plots.plot_confusion_matrix(confusion_matrix, classes, normalize=True)

    with open("../../tmp/k_folds_cm.p", "wb") as f:
        pickle.dump(confusion_matrix, f)
