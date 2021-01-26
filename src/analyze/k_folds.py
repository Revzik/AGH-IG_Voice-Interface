import random
import math
import numpy as np
import os
from src.conf import config


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
