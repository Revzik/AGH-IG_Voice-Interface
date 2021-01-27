import unittest
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt

from src.utils.sound_loader import load_sound_file
from src.analyze.mfcc import mfcc
from src.analyze import k_folds
from src.analyze import train


class KFoldsTets(unittest.TestCase):
    def test_configure_k_folds(self):
        groups = k_folds.configure_k_folds()
        print(groups)

    def test_do_k_folds(self):
        k_folds.do_k_folds()
        # with open("../../tmp/models.p", "rb") as f:
        #     models = pickle.load(f)
        #
        # wave = load_sound_file("D:\Studia\II\IG\BAZA\ZNANI\odstaw\odstaw_2.wav")
        # features = mfcc(wave)
        #
        # scores = np.zeros(len(models.keys()))
        # for i, cls in enumerate(models.keys()):
        #     scores[i] = models[cls].score(features)
        #
        # max_idx, max_label = train.score_sample("D:\Studia\II\IG\BAZA\ZNANI\odstaw\odstaw_2.wav", models)
        #
        # print(scores)

    def test_plot_matrix(self):
        normalize = False

        with open("../../tmp/k_folds_cm.p", "rb") as f:
            cm = pickle.load(f)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]

        classes = ['ciszej', 'do_przodu', 'do_tylu', 'glosniej', 'igla', 'losuj',
                   'odstaw', 'postaw', 'przewin', 'start', 'stop', 'wybierz']
        size = cm.shape[0]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(cm, cmap=plt.get_cmap('BuGn'))
        ax.set_xticks(np.arange(size))
        ax.set_yticks(np.arange(size))
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticklabels(classes)
        ax.set_xticks(np.arange(-0.5, size), minor=True)
        ax.set_yticks(np.arange(-0.5, size), minor=True)
        ax.set_xlabel('Wypowiedziane slowo')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Rozpoznane slowo')
        ax.grid(which='minor')

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        fig.show()
