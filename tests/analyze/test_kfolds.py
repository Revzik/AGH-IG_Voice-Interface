import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.analyze import k_folds


class KFoldsTets(unittest.TestCase):
    def test_configure_k_folds(self):
        groups = k_folds.configure_k_folds()
        print(groups)

    def test_do_k_folds(self):
        k_folds.do_k_folds()


    def test_plot_matrix(self):
        cm = np.identity(14)
        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                   'H', 'I', 'J', 'K', 'L', 'M', 'N']
        size = cm.shape[0]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(cm, cmap=plt.get_cmap('BuGn'))
        # plt.setp(ax, xticks=np.arange(0, cm.shape[0]), xticklabels=classes)
        # plt.setp(ax, yticks=np.arange(0, cm.shape[0]), yticklabels=classes)
        ax.set_xticks(np.arange(size))
        ax.set_yticks(np.arange(size))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xticks(np.arange(-0.5, size), minor=True)
        ax.set_yticks(np.arange(-0.5, size), minor=True)
        ax.set_xlabel('Actual label')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Predicted label')
        ax.grid(which='minor')
        fig.show()
