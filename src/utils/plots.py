import numpy as np
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
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
