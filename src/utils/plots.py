import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(cm)
    plt.setp(ax, xticks=np.arange(0, cm.shape[0]), xticklabels=classes)
    plt.setp(ax, yticks=np.arange(0, cm.shape[0]), yticklabels=classes)
    ax.set_xlabel('Actual label')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Predicted label')
    fig.show()