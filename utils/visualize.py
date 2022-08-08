import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns


def plot_confusion_matrix(cm,
                          classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)
    # print(cm.diag() / cm.sum(1))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f' #'d'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # wandb.log({title: plt})
    return plt
    # plt.savefig('/ws/external/visualization_results/confusion_matrix.png')


def plot_tsne(test_features, targets=None):
    test_features = test_features.cpu().detach().numpy().data
    y = targets.cpu().detach().numpy().data

    tsne = TSNE(n_components=2, perplexity=10, n_iter=300, learning_rate=200.0, init='random')
    tsne_ref = tsne.fit_transform(test_features)

    fig = plt.figure(figsize=(12, 12))
    plt.scatter(tsne_ref[:, 0], tsne_ref[:, 1], marker='.',
                cmap=cm.Paired, c=y)
    plt.title('t-SNE result', weight='bold').set_fontsize('14')
    plt.xlabel('x', weight='bold').set_fontsize('10')
    plt.ylabel('y', weight='bold').set_fontsize('10')
    plt.axis('equal')
    plt.savefig("/ws/data/ai28/hello.png")
    # wandb.log({f"t-sne": self.wandb.Image(fig)})
    plt.close(fig)

