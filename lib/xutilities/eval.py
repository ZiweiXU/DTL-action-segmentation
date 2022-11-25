import numpy as np
import seaborn as sns


def export_gif(frames, path, fps=15):
    """
    Args:
        frames - [np.array], list of frames, data range [0,255]
        path - str
        fps - int
    """
    import imageio
    with imageio.get_writer(path, mode='I', fps=fps) as writer:
        for i in range(len(frames)):
            writer.append_data(frames[i])


def cal_accuracy(preds, labels):
    assert len(preds) == len(labels)
    correct = 0.0
    for i in range(len(preds)):
        correct += 1 if preds[i] == labels[i] else 0
    return correct / len(preds)


def hmean(numbers, eps=1e-8):
    numbers = [number + eps if number < eps else number for number in numbers]
    return len(numbers) / sum([1/i for i in numbers])


def confusion_matrix(preds, labels, num_classes=None, normalize=False):
    """Compute confusion matrix.
    
    Args:
        preds :[]: of integers.
        labels :[]: of integers.
        num_classes :int: number of classes.
        normalize :bool: if true, results will be normalized row-wisely.
    """
    if num_classes is None:
        num_classes = len(list(set(labels)))
    
    matrix = np.zeros((num_classes, num_classes))
    for p, l in zip(preds, labels):
        matrix[l][p] += 1
    
    if normalize:
        matrix /= np.sum(matrix, 1)
    
    return matrix


def plot_confusion_matrix(
    filename, confmat, labels=None, figsize=(30,30), **kwargs):
    """Save the confusion matrix to a file.
    
    Args:
        filename :str: name of the image to be saved.
        confmat :np.array: the confusion matrix of shape MxM.
        labels :list: of categories.
        normalized :bool: optional, whether the input confmat is normalized.
    """
    import matplotlib; matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pyplot as plt

    figsize = figsize
    fig, ax = plt.subplots(figsize=figsize) 
    normalized = kwargs.get('normalized', (np.max(confmat) <= 1))
    fmt = kwargs.get('fmt', '.2f' if normalized else '.0f')
    fig = sns.heatmap(
        confmat, annot=True, fmt=fmt, cmap='YlOrBr',
        xticklabels=labels, yticklabels=labels, ax=ax)
    fig.get_figure().savefig(filename, bbox_inches='tight')
