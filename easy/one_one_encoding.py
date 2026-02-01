import numpy as np


def to_categorical(x, n_col=None):
    col = np.max(x) + 1 if n_col is None else n_col

    one_hot = np.zeros((len(x), col))
    one_hot[np.arange(len(x)), x] = 1

    return one_hot
