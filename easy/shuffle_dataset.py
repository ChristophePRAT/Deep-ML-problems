import numpy as np


def shuffle_data(X, y, seed=None):
    """
    Shuffle the dataset (features and labels) in unison.

    Args:
        X: A 2D list or numpy array representing the features.
        y: A 1D list or numpy array representing the labels.
        seed: An optional integer seed for reproducibility.
    Returns:
        A tuple containing the shuffled features and labels.
    """

    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    shuffled_X = np.array(X)[indices]
    shuffled_y = np.array(y)[indices]
    return shuffled_X.tolist(), shuffled_y.tolist()
