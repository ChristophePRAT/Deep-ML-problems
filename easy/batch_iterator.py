from typing import Any

import numpy as np


def batch_iterator(X, y=None, batch_size=64) -> list[list[Any]]:
    """
    Generate mini-batches from the dataset.

    Args:
        X: A 2D list or numpy array representing the features.
        y: An optional 1D list or numpy array representing the labels.
        batch_size: The size of each mini-batch.
    Returns:
        A list of mini-batches, where each mini-batch is a list containing:
            - A 2D list or numpy array of features for the batch.
            - An optional 1D list or numpy array of labels for the batch (if y is provided).
    """

    num_samples = len(X)
    batches = []

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_X = X[start_idx:end_idx]
        if y is not None:
            batch_y = y[start_idx:end_idx]
            batches.append([batch_X, batch_y])
        else:
            batches.append([batch_X])

    return batches
