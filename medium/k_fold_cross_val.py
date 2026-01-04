import numpy as np


def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Implement k-fold cross-validation by returning train-test indices.
    """
    if shuffle:
        indices = np.random.permutation(len(X))
    else:
        indices = np.arange(len(X))
    fold_sizes = np.full(k, len(X) // k, dtype=int)
    fold_sizes[: len(X) % k] += 1
    current = 0

    returned_folds = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))
        returned_folds.append((train_indices, test_indices))
        current = stop
    return returned_folds
