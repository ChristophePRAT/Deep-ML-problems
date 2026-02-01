import numpy as np


def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    np.random.seed(seed)
    subsets = []
    n_samples = len(X)

    if replacements:
        # Bootstrap sampling: sample n_samples with replacement
        for _ in range(n_subsets):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            subsets.append((X[indices].tolist(), y[indices].tolist()))
    else:
        # Sample n_samples // 2 without replacement for each subset
        subset_size = n_samples // 2

        for _ in range(n_subsets):
            indices = np.random.choice(n_samples, size=subset_size, replace=False)
            subsets.append((X[indices].tolist(), y[indices].tolist()))

    return subsets
