import numpy as np


def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here

    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)

    standardized = (data - means) / std_devs

    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    normalized = (data - min) / (max - min)

    return standardized, normalized
