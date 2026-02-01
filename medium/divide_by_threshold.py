import numpy as np


def divide_on_feature(X, feature_i, threshold):
    """
    Divide the dataset based on a feature threshold.

    Args:
        X: A 2D list or numpy array representing the dataset.
        feature_i: The index of the feature to split on.
        threshold: The threshold value for the split.
    Returns:
        A tuple containing two lists:
            - The first list contains samples where the feature value is less than or equal to the threshold.
            - The second list contains samples where the feature value is greater than the threshold.
    """

    X = np.array(X)
    left_indices = X[:, feature_i] >= threshold
    right_indices = X[:, feature_i] < threshold

    left_split = X[left_indices].tolist()
    right_split = X[right_indices].tolist()

    return left_split, right_split
