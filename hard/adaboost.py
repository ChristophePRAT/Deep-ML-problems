import math
from typing import Dict, List

import numpy as np
import torch


def adaboost_fit(X, y, n_clf) -> List[Dict]:
    """
    Fit an AdaBoost classifier using PyTorch tensors (no sklearn).
    Args:
        X: torch.Tensor or array-like, shape (n_samples, n_features)
        y: torch.Tensor or array-like, shape (n_samples,) with labels (+1/-1)
        n_clf: int, number of weak classifiers
    Returns:
        List of dictionaries with classifier params: 'polarity', 'threshold', 'feature_index', 'alpha'.
    """
    weights = torch.ones(len(y)) / len(y)
    classifiers = []

    for _ in range(n_clf):
        best_clf = {}
        min_error = float("inf")

        for feature_index in range(X.shape[1]):
            feature_values = X[:, feature_index]
            thresholds = torch.unique(feature_values)

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = torch.ones(len(y))
                    predictions[polarity * feature_values < polarity * threshold] = -1

                    misclassified = predictions != y
                    error = torch.sum(weights * misclassified.float())

                    if error < min_error:
                        min_error = error
                        best_clf = {
                            "polarity": polarity,
                            "threshold": threshold.item(),
                            "feature_index": feature_index,
                        }

        EPS = 1e-10
        alpha = 0.5 * math.log((1 - min_error + EPS) / (min_error + EPS))
        best_clf["alpha"] = alpha
        classifiers.append(best_clf)

        predictions = torch.ones(len(y))
        feature_values = X[:, best_clf["feature_index"]]
        threshold = best_clf["threshold"]
        polarity = best_clf["polarity"]
        predictions[polarity * feature_values < polarity * threshold] = -1

        weights *= torch.exp(-alpha * y * predictions)
        weights /= torch.sum(weights)
    return classifiers


# Tests
def test():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 1, -1, -1])
    n_clf = 3

    clfs = adaboost_fit(X, y, n_clf)
    print(clfs)


if __name__ == "__main__":
    test()
