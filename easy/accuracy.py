import numpy as np


def accuracy_score(y_true, y_pred):
    right_predictions = sum(1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred))
    return right_predictions / len(y_true)
