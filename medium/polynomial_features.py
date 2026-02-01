from itertools import combinations_with_replacement

import numpy as np


def polynomial_features(X, degree):
    poly_features = []
    for sample in X:
        features = []
        for deg in range(degree + 1):
            for indices in combinations_with_replacement(range(len(sample)), deg):
                product = 1
                for index in indices:
                    product *= sample[index]
                features.append(product)
        features.sort()
        poly_features.append(features)
    return np.array(poly_features)
