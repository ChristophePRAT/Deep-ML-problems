import math


def single_neuron_model(
    features: list[list[float]], labels: list[int], weights: list[float], bias: float
) -> tuple[list[float], float]:
    """Compute the output of a single neuron model given features, weights, and bias.
    Args:
        features: A list of feature vectors (list of list of floats).
        labels: A list of true labels (list of ints).
        weights: A list of weights (list of floats).
        bias: A float representing the bias term.
    Returns:
        A tuple containing:
            - A list of predicted outputs (list of floats).
            - The mean squared error (float).
    """
    probabilities = []
    mse = 0.0

    for i in range(len(features)):
        z = sum(f * w for f, w in zip(features[i], weights)) + bias
        prob = 1 / (1 + math.exp(-z))  # Sigmoid activation
        probabilities.append(prob)
        mse += (labels[i] - prob) ** 2
    mse /= len(features)
    return probabilities, mse
