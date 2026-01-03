def mean_vector(vectors: list[float]) -> float:
    if not vectors:
        return 0.0
    return sum(vectors) / len(vectors)


def cov(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0

    mean_x = mean_vector(x)
    mean_y = mean_vector(y)

    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)

    return covariance


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    return [[cov(xi, xj) for xi in vectors] for xj in vectors]
