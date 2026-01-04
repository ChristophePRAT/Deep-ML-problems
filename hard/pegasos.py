import numpy as np


def kernel_function(x1, x2, kernel_type="linear", sigma=1.0):
    if kernel_type == "linear":
        return np.dot(x1, x2)
    elif kernel_type == "rbf":
        diff = x1 - x2
        return np.exp(-np.dot(diff, diff) / (2 * sigma**2))
    else:
        raise ValueError("Unsupported kernel type")


def pegasos_kernel_svm(
    data: np.ndarray,
    labels: np.ndarray,
    kernel="linear",
    lambda_val=0.01,
    iterations=100,
    sigma=1.0,
) -> tuple[list, float]:
    alphas = np.zeros(len(data))
    b = 0.0

    for t in range(1, iterations + 1):
        learning_rate = 1 / (lambda_val * t)

        for i in range(len(data)):
            decision_value = (
                sum(
                    alphas[j]
                    * labels[j]
                    * kernel_function(data[j], data[i], kernel, sigma)
                    for j in range(len(data))
                )
                + b
            )

            if labels[i] * decision_value < 1:
                alphas[i] += learning_rate * (labels[i] - lambda_val * alphas[i])
                b += learning_rate * labels[i]

    return (alphas.tolist(), b)


# Tests
def test():
    predicted = pegasos_kernel_svm(
        np.array([[1, 2], [2, 3], [3, 1], [4, 1]]),
        np.array([1, 1, -1, -1]),
        kernel="linear",
        lambda_val=0.01,
        iterations=100,
    )
    expected = ([100.0, 0.0, -100.0, -100.0], -937.4755)
    assert np.allclose(predicted[0], expected[0], atol=1e-4)
    assert abs(predicted[1] - expected[1]) < 1e-4

    predicted_2 = pegasos_kernel_svm(
        np.array([[1, 2], [2, 3], [3, 1], [4, 1]]),
        np.array([1, 1, -1, -1]),
        kernel="rbf",
        lambda_val=0.01,
        iterations=100,
        sigma=0.5,
    )
    expected_2 = ([100.0, 99.0, -100.0, -100.0], -115.0)

    assert np.allclose(predicted_2[0], expected_2[0], atol=1e-4)
    assert abs(predicted_2[1] - expected_2[1]) < 1e-4
    print("All tests passed.")


if __name__ == "__main__":
    test()
