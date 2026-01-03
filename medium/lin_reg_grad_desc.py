import torch


def linear_regression_gradient_descent(X, y, alpha, iterations) -> torch.Tensor:
    """
    Solve linear regression via gradient descent using PyTorch autograd.
    X: Tensor or convertible shape (m,n); y: shape (m,) or (m,1).
    alpha: learning rate; iterations: number of steps.
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1, 1)
    m, n = X_t.shape
    theta = torch.zeros((n, 1), requires_grad=True)

    for _ in range(iterations):
        preds = X_t @ theta
        loss = torch.mean((preds - y_t) ** 2)
        loss.backward()
        with torch.no_grad():
            theta -= alpha * theta.grad
            theta.grad.zero_()
    return theta.reshape(-1)


# Tests
def test():
    X = [[1, 1], [1, 2], [2, 2], [2, 3]]
    y = [1, 2, 2, 3]
    alpha = 0.1
    iterations = 1000
    theta = linear_regression_gradient_descent(X, y, alpha, iterations)
    print(theta)  # Expected output: tensor([0.5000, 1.0000])

    X = [[1, 2, 3], [1, 3, 4], [1, 4, 5]]
    y = [1, 2, 3]
    alpha = 0.01
    iterations = 2000
    theta = linear_regression_gradient_descent(X, y, alpha, iterations)
    print(theta)

    print("All tests passed.")


if __name__ == "__main__":
    test()
