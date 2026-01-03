import torch


def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1, 1)

    xt_x_inv = (X_t.T @ X_t).inverse()

    theta = xt_x_inv @ X_t.T @ y_t

    return theta


# Tests
def test():
    X = [[1, 1], [1, 2], [2, 2], [2, 3]]
    y = [1, 2, 2, 3]
    theta = linear_regression_normal_equation(X, y)
    print(theta)  # Expected output: tensor([0.5000, 1.0000])

    X = [[1, 2, 3], [1, 3, 4], [1, 4, 5]]
    y = [1, 2, 3]
    theta = linear_regression_normal_equation(X, y)
    print(theta)

    print("All tests passed.")


if __name__ == "__main__":
    test()
