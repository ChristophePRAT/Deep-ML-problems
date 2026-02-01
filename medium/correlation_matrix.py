from typing import Optional, Union

import torch


def calculate_correlation_matrix(
    X: Union[torch.Tensor, list],
    Y: Optional[Union[torch.Tensor, list]] = None,
) -> torch.Tensor:
    """
    Compute the correlation matrix of X (and optionally Y) using PyTorch.
    If Y is None, returns the correlation matrix of X with itself.
    """
    Y = X if Y is None else Y

    X_t = torch.as_tensor(X, dtype=torch.float)
    Y_t = torch.as_tensor(Y, dtype=torch.float)

    X_centered = X_t - X_t.mean(dim=0, keepdim=True)
    Y_centered = Y_t - Y_t.mean(dim=0, keepdim=True)
    covariance = X_centered.T @ Y_centered / (X_t.shape[0] - 1)

    X_std = X_t.std(dim=0, unbiased=True)
    Y_std = Y_t.std(dim=0, unbiased=True)

    correlation_matrix = covariance / (X_std[:, None] * Y_std[None, :])

    return correlation_matrix
