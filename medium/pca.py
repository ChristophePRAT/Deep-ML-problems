import torch


def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """

    data = torch.tensor(data, dtype=torch.float32)

    standardized_data = (data - data.mean(dim=0)) / data.std(dim=0)
    covariance_matrix = torch.matmul(standardized_data.T, standardized_data) / (
        standardized_data.shape[0] - 1
    )
    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_k_indices = sorted_indices[:k]
    top_k_eigenvectors = eigenvectors[:, top_k_indices]
    for i in range(top_k_eigenvectors.shape[1]):
        first_non_zero_idx = torch.nonzero(top_k_eigenvectors[:, i], as_tuple=True)[0][
            0
        ]
        if top_k_eigenvectors[first_non_zero_idx, i] < 0:
            top_k_eigenvectors[:, i] = -top_k_eigenvectors[:, i]
    return torch.round(top_k_eigenvectors * 10000) / 10000
