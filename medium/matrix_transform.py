import torch


def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform Tâ»Â¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2Ã2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here

    if torch.det(T_t) == 0 or torch.det(S_t) == 0:
        return torch.tensor(-1.0)
    T_inv = torch.inverse(T_t)
    result = T_inv @ A_t @ S_t
    return result
