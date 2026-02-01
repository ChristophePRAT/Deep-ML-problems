from typing import List

import torch


def log_softmax(scores: List[float]):
    """Computes log softmax over specified dimension of input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to reduce.

    Returns:
        torch.Tensor: Log softmax over specified dimension.
    """
    x = torch.tensor(scores, dtype=torch.float)
    return torch.log(torch.softmax(x, -1))
