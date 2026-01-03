import torch


def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)

    D = torch.diag(torch.diag(A_t))
    R = A_t - D
    D_inv = torch.inverse(D)

    x = torch.zeros_like(b_t)

    for i in range(n):
        x = D_inv @ (b_t - R @ x)
    return x
