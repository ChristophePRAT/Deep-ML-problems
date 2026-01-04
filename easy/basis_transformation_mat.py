import numpy as np


def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    """
    Given two bases B and C of a vector space V, compute the transformation matrix
    that converts coordinates from basis B to basis C.

    Args:
        B: A list of basis vectors for basis B (each vector is a list of ints).
        C: A list of basis vectors for basis C (each vector is a list of ints).
    Returns:
        A transformation matrix (list of list of floats) that converts coordinates from basis B to basis C.
    """

    B_matrix = np.array(B)  # Shape (n, n)
    C_matrix = np.array(C)  # Shape (n, n)

    C_inv = np.linalg.inv(C_matrix)

    T = C_inv @ B_matrix

    return T.tolist()
