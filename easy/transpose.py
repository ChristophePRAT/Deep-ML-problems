from typing import Tuple, Union


def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.

    Args:
        a: A 2D matrix of shape (m, n)

    Returns:
        The transposed matrix of shape (n, m)
    """

    if not a or not a[0]:
        return []

    m, n = len(a), len(a[0])

    transposed = [[a[j][i] for j in range(m)] for i in range(n)]

    return transposed


# Tests


def test():
    a = [[1, 2, 3], [4, 5, 6]]
    expected = [[1, 4], [2, 5], [3, 6]]
    assert transpose_matrix(a) == expected

    a = [[1, 2], [3, 4], [5, 6]]
    expected = [[1, 3, 5], [2, 4, 6]]
    assert transpose_matrix(a) == expected

    a = [[1]]
    expected = [[1]]
    assert transpose_matrix(a) == expected

    a = []
    expected = []
    assert transpose_matrix(a) == expected

    a = [[1, 2, 3]]
    expected = [[1], [2], [3]]
    assert transpose_matrix(a) == expected

    a = [[1], [2], [3]]
    expected = [[1, 2, 3]]
    assert transpose_matrix(a) == expected

    print("All tests passed.")


if __name__ == "__main__":
    test()
