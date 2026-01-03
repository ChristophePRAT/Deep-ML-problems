import numpy as np


def reshape_matrix(
    a: list[list[int | float]], new_shape: tuple[int, int]
) -> list[list[int | float]]:
    # Write your code here and return a python list after reshaping by using numpy's tolist() method

    a_list = np.array(a).flatten().tolist()

    n, m = new_shape

    if n * m != len(a_list):
        return []

    reshaped_matrix = [[a_list[i * m + j] for j in range(m)] for i in range(n)]

    return reshaped_matrix


# Tests


def test():
    a = [[1, 2, 3], [4, 5, 6]]
    new_shape = (3, 2)
    expected = [[1, 2], [3, 4], [5, 6]]
    assert reshape_matrix(a, new_shape) == expected

    a = [[1, 2], [3, 4], [5, 6]]
    new_shape = (2, 3)
    expected = [[1, 2, 3], [4, 5, 6]]
    assert reshape_matrix(a, new_shape) == expected

    a = [[1]]
    new_shape = (1, 1)
    expected = [[1]]
    assert reshape_matrix(a, new_shape) == expected

    a = []
    new_shape = (0, 0)
    expected = []
    assert reshape_matrix(a, new_shape) == expected

    a = [[1, 2, 3, 4]]
    new_shape = (2, 2)
    expected = [[1, 2], [3, 4]]
    assert reshape_matrix(a, new_shape) == expected

    a = [[1], [2], [3], [4]]
    new_shape = (2, 2)
    expected = [[1, 2], [3, 4]]
    assert reshape_matrix(a, new_shape) == expected

    print("All tests passed.")


if __name__ == "__main__":
    test()
