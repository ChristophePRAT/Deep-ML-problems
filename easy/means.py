def mean(numbers: list[float]) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == "column":
        return [mean([row[j] for row in matrix]) for j in range(len(matrix[0]))]
    elif mode == "row":
        return [mean(row) for row in matrix]


# Tests
def test():
    matrix = [[1, 2, 3], [4, 5, 6]]
    mode = "column"
    expected = [2.5, 3.5, 4.5]
    assert calculate_matrix_mean(matrix, mode) == expected

    matrix = [[1, 2, 3], [4, 5, 6]]
    mode = "row"
    expected = [2.0, 5.0]
    assert calculate_matrix_mean(matrix, mode) == expected

    matrix = [[1]]
    mode = "column"
    expected = [1.0]
    assert calculate_matrix_mean(matrix, mode) == expected

    matrix = [[1]]
    mode = "row"
    expected = [1.0]
    assert calculate_matrix_mean(matrix, mode) == expected

    print("All tests passed.")


if __name__ == "__main__":
    test()
