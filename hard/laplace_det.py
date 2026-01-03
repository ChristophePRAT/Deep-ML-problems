def det_matrix(matrix: list[list[int | float]]) -> float:
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    # develop along the first row
    det = 0

    for j in range(n):
        # build the minor matrix
        minor = [row[:j] + row[j + 1 :] for row in matrix[1:]]
        cofactor = ((-1) ** j) * matrix[0][j] * det_matrix(minor)
        det += cofactor

    return det


def determinant_4x4(matrix: list[list[int | float]]) -> float:
    return det_matrix(matrix)
