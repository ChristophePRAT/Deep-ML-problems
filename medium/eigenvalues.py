def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    trace = matrix[0][0] + matrix[1][1]
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    discriminant = trace**2 - 4 * determinant

    if discriminant < 0:
        return []
    elif discriminant == 0:
        eigenvalue = trace / 2
        return [eigenvalue]
    else:
        sqrt_discriminant = discriminant**0.5
        eigenvalue1 = (trace + sqrt_discriminant) / 2
        eigenvalue2 = (trace - sqrt_discriminant) / 2
        return [eigenvalue1, eigenvalue2]
