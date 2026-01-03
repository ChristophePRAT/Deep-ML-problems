def matrixmul(
    a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]]:
    n = len(a)
    m = len(b[0])

    k = len(b)

    if k != len(a[0]):
        return -1

    result = [
        [sum(a[i][p] * b[p][j] for p in range(k)) for j in range(m)] for i in range(n)
    ]
    return result
