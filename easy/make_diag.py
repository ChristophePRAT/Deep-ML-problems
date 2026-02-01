import numpy as np


def make_diagonal(x):
    n = len(x)
    diag = [[0] * n for _ in range(n)]
    for i in range(n):
        diag[i][i] = x[i]
    return diag
