import numpy as np

def crout_decomposition(A, b):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))

    # Dekomposisi Crout
    for j in range(n):
        U[0][j] = A[0][j]

    for i in range(1, n):
        L[i][0] = A[i][0] / U[0][0]

    for i in range(1, n):
        for j in range(1, n):
            if i > j:
                L[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(j))
            else:
                U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))

    # Solusi dengan substitusi maju dan substitusi mundur
    y = np.zeros(n)
    y[0] = b[0] / L[0][0]

    for i in range(1, n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]

    x = np.zeros(n)
    x[n-1] = y[n-1] / U[n-1][n-1]

    for i in range(n-2, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    return x

# Contoh penggunaan
A = np.array([[1, -1, 2],
              [3, 0, 1],
              [1, 0, 2]])
b = np.array([5, 10, 5])
x = crout_decomposition(A, b)
print("Solusi sistem persamaan linear:")
print(x)