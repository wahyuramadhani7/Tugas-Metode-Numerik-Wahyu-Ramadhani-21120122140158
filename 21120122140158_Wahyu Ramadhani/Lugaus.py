import numpy as np

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n, dtype=np.double)
    U = A.copy()

    for k in range(n-1):
        if U[k, k] == 0.0:
            raise ValueError("Matriks A tidak memiliki solusi unik atau sistem persamaan linear tidak konsisten.")

        for i in range(k+1, n):
            if U[i, k] != 0.0:
                lam = U[i, k] / U[k, k]
                L[i, k] = lam
                U[i, k:n] -= lam * U[k, k:n]

    return L, U

def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double)

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]

    return y

def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double)

    for i in range(n-1, -1, -1):
        if U[i, i] == 0.0:
            raise ValueError("Matriks A tidak memiliki solusi unik atau sistem persamaan linear tidak konsisten.")

        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]

    return x

def solve_linear_system(A, b):
    try:
        L, U = lu_decomposition(A)
        y = forward_substitution(L, b)
        x = backward_substitution(U, y)
        return x
    except ValueError as e:
        print(e)
        return None

if __name__ == '__main__':
    A = np.array([[1, -1, 2], [3, 0, 1], [1, 0, 2]], dtype=np.double)
    b = np.array([5, 10, 5], dtype=np.double)

    x = solve_linear_system(A, b)
    if x is not None:
        print("Solusi sistem persamaan linear:")
        print(x)