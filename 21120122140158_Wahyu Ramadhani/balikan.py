import numpy as np

# Fungsi untuk mencari solusi sistem persamaan linear
def solusi_persamaan_linear(A, b):
    # Menghitung determinan matriks A
    det_A = np.linalg.det(A)

    # Memeriksa apakah matriks A memiliki invers
    if det_A == 0:
        print("Matriks A tidak memiliki invers.")
        return None

    # Menghitung invers matriks A
    A_inv = np.linalg.inv(A)

    # Menghitung solusi x
    x = np.dot(A_inv, b)

    return x

# Contoh penggunaan
A = np.array([[1, -1, 2],
              [3, 0, 1],
              [1, 0, 2]])
b = np.array([5, 10, 5])

solusi = solusi_persamaan_linear(A, b)

if solusi is not None:
    print("Solusi sistem persamaan linear:")
    print(solusi)