import numpy as np

def build_hilbert(n: int):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i][j] = 1.0 / (i + j + 1)
    return H


def forward_elimination(A, b):
    n = A.shape[0]
    for k in range(n):
        pivot = A[k, k]
        if abs(pivot) < 1e-15:
            raise ValueError(f"Нулевой ведущий элемент на шаге {k}")

        # Нормировка k-й строки
        for j in range(k, n):
            A[k, j] /= pivot
        b[k] /= pivot

        # Исключение x_k из строк ниже
        for i in range(k + 1, n):
            factor = A[i, k]
            for j in range(k, n):
                A[i, j] -= factor * A[k, j]
            b[i] -= factor * b[k]

    return A, b


def back_substitution(A, b):
    n = A.shape[0]
    x = np.zeros((n, 1))
    for k in range(n - 1, -1, -1):
        s = b[k].copy()
        for j in range(k + 1, n):
            s -= A[k, j] * x[j]
        x[k] = s          
    return x


def solve():
    n = int(input("Введите размерность n: "))
    A = build_hilbert(n)

    print("Матрица A\n", A)

    # b = сумма элементов в каждой строке
    b = A.sum(axis=1, keepdims=True)

    print("Вектор b\n", b)

    A, b = forward_elimination(A, b)
    x = back_substitution(A, b)

    print("Матрица A после метода Гаусса\n", A)

    print("Решение x:")
    print(x)


if __name__ == '__main__':
    solve()