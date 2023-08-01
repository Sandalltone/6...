import numpy as np
from scipy.special import factorial

n = 8
a = np.zeros((n, n))
for i in range(1, n+1):
    for j in range(1, n+1):
        # Исключение отрицательного факториала
        if i == 1 and j == 1:
            continue
        a[i-1, j-1] = factorial(i+j-2) / (factorial(i-1) * factorial(j-1))

shift = 2.0
eigenvalues = np.zeros(3)
eigenvectors = np.zeros((n, 3))
for k in range(3):
    x = np.random.rand(n)
    x /= np.linalg.norm(x)
    for i in range(1000):
        b = np.linalg.inv(a - shift*np.eye(n)).dot(x) # Определение смещения
        x = b / np.linalg.norm(b)
        eigenvalues[k] = shift + 1.0 / np.dot(x, np.linalg.inv(a).dot(x))
        eigenvectors[:, k] = x
        if np.linalg.norm(a.dot(x) - eigenvalues[k]*x) < 1e-8:
            break
    shift = eigenvalues[k]

print("Тремя наименьшими собственными значениями являются:")
print(eigenvalues)
print("Соответствующими собственными векторами являются:")
print(eigenvectors)