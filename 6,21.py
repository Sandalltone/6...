import numpy as np

def tridiagonal_eigenvectors(a, mu):
    n = a.shape[0]
    x = np.random.rand(n)
    x /= np.linalg.norm(x)
    for i in range(1000):
        y = np.linalg.solve(a - mu*np.eye(n), x)
        x = y / np.linalg.norm(y)
        if np.linalg.norm(a.dot(x) - mu*x) < 1e-8:
            break
    return x

def tridiagonal_eigenvalues(a, shift):
    n = a.shape[0]
    for i in range(1000):
        q, r = np.linalg.qr(a - shift*np.eye(n))
        a = r.dot(q) + shift*np.eye(n)
        if np.linalg.norm(np.tril(a, -1)) < 1e-8:
            break
    eigenvalues = np.diag(a)
    return eigenvalues

n = 10
a = np.zeros((n, n))
a[0, 0] = 2
a[0, 1] = -1
a[-1, -1] = 2
a[-1, -2] = -1
for i in range(1, n-1):
    a[i, i] = 2
    a[i, i-1] = -1
    a[i, i+1] = -1

shift = 0.0
eigenvalues = np.zeros(n)
eigenvectors = np.zeros((n, n))
for i in range(n):
    eigenvalues[i] = tridiagonal_eigenvalues(a, shift)[i]
    eigenvectors[:, i] = tridiagonal_eigenvectors(a, eigenvalues[i])
    shift = eigenvalues[i]

print("Первое наименьшее собственное значение равно:")
print(eigenvalues[0])
print("Соответствующий собственный вектор равен:")
print(eigenvectors[:, 0])
