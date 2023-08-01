import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
def runge(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    k = 0
    for i in range(n-1):
        h = (t[i+1] - t[i])
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h/2., t[i] + h/2., *args)
        k3 = f(y[i] + k2 * h/2., t[i] * h/2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h/6.) * (k1 + 2 * k2 * 2 * k3 + k4)
        k +=1
        print(y[i], k)


    return y

t = np.linspace(0, 2, 500)
y0 = np.array([np.e, 1])

def func(y, x):
    return np.array([-2 * x * y[0] * np.log(y[1]), 2 * x * y[1] * np.log(y[0])])


sol= runge(func, y0, t)


plt.plot(t, sol[:, 0], 'b', label = 'y1')
plt.plot(t, sol[:, 1], 'g', label = 'y2')

plt.show()