import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f3(x0, x1):
    r = 2 * x0**2 + x1**2
    ans = r * np.exp(-r)
    return ans

xn = 50
x0 = np.linspace(-2, 2, xn)
x1 = np.linspace(-2, 2, xn)

y = np.zeros((len(x0), len(x1)))

for i0 in range(xn):
    for i1 in range (xn):
        y[i1, i0] = f3(x0[i0], x1[i1])

xx0, xx1 = np.meshgrid(x0, x1)

plt.figure(figsize=(4, 4))
cont = plt.contour(xx0, xx1, y, 5, colors='black')
cont.clabel(fmt='%3.2f', fontsize=8)
plt.xlabel('$x_0$', fontsize=14)
plt.ylabel('$x_1$', fontsize=14)
plt.show()
