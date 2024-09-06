import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f3(x0, x1):
    r = 2 * x0**2 + x1**2
    ans = r * np.exp(-r)
    return ans

xn = 9
x0 = np.linspace(-2, 2, xn)
x1 = np.linspace(-2, 2, xn)
y = np.zeros((len(x0), len(x1)))

for i0 in range(xn):
    for i1 in range (xn):
        y[i1, i0] = f3(x0[i0], x1[i1])

print(x0)
print(x1)
print(np.round(y, 1))

plt.figure(figsize=(3.5, 3))
plt.gray()
plt.pcolor(y)
plt.colorbar()
plt.show()
