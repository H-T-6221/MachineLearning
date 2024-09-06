import math
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.001, 4, 100)
y = np.log(x)
dy = 1 / x

plt.figure(figsize=(4, 4))
plt.plot(x, y, 'gray', linestyle='--', linewidth=3, label='$y=log(x)$')
plt.plot(x, dy, 'black', linewidth=3, label='$y\'=1/x$')
plt.ylim(-8, 8)
plt.ylim(-1, 4)
plt.grid(True)
plt.legend(loc='lower right')
plt.show()
