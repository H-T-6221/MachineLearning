import math
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 100)
a = 3
y = a**x
dy = np.log(a) * y

plt.figure(figsize=(4, 4))
plt.plot(x, y, 'gray', linestyle='--', linewidth=3, label='$y=a^x$')
plt.plot(x, dy, 'black', linewidth=3, label='$y\'=a^xlog$')
plt.ylim(-1, 8)
plt.ylim(-4, 4)
plt.grid(True)
plt.legend(loc='lower right')
plt.show()
