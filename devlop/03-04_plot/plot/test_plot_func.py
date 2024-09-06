import numpy as np
import matplotlib.pyplot as plt

def f2(x, w):
    return (x - w) * x * (x + 2)

x = np.linspace(-3, 3, 100)

plt.plot(x, f2(x, 2), color='black', label='$w=2$')
plt.plot(x, f2(x, 1), color='cornflowerblue', label='$w=1$')

plt.legend(loc="upper left")
plt.ylim(-15, 15)
plt.title('$f_2(2)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(True)

plt.show()
