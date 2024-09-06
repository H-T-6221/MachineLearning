import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

np.random.seed(1)
x = np.arange(10)
y = np.random.rand(10)

plt.plot(x, y)
plt.show()
