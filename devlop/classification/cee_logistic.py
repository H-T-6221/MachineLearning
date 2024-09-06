import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=0)
X_min = 0
X_max = 2.5
X_n = 30
X_col = ['cornflowerblue', 'gray']

X = np.zeros(X_n)
T = np.zeros(X_n, dtype=np.uint8)

Dist_s = [0.4, 0.8]
Dist_w = [0.8, 1.6]
Pi = 0.5

for n in range(X_n):
    wk = np.random.rand()
    T[n] = 0 * (wk < Pi) + 1 * (wk >= Pi)
    X[n] = np.random.rand() * Dist_w[T[n]] + Dist_s[T[n]]

print('X=' + str(np.round(X, 2)))
print('T=' + str(T))

# ロジスティック回帰モデル
def logistic(x, w):
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y

# ロジスティック回帰モデル 表示
def show_logistic(w):
    xb = np.linspace(X_min, X_max, 100)
    y = logistic(xb, w)
    plt.plot(xb, y, color='gray', linewidth=4)
    # 決定境界
    i = np.min(np.where(y > 0.5))
    B = (xb[i - 2] + xb[i]) / 2
    plt.plot([B, B], [-.5, 1.5], color='k', linestyle='-')
    plt.grid(True)
    return B

# 平均交差エントロピー誤差
def cee_lpgistic(w, x, t):
    y = logistic(x, w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n] * np.log(y[n]) + (1 - t[n]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee

# test
W = [1, 1]
print(cee_lpgistic(W, X, T))
