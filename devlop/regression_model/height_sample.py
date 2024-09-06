import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=1)              # 乱数を固定
X_min = 4                           # Xの下限値
X_max = 30                          # Xの上限値
X_n = 16                            # データ個数
X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170, 108, 0.2]             # 生成パラメータ

T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)
np.savez('ch5_data.npz', X=X, X_min=X_min, X_max=X_max, X_n=X_n, T=T)

print(X)
print(np.round(X, 2))
print(np.round(T, 2))

plt.figure(figsize=(4, 4))
plt.plot(X, T, marker='o', linestyle='None', markeredgecolor='black', color='cornflowerblue')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()
