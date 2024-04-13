import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def fit_sin(X, a, b, c, d):
    return -a * np.sin(b*X+c) + d


x_0 = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
y_0 = [1.0, 0.41, 0.50, 0.61, 0.91, 2.02, 2.45]
x = np.linspace(-1, 1, 1000)
p1 = np.poly1d(np.polyfit(x_0, y_0, 3))
p2 = np.poly1d(np.polyfit(x_0, y_0, 4))
valX = np.array(x_0)
valY = np.array(y_0)
best_vals, cover = curve_fit(fit_sin, valX, valY, [1, 1, 1, 1],)
# 表一
plt.subplot(2, 2, 1)
plt.plot(x, p1(x), c='grey', lw=1)
plt.scatter(x_0, y_0, c='grey')
plt.xlim(-0.5, 1.1)
plt.ylim(0, 2.5)
# 表二
plt.subplot(2, 2, 2)
plt.plot(x, p2(x), c='black', ls='--', lw=1)
plt.scatter(x_0, y_0, c='grey')
plt.xlim(-0.5, 1.1)
plt.ylim(0, 2.5)
# 表三
plt.subplot(2, 2, 3)
plt.plot(x, fit_sin(x, best_vals[0], best_vals[1], best_vals[2], best_vals[3]), c='black', ls=':', lw=1)
plt.scatter(x_0, y_0, c='grey')
plt.xlim(-0.5, 1.1)
plt.ylim(0, 2.5)
# 表四
plt.subplot(2, 2, 4)
plt.plot(x, p1(x), label='$ Cubic-curve-fitting $', c='grey', lw=1)
plt.plot(x, p2(x), label='$ Quartic-curve-fitting $', c='black', ls='--', lw=1)
plt.plot(x, fit_sin(x, best_vals[0], best_vals[1], best_vals[2], best_vals[3]), label='$ Fourier-curve-fitting $', c='black', ls=':', lw=1)
plt.xlim(-0.5, 1.1)
plt.ylim(0, 2.5)
plt.scatter(x_0, y_0, c='grey')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()
plt.show()
