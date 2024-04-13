import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 4))
# 曲线拟合
x_0 = [-1+i*0.2 for i in range(11)]
y_0 = [1/(1 + 25*x_0[i]**2) for i in range(11)]
x = np.linspace(-1, 1, 1000)
p1 = np.poly1d(np.polyfit(x_0, y_0, 3))
p2 = np.poly1d(np.polyfit(x_0, y_0, 10))
plt.plot(x, p1(x), label='$ Cubic-curve-fitting $', color='black')
plt.plot(x, p2(x), label='$ High-order-polynomial-interpolation $', color='black', linestyle='--')
plt.scatter(x_0, y_0, c='black')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()
plt.show()
