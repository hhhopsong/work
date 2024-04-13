import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-1, 1, 1000)
plt.figure(figsize=(8, 4))
# 原函数
plt.plot(x, 1/(1+25*x**2), label='$ f(x) $', color='grey', linestyle=':', linewidth=3)
# 分段线性插值
x = -1
dx = 0.1
x_o = []
y_o = []
while x < 1:
    xk = x
    xkk = xk + dx
    x_in = xk
    fk = 1/(1 + 25*xk**2)
    fkk = 1/(1 + 25*xkk**2)
    d2x = dx / 10
    while x_in <= xkk:
        y_in = (x_in - xkk)/(xk - xkk) * fk + (x_in - xk)/(xkk - xk) * fkk
        x_o.append(x_in)
        y_o.append(y_in)
        x_in += d2x
    x += dx
plt.plot(x_o, y_o, label='$ piecewise-linear-interpolation $', color='blue', linewidth=1)
# 高次多项式插值
a = 11
x = -1
x_0 = [-1+i*0.2 for i in range(a)]
y_0 = [1/(1 + 25*x_0[i]**2) for i in range(a)]
x_o = []
y_o = []
while x <= 1.01:
    l_i = []
    for i in range(a):
        up = 1
        down = 1
        for ii in range(a):
            if ii != i:
                up *= (x - x_0[ii])
                down *= (x_0[i] - x_0[ii])
        l_i.append(up / down)
    y = sum([y_0[i] * l_i[i] for i in range(a)])
    x_o.append(x)
    y_o.append(y)
    x += 0.01
plt.plot(x_o, y_o, label='$ High-order-polynomial-interpolation $', color='green', linestyle='-.', linewidth=1)
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()
plt.show()
