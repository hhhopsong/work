import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 4))
# date
x_0 = [0.2, 0.4, 0.6, 0.8, 1.0]
y_0 = [0.98, 0.92, 0.81, 0.64, 0.38]
# Lagrange
x = -0.9
key = 0
x_o = []
y_o = []
while x < 1.5:
    l_i = []
    for i in range(5):
        up = 1
        down = 1
        for ii in range(5):
            if ii != i:
                up *= (x - x_0[ii])
                down *= (x_0[i] - x_0[ii])
        l_i.append(up / down)
    key += 1
    y = sum([y_0[i] * l_i[i] for i in range(5)])
    x_o.append(x)
    y_o.append(y)
    x += 0.01
plt.plot(x_o, y_o, label='$ Lagrange $', color='silver', linestyle='-', linewidth=4)
# Newtown
x = -0.9
key = 0
x_o.clear()
y_o.clear()
while x < 1.5:
    n_l = []
    for j in range(5):
        ano = []
        for i in range(j+1):
            up = y_0[i]
            down = 1
            for ii in range(j+1):
                if ii != i:
                    down *= x_0[i] - x_0[ii]
            ano.append(up / down)
        n_l.append(sum(ano))
    behind = []
    beh = 1
    for i in range(5):
        behind.append(beh)
        beh *= x - x_0[i]
    key += 1
    y = sum([behind[i] * n_l[i] for i in range(5)])
    x_o.append(x)
    y_o.append(y)
    x += 0.01
plt.plot(x_o, y_o, label='$ Newtown $', color='black', linestyle='-.', linewidth=1)
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()
plt.show()
