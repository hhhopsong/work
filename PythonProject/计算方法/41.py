from math import *


def f(x):
    if x != 0:
        return sin(x) / x
    else:
        return 1


# 复化梯形公式
a = eval(input('积分下限:'))
b = eval(input('积分上限:'))
n = eval(input('取n='))
h = (b - a) / n
f_xk = [f(a + i * h) for i in range(1, n)]
I = (h / 2) * (f(a) + 2 * sum(f_xk) + f(b))
print(f'复化梯形公式求解结果:{I}')
print(f'误差:{abs(I - 0.9460831):.16f}')
