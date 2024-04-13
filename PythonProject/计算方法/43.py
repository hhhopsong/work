from math import *


def f(x):
    if x != 0:
        return sin(x) / x
    else:
        return 1


def trapezoid(x, y, n):
    h = (y - x) / n
    f_xk = [f(x + i * h) for i in range(1, n)]
    return h / 2 * (f(x) + 2 * sum(f_xk) + f(y))


def aim_trapezoid(x, y, n=1, rs=0.01):
    global times
    s2 = trapezoid(x, y, n * 2)
    if abs(trapezoid(x, y, n) - s2) < 3 * rs:
        return s2
    else:
        times += 1
        return aim_trapezoid(x, y, n * 2, rs)


# 变步长的梯形公式
a = eval(input('积分下限:'))
b = eval(input('积分上限:'))
RS = 0.1 * 0.1 ** eval(input('有效数字精度为:'))
times = 0
output = aim_trapezoid(a, b, rs=RS)
print(f'二分区间的次数:{times}')
print(f'变步长的梯形公式求解结果:{output:.7f}')
