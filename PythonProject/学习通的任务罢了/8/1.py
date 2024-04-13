def solve(a, b, c):
    o = b ** 2 - 4 * a * c
    if a == 0:
        print('不是一元二次方程！')
        return
    elif o > 0:
        q = (-b - o ** (1 / 2)) / (2 * a)
        w = (-b + o ** (1 / 2)) / (2 * a)
        print(f'方程有两个实根，其中X1={q:.2f} X2={w:.2f}')
    elif o == 0:
        q = (-b) / (2 * a)
        print(f'方程有两个相等实根，X1=X2={q:.2f}')
    elif o < 0:
        q = -b / (2 * a)
        w = -b / (2 * a)
        i = abs(o) ** (1 / 2) / (2 * a)
        print(f'方程有两个根，其中X1={q:.2f} + {i:.2f}i X2={w:.2f} - {i:.2f}i')


a, b, c = eval(input('分别输入a,b,c的值（用逗号隔开）：'))
while a**2 + b ** 2 + c ** 2 != 0:
    solve(a, b, c)
    a, b, c = eval(input('分别输入a,b,c的值（用逗号隔开）：'))
else:
    print('{:*^60}'.format('ending'))
