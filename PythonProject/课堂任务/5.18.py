'''class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def info(self):
        print(f'圆心：({self.x:.2f}, {self.y:.2f})')
        print(f'半径：({self.r:.2f})')

    def area(self):
        return 3.14 * self.r * self.r

    def perimeter(self):
        return 2 * 3.14 * self.r


x = Circle(0, 0, 10)
x.info()
print(x.area())
a, b, c = eval(input('分别输入a,b,c的值（用逗号隔开）：'))
'''

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


a, b, c = 1, 1, 1
while a**2 + b ** 2 + c ** 2 != 0:
    a, b, c = eval(input('分别输入a,b,c的值（用逗号隔开）：'))
    solve(a, b, c)
