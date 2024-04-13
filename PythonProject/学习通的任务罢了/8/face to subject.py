class Equation:
    def __init__(self, a = 0, b = 0, c = 0):
        self.a = a
        self.b = b
        self.c = c

    def solve(self):
        o = self.b ** 2 - 4 * self.a * self.c
        if self.a == 0:
            print('不是一元二次方程！')
            return
        elif o > 0:
            q = (-self.b - o ** (1 / 2)) / (2 * self.a)
            w = (-self.b + o ** (1 / 2)) / (2 * self.a)
            print(f'方程有两个实根，其中X1={q:.2f} X2={w:.2f}')
        elif o == 0:
            q = (-self.b) / (2 * self.a)
            print(f'方程有两个相等实根，X1=X2={q:.2f}')
        elif o < 0:
            q = -self.b / (2 * self.a)
            w = -self.b / (2 * self.a)
            i = abs(o) ** (1 / 2) / (2 * self.a)
            print(f'方程有两个根，其中X1={q:.2f} + {i:.2f}i X2={w:.2f} - {i:.2f}i')


while True:
    a, b, c = eval(input('Enter a, b, c:'))
    if a ** 2 + b ** 2 + c ** 2 == 0:
        print('{:*^60}'.format('ending'))
        break
    Equation(a, b, c).solve()
