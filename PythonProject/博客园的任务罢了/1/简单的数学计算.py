
# 简单的数学计算
from math import sqrt

n = float(input('输入一个数：'))

x = sqrt(n)

print('%f的平方根是：%.2f' % (n, x))   # 输出方式1
print('{}的平方根是{}'.format(n, x))  # 输出方式2
print(f'{n}的平方根是：{x}')          # 输出方式3

# 由此可以求解一元二次方程，我的尝试如下：
print(" ")
print('一个解ax^2+bx+c=0的简单程序')
a, b, c = eval(input('分别输入a,b,c的值（用逗号隔开）：'))
o = b**2-4*a*c
if o > 0:
    q = (-b - o ** (1 / 2)) / (2*a)
    w = (-b + o ** (1 / 2)) / (2*a)
    print(f'方程有两个实根，其中X1={q} X2={w}')
elif o == 0:
    q = (-b)/(2*a)
    print(f'方程有两个相等实根，X1=X2={q}')
else:
    print('方程无实根！！！')

