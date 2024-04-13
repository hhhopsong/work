
# 从键盘输入两个数，交换后打印输出

x, y = eval(input('Enter two numbers:'))
print(f'x = {x},y = {y}')
x, y = y, x
print(f'x = {x},y = {y}')

# 我的实践
print(type(eval('356')))
print(type(eval('365.1')))
print(type(eval('10+5j')))
print(type(eval('654+44+16')), eval('4+6+5'))
print(type(eval('False')))
a = 1
b = 2
c = 3
print(type(eval('a*b*c')), eval('a*b*c'))
print(type(eval("'55'")))
