
a, b = eval(input('请输入两个数，并用逗号隔开：'))
a1, b1 = a, b
if a < b:
    a, b = b, a
while a % b != 0:
    a, b = b, a % b
    if a % b == 0:
        break
z = int(a1 * b1 / b)
print(f'{a1}和{b1}的最大公约数是{b}，最小公倍数是{z}')

