print('1-100之间的所有素数')
num = 1
print(2, end=' ')
for i in range(2, 101):
    n = 1
    while n <= i ** 0.5:
        n += 1
        if i % n == 0:
            break
    else:
        print(i, end=' ')
        num += 1
print(f'\n1-100之间内共有{num}个素数')
