
'''
x = eval(input('整数：'))
m = 2
if x <= 1:
    print(f'{x}不是素数')
else:
    while m <= x ** 0.5:
            if x % m == 0:
                print(f'{x}不是素数')
                break
            else:
                m += 1
    else:
    print(f'{x}是素数')
'''
'''
while True:
    print('{:-^40}'.format('月份天数判断小程序'))
    year = 0
    month = 0
    y, m = eval(input('输入一个年份和月份：'))
    if y % 400 == 0:
        year += 1
    else:
        if y % 4 == 0:
            if y % 100 != 0:
                year += 1
    list1 = [1, 3, 5, 7, 8, 10, 12]
    list2 =[4, 6, 9, 11]
    if m in list1:
        print(f'{y}年{m}月有31天')
    elif m in list2:
        print(f'{y}年{m}月有30天')
    else:
        if year == 1:
            print(f'{y}年{m}月有29天')
        else:
            print(f'{y}年{m}月有28天')
'''
'''
f1, f2 = 1, 1
while f1 < 1000:
    f = f1 + f2
    f1, f2 = f2, f
    print('{:-^6}'.format(f),end=' ')
'''
'''
s = 1
f = 1
for i in range(21):
    f *= i
    s += f
print(s)
'''





