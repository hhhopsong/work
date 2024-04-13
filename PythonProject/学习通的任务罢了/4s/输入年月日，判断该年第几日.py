
while True:
    print('{:*^60}'.format('Remote Interpreter Reinitialized'))
    a, b, c = eval(input('输入年月日：'))
    list1 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    list2 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if a % 4 == 0 and a % 100 != 0 or a % 400 == 0:
        days = sum(list2[0:b - 1]) + c
    else:
        days = sum(list1[0:b - 1]) + c
    print(f'{a}年{b}月{c}日是这一年的第{days}天')
