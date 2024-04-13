
def fun(n):
    if n == 0:
        return 0
    else:
        return 2 * fun(n - 1) + 1


try:
    while True:
        info = int(input('输入一个整数：'))
        print(fun(info))
except ValueError:
    print('不是一个整数！')
