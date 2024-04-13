'''
def x(a, b, c=1):
    y = [chr(i) for i in range(ord(a), ord(b) + 1, c)]
    return y


print(x('x', 'z', 2))
'''
# def函数中u参数的引用是复制，并不对原参数产生影响

'''def fibo(n):
    if n == 1 or n == 2:
        return 1
    else:
        return fibo(n - 1) + fibo(n - 2)
    
    
print(fibo(3))'''

def mypow(x, y):
    if x == 0:
        return 0
    elif y == 0:
        return 1
    elif y > 0:
        return x*mypow(x, y - 1)
    elif y < 0:
        return 1/mypow(x, -y)


