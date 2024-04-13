
'''count = 0
i = '*'
while count < 5:
    count += 1
    print(i*count)'''
'''def printSymbol(a, b):
    for i in range(1, b+1):
        print(a * i)
printSymbol('*', 6)'''

'''def printwishes(sb):
    print(sb, "best wishes")
    return 99
printwishes('sfm')
print( printwishes('lxy') )'''

def isPrime(i):
    if i < 2:
        return False
    else:
        n = 1
        while n <= i ** 0.5 - 1:
            n += 1
            if i % n == 0:
                return False
        else:
            return True
x = []
num = 0
count = 0
for i in range(1, 101):
    if isPrime(i):
        x.append(i)
    num += 1
for i in x:
    count += 1
    if count % 5 == 0:
        print('{:>6}'.format(i))
    else:
        print('{:>6}'.format(i), end='')














