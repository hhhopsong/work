'''
n = 0
i = 0
while n < 100:
    n += 1
    i = i + n
print(i)
'''
'''
a = 0
for i in range(101):
    a += i
print(a)
'''
'''
s = sum(list(range(1,101)))
print(s)
'''
'''
r = 1
j = 1
while r <= 9:
    while j < r:
        a = f'{j}x{r} = {r * j}'
        j += 1
        print('{:<10}'.format(a), end="")
    if j == r:
        a = f'{j}x{r} = {r * j}'
        j += 1
        print('{:<10}'.format(a), end='\n')
    j = 1
    r += 1
'''
'''
for i in range(1,10):
    for j in range(1,i+1):
        print("{}x{}={}\t".format(j,i,i*j),end=" ")
    print()
'''

scores = [91,88,50,32,66]
message = ['pass'if score >= 60 else 'fail' for score in scores ]
print(message)

