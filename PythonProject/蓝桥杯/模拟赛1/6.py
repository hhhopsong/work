s = eval(input())
e = eval(input())
r = (e-7+s)%7
if r == 0:
    r = 7
print((e-7+s)%7)
