
for i in range(100,1000):
    a = eval(str(i)[0])
    b = eval(str(i)[1])
    c = eval(str(i)[2])
    if i == a ** 3 + b ** 3+ c ** 3:
        print(i,end=' ')


