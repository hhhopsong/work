x = 2022
while True:
    x += 1
    a = str(hex(x))[2:]
    print(a)
    flag = 1
    for i in a:
        for j in range(10):
            if i == str(j):
                flag = 0
                break
        if flag == 0:
            break
    if flag == 1:
        print(x)
        break