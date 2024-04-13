info = [eval(i) for i in input('input:').split(',')]
for i in range(len(info) - 2):
    print(info[i] * 0.25 + info[i + 1] * 0.5 + info[i + 2] * 0.25)

