zero = '''99, 22, 51, 72, 61, 20, 88, 40, 21, 63, 30, 11, 18, 99, 12, 93, 16, 7, 53, 64, 9, 28, 84, 34, 96, 52, 82, 51, 77'''
zero = zero.split(', ')
zero = [int(i) for i in zero]
e = 0
for i in range(len(zero)):
    for j in range(i, len(zero)):
        if zero[i] * zero[j] >= 2022:
            e += 1
print(e)
