import numpy as np


def solve(a):
    # 消元过程
    for i in range(len(a)-1):
        max = a[i][i]
        seq = i
        for ii in range(i+1, len(a)):
            if abs(max) < abs(a[ii][i]):
                max = a[ii][i]
                seq = ii
        exc = a[i]
        a[i] = a[seq]
        a[seq] = exc
        for ii in range(i+1, len(a)):
            double = a[i][i]/a[ii][i]
            a[ii] = [a[i][j] - a[ii][j] * double for j in range(len(a[ii]))]
    # 回代过程
    for i in range(len(a)-1, -1, -1):
        for ii in range(i-1, -1, -1):
            double1 = a[ii][i] / a[i][i]
            a[ii] = [a[ii][j] - a[i][j] * double1 for j in range(len(a[i]))]
        double = 1 / a[i][i]
        a[i] = [a[i][j] * double for j in range(len(a[i]))]
    return a


A1 = [[3.01, 6.03, 1.99], [1.27, 4.16, -1.23], [0.987, -4.81, 9.34]]
A2 = [[3.01, 6.03, 1.99], [1.27, 4.16, -1.23], [0.990, -4.81, 9.34]]
b = [[1], [1], [1]]
detA1 = np.linalg.det(np.mat(A1))
detA2 = np.linalg.det(np.mat(A2))
condA1 = np.linalg.cond(np.mat(A1))
condA2 = np.linalg.cond(np.mat(A2))
C1 = [[3.01, 6.03, 1.99, 1], [1.27, 4.16, -1.23, 1], [0.987, -4.81, 9.34, 1]]
C2 = [[3.01, 6.03, 1.99, 1], [1.27, 4.16, -1.23, 1], [0.990, -4.81, 9.34, 1]]
s = solve(C1)
ss = solve(C2)
for i in range(2):
    print(f'{f"方程组{i + 1}":*^18}')
    # 系数矩阵A
    print(f'{f"系数矩阵A{i + 1}":—^18}')
    for ii in range(3):
        A3 = A1+A2
        for iii in A3[i * 3 + ii]:
            print(f'{iii:^6}', end=' ')
        print()
    # b
    print(f'{f"b{i + 1}":—^18}')
    for j in b:
        print(f'{str(j).strip("[]"):^18}')
    # det A
    print(f'{f"det A{i + 1}":—^18}')
    if i == 0:
        print(detA1)
    else:
        print(detA2)
    # 解向量x
    print(f'{f"解向量x{i + 1}":—^18}')
    for ii in range(3):
        sss = s+ss
        print(f'{sss[i * 3 + ii][3]:^18}', end=' ')
        print()
    # 条件数
    print(f'{f"条件数{i + 1}":—^18}')
    if i == 0:
        print(condA1)
    else:
        print(condA2)

    print()
