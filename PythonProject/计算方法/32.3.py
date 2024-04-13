import numpy as np


# 超松弛迭代法
def SOR(n, A, B, x0, x, eps, k, w):
    times = 0
    while times < k:
        for i in range(n):
            temp = 0
            temps = x0.copy()
            for j in range(n):
                if i != j:
                    temp += x0[j] * A[i][j]
            x[i] = (1-w)*x[i]+w * ((B[i] - temp) / A[i][i])
            x0[i] = x[i].copy()
        calTemp = max(abs(x - temps))
        times += 1
        if calTemp < eps:
            print("精确度等于{0}时，逐次超松弛迭代法需要迭代{1}次收敛".format(eps, times))
            return x
        else:
            x0 = x.copy()
    print("在最大迭代次数内不收敛", "最大迭代次数后的结果为", x)
    return None


def main():
    k = 100  # 最大迭代次数
    n = 3
    w = 1.46
    A = np.array([[4, -2, -4], [-2, 17, 10], [-4, 10, 9]])
    B = np.array([10, 3, -7])
    x0 = np.array([1.0, 1, 1])
    x = np.array([0.0, 0, 0])
    eps = 10 ** (-6)
    Sor = SOR(n, A, B, x0, x, eps, k, w)
    print("迭代值为:", Sor)


if __name__ == '__main__':
    main()
