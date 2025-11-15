import numpy as np
import pandas as pd
import pymannkendall as mk


def ws2001(Data):
    """
    消除气象数据自相关对趋势的影响，提高气象统计估计值的准确度过程称作预白化（Pre-whitening)。
    目前预白化有多种方案，这里介绍由Zhang et al., 2000提出，Wang and Swail, 2001改进的预白化。
    考虑到自相关和趋势是共存的，采用迭代方案计算AR(1)进而预白化后的Sen’s slope + Mann-Kendall。
    :param Data: 待处理的数据
    :return: 预白化后的序列
    """
    # 计算一阶自相关系数
    n = len(Data)
    c0 = pd.Series(Data).autocorr(1)
    c = c0
    w = np.zeros(n - 1)
    if c < 0.05:
        for i in range(n - 1):
            w[i] = (Data[i + 1] - c * Data[i]) / (1 - c)  # 预白化序列
        return w
    elif c >= 0.05:
        times = 0
        while c >= 0.05:
            for i in range(n - 1):
                w[i] = (Data[i + 1] - c * Data[i]) / (1 - c)  # 预白化序列
            k, b = mk.sens_slope(w)
            y = np.zeros(n - 1)
            for i in range(n - 1):
                y[i] = w[i] - (i + 1) * k  # 去除趋势的预白化序列
            c0 = c
            c = pd.Series(y).autocorr(1)
            if times == 1:
                if np.abs(c - c0) < 0.01 and np.abs(k - k0) < 0.0001:
                    print(f'waring: 数据白化后仍然存在较高自相关，建议检查数据是否存在异常（c={c:.3f}）')
                    return w
            times = 1
            k0 = k
        return w

