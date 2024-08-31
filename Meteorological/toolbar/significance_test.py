import numpy as np
from scipy.stats import t


def t_test(n, pc, data, p=0.95):
    """
    :param n: 自由度
    :param pc: 时间序列
    :param data: 数据
    :param p: 置信度
    :return: 显著性检验结果
    """
    Lxx = np.sum((pc - np.mean(pc)) ** 2)
    Sr = data ** 2 * Lxx
    St = np.sum((data - np.mean(data, axis=0)) ** 2, axis=0)
    sigma = np.sqrt((St - Sr) / (n - 2))
    t_value = data * np.sqrt(Lxx) / sigma
    t_critical = t.ppf(p, n - 2)
    # 进行显著性检验
    test_results = np.zeros((len(data['lat']), len(data['lon'])))
    test_results.fill(np.nan)
    test_results[np.abs(t_value.to_numpy()) > t_critical] = 1
    return test_results
