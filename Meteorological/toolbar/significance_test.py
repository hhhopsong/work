import numpy as np
from scipy.stats import t


def ols_test(pc, data, alpha=0.05):
    """
    :param pc: 时间序列
    :param data: 线性回归数据(lat, lon)
    :param alpha: 显著性水平
    :return: 显著性检验结果
    """
    n = len(pc)
    Lxx = np.sum((pc - np.mean(pc)) ** 2)
    Sr = data ** 2 * Lxx
    St = np.sum((data - np.mean(data, axis=0)) ** 2, axis=0)
    sigma = np.sqrt((St - Sr) / (n - 2))
    t_value = data * np.sqrt(Lxx) / sigma
    t_critical = t.ppf(1 - (alpha / 2), n - 2)
    # 进行显著性检验
    test_results = np.zeros(data.shape)
    test_results.fill(np.nan)
    test_results[np.abs(t_value.to_numpy()) > t_critical] = 1
    return test_results


def corr_test(pc, data, alpha=0.05):
    """
    :param pc: 时间序列
    :param data: np.array 相关系数数据
    :param alpha: 显著性水平
    :return: 显著性检验结果
    """
    n = len(pc)
    t_critical = t.ppf(alpha / 2, n - 2)  # 双边t检验
    r_critical = np.sqrt(t_critical**2 / (t_critical**2 + n - 2))
    # 进行显著性检验
    test_results = np.zeros(data.shape)
    test_results.fill(np.nan)
    test_results[np.abs(data) > r_critical] = 1
    return test_results
