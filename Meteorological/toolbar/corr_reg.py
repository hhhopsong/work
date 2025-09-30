import numpy as np

def corr(time_series, data):
    # 计算相关系数
    # 将 data 重塑为二维：时间轴为第一个维度
    reshaped_data = data.reshape(len(time_series), -1)

    # 减去均值以标准化
    time_series_mean = time_series - np.mean(time_series)
    data_mean = reshaped_data - np.mean(reshaped_data, axis=0)

    # 计算分子（协方差）
    numerator = np.sum(data_mean * time_series_mean[:, np.newaxis], axis=0)

    # 计算分母（标准差乘积）
    denominator = np.sqrt(np.sum(data_mean ** 2, axis=0)) * np.sqrt(np.sum(time_series_mean ** 2))

    # 相关系数
    correlation = numerator / denominator

    # 重塑为 (lat, lon)
    correlation_map = correlation.reshape(data.shape[1:])
    return correlation_map

def regress(time_series, data):
    # 将 data 重塑为二维：时间轴为第一个维度
    reshaped_data = data.reshape(len(time_series), -1)

    # 减去均值以中心化（标准化自变量和因变量）
    time_series_mean = time_series - np.mean(time_series)
    data_mean = reshaped_data - np.mean(reshaped_data, axis=0)

    # 计算分子（协方差的分子）
    numerator = np.sum(data_mean * time_series_mean[:, np.newaxis], axis=0)

    # 计算分母（自变量的平方和）
    denominator = np.sum(time_series_mean ** 2)

    # 计算回归系数
    regression_coef = numerator / denominator

    # 重塑为 (lat, lon)
    regression_map = regression_coef.reshape(data.shape[1:])

    return regression_map

def cort(time_series, data):
    # 计算 CORT 相关系数
    # 将 data 重塑为二维：时间轴为第一个维度
    reshaped_data = data.reshape(len(time_series), -1)

    diff1 = np.diff(np.array(time_series))
    diff2 = np.diff(np.array(reshaped_data))
    # 计算分子（协方差）
    numerator = np.sum(diff2 * diff1[:, np.newaxis], axis=0)
    # 计算分母（标准差乘积）
    denominator = np.sqrt(np.sum(diff2 ** 2, axis=0)) * np.sqrt(np.sum(diff1 ** 2))
    # CORT 相关系数
    cort = numerator / denominator

    # 重塑为 (lat, lon)
    cort_map = cort.reshape(data.shape[1:])
    return cort_map

if __name__ == '__main__':
    print('Successfully import toolbar.corr_reg!')
