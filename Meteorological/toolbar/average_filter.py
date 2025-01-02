import numpy as np
from scipy.ndimage import generic_filter

def nanmean_filter(data, size):
    """
    对包含 NaN 的二维数组进行均值平滑，忽略 NaN 值。

    参数：
        data: 2D ndarray, 输入数据（包含 NaN 值）
        size: int, 滑动窗口的大小（如 3 表示 3x3 窗口）

    返回：
        result: 2D ndarray, 平滑后的结果
    """
    def nanmean_function(values):
        # 忽略 NaN 值计算均值
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            return np.mean(valid_values)
        else:
            return np.nan  # 如果全是 NaN，返回 NaN

    # 使用滑动窗口进行平滑
    result = generic_filter(data, nanmean_function, size=size, mode='constant', cval=np.nan)
    return result

if __name__ == "__main__":
    # 示例数据
    data = np.array([[1, 2, np.nan],
                     [4, np.nan, 6],
                     [7, 8, 9]])

    # 3x3 窗口大小
    window_size = 3

    # 平滑处理
    smoothed_data = nanmean_filter(data, size=window_size)

    print("原始数据：\n", data)
    print("平滑后的数据：\n", smoothed_data)