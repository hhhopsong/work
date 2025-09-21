import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

##powered by Gemini 2.5 Pro
def mdetrend(series, window_size):
    """
    对一个时间序列使用滑动窗口进行去趋势。
    对于每个点 t，使用 t-window_size 到 t-1 的数据来拟合趋势，并外推到 t。

    Args:
        series (pd.Series): 输入的时间序列数据。
        window_size (int): 滚动的窗口大小。

    Returns:
        pd.Series: 去趋势后的序列
        pd.Series: 趋势值序列
    """
    trend_values = np.zeros_like(series)

    for i in range(len(series)):
        # 确定滑动窗口的起始点
        if i >= window_size >= 2:
            window = series[i-window_size:i]
            # 准备回归数据
            X_train_window = np.arange(len(window)).reshape(-1, 1)
            y_train_window = window

            # 拟合局部线性趋势
            lr = LinearRegression()
            lr.fit(X_train_window, y_train_window)

            # 预测当前时间点的趋势值
            current_year = len(window)
            trend_values[i] = lr.predict([[current_year]])[0]
        elif window_size == 0:
            trend_values[i] = 0
        elif window_size == 1:
            raise ValueError("window_size must be greater than 1 or equal to 0.")
        else:
            trend_values[i] = np.nan
    detrended_series = series - trend_values
    return detrended_series, trend_values

if __name__ == '__main__':
    url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
    dat = np.genfromtxt(url, skip_header=19)
    years = np.arange(len(dat))
    dat = pd.Series(dat, index=years)
    detrended_data, trend = mdetrend(dat, window_size=0)
    print(dat, detrended_data)
    print("Successful install move_detrend!")


