import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

class MovingAverageFilter:
    def __init__(self, filter_value, filter_type, filter_window, fill_nan=0.):
        """
        :param filter_value: 滤波值
        :param filter_type: 滤波器类型[lowpass highpass bandpass bandstop]
        :param filter_window: 滤波器窗口, 必须为奇数
        """
        self.filter_type = filter_type
        self.filter_value = filter_value
        self.filter_window = np.array(filter_window)
        self.fill_nan = fill_nan
        if self.filter_window[0] % 2 == 0:
            raise ValueError("滤波器窗口必须为奇数")

    def __str__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def __repr__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def calculation_section(self, data, time_window):
        temp = np.full((len(data)+1), self.fill_nan)
        temp[1:] = np.cumsum(data)
        return (temp[time_window:] - temp[:-time_window]) / time_window

    def filted(self):
        if self.filter_type == "lowpass":
            return self.lowpass()
        elif self.filter_type == "highpass":
            return self.highpass()
        elif self.filter_type == "bandpass":
            return self.bandpass()
        elif self.filter_type == "bandstop":
            return self.bandstop()
        else:
            raise ValueError("Filter type not supported")

    def lowpass(self):
        if self.filter_window.shape[0] != 1:
            raise ValueError("低通滤波器filter_window参数不应为区间")
        return self.calculation_section(self.filter_value, self.filter_window[0])


    def highpass(self):
        if self.filter_window.shape[0] != 1:
            raise ValueError("高通滤波器filter_window参数不应为区间")
        index = (self.filter_window[0] - 1) // 2
        return self.filter_value[index:-index] - self.lowpass()

    def bandpass(self):
        if self.filter_window.shape[0] != 2:
            raise ValueError("带通滤波器filter_window参数应为区间")
        if self.filter_window[1] % 2 == 0:
            raise ValueError("滤波器窗口必须为奇数")
        if self.filter_window[0] - self.filter_window[1] >= 0:
            raise ValueError("filter_window应为递增区间")
        index = (self.filter_window[1] - 1) // 2 - (self.filter_window[0] - 1) // 2
        return self.calculation_section(self.filter_value, self.filter_window[0])[index:-index] - self.calculation_section(self.filter_value, self.filter_window[1])

    def bandstop(self):
        if self.filter_window.shape[0] != 2:
            raise ValueError("带阻滤波器filter_window参数应为区间")
        if self.filter_window[1] % 2 == 0:
            raise ValueError("滤波器窗口必须为奇数")
        if self.filter_window[0] - self.filter_window[1] >= 0:
            raise ValueError("filter_window应为递增区间")
        index = (self.filter_window[1] - 1) // 2
        return self.filter_value[index:-index] - self.bandpass()

    def response(self):
        N = self.filter_window  # 滤波器窗口
        omega = np.linspace(0, 2 * np.pi, 10000)  # 样本频率密度
        period = 2 * np.pi / omega  # 周期
        if self.filter_type == "lowpass":
            numerator = np.sin(omega * N / 2)
            denominator = N * np.sin(omega / 2)
            magnitude = np.abs(numerator / denominator)
            phase = -omega * (N - 1) / 2
            return magnitude, phase, period
        elif self.filter_type == "highpass":
            numerator = np.sin(omega * N / 2)
            denominator = N * np.sin(omega / 2)
            lowpass_magnitude = np.abs(numerator / denominator)
            lowpass_phase = -omega * (N - 1) / 2
            magnitude = np.sqrt((1 - lowpass_magnitude * np.cos(lowpass_phase)) ** 2 +
                                    (lowpass_magnitude * np.sin(lowpass_phase)) ** 2)
            phase = np.arctan2((lowpass_magnitude * np.sin(lowpass_phase)),
                               (1 - lowpass_magnitude * np.cos(lowpass_phase)))
            return magnitude, phase, period




class LanczosFilter:
    def __init__(self, filter_value, filter_type, filter_window, cutoff):
        """
        :param filter_value: 滤波值
        :param filter_type: 滤波器类型[lowpass highpass bandpass bandstop]
        :param filter_window: 滤波器窗口, 必须为奇数
        :param cutoff: 截止周期
        """
        self.filter_type = filter_type
        self.filter_value = filter_value
        self.filter_window = filter_window
        self.cutoff = np.array(cutoff)
        if self.filter_window % 2 == 0:
            raise ValueError("滤波器窗口必须为奇数")

    def __str__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def __repr__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def calculation_section(self, data, time_window, cutoff):
        # 滤波权重计算 cutoff:截止频率
        cutoff = 1.0 / cutoff
        nwts = time_window
        w = np.zeros([nwts])
        n = nwts // 2
        w[n] = 2 * cutoff
        k = np.arange(1., n)
        sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
        firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
        w[n - 1:0:-1] = firstfactor * sigma
        w[n + 1:-1] = firstfactor * sigma
        # 滤波
        yf = np.convolve(data, w, mode='valid')
        return yf

    def filted(self):
        if self.filter_type == "lowpass":
            return self.lowpass()
        elif self.filter_type == "highpass":
            return self.highpass()
        elif self.filter_type == "bandpass":
            return self.bandpass()
        elif self.filter_type == "bandstop":
            return self.bandstop()
        else:
            raise ValueError("Filter type not supported")

    def lowpass(self):
        if self.cutoff.shape[0] != 1:
            raise ValueError("低通滤波器cutoff参数不应为区间")
        return self.calculation_section(self.filter_value, self.filter_window, self.cutoff)


    def highpass(self):
        if self.cutoff.shape[0] != 1:
            raise ValueError("高通滤波器cutoff参数不应为区间")
        index = self.filter_window // 2
        return self.filter_value[index:-index] - self.lowpass()

    def bandpass(self):
        if self.cutoff.shape[0] != 2:
            raise ValueError("带通滤波器cutoff参数应为区间")
        if self.cutoff[0] - self.cutoff[1] >= 0:
            raise ValueError("cutoff应为递增区间")
        return (self.calculation_section(self.filter_value, self.filter_window, self.cutoff[0])
                - self.calculation_section(self.filter_value, self.filter_window, self.cutoff[1]))

    def bandstop(self):
        if self.cutoff.shape[0] != 2:
            raise ValueError("带阻滤波器cutoff参数应为区间")
        if self.cutoff[0] - self.cutoff[1] >= 0:
            raise ValueError("cutoff应为递增区间")
        index = self.filter_window // 2
        return self.filter_value[index:-index] - self.bandpass()


class ButterworthFilter:
    def __init__(self, filter_value, filter_type, filter_window=3, cutoff=[]):
        """
        :param filter_value: 滤波值
        :param filter_type: 滤波器类型[lowpass highpass bandpass bandstop]
        :param filter_window: 滤波器阶数
        :param cutoff: 截止周期
        """

        self.filter_type = filter_type
        self.filter_value = filter_value
        self.filter_window = filter_window
        self.cutoff = np.array(cutoff)
        if self.filter_window % 2 == 0:
            raise ValueError("滤波器窗口必须为奇数")

    def __str__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def __repr__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def calculation_section(self, cutoff):
        # 归一化截止频率计算
        cutoff = 2 / cutoff
        return cutoff

    def filted(self):
        data = self.filter_value
        N = self.filter_window
        if self.filter_type == "lowpass":
            Wn = self.calculation_section(self.cutoff)
            b, a = signal.butter(N, Wn, btype='lowpass')
            filted_series = signal.filtfilt(b, a, data)
            return filted_series
        elif self.filter_type == "highpass":
            Wn = self.calculation_section(self.cutoff)
            b, a = signal.butter(N, Wn, btype='highpass')
            filted_series = signal.filtfilt(b, a, data)
            return filted_series
        elif self.filter_type == "bandpass":
            if self.cutoff.shape[0] != 2:
                raise ValueError("带通滤波器cutoff参数应为区间")
            if self.cutoff[0] - self.cutoff[1] >= 0:
                raise ValueError("cutoff应为递增区间")
            Wn = [0, 0]
            Wn[0] = self.calculation_section(self.cutoff[1])
            Wn[1] = self.calculation_section(self.cutoff[0])
            b, a = signal.butter(N, Wn, btype='bandpass')
            filted_series = signal.filtfilt(b, a, data)
            return filted_series
        elif self.filter_type == "bandstop":
            if self.cutoff.shape[0] != 2:
                raise ValueError("带通滤波器cutoff参数应为区间")
            if self.cutoff[0] - self.cutoff[1] >= 0:
                raise ValueError("cutoff应为递增区间")
            Wn = [0, 0]
            Wn[0] = self.calculation_section(self.cutoff[1])
            Wn[1] = self.calculation_section(self.cutoff[0])
            b, a = signal.butter(N, Wn, btype='bandstop')
            filted_series = signal.filtfilt(b, a, data)
            return filted_series
        else:
            raise ValueError("Filter type not supported")

    def response(self):
        N = self.filter_window
        if self.filter_type == "lowpass":
            Wn = self.calculation_section(self.cutoff)
            b, a = signal.butter(N, Wn, btype='lowpass')
            w, h = signal.freqz(b, a, 1000)
            return w, h
        elif self.filter_type == "highpass":
            Wn = self.calculation_section(self.cutoff)
            b, a = signal.butter(N, Wn, btype='highpass')
            w, h = signal.freqz(b, a, 1000)
            return w, h
        elif self.filter_type == "bandpass":
            if self.cutoff.shape[0] != 2:
                raise ValueError("带通滤波器cutoff参数应为区间")
            if self.cutoff[0] - self.cutoff[1] >= 0:
                raise ValueError("cutoff应为递增区间")
            Wn = [0, 0]
            Wn[0] = self.calculation_section(self.cutoff[1])
            Wn[1] = self.calculation_section(self.cutoff[0])
            b, a = signal.butter(N, Wn, btype='bandpass')
            w, h = signal.freqz(b, a, 1000)
            return w, h
        elif self.filter_type == "bandstop":
            if self.cutoff.shape[0] != 2:
                raise ValueError("带通滤波器cutoff参数应为区间")
            if self.cutoff[0] - self.cutoff[1] >= 0:
                raise ValueError("cutoff应为递增区间")
            Wn = [0, 0]
            Wn[0] = self.calculation_section(self.cutoff[1])
            Wn[1] = self.calculation_section(self.cutoff[0])
            b, a = signal.butter(N, Wn, btype='bandstop')
            w, h = signal.freqz(b, a, 1000)
            return w, h
        else:
            raise ValueError("Filter type not supported")

    def plot_response(self):
        w, h = self.response()
        # 将角频率w转换为周期period
        period = 2 * np.pi / w
        magnitude = np.abs(h)

        # 限制周期范围，防止显示问题
        valid = period < len(self.filter_value)  # 仅显示周期 < 100 年的部分

        plt.plot(period[valid], magnitude[valid])
        plt.xlabel('Period')
        plt.xlim(0, len(self.filter_value))  # 限制周期范围为 0 到 100 年
        plt.ylabel('Magnitude')
        plt.ylim(0, 1.2)
        plt.title('Frequency Response (Filter)')
        plt.grid(True)
        plt.show()
