import numpy as np


class MovingAverageFilter:
    def __init__(self, filter_value, filter_type, filter_window):
        """
        :param filter_value: 滤波值
        :param filter_type: 滤波器类型[lowpass highpass bandpass bandstop]
        :param filter_window: 滤波器窗口
        """
        self.filter_type = filter_type
        self.filter_value = filter_value
        self.filter_window = np.array(filter_window)
        if self.filter_window[0] - self.filter_window[1] < 0:
            raise ValueError("filter_window应为递增区间")

    def __str__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def __repr__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def calculation_section(self, data, time_window):
        temp = np.zeros(len(data)+1)
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
        return self.calculation_section(self.filter_value, self.filter_window)


    def highpass(self):
        if self.filter_window.shape[0] != 1:
            raise ValueError("高通滤波器filter_window参数不应为区间")
        return self.filter_value - self.lowpass()

    def bandpass(self):
        if self.filter_window.shape[0] != 2:
            raise ValueError("带通滤波器filter_window参数应为区间")
        return self.calculation_section(self.filter_value, self.filter_window[1]) - self.calculation_section(self.filter_value, self.filter_window[0])

    def bandstop(self):
        if self.filter_window.shape[0] != 2:
            raise ValueError("带阻滤波器filter_window参数应为区间")
        return self.filter_value - self.bandpass()
