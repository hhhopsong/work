import cmaps
import numpy as np
from scipy.stats.distributions import chi2
from matplotlib import pyplot as plt

import pycwt as wavelet
from pycwt.helpers import find, rednoise


class WaveletAnalysis:
    """小波分析类, 用于时间序列数据(默认时间间隔均匀)"""
    def __init__(self, data, dt, wavelet='Morlet', signal=0.95, s0=2, dj=12, J=7, normal=True, detrend=False):
        """

        Args:
            data (numpy.array): 一维时间序列数据(时间间隔均匀)
            dt (float): 时间间隔
            wavelet (str): 小波分析函数名称[Morlet Paul DOG MexicanHat]
            signal (float): 显著性水平, 默认0.95
            s0 (float):  最小时间尺度,以时间间隔 dt 为单位
            dj (float): 小波尺度步进 Twelve sub-octaves per octaves
            J (float): 小波尺度阶数 Seven powers of two with dj sub-octaves
            normal (bool): 对数据进行标准化处理, 默认 True
            detrend (bool): 是否去趋势, 默认 False
        """
        self.data = data
        self.var = np.std(self.data) ** 2
        self.std = np.std(self.data)
        self.dt = dt
        self.wavelet = wavelet
        self.signal = signal
        self.s0 = s0 * dt
        self.J = J * dj
        self.dj = 1 / dj
        self.normal = normal
        self.detrend = detrend

        self.period = None
        self.power = None
        self.scales = None
        self.mother = None
        self.global_power = None
        self.alpha = None
        self.wavelet_analysis()

    def detrended(self):
        """去趋势处理"""
        data = (self.data - np.polyfit(np.arange(len(self.data)), self.data, 1)[0] * np.arange(len(self.data))
                     - np.polyfit(np.arange(len(self.data)), self.data, 1)[1])
        return data

    def normalize(self):
        """标准化处理"""
        data = (self.data - np.mean(self.data)) / self.std
        return data

    def noise(self, alpha_red=0.05, alpha_white=0.95):
        """计算噪声 错误

        Returns:
            S_red (numpy.array): 红噪声功率谱

            S_white (numpy.array): 白噪声功率谱
        """
        n = self.data.size
        data = (self.data - np.mean(self.data)) / np.std(self.data)
        r1 = np.zeros((n - 6))
        r2 = np.zeros((n - 7))
        for i in np.arange(0, n - 6):
            r1[i] = np.sum(data[:n - i] * data[i:]) / data[:n - i].shape[0]
        for i in np.arange(1, n - 6):
            r2[i - 1] = np.sum(data[:n - i] * data[i:]) / data[:n - i].shape[0]
        r2 = r2[::-1]
        r = np.hstack((r2, r1))
        l = np.arange(0, self.J + 1, 1)
        tao = np.arange(1, self.J, 1)
        Sl = np.zeros((self.J + 1))
        Tl = np.zeros((self.J + 1))
        S0l = np.zeros((self.J + 1))
        a = np.array((r.shape[0] + 1) / 2).astype('int32')
        r = r[a - 1:a + self.J]
        a = r[1:-1] * (1 + np.cos(np.pi * tao / self.J))
        for i in np.arange(2, self.J + 1, 1):
            Sl[i - 1] = (r[0] + np.sum(a * np.cos(l[i - 1] * np.pi * tao / self.J))) / self.J
        Sl[0] = (r[0] + np.sum(a * np.cos(l[0] * np.pi * tao / self.J))) / (2 * self.J)
        Sl[-1] = (r[0] + np.sum(a * np.cos(l[-1] * np.pi * tao / self.J))) / (2 * self.J)
        for i in range(l.shape[0]):
            Tl[i] = 2 * self.J / l[i]
        f = (2 * n - self.J / 2) / self.J
        S = np.mean(Sl)
        for i in range(l.shape[0]):
            S0l[i] = S * (1 - r[1] * r[1]) / (1 + r[1] * r[1] - 2 * r[1] * np.cos(l[i] * np.pi / self.J))
        S_red = S0l / f
        x2r_high = chi2.ppf(1 - alpha_red, df=f)
        S_red_high = S0l * x2r_high / f
        x2r_low = chi2.ppf(alpha_red, df=f)
        S_red_low = S0l * x2r_low / f
        x2w = chi2.ppf(1 - alpha_white, df=f)
        S_white = S * x2w / f
        r1 = r[1]
        return np.array([S_red, S_red_low, S_red_high]), S_white

    def wavelet_analysis(self):
        """小波分析

        Returns:
            小波分析结果:
            period (numpy.array): 周期

            power (numpy.array): 功率谱

            dt (float): 数据的时间间隔(年)

            mother (pycwt.Morlet): 小波基函数

            sig (numpy.array): 显著性水平

            coi (numpy.array): 中心频率

            global_power (numpy.array): 全局功率谱

            global_signif (numpy.array): 全局显著性水平

            fft_power (numpy.array): 傅里叶功率谱

            fftfreqs (numpy.array): 傅里叶频率

            fft_theor (numpy.array): 傅里叶理论功率谱
        """
        data = self.data
        if self.detrend:
            # 去趋势处理
            data = self.detrended()
        if self.normal:
            # 标准化处理
            data = self.normalize()
        if self.wavelet == 'Morlet':
            self.mother = wavelet.Morlet()
        elif self.wavelet == 'Paul':
            self.mother = wavelet.Paul()
        elif self.wavelet == 'DOG':
            self.mother = wavelet.DOG()
        elif self.wavelet == 'MexicanHat':
            self.mother = wavelet.MexicanHat()
        else:
            raise ValueError("不支持的基函数。")
        self.alpha , _, _ = wavelet.ar1(data)  # 红噪声的一阶滞后自相关 Lag-1 autocorrelation for red noise
        # 计算小波系数
        wave, self.scales, freqs, coi, fft, fft_freqs = wavelet.cwt(data, self.dt, self.dj, self.s0, self.J, self.mother)
        # 计算逆小波系数
        iwave = wavelet.icwt(wave, self.scales, self.dt, self.dj, self.mother) * self.std
        # 计算功率谱
        self.power = (np.abs(wave)) ** 2
        # 计算显著性水平
        signif, fft_theor = wavelet.significance(1.0, self.dt, self.scales, 0, self.alpha,
                                                 significance_level=self.signal, wavelet=self.mother)
        sig = np.ones([1, data.size]) * signif[:, None]
        sig = self.power / sig

        fft_power = np.abs(fft) ** 2
        self.period = 1 / freqs
        #self.power /= self.scales[:, None]
        # 计算全局功率谱
        self.global_power = self.power.mean(axis=1)
        dof = data.size - self.scales
        global_signif, tmp = wavelet.significance(self.var, self.dt, self.scales, 1, self.alpha,
                                                  significance_level=self.signal, dof=dof, wavelet=self.mother)
        return (self.period, self.power, self.dt, self.mother, iwave,
                sig, coi, self.global_power, global_signif, fft_power, fft_freqs, fft_theor)

    def find_periods_power(self, start=2, end=8):
        """计算限定周期范围波动的功率谱"""
        data = self.data
        if self.detrend:
            # 去趋势处理
            data = self.detrended()
        if self.normal:
            # 标准化处理
            data = self.normalize()
        if self.period is None:
            raise ValueError("请先运行 wavelet_analysis 函数。")
        if start < np.min(self.period) and end < np.min(self.period):
            if start > np.max(self.period) and end > np.max(self.period):
                raise ValueError(f"周期范围不在有效范围内({np.min(self.period)} - {np.max(self.period)})。")
        sel = find((self.period >= start) & (self.period < end))
        Cdelta = self.mother.cdelta
        scale_avg = (self.scales * np.ones((data.size, 1))).transpose()
        scale_avg = self.power / scale_avg
        scale_avg = self.var * self.dj * self.dt / Cdelta * scale_avg[sel, :].sum(axis=0)
        scale_avg_signif, tmp = wavelet.significance(self.var, self.dt, self.scales, 2, self.alpha, significance_level=self.signal,
                                                     dof=[self.scales[sel[0]], self.scales[sel[-1]]], wavelet=self.mother)
        return scale_avg_signif, scale_avg

    def plot(self, unit="%", start_year=1961):
        """绘制小波分析结果"""
        data = self.data
        if self.detrend:
            # 去趋势处理
            data = self.detrended()
        if self.normal:
            # 标准化处理
            data = self.normalize()
        plt.close('all')
        plt.ioff()
        figprops = dict(figsize=(11, 8), dpi=72)
        fig = plt.figure(**figprops)
        t = np.arange(0, data.size) * self.dt
        period, power, dt, mother, iwave, sig, coi, glbl_power, glbl_signif, fft_power, fft_freqs, fft_theor= self.wavelet_analysis()

        # 第一个子图，原始时间序列异常和逆小波变换
        ax = plt.axes([0.1, 0.75, 0.65, 0.2])
        # ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.)
        ax.plot(t, data, 'k', linewidth=1.5)
        ax.set_title('a) {}'.format('Raw data'))
        ax.set_ylabel(r'[{}]'.format(unit))
        ax.set_xlim([t.min(), t.max()])
        ax.set_xticklabels(np.array([0, 10, 20, 30, 40, 50, 60]) + start_year)

        # 第二个子图，归一化小波功率谱和显著性水平等值线和影响阴影区域的圆锥体。请注意，周期刻度是对数的。
        bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
        levels = [0, 0.0125, 0.25, 0.5, 1, 2, 4]
        bx_fill = bx.contourf(t, np.log2(period), power, level=levels,
                    extend='both', cmap=cmaps.sunshine_9lev)
        extent = [t.min(), t.max(), 0, max(period)]
        bx.contour(t, np.log2(period), sig, [-99, 1], colors='k', linewidths=2,
                   extent=extent)
        bx.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                                   t[:1] - dt, t[:1] - dt]),
                np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                                   np.log2(period[-1:]), [1e-9]]),
                'k', alpha=0.3, hatch='x')
        bx.set_title('b) Wavelet Power Spectrum ({})'.format(mother.name))
        bx.set_ylabel('Period (years)')
        #
        Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                                   np.ceil(np.log2(period.max())))
        bx.set_yticks(np.log2(Yticks))
        bx.set_yticklabels(Yticks)

        # 第三个子图，全局小波和傅里叶功率谱以及理论噪声谱。请注意，周期刻度是对数的。
        var = self.var
        cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
        #cx.plot(var * fft_theor, np.log2(period), '--', color='red')
        cx.plot(var * glbl_power, np.log2(period), '-', color='#cccccc', linewidth=1.5)
        cx.plot(var * fft_power, np.log2(1. / fft_freqs), '-', color='k', linewidth=1)
        cx.plot(glbl_signif, np.log2(period), ':', color='red', linewidth=1.5)
        cx.set_title('c) Global Wavelet Spectrum')
        cx.set_xlabel(r'Power [({})^2]'.format(unit))
        cx.set_xlim([0, var * np.nanmax(fft_power)])
        cx.set_ylim(np.log2([period.min(), period.max()]))
        cx.set_yticks(np.log2(Yticks))
        cx.set_yticklabels(Yticks)
        plt.setp(cx.get_yticklabels(), visible=False)

        # 第四个子图，比例平均小波谱。
        scale_avg_signif, scale_avg= self.find_periods_power(2, 3.5)
        dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
        dx.axhline(scale_avg_signif, color='red', linestyle=':', linewidth=1.5)
        dx.plot(t, scale_avg, 'k-', linewidth=1.5)
        dx.set_title('d) {}-{} year scale-averaged power'.format(3, 4))
        dx.set_xlabel('Time (year)')
        dx.set_ylabel(r'Average variance [{}]'.format(unit))

        plt.show()


if __name__ == '__main__':
    """测试"""
    # 获取数据
    url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
    dat = np.genfromtxt(url, skip_header=19)
    dat = np.load("D:\PyFile\paper1\OLS35.npy")
    # 小波分析
    wavelet_analysis = WaveletAnalysis(dat, dt=1, detrend=False, normal=True, signal=.95, J=4)
    wavelet_analysis.plot(unit="1")