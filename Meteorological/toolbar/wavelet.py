import numpy as np
from scipy.stats.distributions import chi2
from matplotlib import pyplot as plt

import pycwt as wavelet
from pycwt.helpers import find


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
        self.dj = 1 / dj
        self.J = J / self.dj
        self.normal = normal
        self.detrend = detrend

        self.period = None
        self.power = None
        self.scales = None
        self.mother = None
        self.alpha = None
        self.global_power = None

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

    def red_noise(self):
        """计算红噪声"""
        dof = self.data.size - 200
        S = np.mean(self.power) / dof
        x2r = chi2.ppf(1 - 0.95, df=dof)
        t = np.arange(0, self.data.size + 1)
        Sr = [S * (1 - self.alpha**2) / (1 + self.alpha**2 - 2 * self.alpha * np.cos(t[i] * np.pi / self.scales)) for i in range(len(t))] * x2r / dof
        return Sr


    def wavelet_analysis(self):
        """小波分析

        Returns:
            小波分析结果:
            period (numpy.array): 周期

            power (numpy.array): 功率谱

            dt (float): 时间间隔

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
        # 计算相位谱
        fft_power = np.abs(fft) ** 2
        self.period = 1 / freqs
        self.power /= self.scales[:, None]
        signif, fft_theor = wavelet.significance(1.0, self.dt, self.scales, 0, self.alpha,
                                                 significance_level=self.signal, wavelet=self.mother)
        # 计算显著性水平
        sig = np.ones([1, data.size]) * signif[:, None]
        sig = self.power / sig
        # 计算全局功率谱
        self.global_power = self.power.mean(axis=1)
        dof = data.size - self.scales
        global_signif, red_noise = wavelet.significance(self.var, self.dt, self.scales, 1, self.alpha,
                                                  significance_level=self.signal, dof=dof, wavelet=self.mother)
        return (self.period, self.power, self.dt, self.mother, iwave,
                sig, coi, self.global_power, global_signif, fft_power, fft_freqs, fft_theor, red_noise)

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
        sel = find((self.period >= start) & (self.period < end))
        Cdelta = self.mother.cdelta
        scale_avg = (self.scales * np.ones((data.size, 1))).transpose()
        scale_avg = self.power / scale_avg
        scale_avg = self.var * self.dj * self.dt / Cdelta * scale_avg[sel, :].sum(axis=0)
        scale_avg_signif, red_noise = wavelet.significance(self.var, self.dt, self.scales, 2, self.alpha, significance_level=self.signal,
                                                     dof=[self.scales[sel[0]], self.scales[sel[-1]]], wavelet=self.mother)
        return scale_avg_signif, scale_avg, red_noise

    def plot(self):
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
        var = self.var
        period, power, dt, mother, iwave, sig, coi, glbl_power, glbl_signif, fft_power, fftfreqs, fft_theor, red_noise = self.wavelet_analysis()

        # 第一个子图，原始时间序列异常和逆小波变换
        ax = plt.axes([0.1, 0.75, 0.65, 0.2])
        # ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
        ax.plot(t, data, 'k', linewidth=1.5)
        ax.set_title('a) {}'.format('Raw data'))
        ax.set_ylabel(r'[{}]'.format('℃'))

        # 第二个子图，归一化小波功率谱和显著性水平等值线和影响阴影区域的圆锥体。请注意，周期刻度是对数的。
        bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        bx.contourf(t, np.log2(period), np.log2(power), level=np.log2(levels),
                    extend='both', cmap=plt.cm.viridis)
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
        cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
        cx.plot(self.red_noise(), np.log2(period), '--', color='red')
        cx.plot(var * fft_theor, np.log2(period), '--', color='gray')
        # cx.plot(var * fft_power, np.log2(1. / fftfreqs), '-', color='#cccccc', linewidth=1.)
        cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
        cx.set_title('c) Global Wavelet Spectrum')
        cx.set_xlabel(r'Power [({})^2]'.format('℃'))
        cx.set_xlim([0, glbl_power.max() + var])
        cx.set_ylim(np.log2([period.min(), period.max()]))
        cx.set_yticks(np.log2(Yticks))
        cx.set_yticklabels(Yticks)
        plt.setp(cx.get_yticklabels(), visible=False)

        # 第四个子图，比例平均小波谱。
        scale_avg_signif, scale_avg, _ = self.find_periods_power(2, 8)
        dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
        dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
        dx.plot(t, scale_avg, 'k-', linewidth=1.5)
        dx.set_title('d) {}--{} year scale-averaged power'.format(2, 8))
        dx.set_xlabel('Time (year)')
        dx.set_ylabel(r'Average variance [{}]'.format("℃"))
        ax.set_xlim([t.min(), t.max()])

        plt.show()


if __name__ == '__main__':
    """测试"""
    # 获取数据
    url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
    dat = np.genfromtxt(url, skip_header=19)
    # 小波分析
    wavelet_analysis = WaveletAnalysis(dat, dt=0.25, detrend=False, normal=True, signal=.95)
    wavelet_analysis.plot()