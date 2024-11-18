from toolbar.filter import LanczosFilter
from toolbar.wavelet import *
from toolbar.filter import *


dat = np.load("D:\PyFile\paper1\OLS35.npy")
# 滤波
filter = ButterworthFilter(dat, filter_type="highpass", filter_window=9, cutoff=[10])
dat = filter.filted()
filter.plot_response()
# 小波分析
wavelet_analysis = WaveletAnalysis(dat, wave='Morlet', dt=1, detrend=False, normal=True, signal=.90, J=5)
wavelet_analysis.plot(unit="1")

