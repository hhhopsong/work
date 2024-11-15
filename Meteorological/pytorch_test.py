from toolbar.filter import LanczosFilter
from toolbar.wavelet import *
from toolbar.filter import *


dat = np.load("D:\PyFile\paper1\OLS35.npy")
# 滤波
'''filter = LanczosFilter(dat, filter_type="bandpass", filter_window=9, cutoff=[2, 6])
dat = filter.filted()'''
# 小波分析
wavelet_analysis = WaveletAnalysis(dat, wave='Morlet', dt=1, detrend=False, normal=True, signal=.90, J=5)
wavelet_analysis.plot(unit="1")

