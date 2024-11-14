from toolbar.wavelet import *
from toolbar.filter import *


dat = np.load("D:\PyFile\paper1\OLS35.npy")
# 滤波
filter = MovingAverageFilter(dat, filter_type="bandpass", filter_window=[2, 6])
dat = filter.filter()
# 小波分析
wavelet_analysis = WaveletAnalysis(dat, wavelet='Morlet', dt=1, detrend=False, normal=True, signal=.90, J=4)
wavelet_analysis.plot(unit="1")

