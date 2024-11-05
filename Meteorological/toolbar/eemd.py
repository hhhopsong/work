import numpy as np
import matplotlib.pyplot as plt

from PyEMD import EEMD

signal_init =  np.load("D:\PyFile\paper1\OLS35.npy")
t = np.arange(len(signal_init))

# 创建 EEMD 对象
eemd = EEMD(trials=200, noise_width=0.2)
# 对信号进行 EEMD分解
IMFs = eemd(signal_init)

# 可视化
plt.figure(figsize=(20, 15))
plt.subplot(len(IMFs) + 1, 1, 1)
plt.plot(signal_init, 'r')
plt.title("Raw Data")

for num, imf in enumerate(IMFs):
    plt.subplot(len(IMFs) + 1, 1, num + 2)
    plt.plot(imf)
    plt.title("IMF " + str(num + 1), fontsize=10)
# 增加第一排图和第二排图之间的垂直间距
plt.subplots_adjust(hspace=0.8, wspace=0.2)
plt.show()

