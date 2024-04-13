import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from cartopy import crs as ccrs
from numpy import ma
import pprint


def linear_reg(X, Y):
    X = np.array(X).tolist()
    Y = np.array(Y).tolist()
    x_ = np.mean(X)
    y_ = np.mean(Y)
    b = np.sum((X - x_)*(Y - y_))/np.sum((X - x_)**2)
    a = y_ - b*x_
    return a, b


def moving_average(X, n=5):
    ret = np.cumsum(X, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def cum_anomaly(X):
    X_ = np.mean(X)
    X = np.array(X)
    X_ano = np.cumsum(X - X_, dtype=float)
    return X_ano


# 读取数据
ds = xr.open_dataset(r"D:\2021tj\HADISST_50y.nc")
sst_all = ds['sst']
lon_all = ds['lon']
lat_all = ds['lat']
SST = ds.loc[{'lon': slice(35, 125), 'lat': slice(-20, 20)}]
lon = SST['lon']
lat = SST['lat']
sst = SST['sst']
sst_m = ma.masked_values(sst, -999)
sst_m_all = ma.masked_values(sst_all, -999)
# 平均海温时间序列
ISST = np.mean(sst_m, axis=(1, 2))
# #线性倾向估计
a1, b1 = linear_reg([i+1 for i in range(len(ISST))], ISST)
y_linear = [a1 + b1*i for i in range(len(ISST))]
# #移动平均
y_moving = moving_average(ISST, 5)
# #累积距平
y_cum = cum_anomaly(ISST)
# ##近50年冬季全球海温变化趋势（线性倾向估计）
sst_m_all = np.mean(sst_m_all, axis=(1, 2))
# ##线性倾向估计
a2, b2 = linear_reg([i+1 for i in range(len(sst_m_all))], sst_m_all)
y_linear_all = [a2 + b2*i for i in range(len(sst_m_all))]
# 绘图
# 地图要素
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
year_begin = 1958
# 绘制线性倾向估计
ax1 = fig.add_subplot(221)
ax1.plot([i+year_begin for i in range(len(ISST))], ISST, label='热带印度洋平均海温时间序列')
ax1.plot([i+year_begin for i in range(len(ISST))], y_linear, label='线性倾向估计')
ax1.set_title('热带印度洋海温变化趋势（线性倾向估计）')
ax1.set_xlabel('年份')
ax1.set_ylabel('平均海温')
ax1.legend()
# 绘制滑动平均
ax2 = fig.add_subplot(222)
ax2.plot([i+year_begin for i in range(len(ISST))], ISST, label='热带印度洋平均海温时间序列')
ax2.plot([i+year_begin+2 for i in range(len(ISST)-4)], y_moving, label='滑动平均')
ax2.set_title('热带印度洋海温变化趋势（滑动平均）')
ax2.set_xlabel('年份')
ax2.set_ylabel('平均海温')
ax2.legend()
# 绘制累积距平
ax3 = fig.add_subplot(223)
ax32 = ax3.twinx()
ins1 = ax3.plot([i+year_begin for i in range(len(ISST))], ISST, label='热带印度洋平均海温时间序列')
ins2 = ax32.plot([i+year_begin for i in range(len(ISST))], y_cum, label='累积距平', color='orange')
ax3.set_title('热带印度洋海温变化趋势（累积距平）')
ax3.set_xlabel('年份')
ax3.set_ylabel('平均海温')
ax32.set_ylabel('累积距平海温')
ax3.legend(ins1+ins2, [i.get_label() for i in ins1+ins2])
# 绘制近50年冬季全球海温变化趋势（线性倾向估计）
ax4 = fig.add_subplot(224)
ax4.plot([i+year_begin for i in range(len(sst_m_all))], sst_m_all, label='全球平均海温时间序列')
ax4.plot([i+year_begin for i in range(len(sst_m_all))], y_linear_all, label='线性倾向估计')
ax4.set_title('近50年冬季全球海温变化趋势（线性倾向估计）')
ax4.set_xlabel('年份')
ax4.set_ylabel('平均海温')
ax4.legend()
plt.savefig(r'D:\2021tj\海温变化趋势.png', dpi=1000, bbox_inches='tight')
plt.show()
