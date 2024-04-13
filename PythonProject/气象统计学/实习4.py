import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from cartopy import crs as ccrs
from numpy import ma


def b(X, y=None):
    Xt = X.T
    XtX = np.dot(Xt, X)
    B = np.dot(np.dot(np.linalg.inv(XtX), Xt), y)
    return B


# 地图要素
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据读取
H200 = xr.open_dataset(r"D:/2021tj/NCEP_Z200_30y_Wt.nc")['hgt']
Sea_tp = ma.masked_values(xr.open_dataset(r"D:/2021tj/NCEP_TPSST_30y_Wt.nc")['st'], 1.000e33)
Sea_id = ma.masked_values(xr.open_dataset(r"D:/2021tj/NCEP_IOSST_30y_Wt.nc")['st'], 1.000e33)
# 数据处理
Nino_ave = np.zeros(len(Sea_tp))    # 逐年Nino区平均海温
Iob_ave = np.zeros(len(Sea_id))     # 逐年热带印度洋区域平均海温
for i in range(len(Sea_tp)):
    Nino_ave[i] = np.mean(Sea_tp[i])
    Iob_ave[i] = np.mean(Sea_id[i])
X = np.zeros((len(Nino_ave), 3))    # 生成因子矩阵
X[:, 0] = [1]*len(Nino_ave)        # 常数项
X[:, 1] = Nino_ave           # Nino区平均海温
X[:, 2] = Iob_ave       # 热带印度洋区域平均海温
B = np.zeros((H200.shape[1], H200.shape[2], 3))    # 回归系数
for i in range(H200.shape[1]):
    for j in range(H200.shape[2]):
        B[i, j] = b(X, H200[:, i, j])
H_time_list = np.zeros((H200.shape[1], H200.shape[2], 3))    # 生成时间序列
for i in range(H200.shape[1]):
    for ii in range(H200.shape[2]):
        H_time_list[i, ii] = b(X, H200[:, i, ii])
R = np.zeros((H200.shape[1], H200.shape[2]))    # 生成复相关系数
for i in range(H200.shape[1]):
    for ii in range(H200.shape[2]):
        y_pre = np.dot(X, B[i, ii])    # 预测值
        y_ave = np.mean(H200[:, i, ii])  # 平均值
        y_ave = np.array([y_ave]*len(H200[:, i, ii]))  # 平均值序列
        yi = H200[:, i, ii]    # 实测值
        yd_pre = y_pre - y_ave    # 预测值与平均值的差
        yd = yi - y_ave    # 实测值与平均值的差
        R[i, ii] = np.dot(yd_pre, yd) / (np.sqrt(np.dot(yd_pre, yd_pre)) * np.sqrt(np.dot(yd, yd)))
# 数据可视化
fig = plt.figure(figsize=(16, 9))
levels = 10
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_title(f'200hPa 与 Nino区海温和热带印度洋区域海温 复相关系数场', color='black', fontsize=20)
a = ax.contourf(H200['lon'], H200['lat'], R, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both')
ax.coastlines()
b = ax.contour(H200['lon'], H200['lat'], R, levels=levels, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b, inline=True, fontsize=5, fmt='%.01f', colors='black')
fig.colorbar(orientation='horizontal', shrink=0.8, pad=0.05, aspect=50, mappable=a)
plt.savefig(f'D:/2021tj/200hPa 与 Nino区海温和热带印度洋区域海温 复相关系数场.png', dpi=1500, bbox_inches='tight')
plt.show()
