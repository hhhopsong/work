from xgrads import open_CtlDataset
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from numpy import ma


def Gauss_dist(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

# 数据预处理
open_CtlDataset("D:/2021tj/NCEP_slp_30y_Wt.ctl").to_netcdf("D:/2021tj/NCEP_slp_30y_Wt.nc")
ds = xr.open_dataset("D:/2021tj/NCEP_slp_30y_Wt.nc")
lon = ds['lon']
lat = ds['lat']
time = ds['time']
slp = ma.masked_values(ds['slp'], 1.000e33)
ds.close()

# 地图要素设置
year = eval(input("请输入年份："))
YR = year - 1978
levels = np.arange(980, 1040, 5)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16, 9))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines()
ax.stock_img()
ax.set_title(f'{year}年全球海平面气压场', color='black', fontsize=20)
cbar_kwargs = {
    'orientation': 'vertical',
    'label': '2m temperature (℃)',
    'shrink': 0.8,
    'ticks': levels,
    'pad': 0.05}

# 绘制等值线
SLP = slp[YR]
a = ax.contourf(lon, lat, SLP, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both',
                cbar_kwargs=cbar_kwargs)
b = ax.contour(lon, lat, SLP, levels=levels, cbar_kwargs=cbar_kwargs, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b, inline=True, fontsize=10, fmt='%.0f', colors='black')
plt.savefig(f'D:/2021tj/{year}年全球海平面气压场.png', dpi=1500, bbox_inches='tight')
plt.clf()

# 计算气压场
Global_slp = slp
Global_slp_year_ave_num = np.zeros(len(Global_slp))  # 逐年平均值
Global_slp_ave_num = np.mean(Global_slp)  # 多年平均值
Global_slp_ave = np.mean(Global_slp, axis=0)  # 多年平均场
Global_slp_delta = np.zeros(len(Global_slp))  # 距平
for i in range(len(Global_slp)):
    Global_slp_year_ave_num[i] = np.mean(Global_slp[i])
    Global_slp_delta[i] = Global_slp_year_ave_num[i] - Global_slp_ave_num
Sx = np.sqrt((1 / len(Global_slp)) * np.sum(Global_slp_delta ** 2))  # 标准差
Global_slp_std = Global_slp_delta / Sx  # 标准化
StartYear = 1978
iYear = np.arange(StartYear, StartYear + len(Global_slp))

# 绘制气压场距平图
levels1 = np.arange(-16, 16, 2)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16, 9))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines()
ax.stock_img()
ax.set_title(f'{year}年全球海平面气压距平场', color='black', fontsize=20)
cbar_kwargs1 = {
    'orientation': 'vertical',
    'label': '2m temperature (℃)',
    'shrink': 0.8,
    'ticks': levels1,
    'pad': 0.05}
slp_delta = slp[YR] - Global_slp_ave
a_ave = ax.contourf(lon, lat, slp_delta, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels1, extend='both',
                    cbar_kwargs=cbar_kwargs1)
b_ave = ax.contour(lon, lat, slp_delta, levels=levels1, cbar_kwargs=cbar_kwargs1, linewidths=0.5,
                   transform=ccrs.PlateCarree())
plt.clabel(b_ave, inline=True, fontsize=10, fmt='%+.00f', colors='black')
plt.savefig(f'D:/2021tj/{year}年全球海平面气压距平场.png', dpi=1500, bbox_inches='tight')
plt.clf()

# 绘制指数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
# 原始数据序列
ax1 = fig.add_subplot(221)
ax1.plot(iYear, Global_slp_year_ave_num, color='black', linewidth=1.5)
ax1.set_ylabel('SLP(hPa)')
ax1.axhline(y=Global_slp_ave_num, color='black', linestyle='dotted')
ax1.set_title('Global SLP 原始数据序列', color='black', fontsize=15)
# 距平序列
ax2 = fig.add_subplot(222)
ax2.plot(iYear, Global_slp_delta, color='black', linewidth=1.5)
ax2.set_ylabel('ΔP(hPa)')
ax2.set_title('Global SLP 距平序列', color='black', fontsize=15)
# 标准化时间序列
ax3 = fig.add_subplot(223)
ax3.plot(iYear, Global_slp_std, color='black', linewidth=1.5)
ax3.set_ylabel('标准差')
ax3.axhline(y=0, color='black', linestyle='dotted')
ax3.axhline(y=1, color='red', linestyle='--')
ax3.axhline(y=-1, color='red', linestyle='--')
ax3.set_title('Global SLP 标准化时间序列', color='black', fontsize=15)
# u检验
Global_slp_EL = [Global_slp[i] for i in range(len(Global_slp)) if i in [4, 8, 13, 19, 24]]
Global_slp_LN = [Global_slp[i] for i in range(len(Global_slp)) if i in [6, 10, 20, 21, 29]]
Global_slp_EL_ave_num = np.mean([Global_slp[i] for i in range(len(Global_slp_year_ave_num)) if i in [4, 8, 13, 19, 24]])    # EL总平均值
Global_slp_LN_ave_num = np.mean([Global_slp[i] for i in range(len(Global_slp_year_ave_num)) if i in [6, 10, 20, 21, 29]])    # LN总平均值
Global_slp_EL_year_ave_num = [Global_slp_year_ave_num[i] for i in range(len(Global_slp_year_ave_num)) if i in [4, 8, 13, 19, 24]]   # EL年平均值
Global_slp_LN_year_ave_num = [Global_slp_year_ave_num[i] for i in range(len(Global_slp_year_ave_num)) if i not in [6, 10, 20, 21, 29]]  # LN年平均值
Global_slp_EL_delta = np.zeros(len(Global_slp_EL))
Global_slp_LN_delta = np.zeros(len(Global_slp_LN))
for i in range(len(Global_slp_EL)):
    Global_slp_EL_year_ave_num[i] = np.mean(Global_slp_EL[i])
    Global_slp_EL_delta[i] = Global_slp_EL_year_ave_num[i] - Global_slp_EL_ave_num
for i in range(len(Global_slp_LN)):
    Global_slp_LN_year_ave_num[i] = np.mean(Global_slp_LN[i])
    Global_slp_LN_delta[i] = Global_slp_LN_year_ave_num[i] - Global_slp_LN_ave_num
EL_Sx = np.sqrt((1 / len(Global_slp_EL)) * np.sum(Global_slp_EL_delta ** 2))  # EL年标准差
LN_Sx = np.sqrt((1 / len(Global_slp_LN)) * np.sum(Global_slp_LN_delta ** 2))  # LN年标准差
Uxy = (Global_slp_EL_ave_num - Global_slp_LN_ave_num) / np.sqrt(EL_Sx ** 2 / len(Global_slp_EL) + LN_Sx ** 2 / len(Global_slp_LN))  # U统计量
X = np.linspace(-3, 3, 500)
ax4 = fig.add_subplot(224)
ax4.plot(X, Gauss_dist(X, 0, 1), color='black', linewidth=1.5)
ax4.set_xlabel(' U统计量')
ax4.scatter(Uxy, Gauss_dist(Uxy, 0, 1), color='blue', marker='o')
plt.text(Uxy, Gauss_dist(Uxy, 0, 1) + 0.025, 'U', color='blue', fontsize=10, ha='center', va='center')
ax4.axvline(x=1.96, color='red', linestyle='--')
ax4.axvline(x=-1.96, color='red', linestyle='--')
plt.text(1.96, 0.4, '95%', color='black', fontsize=10, ha='left', va='center')
plt.text(-1.96, 0.4, '95%', color='black', fontsize=10, ha='right', va='center')
ax4.axvline(x=1.645, color='orange', linestyle='--')
ax4.axvline(x=-1.645, color='orange', linestyle='--')
plt.text(1.645, 0.4, '90%', color='black', fontsize=10, ha='right', va='center')
plt.text(-1.645, 0.4, '90%', color='black', fontsize=10, ha='left', va='center')
ax4.set_title('Global SLP U检验', color='black', fontsize=15)
plt.savefig(f'D:/2021tj/Global SLP.png', dpi=1500, bbox_inches='tight')
plt.clf()

# 计算热带太平洋Nino3.4区海温指数（实习一中的结果）与全球海平面气压场之间的相关关系
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
ax_b = fig.add_subplot(111)
ax_b = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax_b.coastlines()
ax_b.stock_img()
ax_b.set_title(f'全球相关系数分布场', color='black', fontsize=20)
with open("D:/2021tj/Nino3.4 SST.csv", 'r', encoding='utf-8') as Nino34_data:
    Nino34逐年气温原始数据 = Nino34_data.read()
海温指数 = [eval(i) for i in Nino34逐年气温原始数据.split()]
海温指数30年平均值 = np.average(海温指数)
海温指数30年距平 = np.zeros(len(海温指数))
for i in range(len(海温指数)):
    海温指数30年距平[i] = 海温指数[i] - 海温指数30年平均值
海温指数标准差 = np.sqrt(1 / len(海温指数) * np.sum(海温指数30年距平 ** 2))
海平面气压场标准差 = Sx
海平面气压场距平 = Global_slp_delta
逐年距平场 = [0 for i in range(30)]
for i in range(30):
    逐年距平场[i] = slp[i] - Global_slp_ave
逐年距平场 = np.array(逐年距平场)
Sxy_list = [0 for i in range(30)]
for i in range(30):
    Sxy_list[i] = 海温指数30年距平[i] * 逐年距平场[i] / len(海温指数)
协方差 = np.zeros((24, 48))
for i in range(30):
    协方差 += Sxy_list[i]
海平面气压场格点标准差 = np.sqrt((1 / len(Global_slp)) * np.sum(逐年距平场 ** 2, 0))  # 标准差
r = 协方差 / (海平面气压场格点标准差 * 海温指数标准差)     # 相关系数
r_level = 10

r_cbar = {'orientation': 'vertical', 'label': '2m temperature (℃)', 'shrink': 0.8, 'ticks': r_level, 'pad': 0.05}
r_a = ax_b.contourf(lon, lat, r, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=r_level, extend='both',
                    cbar_kwargs=r_cbar)
r_b = ax_b.contour(lon, lat, r, levels=r_level, cbar_kwargs=r_cbar, linewidths=0.5,
                   transform=ccrs.PlateCarree())
plt.clabel(r_b, inline=True, fontsize=10, fmt='%+.02f', colors='black')
fig.colorbar(orientation='horizontal', shrink=0.8, pad=0.05, aspect=50, mappable=r_a)
plt.savefig(f'D:/2021tj/Global r.png', dpi=1500, bbox_inches='tight')
plt.clf()
# 分别计算南方涛动指数SOI与Nino3.4区海温指数各自超前滞后相关系数（自相关）及两者之间的超前滞后相关系数（交叉相关）。注意：这里取滞后时间为1~15年。
# 这里我们可以根据题2中确定最大/最小值相关系数的中心位置, 计算该位置海平面气压场差值近似定义SOI指数（东太减西太）。
# 确定最大/最小值相关系数的中心位置
def rxx(x, y, xt, ytj):
    xt = np.array(xt)
    ytj = np.array(ytj)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    r = np.sum((xt - x_mean) * (ytj - y_mean)) / (len(xt) * x_std * y_std)
    return r


r_max = np.max(r)
r_min = np.min(r)
r_max_index = np.where(r == r_max)
r_min_index = np.where(r == r_min)
r_max_lon = r_max_index[1]
r_max_lat = r_max_index[0]
r_min_lon = r_min_index[1]
r_min_lat = r_min_index[0]
# 计算SOI指数
SOI = [0 for i in range(30)]
for i in range(30):
    SOI[i] = slp[i][r_min_lat, r_min_lon] - slp[i][r_max_lat, r_max_lon]
SOI = np.array(SOI)
# 计算Nino3.4区海温指数
Nino34区海温指数 = 海温指数
fig = plt.figure(figsize=(16, 9))
ax_1 = fig.add_subplot(221)
ax_1.set_title('SOI超前滞后相关系数')
ax_1.set_xlabel('滞后时间')
ax_1.set_ylabel('相关系数')
ax_1.set_xlim(1, 15)
ax_1.set_xticks(np.arange(1, 16, 1))
r_soi = [0 for i in range(15)]
for i in range(1, 15):
    r_soi[i-1] = rxx(SOI, SOI, SOI[:30-i], SOI[i:])
r_soi = np.array(r_soi)
ax_1.plot(np.arange(1, 16, 1), r_soi, color='black', label='SOI')
ax_2 = fig.add_subplot(222)
ax_2.set_title('Nino3.4区海温指数超前滞后相关系数')
ax_2.set_xlabel('滞后时间')
ax_2.set_ylabel('相关系数')
ax_2.set_xlim(1, 15)
ax_2.set_xticks(np.arange(1, 16, 1))
r_nino34 = [0 for i in range(15)]
for i in range(1, 15):
    r_nino34[i-1] = rxx(Nino34区海温指数, Nino34区海温指数, Nino34区海温指数[:30-i], Nino34区海温指数[i:])
r_nino34 = np.array(r_nino34)
ax_2.plot(np.arange(1, 16, 1), r_nino34, color='black', label='Nino3.4区海温指数')
ax_3 = fig.add_subplot(223)
ax_3.set_title('SOI与Nino3.4区海温指数超前滞后相关系数')
ax_3.set_xlabel('滞后时间')
ax_3.set_ylabel('相关系数')
ax_3.set_xlim(1, 15)
ax_3.set_xticks(np.arange(1, 16, 1))
r_soi_nino34 = [0 for i in range(15)]
for i in range(1, 15):
    r_soi_nino34[i-1] = rxx(SOI, Nino34区海温指数, SOI[:30-i], Nino34区海温指数[i:])
r_soi_nino34 = np.array(r_soi_nino34)
ax_3.plot(np.arange(1, 16, 1), r_soi_nino34, color='black', label='SOI与Nino3.4区海温指数')
plt.savefig(f'D:/2021tj/SOI&Nino3.4区海温指数自相关及交叉相关.png', dpi=1500, bbox_inches='tight')
plt.show()
