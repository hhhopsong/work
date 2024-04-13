from xgrads import CtlDescriptor, open_CtlDataset
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from numpy import ma

# 数据预处理
ctl = CtlDescriptor(file="D:/2021tj/NCEP_TPSST_30y_Wt.ctl")
dset = open_CtlDataset("D:/2021tj/NCEP_TPSST_30y_Wt.ctl")
open_CtlDataset("D:/2021tj/NCEP_TPSST_30y_Wt.ctl").to_netcdf("D:/2021tj/NCEP_TPSST_30y_Wt.nc")
ds = xr.open_dataset("D:/2021tj/NCEP_TPSST_30y_Wt.nc")
lon = ds['lon'].loc[120:300]
lat = ds['lat'].loc[-20:20]
time = ds['time']
x, y = np.meshgrid(lon, lat)
st = ma.masked_values(ds['st'], 1.000e33)


# 地图要素设置
year = eval(input("请输入年份："))
YR = year - 1978
levels = 10
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines()
ax.stock_img()
ax.set_extent([120, 300, -20, 20], crs=ccrs.PlateCarree(central_longitude=0))
ax.set_title(f'{year}年冬季海表温度原始场', color='black', fontsize=20)
cbar_kwargs = {
    'orientation': 'vertical',
    'label': '2m temperature (℃)',
    'shrink': 0.8,
    'ticks': levels,
    'pad': 0.05}


# 绘制全球海表温度场
Temp = st[YR]
a = ax.contourf(lon, lat, Temp, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both', cbar_kwargs=cbar_kwargs)
b = ax.contour(lon, lat, Temp, levels=levels, cbar_kwargs=cbar_kwargs, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b, inline=True, fontsize=5, fmt='%.01f', colors='black')
plt.savefig(f'D:/2021tj/{year}年冬季海表温度原始场.png', dpi=1500, bbox_inches='tight')
plt.clf()

# 地图要素设置
levels1 = 10
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines()
ax.stock_img()
ax.set_extent([120, 300, -20, 20], crs=ccrs.PlateCarree(central_longitude=0))
ax.set_title(f'{year}年冬季海表温度距平场', color='black', fontsize=20)
cbar_kwargs1 = {
    'orientation': 'vertical',
    'label': '2m temperature (℃)',
    'shrink': 0.8,
    'ticks': levels1,
    'pad': 0.05}

# 绘制全球海表温度距平场
ave = np.mean(st, axis=0)
st_delta = st[YR] - ave
a_ave = ax.contourf(lon, lat, st_delta, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels1, extend='both', cbar_kwargs=cbar_kwargs1)
b_ave = ax.contour(lon, lat, st_delta, levels=levels1, cbar_kwargs=cbar_kwargs1, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b_ave, inline=True, fontsize=5, fmt='%+.01f', colors='black')
plt.savefig(f'D:/2021tj/{year}年冬季海表温度距平场.png', dpi=1500, bbox_inches='tight')
plt.clf()

# 进一步计算Nino3.4区（5S-5N，170W-120W）海温指数（区域平均SST的时间序列）
# 分别给出原始数据序列、距平序列及标准化时间序列
# 根据这些时间序列，判断哪些是El Nino年和La Nina年（这里定义大于（小于）等于一个标准差的为El Nino（La Nina）年）

# 计算逐年Nino3.4区海温指数
Nino34_temp = ds['st'].loc[{'lat': slice(-5, 5), 'lon': slice(190, 240)}]
Nino34_temp_year_ave = np.zeros(len(Nino34_temp))  # 逐年平均海温
Nino34_temp_ave = np.mean(Nino34_temp)  # 多年平均海温
Nino34_temp_delta = np.zeros(len(Nino34_temp))  # 距平海温
for i in range(len(Nino34_temp)):
    Nino34_temp_year_ave[i] = np.mean(Nino34_temp[i])
    Nino34_temp_delta[i] = Nino34_temp_year_ave[i] - Nino34_temp_ave
Sx = np.sqrt((1 / len(Nino34_temp)) * np.sum(Nino34_temp_delta ** 2))  # 标准差
Nino34_temp_std = Nino34_temp_delta / Sx  # 标准化海温
StartYear = 1978
iYear = np.arange(StartYear, StartYear + len(Nino34_temp))

# 绘制Nino3.4区海温指数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
# 原始数据序列
ax1 = fig.add_subplot(221)
ax1.plot(iYear, Nino34_temp_year_ave, color='black', linewidth=1.5)
ax1.set_ylabel('SST(℃)')
ax1.axhline(y=Nino34_temp_ave, color='black', linestyle='dotted')
ax1.set_title('Nino3.4 SST 原始数据序列', color='black', fontsize=15)
# 距平序列
ax2 = fig.add_subplot(222)
ax2.plot(iYear, Nino34_temp_delta, color='black', linewidth=1.5)
ax2.set_ylabel('ΔT(℃)')
ax2.set_title('Nino3.4 SST 距平序列', color='black', fontsize=15)
# 标准化时间序列
ax3 = fig.add_subplot(223)
ax3.plot(iYear, Nino34_temp_std, color='black', linewidth=1.5)
for i in range(len(Nino34_temp_std)):
    if Nino34_temp_std[i] >= 1:
        ax3.scatter(iYear[i], Nino34_temp_std[i], color='red', marker='o')
        plt.text(iYear[i], Nino34_temp_std[i] + 0.2, f'{iYear[i]}', color='red', fontsize=7, ha='center', va='center')
    elif Nino34_temp_std[i] <= -1:
        ax3.scatter(iYear[i], Nino34_temp_std[i], color='blue', marker='o')
        plt.text(iYear[i], Nino34_temp_std[i] - 0.2, f'{iYear[i]}', color='blue', fontsize=7, ha='center', va='center')

ax3.set_ylabel('标准差')
ax3.axhline(y=0, color='black', linestyle='dotted')
ax3.axhline(y=1, color='red', linestyle='--')
ax3.axhline(y=-1, color='red', linestyle='--')
ax3.set_title('Nino3.4 SST 标准化时间序列', color='black', fontsize=15)

plt.savefig(f'D:/2021tj/Nino3.4 SST.png', dpi=1500, bbox_inches='tight')
plt.show()
ds.close()
with open('D:/2021tj/Nino3.4 SST.csv', 'w', encoding='utf-8') as output:
    output.writelines(str(Nino34_temp_year_ave))
