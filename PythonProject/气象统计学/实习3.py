import pprint
from xgrads import CtlDescriptor, open_CtlDataset
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from numpy import ma


slp = xr.open_dataset("D:/2021tj/NCEP_slp_30y_Wt.nc")
sst = xr.open_dataset("D:/2021tj/NCEP_TPSST_30y_Wt.nc")
st = ma.masked_values(sst['st'], 1.000e33)
lon = slp['lon']
lat = slp['lat']
slp = ma.masked_values(slp['slp'], 1.000e33)


# 回归方法
Nino逐年平均值 = np.zeros(len(st))  # 逐年平均值
for i in range(len(st)):
    Nino逐年平均值[i] = np.mean(st[i])
Nino30年平均值 = np.mean(Nino逐年平均值)
x2 = np.sum((Nino逐年平均值 - Nino30年平均值) ** 2)
海平面气压多年平均场 = np.mean(slp, axis=0)
海平面气压多年距平场 = [i for i in range(30)]
for i in range(30):
    海平面气压多年距平场[i] = slp[i] - 海平面气压多年平均场
海平面气压多年距平场 = np.array(海平面气压多年距平场)
nxy = 30 * Nino30年平均值 * 海平面气压多年平均场
xy = [0 for i in range(30)]
for i in range(30):
    xy[i] += slp[i] * Nino逐年平均值[i]
xy = np.array(xy)
b = (sum(xy) - nxy) / x2


# 地图要素设置
levels = np.arange(-12, 12, 1)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines()
ax.stock_img()
ax.set_title(f'b系数的空间分布图', color='black', fontsize=20)
cbar_kwargs = {
    'orientation': 'vertical',
    'label': '2m temperature (℃)',
    'shrink': 0.8,
    'ticks': levels,
    'pad': 0.05}
a = ax.contourf(lon, lat, b, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both', cbar_kwargs=cbar_kwargs)
b = ax.contour(lon, lat, b, levels=levels, cbar_kwargs=cbar_kwargs, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b, inline=True, fontsize=5, fmt='%.01f', colors='black')
plt.savefig(f'D:/2021tj/b系数的空间分布图.png', dpi=1500, bbox_inches='tight')
plt.show()
