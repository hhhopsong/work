import pprint
from cartopy import crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# 数据读取
U = xr.open_dataset("D:/tz_shixi/uwnd.2020.nc")
V = xr.open_dataset("D:/tz_shixi/vwnd.2020.nc")
H = xr.open_dataset("D:/tz_shixi/hgt.2020.nc")      # 高度场(单位:m)
T = xr.open_dataset("D:/tz_shixi/air.2020.nc")      # 温度(单位:K)
q = xr.open_dataset("D:/tz_shixi/rhum.2020.nc")     # 相对湿度(eg:78,67...)
lon = U['lon']
lat = U['lat']
# 调出2020年7月18日200hPa数据(时次:796-799层次:10)
H718 = H['hgt'][796:800][:, 9]
U718 = U['uwnd'][796:800][:, 9]
V718 = V['vwnd'][796:800][:, 9]
本日平均高度场 = np.mean(H718, axis=0)
本日平均纬向风 = np.mean(U718, axis=0)
本日平均经向风 = np.mean(V718, axis=0)
# 地图要素设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_title('2020-07-18 平均高度场', color='black', fontsize=15)
ax1.set_extent([0, -180, 0, 90], crs=ccrs.PlateCarree())
ax1.coastlines()
ax1.stock_img()
# 绘制平均高度场
levels = 50
cbar_kwargs = {'orientation': 'vertical', 'label': '2m temperature (℃)', 'shrink': 0.8, 'ticks': levels, 'pad': 0.05}
a = ax1.contourf(lon, lat, 本日平均高度场, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both', cbar_kwargs=cbar_kwargs)
b = ax1.contour(lon, lat, 本日平均高度场, levels=levels, cbar_kwargs=cbar_kwargs, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b, inline=True, fontsize=5, fmt='%.01f', colors='black')
# 绘制平均水平风场
ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree(central_longitude=180))
ax2.set_title('2020-07-18 平均水平风场', color='black', fontsize=15)
ax2.coastlines()
ax2.stock_img()
ax2.set_extent([0, -178.5, 0, 90], crs=ccrs.PlateCarree())
ax2.barbs(lon, lat, 本日平均纬向风, 本日平均经向风, length=2.2, sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5), linewidth=0.2)
# 绘制日变高分布图(五点差分)
A_x十Δx = H['hgt'][797][9]
A_x一Δx = H['hgt'][799][9]
A_x十2Δx = H['hgt'][796][9]
A_x一2Δx = H['hgt'][800][9]
Δx = 6  # 单位:小时
日变高 = 4 / 3 * (A_x十Δx - A_x一Δx) / 2 * Δx - 1 / 3 * (A_x十2Δx - A_x一2Δx) / 4 * Δx
ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree(central_longitude=180))
ax3.set_title('2020-07-18 日变高分布图', color='black', fontsize=15)
ax3.coastlines()
ax3.stock_img()
ax3.set_extent([0, -180, 0, 90], crs=ccrs.PlateCarree())
levels_ = 20
cbar_kwargs = {'orientation': 'vertical', 'label': '2m temperature (℃)', 'shrink': 0.8, 'ticks': levels_, 'pad': 0.05}
a_ = ax3.contourf(lon, lat, 日变高, cmap='Spectral_r', levels=levels_, extend='both', cbar_kwargs=cbar_kwargs, transform=ccrs.PlateCarree())
b_ = ax3.contour(lon, lat, 日变高, levels=levels_, cbar_kwargs=cbar_kwargs, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b_, inline=True, fontsize=5, fmt='%+.01f', colors='black')
# 绘制日变动能分布图(五点差分)
K = U['uwnd'][796:801][:, 9] ** 2 * 0.5 + V['vwnd'][796:801][:, 9] ** 2 * 0.5
K_x十Δx = K[1]
K_x一Δx = K[3]
K_x十2Δx = K[0]
K_x一2Δx = K[4]
日变动能 = 4 / 3 * (K_x十Δx - K_x一Δx) / 2 * Δx - 1 / 3 * (K_x十2Δx - K_x一2Δx) / 4 * Δx
ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree(central_longitude=180))
ax4.set_title('2020-07-18 日变动能分布图', color='black', fontsize=15)
ax4.coastlines()
ax4.stock_img()
ax4.set_extent([0, -180, 0, 90], crs=ccrs.PlateCarree())
levels__ = 20
cbar_kwargs = {'orientation': 'vertical', 'label': '2m temperature (℃)', 'shrink': 0.8, 'ticks': levels__, 'pad': 0.05}
a__ = ax4.contourf(lon, lat, 日变动能, cmap='Spectral_r', levels=levels__, extend='both', cbar_kwargs=cbar_kwargs, transform=ccrs.PlateCarree())
b__ = ax4.contour(lon, lat, 日变动能, levels=levels__, cbar_kwargs=cbar_kwargs, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b__, inline=True, fontsize=5, fmt='%+.01f', colors='black')
plt.savefig(f'D:/tz_shixi/实习3/实习3.png', dpi=1500, bbox_inches='tight')
plt.show()
