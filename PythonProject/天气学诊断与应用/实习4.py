import pprint
from cartopy import crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def wind_g(φ, ΔH, Δφ, mode='x'):
    # φ:单位°
    φ = φ * np.pi / 180
    Δφ = Δφ * np.pi / 180
    R = 6371 * 10 ** 3  # m
    if mode == 'x':
        Δx = R * np.cos(φ) * Δφ
    elif mode == 'y':
        Δx = R * Δφ
    return 0.67197 * ΔH * 10 ** 5 / (Δx * np.sin(φ))    # m/s


# 数据读取
H = xr.open_dataset("D:/tz_shixi/hgt.2020.nc")  # 高度场(单位:m)
U = xr.open_dataset("D:/tz_shixi/uwnd.2020.nc")  # 纬向风速
V = xr.open_dataset("D:/tz_shixi/vwnd.2020.nc")  # 经向风速
lon = H['lon'][23:66]
lat = H['lat'][7:34]
# 调出2020年7月18日8:00数据(时次:796   层次:10(200hPa)  范围:lat12-32;lon28-56   格点:25 * 41)
H200 = H['hgt'][796][9, 6:35, 22:67]
U200 = U['uwnd'][796][9, 7:34, 23:66]
V200 = V['vwnd'][796][9, 7:34, 23:66]
# u风速
uwind_g = np.zeros((27, 43))
for ii in range(43):
    for i in range(27):
        uwind_g[i, ii] = wind_g(lat[i], H200[i+2][ii+1] - H200[i][ii+1], 5, mode='y')
# v风速
vwind_g = np.zeros((27, 43))
for i in range(27):
    for ii in range(43):
        vwind_g[i, ii] = wind_g(lat[i], H200[i+1][ii+2] - H200[i+1][ii], 5)
# 地图要素设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
# 绘制地转风场
ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
ax1.set_title('2020-07-18 8:00 200hPa地转风场', color='black', fontsize=15)
ax1.coastlines()
ax1.stock_img()
ax1.set_extent([57.5, 162.5, 7.5, 72.5], crs=ccrs.PlateCarree())
ax1.barbs(lon[1:42], lat[1:26], uwind_g[1:26, 1:42], vwind_g[1:26, 1:42], length=4, sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5), linewidth=0.2)
# 计算地转涡度
R = 6371 * 10 ** 3  # m
dvg_dx = np.zeros((25, 41))
dug_dy = np.zeros((25, 41))
for i in range(25):
    for ii in range(41):
        dvg_dx[i, ii] = (vwind_g[i + 1, ii + 2] - vwind_g[i + 1, ii]) / (R * np.cos(lat[i + 1] * np.pi / 180) * 5 * np.pi / 180)
        dug_dy[i, ii] = (uwind_g[i, ii + 1] - uwind_g[i + 2, ii + 1]) / (R * 5 * np.pi / 180)
ξgp = dvg_dx - dug_dy
# 绘制地转涡度场
levels = 10
ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
ax2.set_title(f'2020-07-18 8:00 地转涡度场(×10$^-$$^5$)', color='black', fontsize=15)
ax2.coastlines()
ax2.stock_img()
ax2.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
a2 = ax2.contourf(lon[1:42], lat[1:26], ξgp, cmap='Spectral_r', levels=levels, extend='both', transform=ccrs.PlateCarree())
b2 = ax2.contour(lon[1:42], lat[1:26], ξgp * 10 ** 5, levels=levels, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b2, inline=True, fontsize=5, fmt='%.0f', colors='black')
# 计算相对涡度
R = 6371 * 10 ** 3  # m
dv_dx = np.zeros((25, 41))
du_dy = np.zeros((25, 41))
for i in range(25):
    for ii in range(41):
        dv_dx[i, ii] = (V200[i + 1, ii + 2] - V200[i + 1, ii]) / (R * np.cos(lat[i + 1] * np.pi / 180) * 5 * np.pi / 180)
        du_dy[i, ii] = (U200[i, ii + 1] - U200[i + 2, ii + 1]) / (R * 5 * np.pi / 180)
ξp = dv_dx - du_dy
# 绘制相对涡度场
ax4 = fig.add_subplot(223, projection=ccrs.PlateCarree())
ax4.set_title(f'2020-07-18 8:00 相对涡度场(×10$^-$$^5$)', color='black', fontsize=15)
ax4.coastlines()
ax4.stock_img()
ax4.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
a4 = ax4.contourf(lon[1:42], lat[1:26], ξp, cmap='Spectral_r', levels=levels, extend='both', transform=ccrs.PlateCarree())
b4 = ax4.contour(lon[1:42], lat[1:26], ξp * 10 ** 5, levels=levels, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b4, inline=True, fontsize=5, fmt='%.0f', colors='black')
# 计算散度
R = 6371 * 10 ** 3  # m
du_dx = np.zeros((25, 41))
dv_dy = np.zeros((25, 41))
for i in range(25):
    for ii in range(41):
        du_dx[i, ii] = (U200[i + 2, ii + 2] - U200[i, ii + 1]) / (R * np.cos(lat[i + 1] * np.pi / 180) * 5 * np.pi / 180)
        dv_dy[i, ii] = (V200[i + 1, ii] - V200[i + 1, ii + 2]) / (R * 5 * np.pi / 180)
Dp = du_dx + dv_dy
# 绘制散度场
ax5 = fig.add_subplot(224, projection=ccrs.PlateCarree())
ax5.set_title(f'2020-07-18 8:00 散度场(×10$^-$$^5$)', color='black', fontsize=15)
ax5.coastlines()
ax5.stock_img()
ax5.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
a5 = ax5.contourf(lon[1:42], lat[1:26], Dp, cmap='Spectral_r', levels=levels, extend='both', transform=ccrs.PlateCarree())
b5 = ax5.contour(lon[1:42], lat[1:26], Dp * 10 ** 5, levels=levels, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b5, inline=True, fontsize=5, fmt='%.0f', colors='black')
plt.savefig(f'D:/tz_shixi/实习4/实习4.png', dpi=1500, bbox_inches='tight')
plt.show()
