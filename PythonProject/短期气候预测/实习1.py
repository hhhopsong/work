from xgrads import open_CtlDataset
from cartopy import crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pprint

# Data-Read
open_CtlDataset('D:/short_term_pre/task1/hgt4821-500.ctl').to_netcdf('D:/short_term_pre/task1/hgt4821-500.nc')
data = xr.open_dataset('D:/short_term_pre/task1/hgt4821-500.nc')
H = data['hgt']
lon = data['lon']
lat = data['lat']
day = eval(input()) - 1948
# Data-Processing
H_ave = np.mean(H, axis=0)    # 平均高度场
H_A = H[day] - H_ave     # 高度距平场
H_LA = H[day, :, :] - np.mean(H[day, :, :], axis=1)    # 高度场纬偏值
# Map-Setting
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
# Map_1
ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_title('1948-2020年 1月500hPa平均高度场', color='black', fontsize=15)
ax1.set_extent([0, -180, -10, 90], crs=ccrs.PlateCarree())
ax1.coastlines()
ax1.stock_img()
#a1 = ax1.contourf(lon, lat, H_ave, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=15, extend='both')
b1 = ax1.contour(lon, lat, H_ave, transform=ccrs.PlateCarree(), lw=0.5,levels=15, extend='both')
plt.clabel(b1, inline=True, fontsize=5, fmt='%.00f', colors='black')
# Map_2
ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree(central_longitude=180))
ax2.set_title(f'{day+1948}年 1月500hPa高度距平场', color='black', fontsize=15)
ax2.set_extent([0, -180, -10, 90], crs=ccrs.PlateCarree())
ax2.coastlines()
ax2.stock_img()
#a2 = ax2.contourf(lon, lat, H_A, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=15, extend='both')
b2 = ax2.contour(lon, lat, H_A, transform=ccrs.PlateCarree(), lw=0.5,levels=15, extend='both')
plt.clabel(b2, inline=True, fontsize=5, fmt='%.00f', colors='black')
# Map_3
ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree(central_longitude=180))
ax3.set_title(f'{day+1948}年 1月500hPa高度纬偏场', color='black', fontsize=15)
ax3.set_extent([0, -180, -10, 90], crs=ccrs.PlateCarree())
ax3.coastlines()
ax3.stock_img()
#a3 = ax3.contourf(lon, lat, H_LA, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=15, extend='both')
b3 = ax3.contour(lon, lat, H_LA, transform=ccrs.PlateCarree(), lw=0.5,levels=15, extend='both')
plt.clabel(b3, inline=True, fontsize=5, fmt='%.00f', colors='black')
# Print
plt.savefig(f'D:/short_term_pre/task1/实习1.png', dpi=1500)
plt.show()
