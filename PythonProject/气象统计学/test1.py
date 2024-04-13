from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from numpy import ma
from pprint import pprint


with open("C:/Users/admin/OneDrive/桌面/2021072012.txt", 'r', encoding='utf-8') as f:
    data = f.readlines()[4:]
    data = [ii.strip().split() for ii in [i for i in data]]

for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = eval(data[i][j])
for i in range(len(data)):
    try:
        birdge = []
        for ii in range(8):
            birdge += data[i * 8 + ii]
        data[i] = birdge
    except:
        index = i
        break

del data[index:]
data = np.array(data)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16, 9))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines()
ax.stock_img()
ax.set_extent([0, 180, 0, 90], crs=ccrs.PlateCarree())
cbar_kwargs = {
    'orientation': 'vertical',
    'label': '2m temperature (℃)',
    'shrink': 0.8,
    'ticks': 16,
    'pad': 0.05}

# 绘制等值线
lon = np.arange(0, 182.5, 2.5)
lat = np.arange(90, -2.5, -2.5)
#a = ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=16, extend='both',cbar_kwargs=cbar_kwargs)
b = ax.contour(lon, lat, data, colors='black', levels=16, linewidths=1, transform=ccrs.PlateCarree())
plt.clabel(b, inline=True, fontsize=10, fmt='%.0f', colors='black')
plt.show()
