from xgrads import open_CtlDataset
from cartopy import crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pprint


# Data-Read
open_CtlDataset('D:/short_term_pre/task5/ersstwin.ctl').to_netcdf('D:/short_term_pre/task5/ersstwin.nc')
data = xr.open_dataset('D:/short_term_pre/task5/ersstwin.nc')
lon = data['lon']
lat = data['lat']
sst = np.ma.masked_values(data['sst'], 32767)
sst_avg = np.mean(sst, axis=0)
# 逐年距平场以及t检验
sst_anom = sst - sst_avg


# 地图要素设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
levels = 15
# 绘制地图
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_title('全球海温距平分布图')
ax.set_extent([120, 300, -10, 60], crs=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
# 绘制等值线
cs = ax.contourf(lon, lat, sst_avg, levels=levels, cmap='jet', transform=ccrs.PlateCarree())
plt.show()
pprint.pprint(sst_anom.shape)
