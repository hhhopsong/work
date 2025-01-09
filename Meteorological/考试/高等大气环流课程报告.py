import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import pandas as pd

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def round_boundary(lon, lat, dx=1):
    """
    对经纬度进行等角度划分，返回划分后的经纬度
    :param lon: 经度
    :param lat: 纬度
    :param n: 划分的角度
    :return: 划分后的经纬度
    """
    latmin = lat[0]
    latmax = lat[1]
    lonmin = lon[0]
    lonmax = lon[1]
    vertices = [(lon, latmin) for lon in range(lonmin, lonmax + 1, dx)] + [(lon, latmax) for lon in range(lonmax, lonmin - 1, -dx)]
    boundary = Path(vertices)
    return boundary


DOE_R2 = xr.open_dataset(r"E:\data\NOAA\DOE\hgt.mon.mean.nc")
# 绘制冬季（12~1月）和夏季（6~8月）平均北半球（20°N以北） 200hPa 位势高度显示的大气定常波
hgt_winter = DOE_R2.sel(level=200, time=DOE_R2['time.month'].isin([12, 1])).mean('time')
hgt_summer = DOE_R2.sel(level=200, time=DOE_R2['time.month'].isin([6, 7, 8])).mean('time')
# 绘图，左图显示夏天，右图显示冬天,兰勃托投影
fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
proj = ccrs.LambertConformal(central_longitude=180, central_latitude=50)
ax1 = fig.add_subplot(gs[0], projection=proj)
ax2 = fig.add_subplot(gs[1], projection=proj)
# 设置地图范围
ax1.set_extent([-180, 180, 20, 80], crs=ccrs.PlateCarree())
ax1.set_boundary(round_boundary([-180, 180], [20, 80]), transform=ccrs.PlateCarree())
ax2.set_extent([-180, 180, 20, 80], crs=ccrs.PlateCarree())
ax2.set_boundary(round_boundary([-180, 180], [20, 80]), transform=ccrs.PlateCarree())
# 添加地图要素
ax1.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5)
ax2.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5)
# 设置刻度线
ax1.gridlines(  xlocs=np.arange(-180, 180 + 1, 60), ylocs=np.arange(20, 80 + 1, 20),
                draw_labels=True, x_inline=False, y_inline=False,
                linewidth=0.5, linestyle='--', color='gray')
ax2.gridlines(  xlocs=np.arange(-180, 180 + 1, 60), ylocs=np.arange(20, 80 + 1, 20),
                draw_labels=True, x_inline=False, y_inline=False,
                linewidth=0.5, linestyle='--', color='gray')
# 绘制等值线
clevs = 10
summer, lon_s = add_cyclic_point(hgt_summer['hgt'].data, coord=hgt_summer['lon'])
winter, lon_w = add_cyclic_point(hgt_winter['hgt'].data, coord=hgt_winter['lon'])
contourf1 = ax1.contourf(lon_s, hgt_summer['lat'], summer, clevs, cmap=cmaps.GMT_polar[4:-4], transform=ccrs.PlateCarree())
contourf2 = ax2.contourf(lon_w, hgt_winter['lat'], winter, clevs, cmap=cmaps.GMT_polar[4:-4], transform=ccrs.PlateCarree())
contour1 = ax1.contour(lon_s, hgt_summer['lat'], summer, clevs, colors='w', linewidths=0.5, transform=ccrs.PlateCarree())
contour2 = ax2.contour(lon_w, hgt_winter['lat'], winter, clevs, colors='w', linewidths=0.5, transform=ccrs.PlateCarree())
plt.show()


# 冬季（12~1月）和夏季（6~8月）平均北半球（20°N以北） 200hPa 位势高度沿不同纬度（25°N、45°N 和 60°N） 100hPa-1000hPa 的纬向—垂直剖面图
winter_p = DOE_R2.sel(lat=[25, 45, 60],level=slice(1000, 100), time=DOE_R2['time.month'].isin([12, 1])).mean('time')
summer_p = DOE_R2.sel(lat=[25, 45, 60],level=slice(1000, 100), time=DOE_R2['time.month'].isin([6, 7, 8])).mean('time')

# 绘图

# 设置地图范围
