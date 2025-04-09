import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MultipleLocator, FixedLocator


def latlon_fmt(ax, xticks1, yticks1, xmajorLocator, xminorLocator, ymajorLocator, yminorLocator):
    if yticks1 is not None: ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
    if xticks1 is not None: ax.set_xticks(xticks1, crs=ccrs.PlateCarree())
    if xticks1 is not None: lon_formatter = LongitudeFormatter()
    if yticks1 is not None: lat_formatter = LatitudeFormatter()
    if yticks1 is not None: ax.yaxis.set_major_formatter(lat_formatter)
    if xticks1 is not None: ax.xaxis.set_major_formatter(lon_formatter)
    if yticks1 is not None: ax.yaxis.set_major_locator(ymajorLocator)
    if yticks1 is not None: ax.yaxis.set_minor_locator(yminorLocator)
    if xticks1 is not None: ax.xaxis.set_major_locator(xmajorLocator)
    if xticks1 is not None: ax.xaxis.set_minor_locator(xminorLocator)
    ax.tick_params(which='major', length=4, width=.5, color='black')
    ax.tick_params(which='minor', length=2, width=.2, color='black')
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'
    ax.tick_params(axis='both', labelsize=8, colors='black')


# 创建一个新的图形
fig = plt.figure(figsize=(5, 7))
fig.subplots_adjust(hspace=0.3)
xticks1 = np.arange(-180, 181, 30)
yticks1 = np.arange(-30, 81, 15)
xmajor = [-180, -120, -60, 0, 60, 120, 180]
xminor = np.arange(-180, 181, 10)
ymajor = [-30, -15, 0, 15, 30, 45, 60, 75]
yminor = np.arange(-90, 81, 3)
ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree(central_longitude=90))
ax1.set_extent([-70, 360 - 130, -20, 80], crs=ccrs.PlateCarree(central_longitude=0))  # 设置经纬度范围
ax1.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
ax1.set_title('a)Type1', fontsize=14, loc='left')
ax1.set_aspect('auto')
ax1.add_geometries(Reader(r'D:\Code\work\Meteorological\p2\map\EYTR\长江_tp.shp').geometries(),
                   ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.5)
ax1.add_geometries(Reader(r'D:\Code\work\Meteorological\p2\map\WYTR\长江_tp.shp').geometries(),
                   ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.5)
ax1.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(),
                   ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.5)
ax1.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_长江流域\TPBoundary2500m_长江流域.shp').geometries(),
                   ccrs.PlateCarree(central_longitude=0), facecolor='gray', edgecolor='black', linewidth=.5, zorder=11)
ax1.add_geometries(Reader(r'D:\PyFile\map\地图线路数据\长江\长江.shp').geometries(),
                   ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='blue', linewidth=0.2, zorder=12)
latlon_fmt(ax1, xticks1, yticks1, FixedLocator(xmajor), FixedLocator(xminor), FixedLocator(ymajor), FixedLocator(yminor))

ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree(central_longitude=-30))
ax2.set_extent([-205, 140, -20, 80], crs=ccrs.PlateCarree(central_longitude=0))  # 设置经纬度范围
ax2.set_title('b)Type2', fontsize=12, loc='left')
ax2.set_aspect('auto')
ax2.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
ax2.add_geometries(Reader(r'D:\Code\work\Meteorological\p2\map\EYTR\长江_tp.shp').geometries(),
                   ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.5)
ax2.add_geometries(Reader(r'D:\Code\work\Meteorological\p2\map\WYTR\长江_tp.shp').geometries(),
                     ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.5)
ax2.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(),
                     ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.5)
ax2.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_长江流域\TPBoundary2500m_长江流域.shp').geometries(),
                     ccrs.PlateCarree(central_longitude=0), facecolor='gray', edgecolor='black', linewidth=.5, zorder=11)
ax2.add_geometries(Reader(r'D:\PyFile\map\地图线路数据\长江\长江.shp').geometries(),
                     ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='blue', linewidth=0.2, zorder=12)
latlon_fmt(ax2, xticks1, yticks1, FixedLocator(xmajor), FixedLocator(xminor), FixedLocator(ymajor), FixedLocator(yminor))

ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree(central_longitude=90))
ax3.set_extent([-70, 360 - 130, -20, 80], crs=ccrs.PlateCarree(central_longitude=0))  # 设置经纬度范围
ax3.set_title('c)Type3', fontsize=10, loc='left')
ax3.set_aspect('auto')
ax3.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
ax3.add_geometries(Reader(r'D:\Code\work\Meteorological\p2\map\EYTR\长江_tp.shp').geometries(),
                     ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.5)
ax3.add_geometries(Reader(r'D:\Code\work\Meteorological\p2\map\WYTR\长江_tp.shp').geometries(),
                        ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.5)
ax3.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(),
                        ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.5)
ax3.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_长江流域\TPBoundary2500m_长江流域.shp').geometries(),
                        ccrs.PlateCarree(central_longitude=0), facecolor='gray', edgecolor='black', linewidth=.5, zorder=11)
ax3.add_geometries(Reader(r'D:\PyFile\map\地图线路数据\长江\长江.shp').geometries(),
                        ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='blue', linewidth=0.2, zorder=12)
latlon_fmt(ax3, xticks1, yticks1, FixedLocator(xmajor), FixedLocator(xminor), FixedLocator(ymajor), FixedLocator(yminor))

plt.savefig('D:\PyFile\p2\pic\示意图.svg', bbox_inches='tight')