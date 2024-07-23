from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from cnmaps import get_adm_maps, draw_maps
from matplotlib import ticker
import cmaps
from matplotlib.ticker import MultipleLocator
from toolbar.masked import masked   # 气象工具函数


# 数据读取
# 读取CN05.1逐日最高气温数据
CN051_1 = xr.open_dataset(r"E:\data\CN05.1\1961_2021\CN05.1_Tmax_1961_2021_daily_025x025.nc")
CN051_2 = xr.open_dataset(r"E:\data\CN05.1\2022\CN05.1_Tmax_2022_daily_025x025.nc")
Tmax_cn051 = xr.concat([CN051_1, CN051_2], dim='time')
if eval(input("是否计算全国极端高温95分位相对阈值(0/1)?\n")):
    Tmax_sort95 = Tmax_cn051['tmax'].sel(time=slice('1979-01-01', '2022-12-31')).quantile(0.95, dim='time')     # 全国极端高温95分位相对阈值
    Tmax_sort95.to_netcdf(r"D:\CODES\Python\Meteorological\paper1\cache\NationalHighTemp_95threshold.nc")
    del Tmax_sort95     # 释放Tmax_sort95占用内存,优化代码性能
Tmax_sort95 = xr.open_dataset(r"cache\NationalHighTemp_95threshold.nc")     # 读取缓存
Tmax_sort95 = masked(Tmax_sort95, r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")


# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
extent_CN = [88, 124, 22, 38]  # 中国大陆经度范围，纬度范围
proj = ccrs.PlateCarree()   # 投影方式
fig = plt.figure(figsize=(16, 9))   # 创建画布
ax1 = fig.add_subplot(111, projection=proj)  # 添加子图
ax1.set_extent(extent_CN, crs=proj) # 设置地图范围
a1 = ax1.contourf(Tmax_sort95['lon'], Tmax_sort95['lat'], Tmax_sort95['tmax'], cmap=cmaps.WhiteBlueGreenYellowRed, levels=15, extend='both', transform=proj)
ax1.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
ax1.add_geometries(Reader(r'C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4)
ax1.add_geometries(Reader(r'D:\CODES\Python\Meteorological\maps\cnriver\长江\长江.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=0.2)
plt.savefig(r'C:\Users\10574\desktop\图2.png', dpi=1500, bbox_inches='tight')
plt.show()
pass
