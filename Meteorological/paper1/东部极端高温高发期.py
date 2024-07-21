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
if eval(input("是否计算全国极端高温95分位相对阈值(0/1)?")):
    Tmax_sort95 = Tmax_cn051['tmax'].sel(time=slice('1979-01-01', '2022-12-31')).quantile(0.95, dim='time')     # 全国极端高温95分位相对阈值
    Tmax_sort95.to_netcdf(r"D:\CODES\Python\Meteorological\paper1\cache\NationalHighTemp_95threshold.nc")
    del Tmax_sort95     # 释放Tmax_sort95占用内存,优化代码性能
Tmax_sort95 = xr.open_dataset(r"cache\NationalHighTemp_95threshold.nc")     # 读取缓存
Tmax_sort95 = masked(Tmax_sort95, r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")
pass
