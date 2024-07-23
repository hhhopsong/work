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
import pandas as pd
import tqdm


# 数据读取
data_year = ['1979', '2022']
# 读取CN05.1逐日最高气温数据
CN051_1 = xr.open_dataset(r"E:\data\CN05.1\1961_2021\CN05.1_Tmax_1961_2021_daily_025x025.nc")
CN051_2 = xr.open_dataset(r"E:\data\CN05.1\2022\CN05.1_Tmax_2022_daily_025x025.nc")
Tmax_cn051 = xr.concat([CN051_1, CN051_2], dim='time')
if eval(input("1)是否计算全国极端高温95分位相对阈值(0/1)?\n")):
    Tmax_sort95 = Tmax_cn051['tmax'].sel(time=slice(data_year[0]+'-01-01', data_year[1]+'-12-31')).quantile(0.95, dim='time')     # 全国极端高温95分位相对阈值
    Tmax_sort95.to_netcdf(r"D:\CODES\Python\Meteorological\paper1\cache\NationalHighTemp_95threshold.nc")
    del Tmax_sort95     # 释放Tmax_sort95占用内存,优化代码性能
Tmax_sort95 = xr.open_dataset(r"cache\NationalHighTemp_95threshold.nc")  # 读取缓存
if eval(input("2)是否计算全国极端高温日温度距平(0/1)?\n")):
    EHD = Tmax_cn051.sel(time=slice(data_year[0]+'-01-01', data_year[1]+'-12-31')) - Tmax_sort95  # 温度距平
    EHD = EHD.where(EHD >= 0, np.nan)  # 极端高温日温度距平
    EHD = EHD-EHD+1  # 数据二值化处理(1:极端高温,np.nan:非极端高温)
    EHD.to_netcdf(r"D:\CODES\Python\Meteorological\paper1\cache\EHD.nc")
    del EHD  # 释放EHD占用内存,优化代码性能
EHD = xr.open_dataset(r"cache\EHD.nc")  # 读取缓存
EHD = masked(EHD, r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
EHD = EHD.sel(time=EHD['time.month'].isin([6, 7, 8, 9]))  # 选择6、7、8、9月数据  # 站点数
station_num = masked(CN051_2.sel(time='2022-01-01'), r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域站点数
station_num = station_num.sum()  # 长江流域站点数
print('EHD')
EHDstations_zone = EHD.sum(dim=['lat', 'lon'])  # 长江流域逐日极端高温站点数
'''# 将数据按月分组
EHD_6 = EHD.sel(time=EHD['time.month'] == 6)
EHD_7 = EHD.sel(time=EHD['time.month'] == 7)
EHD_8 = EHD.sel(time=EHD['time.month'] == 8)
EHD_9 = EHD.sel(time=EHD['time.month'] == 9)
# 将数据按日分组,并计算每日的年平均极端高温日数
EHD_6 = EHD_6.groupby('time.day').sum()/(eval(data_year[1])-eval(data_year[0])+1)
EHD_7 = EHD_7.groupby('time.day').sum()/(eval(data_year[1])-eval(data_year[0])+1)
EHD_8 = EHD_8.groupby('time.day').sum()/(eval(data_year[1])-eval(data_year[0])+1)
EHD_9 = EHD_9.groupby('time.day').sum()/(eval(data_year[1])-eval(data_year[0])+1)
'''
print("数据处理完成")
pass
