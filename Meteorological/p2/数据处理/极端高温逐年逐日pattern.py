import xarray as xr
import numpy as np
import tqdm as tq
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from toolbar.filter import MovingAverageFilter
from toolbar.masked import masked   # 气象工具函数
from toolbar.K_Mean import K_Mean, plot_test

# 数据读取
data_year = ['1961', '2022']
# 读取CN05.1逐日最高气温数据
CN051_1 = xr.open_dataset(r"E:\data\CN05.1\1961_2021\CN05.1_Tmax_1961_2021_daily_025x025.nc")
CN051_2 = xr.open_dataset(r"E:\data\CN05.1\2022\CN05.1_Tmax_2022_daily_025x025.nc")
CN051 = xr.concat([CN051_1, CN051_2], dim='time')
try:
    Tmax_5Day_filt = xr.open_dataarray(fr"D:\PyFile\p2\data\Tmax_5Day_filt.nc")
except:
    Tmax = xr.concat([CN051_1, CN051_2], dim='time')
    Tmax = masked(Tmax, r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域温度
    Tmax = Tmax.sel(time=Tmax['time.month'].isin([6, 7, 8])).groupby('time.year')  # 截取夏季数据
    Tmax_5Day_filt = np.array([[[MovingAverageFilter(iyear[1]['tmax'].data[:, i, j], 'lowpass', [5], np.nan).filted() for j in range(283)] for i in range(163)] for iyear in tq.tqdm(Tmax)])  # 5天滑动平均
    Tmax_5Day_filt = Tmax_5Day_filt.transpose(0, 3, 1, 2) # 转换为(year, day, lat, lon)格式
    Tmax_5Day_filt = xr.DataArray(Tmax_5Day_filt,
                                  coords=[[str(i) for i in range(eval(data_year[0]), eval(data_year[1]) + 1)],
                                          [str(i) for i in range(1, 88 + 1)],
                                          CN051_2['lat'].data,
                                          CN051_2['lon'].data],
                                  dims=['year', 'day', 'lat', 'lon'], )
    Tmax_5Day_filt.to_netcdf(fr"D:\PyFile\p2\data\Tmax_5Day_filt.nc")
    del Tmax

zone_stations = masked((CN051_2-CN051_2+1).sel(time='2022-01-01'), r"D:\PyFile\map\self\长江_TP\长江_tp.shp").sum()['tmax'].data
t95 = CN051.quantile(0.95, dim='time')
EHD = Tmax_5Day_filt - t95
EHD = EHD.where(EHD >= 0, -99999)  # 极端高温日温度距平
EHD = EHD.where(EHD == -99999, 1)  # 数据二值化处理(1:极端高温, -99999:非极端高温)
EHD = EHD.where(EHD != -99999, 0)  # 数据二值化处理(1:极端高温, 0:非极端高温)
EHD = masked(EHD, r"D:\PyFile\map\self\长江_TP\长江_tp.shp")  # 掩膜处理得长江流域EHD温度距平
EHDstations_zone = EHD.sum(dim=['lat', 'lon']) / zone_stations  # 长江流域逐日极端高温格点占比
EHD20 = EHD.where(EHDstations_zone >= 0.5, 0)  # 提取极端高温日占比大于20%
EHD20 = masked(EHD20, r"D:\PyFile\map\self\长江_TP\长江_tp.shp")  # 减去非研究地区
EHD20 = EHD20['tmax'].data.reshape(-1, 163*283)
EHD20_ = pd.DataFrame(EHD20).dropna(axis=1, how='all')
plot_test(EHD20_.to_numpy(), max_clusters=20)
K = K_Mean(EHD20_.to_numpy(), 2)

# 绘制三种聚类的平均分布图
fig = plt.figure(figsize=(10, 5))
for cluster in range(2):
    ax = fig.add_subplot(1, 3, cluster+1, projection=ccrs.PlateCarree())
    ax.set_title(f"Cluster {cluster} 平均分布")
    ax.coastlines()
    ax.set_extent([90, 135, 18, 44])
    KM = EHD20[K[cluster]['indices']].mean(axis=0)
    ax.contourf(CN051_2['lon'], CN051_2['lat'], KM.reshape(163, 283),
                cmap='hot', transform=ccrs.PlateCarree())
plt.show()
