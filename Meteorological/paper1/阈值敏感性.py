import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from toolbar.masked import masked

import tqdm as tq

def time_series(info, EHD, station_happended_num):
    EHD = EHD - info
    EHD = EHD.where(EHD >= 0, np.nan)  # 极端高温日温度距平
    EHD = EHD - EHD + 1  # 数据二值化处理(1:极端高温,np.nan:非极端高温)
    EHD = masked(EHD, r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
    EHD = EHD.sel(time=EHD['time.month'].isin([7, 8]))

    EHDstations_happended_zone = EHD.sum(dim=['lat', 'lon']) / station_happended_num
    # 将数据按日序分组，并转换为DataArray格式
    EHDstations_happended_zone = xr.DataArray(
        EHDstations_happended_zone['tmax'].to_numpy().reshape([eval(data_year[1]) - eval(data_year[0]) + 1, 62]),
        coords=[[str(i) for i in range(eval(data_year[0]), eval(data_year[1]) + 1)], [str(i) for i in range(1, 62 + 1)]],
        dims=['year', 'day'])
    return EHDstations_happended_zone


def corr_series(base_series, start, end, delta, EHD, station_happended_num):
    thresholds = np.arange(start, end + delta, delta)
    corr = np.zeros(len(thresholds))
    index = 0
    for threshold in tq.tqdm(thresholds):
        t_s = time_series(threshold, EHD, station_happended_num)
        series = t_s.mean('day').to_numpy()
        corr[index] = np.corrcoef(base_series, (series - series.mean()) / series.std())[0, 1]
        index += 1
    return corr


# 数据读取
data_year = ['1961', '2022']
# 读取CN05.1逐日最高气温数据
CN051_1 = xr.open_dataset(r"E:\data\CN05.1\1961_2021\CN05.1_Tmax_1961_2021_daily_025x025.nc")
CN051_2 = xr.open_dataset(r"E:\data\CN05.1\2022\CN05.1_Tmax_2022_daily_025x025.nc")
CN051_3 = xr.open_dataset(r"E:\data\CN05.1\2023\CN05.1_Tmax_2023_daily_025x025.nc")
Tmax_cn051 = xr.concat([CN051_1, CN051_2, CN051_3], dim='time')
EHD = Tmax_cn051.sel(time=slice(data_year[0] + '-01-01', data_year[1] + '-12-31'))

EHD35 = xr.open_dataset(fr"D:\PyFile\paper1\EHD35.nc")
EHD35 = EHD35.sel(time=EHD35['time.month'].isin([7, 8]))  # 读取缓存
EHD35 = masked(EHD35, r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")
EHD_index = EHD35.groupby('time.year').sum('time').mean('year')  # 截取7-8月发生过高温的格点
EHD_index = EHD_index.where(EHD_index > 0)
station_happended_num = (EHD_index - EHD_index + 1)['tmax']  # 掩膜处理得长江流域极端高温站点数
station_happended_num = station_happended_num.sum()  # 长江流域发生过极端高温的格点数

ols = np.load(r"D:\PyFile\paper1\OLS35.npy")

corr = corr_series(ols, 30, 40, 0.1, EHD, station_happended_num)
np.save(r"D:\PyFile\paper1\thresholds_corr.npy", corr)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(30, 40.1, 0.1), corr)
plt.show()