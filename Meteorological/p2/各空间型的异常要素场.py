from cartopy import crs as ccrs
import cartopy.feature as cfeature
import multiprocessing
import sys
import cartopy.feature as cfeature
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tq
import xarray as xr
from cartopy import crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
from cartopy.util import add_cyclic_point
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.pyplot import quiverkey
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import filters

from toolbar.significance_test import corr_test
from toolbar.TN_WaveActivityFlux import TN_WAF_3D
from toolbar.curved_quivers.modplot import *
from toolbar.data_read import *

def latlon_fmt(ax):
    pass

def corr(time_series, data):
    # 计算相关系数
    # 将 data 重塑为二维：时间轴为第一个维度
    reshaped_data = data.reshape(len(time_series), -1)

    # 减去均值以标准化
    time_series_mean = time_series - np.mean(time_series)
    data_mean = reshaped_data - np.mean(reshaped_data, axis=0)

    # 计算分子（协方差）
    numerator = np.sum(data_mean * time_series_mean[:, np.newaxis], axis=0)

    # 计算分母（标准差乘积）
    denominator = np.sqrt(np.sum(data_mean ** 2, axis=0)) * np.sqrt(np.sum(time_series_mean ** 2))

    # 相关系数
    correlation = numerator / denominator

    # 重塑为 (lat, lon)
    correlation_map = correlation.reshape(data.shape[1:])
    return correlation_map

K_type = xr.open_dataset(r"D:/PyFile/p2/data/Time_type_35_0.1_4.nc")
# z
z_low = era5_p("E:/data/ERA5/ERA5_pressLev/era5_pressLev.nc", 1961, 2022, [200, 500, 850], 'z')
z_high = era5_hp("E:/data/ERA5/ERA5_pressLev/era5_pressLev_high.nc", 1961, 2022, [100, 150], 'z')
z = xr.concat([z_high, z_low], dim='level')
# u
u_low = era5_p("E:/data/ERA5/ERA5_pressLev/era5_pressLev.nc", 1961, 2022, [200, 500, 850], 'u')
u_high = era5_hp("E:/data/ERA5/ERA5_pressLev/era5_pressLev_high.nc", 1961, 2022, [100, 150], 'u')
u = xr.concat([u_high, u_low], dim='level')
# v
v_low = era5_p("E:/data/ERA5/ERA5_pressLev/era5_pressLev.nc", 1961, 2022, [200, 500, 850], 'v')
v_high = era5_hp("E:/data/ERA5/ERA5_pressLev/era5_pressLev_high.nc", 1961, 2022, [100, 150], 'v')
v = xr.concat([v_high, v_low], dim='level')
# t
t_low = era5_p("E:/data/ERA5/ERA5_pressLev/era5_pressLev.nc", 1961, 2022, [200, 500, 850], 't')
t_high = era5_hp("E:/data/ERA5/ERA5_pressLev/era5_pressLev_high.nc", 1961, 2022, [100, 150], 't')
t = xr.concat([t_high, t_low], dim='level')
# pre
pre = prec("E:/data/NOAA/PREC/precip.mon.anom.nc", 1961, 2022)
# sst
sst = ersst("E:/data/NOAA/ERSSTv5/sst.mnmean.nc", 1961, 2022)

# 计算相关系数
corr_z = np.zeros((len(K_type['type']), len(z['level']), len(z['lat']), len(z['lon'])))
corr_u = np.zeros((len(K_type['type']), len(u['level']), len(u['lat']), len(u['lon'])))
corr_v = np.zeros((len(K_type['type']), len(v['level']), len(v['lat']), len(v['lon'])))
corr_t = np.zeros((len(K_type['type']), len(t['level']), len(t['lat']), len(t['lon'])))
corr_pre = np.zeros((len(K_type['type']), len(pre['lat']), len(pre['lon'])))
corr_sst = np.zeros((len(K_type['type']), len(sst['lat']), len(sst['lon'])))

for i in tq.trange(len(K_type['type'])):
    time_series = K_type.sel(type=i+1)['K'].data
    time_series = time_series - np.polyval(np.polyfit(range(len(time_series)), time_series, 1), range(len(time_series)))
    for j in tq.trange(len(z['level'])):
        lev = z['level'][j].data
        corr_z[i, j] = corr(time_series, z['z'].sel(level=lev[j]).data)
        corr_u[i, j] = corr(time_series, u['u'].sel(level=lev[j]).data)
        corr_v[i, j] = corr(time_series, v['v'].sel(level=lev[j]).data)
        corr_t[i, j] = corr(time_series, t['t'].sel(level=lev[j]).data)
    corr_pre[i] = corr(time_series, pre['pre'].data)
    corr_sst[i] = corr(time_series, sst['sst'].data)