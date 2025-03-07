import xarray as xr
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
from metpy.calc import vertical_velocity
from metpy.units import units

from toolbar.TN_WaveActivityFlux import TN_WAF_3D
from toolbar.curved_quivers.modplot import *
from toolbar.data_read import *
from toolbar.lonlat_transform import *


def regress(time_series, data):
    # 将 data 重塑为二维：时间轴为第一个维度
    reshaped_data = data.reshape(len(time_series), -1)

    # 减去均值以中心化（标准化自变量和因变量）
    time_series_mean = time_series - np.mean(time_series)
    data_mean = reshaped_data - np.mean(reshaped_data, axis=0)

    # 计算分子（协方差的分子）
    numerator = np.sum(data_mean * time_series_mean[:, np.newaxis], axis=0)

    # 计算分母（自变量的平方和）
    denominator = np.sum(time_series_mean ** 2)

    # 计算回归系数
    regression_coef = numerator / denominator
    correlation = numerator / (np.sqrt(np.sum(data_mean ** 2, axis=0)) * np.sqrt(np.sum(time_series_mean ** 2)))
    # 重塑为 (lat, lon)
    regression_map = regression_coef.reshape(data.shape[1:])
    correlation_map = correlation.reshape(data.shape[1:])
    return regression_map, correlation_map


K_type = xr.open_dataset(r"D:/PyFile/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc")
Z = xr.open_dataset(r"D:/PyFile/p2/data/Z.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(5, -5), lon=slice(0, 360)) / 9.8
U = xr.open_dataset(r"D:/PyFile/p2/data/U.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(5, -5), lon=slice(0, 360))
T = xr.open_dataset(r"D:/PyFile/p2/data/T.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(5, -5), lon=slice(0, 360))
W = xr.open_dataset(r"D:/PyFile/p2/data/W.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(5, -5), lon=slice(0, 360))
W = vertical_velocity(W * units('Pa/s') , W['level'] * units.hPa, T['t'] * units.degC)
Pre = xr.open_dataset(r"D:/PyFile/p2/data/pre.nc").sel(lat=slice(5, -5), lon=slice(0, 360))
Sst = xr.open_dataset(r"D:/PyFile/p2/data/sst.nc").sel(lat=slice(5, -5), lon=slice(0, 360))
Terrain = xr.open_dataset(r"E:\data\NOAA\ETOPO\ETOPO_2022_v1_30s_N90W180_bed.nc").sel(lat=slice(-5, 5), lon=slice(-180, 180))['z'].astype(np.float64).mean(dim='lat', skipna=True)
Z = transform(Z['z'], lon_name='lon', type='180->360')
U = transform(U['u'], lon_name='lon', type='180->360')
W = transform(W, lon_name='lon', type='180->360')
Pre = transform(Pre['pre'], lon_name='lon', type='180->360')
Sst = transform(Sst['sst'], lon_name='lon', type='180->360')
Terrain = transform(Terrain, lon_name='lon', type='180->360')

Terrain_ver = np.array(Terrain)
Terrain_ver = 1013 * (1 - 6.5/288000 * Terrain_ver)**5.255
lon_Terrain = Terrain.lon


fig = plt.figure(figsize=(16, 9))
for i in range(len(K_type['type'])):
    K = K_type['K'].sel(type=i+1).data
    if i == 1: K = K - np.polyval(np.polyfit(range(len(K)), K, 1), range(len(K)))  # 去除全域一致型的趋势
    K = (K - np.mean(K)) / np.var(K)
    Z_reg, Z_cor = regress(K, Z['z'].data)
    Z_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], Z_reg),
                        'corr': (['level', 'lat', 'lon'], Z_cor)},
                      coords={'level': Z['level'], 'lat': Z['lat'], 'lon': Z['lon']})
    del Z_reg, Z_cor
    U_reg, U_cor = regress(K, U['u'].data)
    U_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], U_reg),
                       'corr': (['level', 'lat', 'lon'], U_cor)},
                      coords={'level': U['level'], 'lat': U['lat'], 'lon': U['lon']})
    del U_reg, U_cor
    W_reg, W_cor = regress(K, W['w'].data)
    W_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], W_reg),
                       'corr': (['level', 'lat', 'lon'], W_cor)},
                      coords={'level': W['level'], 'lat': W['lat'], 'lon': W['lon']})
    del W_reg, W_cor
    Pre_reg, Pre_cor = regress(K, Pre['pre'].data)
    Pre_nc = xr.Dataset({'reg': (['lat', 'lon'], Pre_reg),
                         'corr': (['lat', 'lon'], Pre_cor)},
                        coords={'lat': Pre['lat'], 'lon': Pre['lon']})
    del Pre_reg, Pre_cor
    Sst_reg, Sst_cor = regress(K, Sst['sst'].data)
    Sst_nc = xr.Dataset({'reg': (['lat', 'lon'], Sst_reg),
                         'corr': (['lat', 'lon'], Sst_cor)},
                        coords={'lat': Sst['lat'], 'lon': Sst['lon']})
    del Sst_reg, Sst_cor

    f_ax = fig.add_subplot(111)
    f_ax.set_title('(a) lev-lon', loc='left', fontsize=18)
    f_ax.set_yscale('symlog')
    f_ax.set_xlim(40, 140)
    f_ax.set_yticks([1000, 850, 700, 500, 300, 200, 100])
    f_ax.set_yticklabels(['1000','850', '700', '500', '300', '200', '100'])
    f_ax.invert_yaxis()
    f_ax.set_ylabel('Level (hPa)', fontsize=18)
    f_ax.set_xlabel('Longitude', fontsize=18)
    f_ax.fill_between(lon_Terrain, Terrain_ver, 1000, where=Terrain_ver < 1000, facecolor='k', zorder=10)


