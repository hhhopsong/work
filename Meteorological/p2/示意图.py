from adodbapi.ado_consts import directions
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
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
from cartopy.util import add_cyclic_point
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.pyplot import quiverkey
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import BoundaryNorm
from matplotlib import colors
from scipy.ndimage import filters

from toolbar.significance_test import corr_test
from toolbar.TN_WaveActivityFlux import TN_WAF_3D, TN_WAF
from toolbar.curved_quivers.modplot import *
from toolbar.data_read import *
from toolbar.significance_test import r_test

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
Z = xr.open_dataset(r"D:/PyFile/p2/data/Z.nc").sel(level=[100, 150, 200, 500, 850])
U = xr.open_dataset(r"D:/PyFile/p2/data/U.nc").sel(level=[100, 150, 200, 500, 850])
V = xr.open_dataset(r"D:/PyFile/p2/data/V.nc").sel(level=[100, 150, 200, 500, 850])
T = xr.open_dataset(r"D:/PyFile/p2/data/T.nc").sel(level=[100, 150, 200, 500, 850])
Pre = xr.open_dataset(r"D:/PyFile/p2/data/pre.nc")
Sst = xr.open_dataset(r"D:/PyFile/p2/data/sst.nc")
# 计算相关系数
corr_z = np.zeros((2, len(K_type['type']), len(Z['level']), len(Z['lat']), len(Z['lon'])))
reg_z = np.zeros((len(K_type['type']), len(Z['level']), len(Z['lat']), len(Z['lon'])))
corr_u = np.zeros((2, len(K_type['type']), len(U['level']), len(U['lat']), len(U['lon'])))
corr_v = np.zeros((2, len(K_type['type']), len(V['level']), len(V['lat']), len(V['lon'])))
corr_t = np.zeros((2, len(K_type['type']), len(T['level']), len(T['lat']), len(T['lon'])))
corr_pre = np.zeros((2, len(K_type['type']), len(Pre['lat']), len(Pre['lon'])))
corr_sst = np.zeros((2, len(K_type['type']), len(Sst['lat']), len(Sst['lon'])))

for i in tq.trange(len(K_type['type'])):
    time_series = K_type.sel(type=i+1)['K'].data
    if i != 1:
        time_series = (time_series - np.mean(time_series))/np.std(time_series)
        for j in tq.trange(len(Z['level'])):
            lev = Z['level'][j].data
            corr_z[0, i, j], corr_z[1, i, j] = regress(time_series, Z['z'].sel(level=lev).data)
            corr_u[0, i, j], corr_u[1, i, j] = regress(time_series, U['u'].sel(level=lev).data)
            corr_v[0, i, j], corr_v[1, i, j] = regress(time_series, V['v'].sel(level=lev).data)
            corr_t[0, i, j], corr_t[1, i, j] = regress(time_series, T['t'].sel(level=lev).data)
            reg_z[i, j] = np.array([np.polyfit(time_series, f, 1)[0] for f in Z['z'].sel(level=lev).transpose('lat', 'lon', 'year').data.reshape(-1,len(time_series))]).reshape(Z['z'].sel(level=lev).data.shape[1], Z['z'].sel(level=lev).data.shape[2])
        corr_pre[0, i], corr_pre[1, i] = regress(time_series, Pre['pre'].data)
        corr_sst[0, i], corr_sst[1, i] = regress(time_series, Sst['sst'].data)
    else:
        time_series = time_series[:-1]
        time_series = (time_series - np.mean(time_series))/np.std(time_series)
        for j in tq.trange(len(Z['level'])):
            lev = Z['level'][j].data
            corr_z[0, i, j], corr_z[1, i, j] = regress(time_series, Z['z'].sel(level=lev, year=slice(1961, 2021)).data)
            corr_u[0, i, j], corr_u[1, i, j] = regress(time_series, U['u'].sel(level=lev, year=slice(1961, 2021)).data)
            corr_v[0, i, j], corr_v[1, i, j] = regress(time_series, V['v'].sel(level=lev, year=slice(1961, 2021)).data)
            corr_t[0, i, j], corr_t[1, i, j] = regress(time_series, T['t'].sel(level=lev, year=slice(1961, 2021)).data)
            reg_z[i, j] = np.array([np.polyfit(time_series, f, 1)[0] for f in Z['z'].sel(level=lev, year=slice(1961, 2021)).transpose('lat', 'lon', 'year').data.reshape(-1,len(time_series))]).reshape(Z['z'].sel(level=lev, year=slice(1961, 2021)).data.shape[1], Z['z'].sel(level=lev, year=slice(1961, 2021)).data.shape[2])
        corr_pre[0, i], corr_pre[1, i] = regress(time_series, Pre['pre'].sel(year=slice(1961, 2021)).data)
        corr_sst[0, i], corr_sst[1, i] = regress(time_series, Sst['sst'].sel(year=slice(1961, 2021)).data)

corr_z = xr.Dataset({'corr': (['type', 'level', 'lat', 'lon'], corr_z[1]),
                     'reg': (['type', 'level', 'lat', 'lon'], corr_z[0])},
                    coords={'type': K_type['type'].data,
                            'level': Z['level'].data,
                            'lat': Z['lat'].data,
                            'lon': Z['lon'].data}).interp(lon=np.arange(0, 360, 0.5), lat=np.arange(-90, 90.1, 0.5))
corr_u = xr.Dataset({'corr': (['type', 'level',  'lat', 'lon'], corr_u[1]),
                    'reg': (['type', 'level', 'lat', 'lon'], corr_u[0])},
                    coords={'type': K_type['type'].data,
                            'level': U['level'].data,
                            'lat': U['lat'].data,
                            'lon': U['lon'].data}).interp(lon=np.arange(0, 360, 0.5), lat=np.arange(-90, 90.1, 0.5))
corr_v = xr.Dataset({'corr': (['type', 'level', 'lat', 'lon'], corr_v[1]),
                   'reg': (['type', 'level', 'lat', 'lon'], corr_v[0])},
                    coords={'type': K_type['type'].data,
                            'level': V['level'].data,
                            'lat': V['lat'].data,
                            'lon': V['lon'].data}).interp(lon=np.arange(0, 360, 0.5), lat=np.arange(-90, 90.1, 0.5))
corr_t = xr.Dataset({'corr': (['type', 'level', 'lat', 'lon'], corr_t[1]),
                   'reg': (['type', 'level', 'lat', 'lon'], corr_t[0])},
                    coords={'type': K_type['type'].data,
                            'level': T['level'].data,
                            'lat': T['lat'].data,
                            'lon': T['lon'].data}).interp(lon=np.arange(0, 360, 0.5), lat=np.arange(-90, 90.1, 0.5))
corr_pre = xr.Dataset({'corr': (['type', 'lat', 'lon'], corr_pre[1]),
                      'reg': (['type', 'lat', 'lon'], corr_pre[0])},
                      coords={'type': K_type['type'].data,
                              'lat': Pre['lat'].data,
                              'lon': Pre['lon'].data}).interp(lon=np.arange(0, 360, 0.5), lat=np.arange(-90, 90.1, 0.5), kwargs={"fill_value": "extrapolate"})
corr_sst = xr.Dataset({'corr': (['type', 'lat', 'lon'], corr_sst[1]),
                     'reg': (['type', 'lat', 'lon'], corr_sst[0])},
                      coords={'type': K_type['type'].data,
                              'lat': Sst['lat'].data,
                              'lon': Sst['lon'].data})
def latlon_fmt(ax, xticks1, yticks1, xmajorLocator, xminorLocator, ymajorLocator, yminorLocator):
    ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks1, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.tick_params(which='major', length=4, width=.5, color='black')
    ax.tick_params(which='minor', length=2, width=.2, color='black')
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'
    ax.tick_params(axis='both', labelsize=6, colors='black')

U['u'].sel(level=[150, 200, 500]).mean('year').data


# 绘图
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(hspace=0.4)  # Increase vertical spacing between subplots
gs = gridspec.GridSpec(3, 1)

xticks1 = np.arange(-180, 180, 10)
yticks1 = np.arange(-30, 81, 30)
ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=180-70))
ax1.set_title(f"a)Type1", fontsize=8, loc='left')
# ax1.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.25)
ax1.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),
                   facecolor='none', edgecolor='black', linewidth=.5)
ax1.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
ax1.set_extent([-180, 180, -30, 80], crs=ccrs.PlateCarree(central_longitude=0))
latlon_fmt(ax1, xticks1, yticks1, MultipleLocator(60), MultipleLocator(10), MultipleLocator(30),
           MultipleLocator(10))
# z
positive_values = corr_z['reg'].sel(type=1, level=500).data[
    corr_z['reg'].sel(type=1, level=500).data > 0]
q_positive = np.round(np.percentile(positive_values, 30) / 9.8) if positive_values.size > 0 else 0
positive_values = corr_z['reg'].sel(type=1, level=500).data[
    corr_z['reg'].sel(type=1, level=500).data < 0]
q_positive_ = np.round(np.percentile(positive_values, 30) / 9.8) if positive_values.size > 0 else 0
z_high = ax1.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(type=1, level=500) / 9.8, colors='red',
                     levels=[q_positive], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
z_low = ax1.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(type=1, level=500) / 9.8, colors='blue',
                    levels=[q_positive_], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
z_high.clabel(inline=1, fontsize=3)
z_low.clabel(inline=1, fontsize=3)
reg_sst_, lon = add_cyclic_point(corr_sst['reg'].sel(type=1), coord=corr_sst['lon'])
# sst
lev_sst = np.array([-.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4])
reg_sst_ = np.where((np.abs(reg_sst_) < 0.05), np.nan, reg_sst_)
sst = ax1.contourf(lon, corr_sst['lat'], reg_sst_,
                   cmap=cmaps.GMT_polar[2:10 - 2] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10 + 2:-2], levels=lev_sst,
                   extend='both', transform=ccrs.PlateCarree(central_longitude=0), alpha=0.75)

ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree(central_longitude=180-70))
ax2.set_title(f"b)Type2", fontsize=8, loc='left')
# ax2.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.25)
ax2.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
ax2.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
ax2.set_extent([-180, 180, -30, 80], crs=ccrs.PlateCarree(central_longitude=0))
latlon_fmt(ax2, xticks1, yticks1, MultipleLocator(60), MultipleLocator(10), MultipleLocator(30), MultipleLocator(10))
# z
positive_values = corr_z['reg'].sel(type=2, level=500).data[corr_z['reg'].sel(type=2, level=500).data > 0]
q_positive = np.round(np.percentile(positive_values, 70)/9.8) if positive_values.size > 0 else 0
positive_values = corr_z['reg'].sel(type=2, level=500).data[corr_z['reg'].sel(type=2, level=500).data < 0]
q_positive_ = np.round(np.percentile(positive_values, 70)/9.8) if positive_values.size > 0 else 0
z_high = ax2.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(type=2, level=500)/9.8, colors='red', levels=[q_positive], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
z_low = ax2.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(type=2, level=500)/9.8, colors='blue', levels=[q_positive_], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
z_high.clabel(inline=1, fontsize=3)
z_low.clabel(inline=1, fontsize=3)
reg_sst_, lon = add_cyclic_point(corr_sst['reg'].sel(type=2), coord=corr_sst['lon'])
# sst
lev_sst = np.array([-.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4])
reg_sst_ = np.where((np.abs(reg_sst_) < 0.05), np.nan, reg_sst_)
sst = ax2.contourf(lon, corr_sst['lat'], reg_sst_, cmap=cmaps.GMT_polar[2:10-2] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10+2:-2], levels=lev_sst, extend='both', transform=ccrs.PlateCarree(central_longitude=0), alpha=0.75)

ax3 = fig.add_subplot(gs[2], projection=ccrs.PlateCarree(central_longitude=110))
ax3.set_title(f"c)Type3", fontsize=8, loc='left')
ax3.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
ax3.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
ax3.set_extent([-180, 180, -30, 80], crs=ccrs.PlateCarree(central_longitude=0))
latlon_fmt(ax3, xticks1, yticks1, MultipleLocator(60), MultipleLocator(10), MultipleLocator(30), MultipleLocator(10))
# z
positive_values = corr_z['reg'].sel(type=3, level=500).data[corr_z['reg'].sel(type=3, level=500).data > 0]
q_positive = np.round(np.percentile(positive_values, 50)/9.8) if positive_values.size > 0 else 0
positive_values = corr_z['reg'].sel(type=3, level=500).data[corr_z['reg'].sel(type=3, level=500).data < 0]
q_positive_ = np.round(np.percentile(positive_values, 30)/9.8) if positive_values.size > 0 else 0
z_high = ax3.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(type=3, level=500)/9.8, colors='red', levels=[q_positive], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
z_low = ax3.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(type=3, level=500)/9.8, colors='blue', levels=[q_positive_], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
z_high.clabel(inline=1, fontsize=3)
z_low.clabel(inline=1, fontsize=3)
reg_sst_, lon = add_cyclic_point(corr_sst['reg'].sel(type=3), coord=corr_sst['lon'])
# sst
lev_sst = np.array([-.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4])
reg_sst_ = np.where((np.abs(reg_sst_) < 0.05), np.nan, reg_sst_)
sst = ax3.contourf(lon, corr_sst['lat'], reg_sst_, cmap=cmaps.GMT_polar[2:10-2] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10+2:-2], levels=lev_sst, extend='both', transform=ccrs.PlateCarree(central_longitude=0), alpha=0.75)

plt.savefig(f'D:/PyFile/p2/pic/示意图.svg', bbox_inches='tight')
plt.show()