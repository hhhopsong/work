import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import tqdm as tq

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator

from toolbar.curved_quivers.modplot import *
from toolbar.significance_test import r_test

nanmax = None
def pic(fig, pic_loc, lat, lon, corr_u, corr_v, corr_z, corr_t2m):
    global lev_t, nanmax
    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    ax.set_title(f'{str(picloc)[2]}) Type{str(picloc)[2]}', loc='left', fontsize=10)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    contf = ax.contourf(t2m['lon'], t2m['lat'], corr_t2m[0], cmap=cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
                        levels=lev_t, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
    # 显著性打点
    p_test = np.where(np.abs(corr_t2m[1]) >= r_test(62), 0, np.nan)
    p = ax.quiver(t2m['lon'], t2m['lat'], p_test, p_test, transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=60, color='k', scale=20, headlength=2, headaxislength=2)
    cont = ax.contour(lon, lat, corr_z[0], colors='red', levels=[20, 40, 60], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(lon, lat, corr_z[0], colors='blue', levels=[-60, -40, -20], linestyles='--', linewidths=0.4,
                       transform=ccrs.PlateCarree(central_longitude=0))
    cont.clabel(inline=1, fontsize=4)
    cont_.clabel(inline=1, fontsize=4)
    #cont_clim = ax.contour(lon, lat, uvz_clim['z'], colors='k', levels=20, linewidths=0.6, transform=ccrs.PlateCarree(central_longitude=0))
    if nanmax:
        Cq = Curlyquiver(ax, lon, lat, corr_u[0], corr_v[0], center_lon=110, scale=20, linewidth=0.2, arrowsize=.5,
                         regrid=15, color='k', nanmax=nanmax)
    else:
        Cq = Curlyquiver(ax, lon, lat, corr_u[0], corr_v[0], center_lon=110, scale=20, linewidth=0.2, arrowsize=.5,
                         regrid=15, color='k')
    Cq.key(fig, U=1, label='1 m/s', color='k')
    nanmax = Cq.nanmax
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.2)
    ax.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_长江流域\TPBoundary2500m_长江流域.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='gray', edgecolor='black', linewidth=.5)

    # 刻度线设置
    xticks1 = np.arange(60, 160, 20)
    yticks1 = np.arange(0, 60, 15)
    if pic_loc == 131: ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks1, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)

    ymajorLocator = MultipleLocator(15)  # 先定义xmajorLocator，再进行调用
    ax.yaxis.set_major_locator(ymajorLocator)  # x轴最大刻度
    yminorLocator = MultipleLocator(5)
    ax.yaxis.set_minor_locator(yminorLocator)  # x轴最小刻度
    xmajorLocator = MultipleLocator(30)  # 先定义xmajorLocator，再进行调用
    xminorLocator = MultipleLocator(10)
    ax.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
    ax.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
    # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
    ax.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax.tick_params(axis='both', labelsize=8, colors='black')

    return contf

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


K_type = xr.open_dataset(r"D:\PyFile\p2\data\Time_type_AverFiltAll0.9%_0.3%_3.nc")

try:
    uvz = xr.open_dataset(r"D:\PyFile\p2\data\uvz_78.nc")
except:
    uvz = xr.open_dataset(r"E:\data\ERA5\ERA5_pressLev\era5_pressLev.nc").sel(
        date=slice('1961-01-01', '2023-12-31'),
        pressure_level=[200, 300, 400, 500, 600, 700, 850],
        latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])
    uvz = xr.Dataset(
            {'u': (['time', 'p', 'lat', 'lon'],uvz['u'].data),
                      'v': (['time', 'p', 'lat', 'lon'],uvz['v'].data),
                      'z': (['time', 'p', 'lat', 'lon'],uvz['z'].data)},
                     coords={'time': pd.to_datetime(uvz['date'], format="%Y%m%d"),
                             'p': uvz['pressure_level'].data,
                             'lat': uvz['latitude'].data,
                             'lon': uvz['longitude'].data})
    uvz = uvz.sel(time=slice('1961-01-01', '2022-12-31'))
    uvz = uvz.sel(time=uvz['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    uvz.to_netcdf(r"D:\PyFile\p2\data\uvz_78.nc")
uvz = uvz.sel(p=500).transpose('year', 'lat', 'lon')  # 500hPa
uvz_clim = uvz.mean('year')

try:
    t2m = xr.open_dataset(r"D:\PyFile\p2\data\t2m_78.nc")
except:
    t2m = xr.open_dataset(r"E:\data\ERA5\ERA5_singleLev\ERA5_sgLEv.nc")['t2m']
    t2m = t2m.sel(date=slice('1961-01-01', '2023-12-31'))
    t2m = xr.Dataset(
            {'t2m': (['time', 'lat', 'lon'],t2m.data)},
                     coords={'time': pd.to_datetime(t2m['date'], format="%Y%m%d"),
                             'lat': t2m['latitude'].data,
                             'lon': t2m['longitude'].data})
    t2m = t2m.sel(time=slice('1961-01-01', '2022-12-31'))
    t2m = t2m.sel(time=t2m['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    t2m.to_netcdf(r"D:\PyFile\p2\data\t2m_78.nc")
t2m = t2m.transpose('year', 'lat', 'lon')
t2m_clim = t2m.mean('year')

fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(wspace=0.1)
lev_t = np.array([-.5, -.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4, .5])
for i in K_type['type']:
    picloc = int(100 + len(K_type['type'])*10 + i)
    time_ser = K_type.sel(type=i)['K'].data
    # if i == 2:
    #     time_ser = time_ser - np.polyval(np.polyfit(range(len(time_ser)), time_ser, 1), range(len(time_ser)))
    if i!=2:
        time_ser = (time_ser - time_ser.mean()) / time_ser.std()
        reg_K_u = regress(time_ser, uvz['u'].data)
        reg_K_v = regress(time_ser, uvz['v'].data)
        reg_K_z = regress(time_ser, uvz['z'].data)
        reg_K_t2m = regress(time_ser, t2m['t2m'].data)
    else:
        time_ser = time_ser[:-1]
        time_ser = (time_ser - time_ser.mean()) / time_ser.std()
        reg_K_u = regress(time_ser, uvz['u'].sel(year=slice(1961, 2021)).data)
        reg_K_v = regress(time_ser, uvz['v'].sel(year=slice(1961, 2021)).data)
        reg_K_z = regress(time_ser, uvz['z'].sel(year=slice(1961, 2021)).data)
        reg_K_t2m = regress(time_ser, t2m['t2m'].sel(year=slice(1961, 2021)).data)
    contourfs = pic(fig, picloc, uvz['lat'], uvz['lon'], reg_K_u, reg_K_v, reg_K_z, reg_K_t2m)

# 添加全局colorbar  # 为colorbar腾出空间
cbar_ax = fig.add_axes([0.25, 0.38, 0.5, 0.01]) # [left, bottom, width, height]
cbar = fig.colorbar(contourfs, cax=cbar_ax, orientation='horizontal', drawedges=True)
cbar.locator = ticker.FixedLocator(lev_t)
cbar.set_ticklabels([str(i) for i in lev_t])
cbar.ax.tick_params(labelsize=8, length=0)


plt.savefig(r"D:\PyFile\p2\pic\图4.png", dpi=600, bbox_inches='tight')
plt.show()