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
from scipy.ndimage import filters

from toolbar.masked import masked
from toolbar.significance_test import r_test
from toolbar.TN_WaveActivityFlux import TN_WAF_3D
from toolbar.curved_quivers.modplot import *


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


time = [1961, 2022]
t_budget = xr.open_dataset(r'E:\data\ERA5\ERA5_pressLev\single_var\t_budget_1961_2022.nc').sel(
    time=slice(str(time[0]) + '-01', str(time[1]) + '-12'))
dTdt_78 = t_budget['dTdt'].sel(time=t_budget['time.month'].isin([6, 7])).groupby('time.year').mean('time').data
adv_T_78 = t_budget['adv_T'].sel(time=t_budget['time.month'].isin([7, 8])).groupby('time.year').mean('time').data
ver_78 = t_budget['ver'].sel(time=t_budget['time.month'].isin([7, 8])).groupby('time.year').mean('time').data
Q_78 = dTdt_78 - adv_T_78 - ver_78
K_type = xr.open_dataset(r"D:/PyFile/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc")


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


reg_map, corr_map = np.zeros(
    (3, 4, len(t_budget['Q'].level), len(t_budget['Q'].lat), len(t_budget['Q'].lon))), np.zeros(
    (3, 4, len(t_budget['Q'].level), len(t_budget['Q'].lat), len(t_budget['Q'].lon)))

for i in tq.trange(len(K_type['type'])):
    K_series = K_type.sel(type=i + 1)['K'].data
    if i == 1: K_series = K_series - np.polyval(np.polyfit(range(len(K_series)), K_series, 1), range(len(K_series)))
    K_series = (K_series - np.mean(K_series)) / np.std(K_series)
    reg_map[i, 0], corr_map[i, 0] = regress(K_series, dTdt_78)
    reg_map[i, 1], corr_map[i, 1] = regress(K_series, adv_T_78)
    reg_map[i, 2], corr_map[i, 2] = regress(K_series, ver_78)
    reg_map[i, 3], corr_map[i, 3] = regress(K_series, Q_78)

reg_map = xr.Dataset({'dTdt': (['type', 'level', 'lat', 'lon'], reg_map[:, 0]),
                      'adv_T': (['type', 'level', 'lat', 'lon'], reg_map[:, 1]),
                      'ver': (['type', 'level', 'lat', 'lon'], reg_map[:, 2]),
                      'Q': (['type', 'level', 'lat', 'lon'], reg_map[:, 3])},
                     coords={'type': K_type['type'], 'level': t_budget['Q'].level, 'lat': t_budget['Q'].lat,
                             'lon': t_budget['Q'].lon})
corr_map = xr.Dataset({'dTdt': (['type', 'level', 'lat', 'lon'], corr_map[:, 0]),
                       'adv_T': (['type', 'level', 'lat', 'lon'], corr_map[:, 1]),
                       'ver': (['type', 'level', 'lat', 'lon'], corr_map[:, 2]),
                       'Q': (['type', 'level', 'lat', 'lon'], corr_map[:, 3])},
                      coords={'type': K_type['type'], 'level': t_budget['Q'].level, 'lat': t_budget['Q'].lat,
                              'lon': t_budget['Q'].lon})
p_th = r_test(62, 0.05)  # 62为样本量，0.1为显著性水平

for KType in range(1, 4):
    var = ['dTdt', 'adv_T', 'ver', 'Q']
    var_out = [r'$\frac{\partial T}{\partial t}$', r'$-V \cdot \nabla T$', r'$\omega \sigma$', r'$\dot{Q}$']
    if KType == 3:
        reg_map_ = masked(reg_map, r"D:\Code\work\Meteorological\p2\map\WYTR\长江_tp.shp")
    elif KType == 1:
        reg_map_ = masked(reg_map, r"D:\Code\work\Meteorological\p2\map\EYTR\长江_tp.shp")
    elif KType == 2:
        reg_map_ = masked(reg_map, r'D:\PyFile\map\self\长江_TP\长江_tp.shp')
    interp_lon = np.linspace(88, 124, 1000)
    interp_lat = np.linspace(22, 38, 500)
    reg = masked(reg_map.interp(lon=interp_lon, lat=interp_lat), r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").sel(type=KType,level=[850, 900]).mean(dim='level')
    reg_ = reg_map_.sel(type=KType, level=[850, 900]).mean(dim='level')
    corr = corr_map.sel(type=KType, level=850)
    # 绘图
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(hspace=0.2)
    extent_CN = [88, 124, 22, 38]  # Increase vertical spacing between subplots
    gs = gridspec.GridSpec(2, 2)
    xticks1 = np.arange(extent_CN[0], extent_CN[1] + 1, 10)
    yticks1 = np.arange(extent_CN[2], extent_CN[3] + 1, 10)

    for ipic in tq.trange(4):
        ax1 = fig.add_subplot(gs[ipic], projection=ccrs.PlateCarree(central_longitude=180 - 70))
        ax1.set_title(f"{chr(ord('a') + ipic)})Reg. {var_out[ipic]} onto type{KType}", fontsize=12, loc='left')
        ax1.set_extent(extent_CN, crs=ccrs.PlateCarree(central_longitude=0))
        ####
        num_times = [0.1, 10, 10, 10]
        if KType == 2: num_times = [0.4, 10, 10, 10]
        lev_dTdt = np.array([-1., -.8, -.6, -.4, -.2, -.05, .05, .2, .4, .6, .8, 1.]) * num_times[0]
        lev_adv_T = np.array([-1., -.8, -.6, -.4, -.2, -.05, .05, .2, .4, .6, .8, 1.]) * num_times[1]
        lev_ver = np.array([-1., -.8, -.6, -.4, -.2, -.05, .05, .2, .4, .6, .8, 1.]) * num_times[2]
        lev_Q = np.array([-1., -.8, -.6, -.4, -.2, -.05, .05, .2, .4, .6, .8, 1.]) * num_times[3]
        lev = np.array([lev_dTdt, lev_adv_T, lev_ver, lev_Q])
        # lev= 10
        contf = ax1.contourf(reg['lon'], reg['lat'], reg[var[ipic]] * 86400 * 31,
                             cmap=cmaps.GMT_polar[0:9] + cmaps.CBR_wet[0] + cmaps.GMT_polar[11:],
                             levels=lev[ipic], extend='both', transform=ccrs.PlateCarree(central_longitude=0))
        cont = ax1.contour(reg['lon'], reg['lat'], reg[var[ipic]] * 86400 * 31, linestyles='solid',
                           levels=lev[ipic], colors='w', linewidths=0.25, transform=ccrs.PlateCarree(central_longitude=0), zorder=10)
        # 在右上角添加标注 (坐标范围是轴坐标0-1)
        ax1.text(0.75, 1.03,
                 fr'$\times {num_times[ipic]} K/month$',
                 transform=ax1.transAxes,
                 fontsize=10,
                 color='black',
                 bbox=dict(facecolor='none', alpha=0.0, edgecolor='none'))
        ####
        ax1.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),
                           facecolor='none', edgecolor='black', linewidth=.5)
        ax1.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(),
                           ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=.5)
        ax1.add_geometries(Reader(
            r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_长江流域\TPBoundary2500m_长江流域.shp').geometries(),
                           ccrs.PlateCarree(), facecolor='gray', edgecolor='black', linewidth=.5, zorder=11)
        ax1.add_geometries(Reader(r'D:\PyFile\map\地图线路数据\长江\长江.shp').geometries(), ccrs.PlateCarree(),
                           facecolor='none', edgecolor='blue', linewidth=0.2, zorder=12)
        latlon_fmt(ax1, xticks1, yticks1, MultipleLocator(5), MultipleLocator(1), MultipleLocator(4),
                   MultipleLocator(1))
        # 色条
        # 边框显示为黑色
        ax1.spines['top'].set_color('black')
        ax1.spines['right'].set_color('black')
        ax1.spines['bottom'].set_color('black')
        ax1.spines['left'].set_color('black')

    ax1_colorbar = fig.add_axes([0.25, 0.03, 0.5, 0.02])
    cb1 = plt.colorbar(contf, cax=ax1_colorbar, orientation='horizontal', drawedges=True)
    cb1.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
    cb1.dividers.set_color('black')  # 将colorbar内间隔线调为黑色
    cb1.locator = ticker.FixedLocator(lev[ipic])
    cb1.set_ticklabels(['-1', '-0.8', '-0.6', '-0.4', '-0.2', '-0.05', '0.05', '0.2', '0.4', '0.6', '0.8', '1'])
    cb1.ax.tick_params(length=0, labelsize=8)  # length为刻度线的长度
    plt.savefig(f'D:/PyFile/p2/pic/温度收支type{KType}.png', dpi=600, bbox_inches='tight')
    plt.show()

# 创建柱状图
fig = plt.figure(figsize=(9, 3))

for KType in range(1, 4):
    if KType == 3:
        reg_map_ = masked(reg_map, r"D:\Code\work\Meteorological\p2\map\WYTR\长江_tp.shp")
    elif KType == 1:
        reg_map_ = masked(reg_map, r"D:\Code\work\Meteorological\p2\map\EYTR\长江_tp.shp")
    elif KType == 2:
        reg_map_ = masked(reg_map, r'D:\PyFile\map\self\长江_TP\长江_tp.shp')
    reg = masked(reg_map, r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").sel(
        type=KType, level=[850, 900]).mean(dim='level')
    reg_ = reg_map_.sel(type=KType, level=[850, 900]).mean(dim='level')

    adv_X_dTdt = np.nanmean(reg_['adv_T']) * 86400 * 31
    ver_X_dTdt = np.nanmean(reg_['ver']) * 86400 * 31
    Q_X_dTdt = np.nanmean(reg_['Q']) * 86400 * 31

    # 在计算完三个变量之后添加以下代码

    # 准备绘图数据
    variables = ['Adv', 'Ver', 'Q']
    values = [adv_X_dTdt, ver_X_dTdt, Q_X_dTdt]
    colors = ['red' if val > 0 else 'blue' for val in values]


    # 添加标题和网格
    ax = fig.add_subplot(1, 3, KType)
    ax.set_title(f'{chr(ord("a") + KType - 1)})Temp. pert budget of type{KType}', fontsize=12,loc='left')
    ax.grid(True, linestyle='--', zorder=0, axis='y')

    bars = ax.bar(range(3), values, width=0.3, color=colors, edgecolor='black', zorder=2)

    # 设置坐标轴标签,字体为 Times New Roman
    ax.set_xticks(range(3))
    ax.set_xticklabels([r'$-(\mathbf{v} \cdot \nabla T)^{\prime}$',
                        r'$(\omega \sigma)^{\prime}$',
                        r'${\dot{Q}}^{\prime}$'], fontsize=12, fontname='Times New Roman')

    # 设置y轴范围
    ymax = 3
    ax.set_ylim(-ymax, ymax)

    #仅当 KType == 1 时添加y刻度标签
    if KType == 1:
        ax.set_yticks(np.arange(-3, 4, 1))
        ax.set_yticklabels(np.arange(-3, 4, 1), fontsize=12)
    else:
        ax.set_yticks(np.arange(-3, 4, 1))
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)

    # 添加零线
    ax.axhline(0, color='black', lw=1)

    # 边框显示为黑色
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

plt.tight_layout()
plt.savefig(f'D:/PyFile/p2/pic/温度收支type_bar.png', dpi=600, bbox_inches='tight')
plt.show()