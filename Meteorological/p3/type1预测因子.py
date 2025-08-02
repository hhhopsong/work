from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.lines import lineStyles
from matplotlib.pyplot import quiverkey
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmaps

from scipy.ndimage import filters
import xarray as xr
import numpy as np
import multiprocessing
import sys
import tqdm as tq
import time

from toolbar.significance_test import corr_test, r_test
from toolbar.TN_WaveActivityFlux import TN_WAF_3D, TN_WAF
from toolbar.curved_quivers.modplot import *
from toolbar.data_read import *
from toolbar.masked import masked
from toolbar.corr_reg import corr, regress


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


def rec(ax, point, color='blue', ls='--', lw=0.5):
    x1, x2 = point[:2]
    y1, y2 = point[2:]
    x, y = [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1]
    ax.plot(x, y, color=color, linestyle=ls, lw=lw, transform=ccrs.PlateCarree())


def sub_pic(fig, axes_sub, title, extent, geoticks,
            shading, shading_levels, shading_cmap, shading_corr, p_test_drawSet, edgedraw,
            shading2, shading2_levels, shading2_cmap, shading2_corr, p_test_drawSet2, edgedraw2,
            contour, contour_levels, contour_cmap,
            wind_1, wind_1_set, wind_1_key_set,
            wind_2, wind_2_set, wind_2_key_set,
            rec_Set):
    """
    子图绘制函数
    :param fig: 图形对象, fig = plt.figure(figsize=(10, 5))
    :param axes_sub: axes对象, Axes = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=180))
    :param title: 子图标题
    :param extent: 子图范围, [xmin, xmax, ymin, ymax], such as [-180, 180, -30, 80]
    :param geoticks: 地理坐标刻度, {'x', 'y', 'xmajor', 'xminor', 'ymajor', 'yminor'},
                                such as {'x': xticks, 'y': yticks,
                                'xmajor': xmajorLocator, 'xminor': xminorLocator,
                                'ymajor': ymajorLocator, 'yminor': yminorLocator}
    :param shading:  xr.DataArray对象, ['lat', 'lon']
    :param shading_levels:  shading级别
    :param shading_cmap:  shading颜色映射
    :param shading_corr:  shading相关系数结果 ['lat', 'lon']
    :param p_test_drawSet:  显著性绘制设置, {N, alpha, lw, color}, such as {'N': 60, 'alpha': 0.1, 'lw': 0.2, 'color': '#FFFFFF'}
    :param edgedraw:  shading是否有边缘绘制, bool类型
    :param shading2:  xr.DataArray对象, ['lat', 'lon']
    :param shading2_levels:  shading2级别
    :param shading2_cmap:  shading2颜色映射
    :param shading2_corr:  shading2相关系数结果 ['lat', 'lon']
    :param p_test_drawSet2:  显著性绘制设置, {N, alpha, lw, color}, such as {'N': 60, 'alpha': 0.1, 'lw': 0.2, 'color': '#FFFFFF'}
    :param edgedraw2:  shading2是否有边缘绘制, bool类型
    :param contour:  xr.DataArray对象, ['lat', 'lon']
    :param contour_levels:  contour级别, [[负等值线], [正等值线]], such as [[-1, -0.5, -0.2], [0.2, 0.5, 1]]
    :param contour_cmap:  contour颜色, [负等值线颜色, 正等值线颜色], such as ['blue', 'red']
    :param wind_1:  xr.DataArray对象, ['lat', 'lon', 'u', 'v']
    :param wind_1_set:  风矢量设置, {center_lon, regrid, arrowsize, scale, lw,
                                  color, thinning, nanmax, MinDistance},
                                  such as {'center_lon': 180, 'regrid': 1, 'arrowsize': 100, 'scale': 100, 'lw': 0.5,
                                  'color': 'black', 'thinning': 1, 'nanmax': 1, 'MinDistance': 1}
    :param wind_1_key_set:  风矢量图例设置, {U, label, ud, lr, arrowsize, edgecolor, lw},
                                    such as {'U': 1, 'label': '1 m/s', 'ud': 7.7, 'lr': 1.7, 'arrowsize': 0.5, 'edgecolor': 'none', 'lw': 0.5}
    :param wind_2:  xr.DataArray对象, ['lat', 'lon', 'u', 'v']
    :param wind_2_set:  风矢量设置, 同上
    :param wind_2_key_set:  风矢量图例设置, 同上
    :param rec_Set:  矩形框设置, {point, color, ls, lw}, such as {'point': [105, 120, 20, 30], 'color': 'blue', 'ls': '--', 'lw': 0.5}
    :return:
    """
    start_time = time.perf_counter()
    plt.rcParams['hatch.linewidth'] = p_test_drawSet['lw']
    plt.rcParams['hatch.color'] = p_test_drawSet['color']
    axes_sub.set_title(title, fontsize=8, loc='left')
    latlon_fmt(axes_sub, geoticks['x'], geoticks['y'], MultipleLocator(geoticks['xmajor']), MultipleLocator(geoticks['xminor']),
               MultipleLocator(geoticks['ymajor']), MultipleLocator(geoticks['yminor']))
    axes_sub.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.15)
    axes_sub.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    axes_sub.add_geometries(Reader(r'D:\PyFile\map\地图线路数据\长江\长江.shp').geometries(), ccrs.PlateCarree(),
                       facecolor='none', edgecolor='blue', linewidth=0.6)
    axes_sub.add_geometries(Reader(r'D:\PyFile\map\地图线路数据\长江干流_lake\lake_wsg84.shp').geometries(),
                       ccrs.PlateCarree(), facecolor='blue', edgecolor='blue', linewidth=0.2)
    if rec_Set is not None: rec(axes_sub, rec_Set['point'], rec_Set['color'], rec_Set['ls'], rec_Set['lw'])  # 绘制矩形框

    # 判断是否绘制
    shading_signal = True if isinstance(shading, xr.DataArray) else False
    shading_corr_signal = True if isinstance(shading_corr, xr.DataArray) else False
    shading2_signal = True if isinstance(shading2, xr.DataArray) else False
    shading2_corr_signal = True if isinstance(shading2_corr, xr.DataArray) else False
    contour_signal = True if isinstance(contour, xr.DataArray) else False
    wind_1_signal = True if isinstance(wind_1, xr.DataArray) else False
    wind_2_signal = True if isinstance(wind_2, xr.DataArray) else False

    # 裁剪多余数据, 缩减绘制元素
    axes_sub.set_extent(extent, crs=ccrs.PlateCarree(central_longitude=0))
    roi_shape = ((extent[0], extent[2]), (extent[1], extent[3]))
    shading = shading.salem.roi(corners=roi_shape) if shading_signal else None
    shading2 = shading2.salem.roi(corners=roi_shape) if shading2_signal else None
    contour = contour.salem.roi(corners=roi_shape) if contour_signal else None
    wind_1 = wind_1.salem.roi(corners=roi_shape) if wind_1_signal else None
    wind_2 = wind_2.salem.roi(corners=roi_shape) if wind_2_signal else None

    # 阴影
    if shading_signal:
        shading_draw = axes_sub.contourf(shading['lon'], shading['lat'], shading.data,
                                               levels=shading_levels,
                                               cmap=shading_cmap,
                                               extend='both', alpha=.75,
                                               transform=ccrs.PlateCarree(central_longitude=0))
    else:
        shading_draw = False

    # 阴影图边缘绘制
    if shading_signal and shading_draw:  axes_sub.contour(shading['lon'], shading['lat'], shading.data, colors='white', levels=shading_levels,
                                         linestyles='solid', linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))

    # 显著性检验
    if shading_corr_signal:
        p_test = np.where(np.abs(shading_corr) > r_test(p_test_drawSet['N'], p_test_drawSet['alpha']), 0, np.nan)    # 显著性
        axes_sub.contourf(shading_corr['lon'], shading_corr['lat'], p_test, levels=[0, 1], hatches=['////////////', None],
                                  colors="none", add_colorbar=False, transform=ccrs.PlateCarree(central_longitude=0), edgecolor='none', linewidths=0)

    # 阴影2
    if shading2_signal:
        shading2_draw = axes_sub.contourf(shading2['lon'], shading2['lat'], shading2.data,
                                               levels=shading2_levels,
                                               cmap=shading2_cmap,
                                               extend='both', alpha=.75,
                                               transform=ccrs.PlateCarree(central_longitude=0))
    else:
        shading2_draw = False

    # 阴影2图边缘绘制
    if shading2_signal and shading2_draw:  axes_sub.contour(shading2['lon'], shading2['lat'], shading2.data, colors='white', levels=shading2_levels,
                                            linestyles='solid', linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))

    # 显著性检验2
    if shading2_corr_signal:
        p_test2 = np.where(np.abs(shading2_corr) > r_test(p_test_drawSet['N'], p_test_drawSet['alpha']), 0, np.nan)    # 显著性
        axes_sub.contourf(shading2_corr['lon'], shading2_corr['lat'], p_test2, levels=[0, 1], hatches=['////////////', None],
                                  colors="none", add_colorbar=False, transform=ccrs.PlateCarree(central_longitude=0), edgecolor='none', linewidths=0)

    # 等值线
    if contour_signal:
        contour_low = axes_sub.contour(contour['lon'], contour['lat'], contour.data, colors=contour_cmap[0],
                                       levels=contour_levels[0], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
        contour_high = axes_sub.contour(contour['lon'], contour['lat'], contour.data, colors=contour_cmap[1],
                                        levels=contour_levels[1], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
        contour_low.clabel(inline=1, fontsize=3)
        contour_high.clabel(inline=1, fontsize=3)

    # 风矢量No.1
    if wind_1_signal:
        wind1 = Curlyquiver(axes_sub, wind_1['lon'], wind_1['lat'], wind_1['u'], wind_1['v'],
                            center_lon=wind_1_set['center_lon'], arrowsize=wind_1_set['arrowsize'],
                            scale=wind_1_set['scale'], linewidth=wind_1_set['lw'], regrid=wind_1_set['regrid'],
                            color=wind_1_set['color'], thinning=wind_1_set['thinning'], nanmax=wind_1_set['nanmax'],
                            MinDistance=wind_1_set['MinDistance'])
        if wind_1_key_set['lr'] is not None:
            wind1.key(fig, U=wind_1_key_set['U'], label=wind_1_key_set['label'], lr=wind_1_key_set['lr'], ud=wind_1_key_set['ud'],
                      edgecolor=wind_1_key_set['edgecolor'], arrowsize=wind_1_key_set['arrowsize'], linewidth=wind_1_key_set['lw'])
        else:
            wind1.key(fig, U=wind_1_key_set['U'], label=wind_1_key_set['label'], ud=wind_1_key_set['ud'],
                      edgecolor=wind_1_key_set['edgecolor'], arrowsize=wind_1_key_set['arrowsize'], linewidth=wind_1_key_set['lw'])

    # 风矢量No.2
    if wind_2_signal:
        wind2 = Curlyquiver(axes_sub, wind_2['lon'], wind_2['lat'], wind_2['u'], wind_2['v'],
                            center_lon=wind_2_set['center_lon'], arrowsize=wind_2_set['arrowsize'],
                            scale=wind_2_set['scale'], linewidth=wind_2_set['lw'], regrid=wind_2_set['regrid'],
                            color=wind_2_set['color'], thinning=wind_2_set['thinning'], nanmax=wind_2_set['nanmax'],
                            MinDistance=wind_2_set['MinDistance'])
        wind2.key(fig, U=wind_2_key_set['U'], label=wind_2_key_set['label'], lr=wind_2_key_set['lr'], ud=wind_2_key_set['ud'],
                  edgecolor=wind_2_key_set['edgecolor'], arrowsize=wind_2_key_set['arrowsize'], linewidth=wind_2_key_set['lw'])
    # 边框显示为黑色
    axes_sub.grid(False)
    for spine in axes_sub.spines.values():
        spine.set_edgecolor('black')
    # 色标
    if shading_signal:
        ax_colorbar = inset_axes(axes_sub, width="3%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1),
                                 bbox_transform=axes_sub.transAxes, borderpad=0)
        cb1 = plt.colorbar(shading_draw, cax=ax_colorbar, orientation='vertical', drawedges=True)
        cb1.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
        cb1.dividers.set_color('black') # 将colorbar内间隔线调为黑色
        cb1.locator = ticker.FixedLocator(shading_levels)
        cb1.set_ticklabels([str(lev) for lev in shading_levels])
        cb1.ax.tick_params(length=0, labelsize=6)  # length为刻度线的长度

        # 阴影2色标
        if shading2_signal:
            ax_colorbar2 = inset_axes(axes_sub, width="3%", height="100%", loc='lower left', bbox_to_anchor=(1.10, 0., 1, 1),
                                 bbox_transform=axes_sub.transAxes, borderpad=0)
            cb2 = plt.colorbar(shading2_draw, cax=ax_colorbar2, orientation='vertical', drawedges=True)
            cb2.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
            cb2.dividers.set_color('black') # 将colorbar内间隔线调为黑色
            cb2.locator = ticker.FixedLocator(shading2_levels)


    for artist in axes_sub.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)
    # 计算函数运行时长
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"子图 '{title}' 绘制完成, 耗时: {duration:.2f}秒")
    return 0

# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
xticks = np.arange(-180, 180, 10)
yticks = np.arange(-30, 81, 30)

# 下列参数的默认值
# center_lon
# extent, geoticks
# shading, shading_levels, shading_cmap, shading_corr, p_test_drawSet, edgedraw
# contour, contour_levels, contour_cmap
# wind_1, wind_1_set, wind_1_key_set
# wind_2, wind_2_set, wind_2_key_set
# rec_Set
default_clon = 180
## 填色图
default_extent = [-180, 180, -30, 80]  # 子图范围
default_geoticks = {'x': xticks, 'y': yticks,
                    'xmajor': 30, 'xminor': 10,
                    'ymajor': 30, 'yminor': 10}  # 地理坐标刻度
default_shading = False # 填色图数据
default_shading_levels = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])
default_shading_cmap = cmaps.temp_diff_18lev[5:-5]
default_shading_corr = False
default_p_test_drawSet = {'N': 60, 'alpha': 0.1, 'lw': 0.2, 'color': '#FFFFFF'} # 显著性绘制设置, 可为False
default_edgedraw = False # 填色图边缘绘制
## 填色图2
default_extent2 = [-180, 180, -30, 80]  # 子图范围
default_geoticks2 = {'x': xticks, 'y': yticks,
                    'xmajor': 30, 'xminor': 10,
                    'ymajor': 30, 'yminor': 10}  # 地理坐标刻度
default_shading2 = False # 填色图数据
default_shading_levels2 = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])
default_shading_cmap2 = cmaps.temp_diff_18lev[5:-5]
default_shading_corr2 = False
default_p_test_drawSet2 = {'N': 60, 'alpha': 0.1, 'lw': 0.2, 'color': '#FFFFFF'} # 显著性绘制设置, 可为False
default_edgedraw2 = False # 填色图边缘绘制
## 等值线
default_contour = False # 等值线数据
default_contour_levels = [[-1, -0.5, -0.2], [0.2, 0.5, 1]]
default_contour_cmap = ['blue', 'red']
## 风矢量_1
default_wind_1 = False # 风矢量No.1数据
default_wind_1_set = {'center_lon': default_clon, 'regrid': 15, 'arrowsize': 0.5, 'scale': 0.5, 'lw': 0.15,
                      'color': 'black', 'thinning': ['50%', 'min'], 'nanmax': 20/3, 'MinDistance': [0.2, 0.1]}
default_wind_1_key_set = {'U': 1, 'label': '1 m/s', 'ud': 7.7, 'lr': None, 'arrowsize': 0.5, 'edgecolor': 'none', 'lw': 0.5}
## 风矢量_2
default_wind_2 = False # 风矢量No.2数据
default_wind_2_set = {'center_lon': default_clon, 'regrid': 12, 'arrowsize': 0.5, 'scale': 5, 'lw': 0.4,
                      'color': 'purple', 'thinning': ['40%', 'min'], 'nanmax': 0.1, 'MinDistance': [0.2, 0.1]}
default_wind_2_key_set = {'U': 0.03, 'label': '0.03 m$^2$/s$^2$', 'ud': 7.7, 'lr': 1.7, 'arrowsize': 0.5, 'edgecolor': 'none', 'lw': 0.5}
# 矩形框设置, 可为False
default_rec_Set = {'point': [105, 120, 20, 30], 'color': 'blue', 'ls': '--', 'lw': 0.5}

typesTimeSer = xr.open_dataset(r"D:/PyFile/p3/time_ser/typesTimeSer.nc")
# 2mT
t2m = era5_land("E:/data/ERA5/ERA5_land/uv_2mTTd_sfp_pre_0.nc", 1961, 2022, 't2m')
# SLP
slp = era5_s("E:/data/ERA5/ERA5_singleLev/ERA5_sgLEv.nc", 1961, 2022, 'msl')
# sst
sst = ersst("E:/data/NOAA/ERSSTv5/sst.mnmean.nc", 1961, 2022)
# %% 计算6月的z的年平均
t2m_6 = t2m.sel(time=t2m['time.month'].isin([6])).groupby('time.year').mean('time').transpose('year', 'lat', 'lon')
slp_6 = slp.sel(time=slp['time.month'].isin([6])).groupby('time.year').mean('time').transpose('year', 'lat', 'lon')
sst_6 = sst.sel(time=sst['time.month'].isin([6])).groupby('time.year').mean('time').transpose('year', 'lat', 'lon')
# %%
timeSerie = typesTimeSer.sel(type=1)['K'].data
timeSerie = (timeSerie - np.mean(timeSerie))/np.std(timeSerie) # 标准化处理
t2mReg, t2mCorr = regress(timeSerie, t2m_6['t2m'].data), corr(timeSerie, t2m_6['t2m'].data)
slpReg, slpCorr = regress(timeSerie, slp_6['msl'].data), corr(timeSerie, slp_6['msl'].data)
sstReg, sstCorr = regress(timeSerie, sst_6['sst'].data), corr(timeSerie, sst_6['sst'].data)
# %%
t2mReg = xr.DataArray(t2mReg, coords=[t2m_6['lat'], t2m_6['lon']],
                      dims=['lat', 'lon'], name='t2m_reg')
slpReg = xr.DataArray(slpReg, coords=[slp_6['lat'], slp_6['lon']],
                      dims=['lat', 'lon'], name='slp_reg')
sstReg = xr.DataArray(sstReg, coords=[sst_6['lat'], sst_6['lon']],
                      dims=['lat', 'lon'], name='sst_reg')
t2mCorr = xr.DataArray(t2mCorr, coords=[t2m_6['lat'], t2m_6['lon']],
                      dims=['lat', 'lon'], name='t2m_corr')
slpCorr = xr.DataArray(slpCorr, coords=[slp_6['lat'], slp_6['lon']],
                      dims=['lat', 'lon'], name='slp_corr')
sstCorr = xr.DataArray(sstCorr, coords=[sst_6['lat'], sst_6['lon']],
                      dims=['lat', 'lon'], name='sst_corr')
# %%
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(hspace=0.4)  # Increase vertical spacing between subplots
gs = gridspec.GridSpec(3, 1)
lbm = xr.open_dataset(r'D:\PyFile\p2\lbm\type1_apre.nc')
u = lbm['u'][19:25].sel(lev=200).mean('time')
v = lbm['v'][19:25].sel(lev=200).mean('time')
uv = xr.merge([u, v])
# 绘制子图1
ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=180-70))
ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree(central_longitude=180-70))
ax3 = fig.add_subplot(gs[2], projection=ccrs.PlateCarree(central_longitude=180-70))
sub_pic(fig, ax1, title='子图1', extent=[-180, 180, -30, 80],
        geoticks={'x': xticks, 'y': yticks, 'xmajor': 30, 'xminor': 10, 'ymajor': 30, 'yminor': 10},
        shading=t2mReg, shading_levels=default_shading_levels, shading_cmap=default_shading_cmap,
        shading_corr=t2mCorr, p_test_drawSet=default_p_test_drawSet, edgedraw=default_edgedraw,
        shading2=sstReg, shading2_levels=default_shading_levels2, shading2_cmap=default_shading_cmap2,
        shading2_corr=sstCorr, p_test_drawSet2=default_p_test_drawSet2, edgedraw2=default_edgedraw2,
        contour=slpReg, contour_levels=default_contour_levels, contour_cmap=default_contour_cmap,
        wind_1=uv, wind_1_set=default_wind_1_set, wind_1_key_set=default_wind_1_key_set,
        wind_2=default_wind_2, wind_2_set=default_wind_2_set, wind_2_key_set=default_wind_2_key_set,
        rec_Set=default_rec_Set)
plt.show()

