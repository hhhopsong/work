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
from matplotlib.ticker import MultipleLocator, FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmaps

from scipy.ndimage import filters
import xarray as xr
import numpy as np
import multiprocessing
import sys
import tqdm as tq
import time
import pandas as pd

from climkit.significance_test import corr_test, r_test
from climkit.TN_WaveActivityFlux import TN_WAF_3D, TN_WAF
from climkit.Cquiver import *
from climkit.data_read import *
from climkit.masked import masked
from climkit.corr_reg import corr, regress
from climkit.lonlat_transform import transform


#—————————————————————————————————————————————————————绘图默认配置————————————————————————————————————————————————————————
PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
plt.rcParams['font.family'] = ['AVHershey Simplex', 'AVHershey Duplex', 'Helvetica']    # 字体为Hershey
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
xticks = np.arange(-180, 181, 30)
yticks = np.arange(-30, 81, 30)
warnings.filterwarnings("ignore", category=RuntimeWarning)



default_clon = 180
## 填色图
default_extent = [-180, 180, -30, 80]  # 子图范围
default_geoticks = {'x': xticks, 'y': yticks,
                    'xmajor': 30, 'xminor': 10,
                    'ymajor': 30, 'yminor': 10}  # 地理坐标刻度
default_fontsize_times = 1.0  # 字体大小倍数
default_shading = False # 填色图数据
default_shading_levels = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])
default_shading_cmap = cmaps.temp_diff_18lev[5:-5]
default_shading_corr = False
default_p_test_drawSet = {'N': 60, 'alpha': 0.1, 'lw': 0.2, 'color': '#FFFFFF'} # 显著性绘制设置, 可为False
default_edgedraw = False # 填色图边缘绘制
## 填色图2
default_extent2 = [-180, 180, -30, 80]  # 子图范围
default_geoticks2 = {'x': xticks, 'y': yticks,
                    'xminor': 10, 'yminor': 10}  # 地理坐标刻度
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
default_contour_corr = False # 等值线相关系数结果
default_p_test_drawSet_corr = {'N': 60, 'alpha': 0.1} # 显著
## 风矢量_1
default_wind_1 = False # 风矢量No.1数据
default_wind_1_set = {'regrid': 15, 'arrowsize': 0.5, 'scale': 5, 'lw': 0.15,
                      'color': 'black', 'thinning': ['25%', 'min'], 'nanmax': 20/3, 'MinDistance': [0.2, 0.1]}
default_wind_1_key_set = {'U': 1, 'label': '1 m/s', 'ud': 7.7, 'lr': None, 'arrowsize': 0.5, 'edgecolor': 'none', 'lw': 0.5}
## 风矢量_2
default_wind_2 = False # 风矢量No.2数据
default_wind_2_set = {'regrid': 12, 'arrowsize': 0.5, 'scale': 5, 'lw': 0.4,
                      'color': 'purple', 'thinning': ['40%', 'min'], 'nanmax': 0.1, 'MinDistance': [0.2, 0.1]}
default_wind_2_key_set = {'U': 0.03, 'label': '0.03 m$^2$/s$^2$', 'ud': 7.7, 'lr': 1.7, 'arrowsize': 0.5, 'edgecolor': 'none', 'lw': 0.5}


def sub_pic(axes_sub, title, extent, geoticks, fontsize_times,
            shading, shading_levels, shading_cmap, shading_corr, cb_draw, p_test_drawSet, edgedraw,
            shading2, shading2_levels, shading2_cmap, shading2_corr, p_test_drawSet2, edgedraw2,
            contour, contour_levels, contour_cmap, contour_corr, cb_draw2, p_test_drawSet_corr,
            wind_1, wind_1_set, wind_1_key_set, bbox_to_anchor_1, loc1,
            wind_2, wind_2_set, wind_2_key_set, bbox_to_anchor_2, loc2,
            rec_Set):
    """
    子图绘制函数
    :param axes_sub: axes对象, Axes = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=180))
    :param title: 子图标题
    :param extent: 子图范围, [xmin, xmax, ymin, ymax], such as [-180, 180, -30, 80]
    :param geoticks: 地理坐标刻度, {'x', 'y', 'xminor', 'yminor'},
                                such as {'x': xticks, 'y': yticks,
                                'xminor': xminorLocator, 'yminor': yminorLocator}
    :param shading:  xr.DataArray对象, ['lat', 'lon']
    :param shading_levels:  shading级别
    :param shading_cmap:  shading颜色映射
    :param cb_draw:  是否绘制色标, bool类型
    :param shading_corr:  shading相关系数结果 ['lat', 'lon']
    :param p_test_drawSet:  显著性绘制设置, {N, alpha, lw, color}, such as {'N': 60, 'alpha': 0.1, 'lw': 0.2, 'color': '#FFFFFF'}
    :param edgedraw:  shading是否有边缘绘制, bool类型
    :param shading2:  xr.DataArray对象, ['lat', 'lon']
    :param shading2_levels:  shading2级别
    :param shading2_cmap:  shading2颜色映射
    :param cb_draw2:  是否绘制色标, bool类型
    :param shading2_corr:  shading2相关系数结果 ['lat', 'lon']
    :param p_test_drawSet2:  显著性绘制设置, {N, alpha, lw, color}, such as {'N': 60, 'alpha': 0.1, 'lw': 0.2, 'color': '#FFFFFF'}
    :param edgedraw2:  shading2是否有边缘绘制, bool类型
    :param contour:  xr.DataArray对象, ['lat', 'lon']
    :param contour_levels:  contour级别, [[负等值线], [正等值线]], such as [[-1, -0.5, -0.2], [0.2, 0.5, 1]]
    :param contour_cmap:  contour颜色, [负等值线颜色, 正等值线颜色], such as ['blue', 'red']
    :param contour_corr:  contour相关系数结果 ['lat', 'lon']
    :param wind_1:  xr.DataArray对象, ['lat', 'lon', 'u', 'v']
    :param wind_1_set:  风矢量设置, {center_lon, regrid, arrowsize, scale, lw,
                                  color, thinning, nanmax, MinDistance},
                                  such as {'center_lon': 180, 'regrid': 1, 'arrowsize': 100, 'scale': 100, 'lw': 0.5,
                                  'color': 'black', 'thinning': 1, 'nanmax': 1, 'MinDistance': 1}
    :param wind_1_key_set:  风矢量图例设置, {U, label, ud, lr, arrowsize, edgecolor, lw},
                                    such as {'U': 1, 'label': '1 m/s', 'ud': 7.7, 'lr': 1.7, 'arrowsize': 0.5, 'edgecolor': 'none', 'lw': 0.5}
    :param bbox_to_anchor_1:  风矢量图例位置, [x0, y0, width, height], such as [0.15, 0.15, 0.1, 0.1]
    :param wind_2:  xr.DataArray对象, ['lat', 'lon', 'u', 'v']
    :param wind_2_set:  风矢量设置, 同上
    :param wind_2_key_set:  风矢量图例设置, 同上
    :param bbox_to_anchor_2:  风矢量图例位置, 同上
    :param rec_Set:  矩形框设置, [{point, color, ls, lw}, such as {'point': [105, 120, 20, 30], 'color': 'blue', 'ls': '--', 'lw': 0.5}, {...}]
    :return:
    """
    # 屏蔽运行时警告 (主要解决 shapely 和 numpy 的除0/buffer 警告)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    def latlon_fmt(ax, xticks1, yticks1, xminorLocator, yminorLocator):
        ax.set_yticks(yticks1)
        ax.set_xticks(xticks1)
        ax.xlocator = FixedLocator(xticks1)
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_minor_locator(yminorLocator)
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.tick_params(axis='both', which='major', direction='out', length=4, width=.5, color='black', bottom=True,
                       left=True)
        ax.tick_params(axis='both', which='minor', direction='out', length=2, width=.2, color='black', bottom=True,
                       left=True)
        ax.tick_params(axis='both', labelsize=6 * fontsize_times, colors='black')

    def rec(ax, point, color='blue', ls='--', lw=0.5):
        x1, x2 = point[:2]
        y1, y2 = point[2:]
        x, y = [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1]
        ax.plot(x, y, color=color, linestyle=ls, lw=lw, transform=ccrs.PlateCarree())

    def dlon(data):
        return data.lon[1] - data.lon[0]

    start_time = time.perf_counter()
    plt.rcParams['hatch.linewidth'] = p_test_drawSet['lw']
    plt.rcParams['hatch.color'] = p_test_drawSet['color']
    axes_sub.set_aspect('auto')
    axes_sub.set_title(title, fontsize=8*fontsize_times, loc='left')
    latlon_fmt(axes_sub, geoticks['x'], geoticks['y'],  MultipleLocator(geoticks['xminor']),
               MultipleLocator(geoticks['yminor']))
    axes_sub.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.7, color='#757575', alpha=0.75)
    axes_sub.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    axes_sub.add_geometries(Reader(fr'{PYFILE}/map/地图线路数据/长江/长江.shp').geometries(), ccrs.PlateCarree(),
                       facecolor='none', edgecolor='blue', linewidth=0.2)
    axes_sub.add_geometries(Reader(fr'{PYFILE}/map/地图线路数据/长江干流_lake/lake_wsg84.shp').geometries(),
                       ccrs.PlateCarree(), facecolor='blue', edgecolor='blue', linewidth=0.05)
    axes_sub.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp').geometries(),
                       ccrs.PlateCarree(), facecolor='gray', edgecolor='gray', linewidth=.1, hatch='.', zorder=10)
    if rec_Set is not None:
        for rec_set in rec_Set:
            rec(axes_sub, rec_set['point'], rec_set['color'], rec_set['ls'], rec_set['lw'])  # 绘制矩形框

    # 判断是否绘制
    shading_signal = True if isinstance(shading, xr.DataArray) or isinstance(shading, xr.Dataset) else False
    shading_corr_signal = True if isinstance(shading_corr, xr.DataArray) or isinstance(shading_corr, xr.Dataset) else False
    shading2_signal = True if isinstance(shading2, xr.DataArray) or isinstance(shading2, xr.Dataset) else False
    shading2_corr_signal = True if isinstance(shading2_corr, xr.DataArray) or isinstance(shading2_corr, xr.Dataset) else False
    contour_signal = True if isinstance(contour, xr.DataArray) or isinstance(contour, xr.Dataset) else False
    contour_corr_signal = True if isinstance(contour_corr, xr.DataArray) or isinstance(contour_corr, xr.Dataset) else False
    wind_1_signal = True if isinstance(wind_1, xr.DataArray) or isinstance(wind_1, xr.Dataset) else False
    wind_2_signal = True if isinstance(wind_2, xr.DataArray) or isinstance(wind_2, xr.Dataset) else False

    # 经度转换
    shading = transform(shading, lon_name='lon', type='360->180') if shading_signal else None
    shading2 = transform(shading2, lon_name='lon', type='360->180') if shading2_signal else None
    contour = transform(contour, lon_name='lon', type='360->180') if contour_signal else None
    wind_1 = transform(wind_1, lon_name='lon', type='360->180') if wind_1_signal else None
    wind_2 = transform(wind_2, lon_name='lon', type='360->180') if wind_2_signal else None
    shading_corr = transform(shading_corr, lon_name='lon', type='360->180') if shading_corr_signal else None
    shading2_corr = transform(shading2_corr, lon_name='lon', type='360->180') if shading2_corr_signal else None
    contour_corr = transform(contour_corr, lon_name='lon', type='360->180') if contour_corr_signal else None

    # 裁剪多余数据, 缩减绘制元素
    if extent[1] == extent[0] + 360:
        axes_sub.set_xlim(extent[0], extent[1])
        axes_sub.set_ylim(extent[2], extent[3])
        shading = shading.salem.roi(corners=((extent[0]+dlon(shading)+1e-5, extent[2]), (extent[1], extent[3]))) if shading_signal else None
        shading2 = shading2.salem.roi(corners=((extent[0]+dlon(shading2)+1e-5, extent[2]), (extent[1], extent[3]))) if shading2_signal else None
        contour = contour.salem.roi(corners=((extent[0]+dlon(contour)+1e-5, extent[2]), (extent[1], extent[3]))) if contour_signal else None
        wind_1 = wind_1.salem.roi(corners=((extent[0]+dlon(wind_1)+1e-5, extent[2]), (extent[1], extent[3]))) if wind_1_signal else None
        wind_2 = wind_2.salem.roi(corners=((extent[0]+dlon(wind_2)+1e-5, extent[2]), (extent[1], extent[3]))) if wind_2_signal else None
        shading_corr = shading_corr.salem.roi(corners=((extent[0]+dlon(shading_corr)+1e-5, extent[2]), (extent[1], extent[3]))) if shading_corr_signal else None
        shading2_corr = shading2_corr.salem.roi(corners=((extent[0]+dlon(shading2_corr)+1e-5, extent[2]), (extent[1], extent[3]))) if shading2_corr_signal else None
        contour_corr = contour_corr.salem.roi(corners=((extent[0]+dlon(contour_corr)+1e-5, extent[2]), (extent[1], extent[3]))) if contour_corr_signal else None
    else:
        axes_sub.set_extent(extent, crs=ccrs.PlateCarree(central_longitude=0))
        roi_shape = ((extent[0], extent[2]), (extent[1], extent[3]))
        shading = shading.salem.roi(corners=roi_shape) if shading_signal else None
        shading2 = shading2.salem.roi(corners=roi_shape) if shading2_signal else None
        contour = contour.salem.roi(corners=roi_shape) if contour_signal else None
        wind_1 = wind_1.salem.roi(corners=roi_shape) if wind_1_signal else None
        wind_2 = wind_2.salem.roi(corners=roi_shape) if wind_2_signal else None
        shading_corr = shading_corr.salem.roi(corners=roi_shape) if shading_corr_signal else None
        shading2_corr = shading2_corr.salem.roi(corners=roi_shape) if shading2_corr_signal else None
        contour_corr = contour_corr.salem.roi(corners=roi_shape) if contour_corr_signal else None

    # 阴影
    if shading_signal:
        shading_data, shading_lon = add_cyclic_point(shading, shading['lon'])
        shading_draw = axes_sub.contourf(shading_lon, shading['lat'], shading_data,
                                               levels=shading_levels,
                                               cmap=shading_cmap,
                                               extend='both', alpha=.75, norm=mcolors.BoundaryNorm(boundaries=shading_levels, ncolors=shading_cmap.N, clip=False),
                                               transform=ccrs.PlateCarree(central_longitude=0))
    else:
        shading_draw = False

    # 阴影图边缘绘制
    if shading_signal and edgedraw:
        axes_sub.contour(shading_lon, shading['lat'], shading_data, colors='white', levels=shading_levels,
                                         linestyles='solid', linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))

    # 显著性检验
    if shading_corr_signal:
        # 去除白线
        shading_corr_data, shading_corr_lon = add_cyclic_point(shading_corr, shading_corr['lon'])
        p_test = np.where(np.abs(shading_corr_data) > r_test(p_test_drawSet['N'], p_test_drawSet['alpha']), 0, np.nan)    # 显著性
        axes_sub.contourf(shading_corr_lon, shading_corr['lat'], p_test, levels=[0, 1], hatches=['////////////', None],
                                  colors="none", add_colorbar=False, transform=ccrs.PlateCarree(central_longitude=0), edgecolor='none', linewidths=0)

    # 阴影2
    if shading2_signal:
        # 去除白线
        shading2_data, shading2_lon = add_cyclic_point(shading2, shading2['lon'])
        shading2_draw = axes_sub.contourf(shading2_lon, shading2['lat'], shading2_data,
                                               levels=shading2_levels,
                                               cmap=shading2_cmap,
                                               extend='both', alpha=.75, norm=mcolors.BoundaryNorm(boundaries=shading2_levels, ncolors=shading2_cmap.N, clip=False),
                                               transform=ccrs.PlateCarree(central_longitude=0))
    else:
        shading2_draw = False

    # 阴影2图边缘绘制
    if shading2_signal and edgedraw2:
        axes_sub.contour(shading2_lon, shading2['lat'], shading2_data, colors='white', levels=shading2_levels,
                                            linestyles='solid', linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))

    # 显著性检验2
    if shading2_corr_signal:
        # 去除白线
        shading2_corr_data, shading2_corr_lon = add_cyclic_point(shading2_corr, shading2_corr['lon'])
        p_test2 = np.where(np.abs(shading2_corr_data) > r_test(p_test_drawSet2['N'], p_test_drawSet2['alpha']), 0, np.nan)    # 显著性
        axes_sub.contourf(shading2_corr_lon, shading2_corr['lat'], p_test2, levels=[0, 1], hatches=['////////////', None],
                                  colors="none", add_colorbar=False, transform=ccrs.PlateCarree(central_longitude=0), edgecolor='none', linewidths=0)

    # 等值线
    if contour_signal:
        # 去除白线
        contour_data, contour_lon = add_cyclic_point(contour, contour['lon'])
        contour_low = axes_sub.contour(contour_lon, contour['lat'], contour_data, colors=contour_cmap[0], linestyles='solid',
                                       levels=contour_levels[0], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
        contour_high = axes_sub.contour(contour_lon, contour['lat'], contour_data, colors=contour_cmap[1], linestyles='solid',
                                        levels=contour_levels[1], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
        clabel1 = contour_low.clabel(inline=1, fontsize=3*fontsize_times, inline_spacing=0)
        clabel2 = contour_high.clabel(inline=1, fontsize=3*fontsize_times, inline_spacing=0)
        clabels = clabel1 + clabel2

        # 循环遍历每个标签，并为它设置一个带白色背景的边界框
        for label in clabels:
            label.set_bbox(dict(facecolor='white',  # 背景色为白色
                                edgecolor='none',  # 无边框
                                pad=0,  # 标签与背景的间距
                                alpha=1  # 背景的透明度 (0.8表示80%不透明)
                                ))

    # 显著性检验2
    if contour_corr_signal:
        # 去除白线
        contour_corr_data, contour_corr_lon = add_cyclic_point(contour_corr, contour_corr['lon'])
        p_test_corr = np.where(contour_corr_data > r_test(p_test_drawSet_corr['N'], p_test_drawSet_corr['alpha']), 0,
                           np.nan)  # 显著性 正
        axes_sub.quiver(contour_corr_lon, contour_corr['lat'], p_test_corr, p_test_corr,
                        transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=40,
                        color=contour_cmap[1], scale=10, width=0.0025)
        p_test_corr = np.where(contour_corr_data < -r_test(p_test_drawSet_corr['N'], p_test_drawSet_corr['alpha']), 0, np.nan)  # 显著性 负
        axes_sub.quiver(contour_corr_lon, contour_corr['lat'], p_test_corr, p_test_corr,
                        transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=40,
                        color=contour_cmap[0], scale=10, width=0.0025)


    # 风矢量No.1
    if wind_1_signal:
        wind1 = axes_sub.Curlyquiver(wind_1['lon'], wind_1['lat'], wind_1['u'], wind_1['v'],
                            arrowsize=wind_1_set['arrowsize'], transform=ccrs.PlateCarree(central_longitude=0),
                            scale=wind_1_set['scale'], linewidth=wind_1_set['lw'], regrid=wind_1_set['regrid'],
                            color=wind_1_set['color'], thinning=wind_1_set['thinning'], nanmax=wind_1_set['nanmax'],
                            MinDistance=wind_1_set['MinDistance'])
        if wind_1_key_set['lr'] is not None:
            wind1.key(U=wind_1_key_set['U'], label=wind_1_key_set['label'], bbox_to_anchor=bbox_to_anchor_1, loc=loc1,
                      edgecolor=wind_1_key_set['edgecolor'], arrowsize=wind_1_key_set['arrowsize'], linewidth=wind_1_key_set['lw'])
        else:
            wind1.key(U=wind_1_key_set['U'], label=wind_1_key_set['label'], bbox_to_anchor=bbox_to_anchor_1, loc=loc1,
                      edgecolor=wind_1_key_set['edgecolor'], arrowsize=wind_1_key_set['arrowsize'], linewidth=wind_1_key_set['lw'])

    # 风矢量No.2
    if wind_2_signal:
        wind2 = axes_sub.Curlyquiver(wind_2['lon'], wind_2['lat'], wind_2['u'], wind_2['v'],
                            arrowsize=wind_2_set['arrowsize'], transform=ccrs.PlateCarree(central_longitude=0),
                            scale=wind_2_set['scale'], linewidth=wind_2_set['lw'], regrid=wind_2_set['regrid'],
                            color=wind_2_set['color'], thinning=wind_2_set['thinning'], nanmax=wind_2_set['nanmax'],
                            MinDistance=wind_2_set['MinDistance'])
        wind2.key(U=wind_2_key_set['U'], label=wind_2_key_set['label'], bbox_to_anchor=bbox_to_anchor_2, loc=loc2,
                  edgecolor=wind_2_key_set['edgecolor'], arrowsize=wind_2_key_set['arrowsize'], linewidth=wind_2_key_set['lw'])
    # 边框显示为黑色
    axes_sub.grid(False)
    for spine in axes_sub.spines.values():
        spine.set_edgecolor('black')
    # 色标
    if shading_signal and cb_draw:
        ax_colorbar = inset_axes(axes_sub, width="3%", height="100%", loc='lower left', bbox_to_anchor=(1.03, 0., 1, 1),
                                 bbox_transform=axes_sub.transAxes, borderpad=0)
        cb1 = plt.colorbar(shading_draw, cax=ax_colorbar, orientation='vertical', drawedges=True)
        cb1.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
        cb1.dividers.set_color('black') # 将colorbar内间隔线调为黑色
        cb1.locator = ticker.FixedLocator(shading_levels)
        cb1.set_ticklabels([str(lev) for lev in shading_levels])
        cb1.ax.tick_params(length=0, labelsize=6*fontsize_times)  # length为刻度线的长度

    # 阴影2色标
    if shading2_signal and cb_draw2:
        ax_colorbar2 = inset_axes(axes_sub, width="3%", height="100%", loc='lower left', bbox_to_anchor=(1.13, 0., 1, 1),
                             bbox_transform=axes_sub.transAxes, borderpad=0) if shading_signal else inset_axes(axes_sub, width="3%", height="100%", loc='lower left', bbox_to_anchor=(1.03, 0., 1, 1), bbox_transform=axes_sub.transAxes, borderpad=0)
        cb2 = plt.colorbar(shading2_draw, cax=ax_colorbar2, orientation='vertical', drawedges=True)
        cb2.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
        cb2.dividers.set_color('black') # 将colorbar内间隔线调为黑色
        cb2.locator = ticker.FixedLocator(shading2_levels)
        cb2.set_ticklabels([str(lev) for lev in shading2_levels])
        cb2.ax.tick_params(length=0, labelsize=6*fontsize_times)  # length为刻度线的长度


    for artist in axes_sub.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)
    # 计算函数运行时长
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"子图 '{title}' 绘制完成, 耗时: {duration:.2f}秒")
    return 0

TR_time = [1961, 2004]
sst = ersst(f"{DATA}/NOAA/ERSSTv5/sst.mnmean.nc", 1961, 2022)
# SLP
slp = era5_s(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc", 1961, 2022, 'msl')
#—————————————————————————————————————————————————————预测因子指数————————————————————————————————————————————————————————
EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
EHCI30 = EHCI.apply(lambda x: (x > 0.6).sum())
EHCI30 = (EHCI30 - EHCI30.mean()) / EHCI30.std('year')
EHCI30 = EHCI30['EHCI'].sel(year=slice(f'1961', f'{TR_time[1]}'))
timeSerie = EHCI30.values

#———————————————————因子1——————————————————
slp_1_0 = slp.sel(time=slp['time.month'].isin([12])).groupby('time.year').mean('time').transpose('year', 'lat', 'lon').shift(year=1).sel(year=slice(f'{TR_time[0]+1}', f'{TR_time[1]}'))
slp_1_1 = slp.sel(time=slp['time.month'].isin([5])).groupby('time.year').mean('time').transpose('year', 'lat', 'lon').sel(year=slice(f'{TR_time[0]+1}', f'{TR_time[1]}'))
slp_1 = slp_1_1 - slp_1_0

X1_zone = [34.0,	-20,	210,	290]
slpReg1, slpCorr1 = regress(timeSerie[1:], slp_1['msl'].data), corr(timeSerie[1:], slp_1['msl'].data)
slpReg1 = xr.DataArray(slpReg1, coords=[slp_1['lat'], slp_1['lon']],
                       dims=['lat', 'lon'], name='slp_reg')
slpCorr1 = xr.DataArray(slpCorr1, coords=[slp_1['lat'], slp_1['lon']],
                        dims=['lat', 'lon'], name='slp_corr')
slpWeight_1 = slpCorr1.sel(lat=slice(X1_zone[0], X1_zone[1]), lon=slice(X1_zone[2], X1_zone[3]))
X1 = slp_1.sel(lat=slice(X1_zone[0], X1_zone[1]), lon=slice(X1_zone[2], X1_zone[3])) * np.where(np.abs(slpWeight_1) > r_test(TR_time[1] - TR_time[0] + 1 - 1, 0.1), slpWeight_1, np.nan)
X1 = X1.mean(['lat', 'lon'])
X1_mean, X1_std = X1.mean(), X1.std()  # 均值和标准差
X1 = (X1 - X1_mean) / X1_std  # 标准化
#———————————————————因子2——————————————————
sst_2_0 = sst.sel(time=sst['time.month'].isin([2])).groupby('time.year').mean('time').transpose('year', 'lat', 'lon').sel(year=slice(f'{TR_time[0]+1}', f'{TR_time[1]}'))
sst_2_1 = sst.sel(time=sst['time.month'].isin([4])).groupby('time.year').mean('time').transpose('year', 'lat', 'lon').sel(year=slice(f'{TR_time[0]+1}', f'{TR_time[1]}'))
sst_2 = sst_2_1 - sst_2_0
X2_zone = [30.0, 10.0, 194.0, 242.0]
sstReg2, sstCorr2 = regress(timeSerie[1:], sst_2['sst'].data), corr(timeSerie[1:], sst_2['sst'].data)
sstReg2 = xr.DataArray(sstReg2, coords=[sst_2['lat'], sst_2['lon']],
                      dims=['lat', 'lon'], name='sst_reg')
sstCorr2 = xr.DataArray(sstCorr2, coords=[sst_2['lat'], sst_2['lon']],
                      dims=['lat', 'lon'], name='sst_corr')
sstWeight_2 = sstCorr2.sel(lat=slice(X2_zone[0], X2_zone[1]), lon=slice(X2_zone[2], X2_zone[3]))
X2 = sst_2.sel(lat=slice(X2_zone[0], X2_zone[1]), lon=slice(X2_zone[2], X2_zone[3])) * np.where(np.abs(sstWeight_2)>r_test(TR_time[1]-TR_time[0]+1-1, 0.1), sstWeight_2, np.nan)
X2 = X2.mean(['lat', 'lon'])
X2_mean, X2_std = X2.mean(), X2.std()  # 均值和标准差
X2 = (X2 - X2_mean) / X2_std  # 标准化

#————————————————————————————————————————————————————————绘图————————————————————————————————————————————————————————————
fig = plt.figure(figsize=np.array([12, 5])*0.7)
fig.subplots_adjust(hspace=0.35)
gs = gridspec.GridSpec(2, 3)  # 设置子图的高度比例
# 绘制子图
for imonth in [6, 7, 8]:
    i = imonth-6
    ax1 = fig.add_subplot(gs[i], projection=ccrs.PlateCarree(central_longitude=180-70))
    X1_slpReg, X1_slpCorr = (regress(X1["msl"].values, slp['msl'].sel(time=sst['time.month'].isin([imonth])).sel(time=slice(f"{TR_time[0] + 1}-01-01", f"{TR_time[1]}-12-31")).values),
                             corr(X1["msl"].values, slp['msl'].sel(time=sst['time.month'].isin([imonth])).sel(time=slice(f"{TR_time[0]+1}-01-01", f"{TR_time[1]}-12-31")).values))
    X1_slpReg = xr.DataArray(X1_slpReg, coords=[slp['lat'], slp['lon']],
                             dims=['lat', 'lon'], name='slp_reg')
    X1_slpCorr = xr.DataArray(X1_slpCorr, coords=[slp['lat'], slp['lon']],
                              dims=['lat', 'lon'], name='slp_corr')
    sub_pic(ax1, title=f'({chr(ord('a')+i)}) Reg_{imonth}_SLP onto X1', extent=[-180, 180, -50, 80],
            geoticks={'x': np.arange(-180, 181, 60), 'y': yticks, 'xminor': 10, 'yminor': 10}, fontsize_times=default_fontsize_times,
            shading=None, shading_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]), shading_cmap=cmaps.GreenMagenta16[8-5:8] + cmaps.GMT_red2green_r[11:11+4],
            shading_corr=None, p_test_drawSet={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'}, edgedraw=False, cb_draw=True,
            shading2=X1_slpCorr, shading2_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]), shading2_cmap=cmaps.BlueWhiteOrangeRed[40:-40],
            shading2_corr=X1_slpCorr, p_test_drawSet2={'N': TR_time[1] - TR_time[0] + 1 - 1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'}, edgedraw2=False, cb_draw2=True if i == 2 else False,
            contour=None, contour_levels=np.array([[-50, -20], [20, 50]])*0.0005, contour_cmap=default_contour_cmap,
            contour_corr=None, p_test_drawSet_corr={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1},
            wind_1=default_wind_1, wind_1_set=default_wind_1_set, wind_1_key_set=default_wind_1_key_set, bbox_to_anchor_1=None, loc1='upper right',
            wind_2=default_wind_2, wind_2_set=default_wind_2_set, wind_2_key_set=default_wind_2_key_set, bbox_to_anchor_2=None, loc2='upper right',
            rec_Set=None)

    ii = imonth-6+3
    ax2 = fig.add_subplot(gs[ii], projection=ccrs.PlateCarree(central_longitude=180-70))
    X2_sstReg, X2_sstCorr = (regress(X2["sst"].values, sst['sst'].sel(time=sst['time.month'].isin([imonth])).sel(time=slice(f"{TR_time[0]+1}-01-01", f"{TR_time[1]}-12-31")).values),
                             corr(X2["sst"].values, sst['sst'].sel(time=sst['time.month'].isin([imonth])).sel(time=slice(f"{TR_time[0]+1}-01-01", f"{TR_time[1]}-12-31")).values))
    X2_sstReg = xr.DataArray(X2_sstReg, coords=[sst['lat'], sst['lon']],
                           dims=['lat', 'lon'], name='sst_reg')
    X2_sstCorr = xr.DataArray(X2_sstCorr, coords=[sst['lat'], sst['lon']],
                            dims=['lat', 'lon'], name='sst_corr')
    sub_pic(ax2, title=f'({chr(ord('a')+ii)}) Reg_{imonth}_SST onto X2', extent=[-180, 180, -50, 80],
            geoticks={'x': np.arange(-180, 181, 60), 'y': yticks, 'xminor': 10, 'yminor': 10}, fontsize_times=default_fontsize_times,
            shading=None, shading_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]), shading_cmap=cmaps.GreenMagenta16[8-5:8] + cmaps.GMT_red2green_r[11:11+4],
            shading_corr=None, p_test_drawSet={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'}, edgedraw=False, cb_draw=True,
            shading2=X2_sstCorr, shading2_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]), shading2_cmap=cmaps.BlueWhiteOrangeRed[40:-40],
            shading2_corr=X2_sstCorr, p_test_drawSet2={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'}, edgedraw2=False, cb_draw2=True if ii==5 else False,
            contour=None, contour_levels=np.array([[-50, -20], [20, 50]])*0.0005, contour_cmap=default_contour_cmap,
            contour_corr=None, p_test_drawSet_corr={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1},
            wind_1=default_wind_1, wind_1_set=default_wind_1_set, wind_1_key_set=default_wind_1_key_set, bbox_to_anchor_1=None, loc1='upper right',
            wind_2=default_wind_2, wind_2_set=default_wind_2_set, wind_2_key_set=default_wind_2_key_set, bbox_to_anchor_2=None, loc2='upper right',
            rec_Set=None)

plt.savefig(f'{PYFILE}/p5/pic/因子发展.pdf', bbox_inches='tight')
plt.savefig(f'{PYFILE}/p5/pic/因子发展.png', dpi=600, bbox_inches='tight')
