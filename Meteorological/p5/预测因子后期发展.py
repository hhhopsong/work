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
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
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
## 风��量_2
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

def prepare_swvl_dataset(swvl_like):
    """
    Normalize soil moisture field to a stable {'swvl': [time, lat, lon]} schema.
    支持 DataArray / Dataset
    """
    if isinstance(swvl_like, xr.DataArray):
        da = swvl_like.copy()
        if da.name is None:
            da = da.rename('swvl')
        elif da.name != 'swvl':
            da = da.rename('swvl')
        ds = da.to_dataset()

    elif isinstance(swvl_like, xr.Dataset):
        ds = swvl_like.copy()

        rename_map = {}
        if 'latitude' in ds.coords and 'lat' not in ds.coords:
            rename_map['latitude'] = 'lat'
        if 'longitude' in ds.coords and 'lon' not in ds.coords:
            rename_map['longitude'] = 'lon'
        if len(rename_map) > 0:
            ds = ds.rename(rename_map)

        if 'swvl' not in ds.data_vars:
            for cand in ('swvl1', 'soil_moisture', 'sm'):
                if cand in ds.data_vars:
                    ds = ds.rename({cand: 'swvl'})
                    break
    else:
        raise TypeError("swvl_like must be xarray.DataArray or xarray.Dataset")

    rename_map = {}
    if 'latitude' in ds.coords and 'lat' not in ds.coords:
        rename_map['latitude'] = 'lat'
    if 'longitude' in ds.coords and 'lon' not in ds.coords:
        rename_map['longitude'] = 'lon'
    if len(rename_map) > 0:
        ds = ds.rename(rename_map)

    if 'swvl' not in ds.data_vars:
        raise ValueError(f"SWVL variable not found in dataset. Available vars: {list(ds.data_vars)}")

    out = xr.Dataset({'swvl': ds['swvl']})
    out = out.sortby('lat').sortby('lon')
    return out

TR_time = [1962, 2006]
sst = ersst(f"{DATA}/NOAA/ERSSTv5/sst.mnmean.nc", 1962, 2022)
# SLP
slp = era5_s(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc", 1961, 2022, 'msl')
swvl1 = era5_land(fr"{DATA}/ERA5/ERA5_land/sm.nc", 1961, 2022, 'swvl1')
swvl2 = era5_land(fr"{DATA}/ERA5/ERA5_land/sm.nc", 1961, 2022, 'swvl2')
if isinstance(swvl1, xr.Dataset):
    swvl1_da = swvl1['swvl1']
else:
    swvl1_da = swvl1
if isinstance(swvl2, xr.Dataset):
    swvl2_da = swvl2['swvl2']
else: swvl2_da = swvl2
sm = prepare_swvl_dataset((swvl1_da + swvl2_da).rename('swvl'))
#—————————————————————————————————————————————————————预测因子指数————————————————————————————————————————————————————————
EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
EHCI30 = EHCI.apply(lambda x: (x > 0.6).sum())
EHCI30 = (EHCI30 - EHCI30.mean()) / EHCI30.std('year')
EHCI30 = EHCI30['EHCI'].sel(year=slice(f'{TR_time[0]}', f'{TR_time[1]}'))
timeSerie = EHCI30.values

#———————————————————因子1——————————————————
sm_1 = sm.sel(time=sm['time.month'].isin([5, 6])).groupby('time.year').mean('time').transpose('year', 'lat', 'lon').sel(year=slice(f'{TR_time[0]}', f'{TR_time[1]}'))

X1_zone = [50,  68,  96,  136]
slpReg1, slpCorr1 = regress(timeSerie, sm_1['swvl'].data), corr(timeSerie, sm_1['swvl'].data)
slpReg1 = xr.DataArray(slpReg1, coords=[sm_1['lat'], sm_1['lon']],
                       dims=['lat', 'lon'], name='slp_reg')
slpCorr1 = xr.DataArray(slpCorr1, coords=[sm_1['lat'], sm_1['lon']],
                        dims=['lat', 'lon'], name='slp_corr')
slpWeight_1 = slpCorr1.sel(lat=slice(X1_zone[0], X1_zone[1]), lon=slice(X1_zone[2], X1_zone[3]))
X1 = sm_1.sel(lat=slice(X1_zone[0], X1_zone[1]), lon=slice(X1_zone[2], X1_zone[3])) * np.where(np.abs(slpWeight_1) > r_test(TR_time[1] - TR_time[0] + 1, 0.1), slpWeight_1, np.nan)
X1 = X1.mean(['lat', 'lon'])
X1_mean, X1_std = X1.mean(), X1.std()  # 均值和标准差
X1 = (X1 - X1_mean) / X1_std  # 标准化
#———————————————————因子2——————————————————
sst_2 = sst.sel(time=sst['time.month'].isin([5, 6])).groupby('time.year').mean('time').transpose('year', 'lat', 'lon').sel(year=slice(f'{TR_time[0]}', f'{TR_time[1]}'))
X2_zone = [10, -10, 360-150,  360-80]
sstReg2, sstCorr2 = regress(timeSerie, sst_2['sst'].data), corr(timeSerie, sst_2['sst'].data)
sstReg2 = xr.DataArray(sstReg2, coords=[sst_2['lat'], sst_2['lon']],
                      dims=['lat', 'lon'], name='sst_reg')
sstCorr2 = xr.DataArray(sstCorr2, coords=[sst_2['lat'], sst_2['lon']],
                      dims=['lat', 'lon'], name='sst_corr')
sstWeight_2 = sstCorr2.sel(lat=slice(X2_zone[0], X2_zone[1]), lon=slice(X2_zone[2], X2_zone[3]))
X2 = sst_2.sel(lat=slice(X2_zone[0], X2_zone[1]), lon=slice(X2_zone[2], X2_zone[3])) * np.where(np.abs(sstWeight_2)>r_test(TR_time[1]-TR_time[0]+1, 0.1), sstWeight_2, np.nan)
X2 = X2.mean(['lat', 'lon'])
X2_mean, X2_std = X2.mean(), X2.std()  # 均值和标准差
X2 = (X2 - X2_mean) / X2_std  # 标准化

#————————————————————————————————————————————————————————7-8月平均综合绘图————————————————————————————————————————————————————————————
import warnings
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore", category=RuntimeWarning)

Z_TO_GPM = True
G0 = 9.80665


def ensure_cmap(cmap_like, name='custom_cmap'):
    """
    保证传入 sub_pic 的 cmap 具有 .N 属性。
    cmaps 切片有时返回 list，需要转成 ListedColormap。
    """
    if hasattr(cmap_like, 'N'):
        return cmap_like
    return mcolors.ListedColormap(cmap_like, name=name)


def rename_common(ds):
    """统一 ERA5/ERSST 常见维度和坐标名"""
    rename_map = {}

    pairs = [
        ('latitude', 'lat'),
        ('longitude', 'lon'),
        ('pressure_level', 'p'),
        ('level', 'p'),
        ('date', 'time')
    ]

    for old, new in pairs:
        if old in ds.dims or old in ds.coords:
            if new not in ds.dims and new not in ds.coords:
                rename_map[old] = new

    if rename_map:
        ds = ds.rename(rename_map)

    return ds


def to_78_yearly(ds, varname=None, start=1961, end=2022):
    """
    输入 Dataset/DataArray，输出 7-8 月平均年序列 Dataset: [year, lat, lon]
    如果输入已经是 year 维度，则只做命名整理和裁剪。
    """
    if isinstance(ds, xr.DataArray):
        if varname is None:
            varname = ds.name if ds.name is not None else 'var'
        ds = ds.rename(varname).to_dataset()

    ds = rename_common(ds)

    if varname is not None:
        if varname not in ds.data_vars:
            raise ValueError(f"{varname} not found. Available vars: {list(ds.data_vars)}")
        ds = ds[[varname]]

    if 'time' in ds.dims or 'time' in ds.coords:
        tvals = ds['time'].values

        if np.issubdtype(tvals.dtype, np.integer):
            ds = ds.assign_coords(time=pd.to_datetime(tvals.astype(str), format="%Y%m%d"))
        else:
            ds = ds.assign_coords(time=pd.to_datetime(tvals))

        ds = ds.sel(time=slice(f'{start}-01-01', f'{end}-12-31'))
        ds = ds.sel(time=ds['time.month'].isin([7, 8])).groupby('time.year').mean('time')

    if 'year' not in ds.dims and 'year' not in ds.coords:
        raise ValueError("输入数据既没有 time 维，也没有 year 维，无法处理为 7-8 月平均。")

    ds = ds.sel(year=slice(start, end))
    ds = ds.sortby('lat').sortby('lon')

    return ds


def get_factor_da(factor_obj, preferred_name):
    """从 X1/X2 中取出因子序列 DataArray"""
    if isinstance(factor_obj, xr.Dataset):
        if preferred_name in factor_obj.data_vars:
            return factor_obj[preferred_name]
        return factor_obj[list(factor_obj.data_vars)[0]]
    return factor_obj


def regcorr_2d(index_da, field_da, varname):
    """
    index_da: [year]
    field_da: [year, lat, lon]
    返回 regression 和 correlation 的 DataArray
    """
    field_da = field_da.sel(year=slice(TR_time[0], TR_time[1])).transpose('year', 'lat', 'lon')
    index_da = index_da.sel(year=field_da['year'])

    reg_data = regress(index_da.values, field_da.values)
    corr_data = corr(index_da.values, field_da.values)

    reg_da = xr.DataArray(
        reg_data,
        coords=[field_da['lat'], field_da['lon']],
        dims=['lat', 'lon'],
        name=f'{varname}_reg'
    )

    corr_da = xr.DataArray(
        corr_data,
        coords=[field_da['lat'], field_da['lon']],
        dims=['lat', 'lon'],
        name=f'{varname}_corr'
    )

    return reg_da, corr_da


def make_waf_200(z200_reg, uvz78):
    """
    用 200 hPa 气候态 U/V 和 200 hPa Z 回归场计算 TN-WAF。
    返回满足 sub_pic() 的风矢量格式 Dataset: {'u', 'v'}。
    """
    uvz200_clim = uvz78.sel(p=200).mean('year')

    Uc = xr.DataArray(
        uvz200_clim['u'].values[np.newaxis, :, :],
        coords=[
            ('level', [200]),
            ('lat', uvz78['lat'].values),
            ('lon', uvz78['lon'].values)
        ]
    )

    Vc = xr.DataArray(
        uvz200_clim['v'].values[np.newaxis, :, :],
        coords=[
            ('level', [200]),
            ('lat', uvz78['lat'].values),
            ('lon', uvz78['lon'].values)
        ]
    )

    GEOa = xr.DataArray(
        z200_reg.values[np.newaxis, :, :],
        coords=[
            ('level', [200]),
            ('lat', z200_reg['lat'].values),
            ('lon', z200_reg['lon'].values)
        ]
    )

    waf_x, waf_y = TN_WAF_3D(Uc, Vc, GEOa)

    if isinstance(waf_x, xr.DataArray):
        waf_x_da = waf_x.squeeze(drop=True)
    else:
        waf_x_da = xr.DataArray(
            np.squeeze(waf_x),
            coords=[z200_reg['lat'], z200_reg['lon']],
            dims=['lat', 'lon']
        )

    if isinstance(waf_y, xr.DataArray):
        waf_y_da = waf_y.squeeze(drop=True)
    else:
        waf_y_da = xr.DataArray(
            np.squeeze(waf_y),
            coords=[z200_reg['lat'], z200_reg['lon']],
            dims=['lat', 'lon']
        )

    waf = xr.Dataset(
        {
            'u': waf_x_da,
            'v': waf_y_da
        }
    )

    waf = waf.assign_coords(lat=z200_reg['lat'], lon=z200_reg['lon'])
    return waf


#————————————————————读取 / 整理 7-8 月平均资料————————————————————

# 200/500 hPa U/V/Z
try:
    uvz78 = xr.open_dataset(fr"{PYFILE}/p2/data/uvz_78.nc")
    uvz78 = rename_common(uvz78)

except FileNotFoundError:
    uvz_raw = xr.open_dataset(fr"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc").sel(
        date=slice('1961-01-01', '2022-12-31'),
        pressure_level=[200, 500],
        latitude=[90 - i * 0.5 for i in range(361)],
        longitude=[i * 0.5 for i in range(720)]
    )

    uvz78 = xr.Dataset(
        {
            'u': (['time', 'p', 'lat', 'lon'], uvz_raw['u'].data),
            'v': (['time', 'p', 'lat', 'lon'], uvz_raw['v'].data),
            'z': (['time', 'p', 'lat', 'lon'], uvz_raw['z'].data)
        },
        coords={
            'time': pd.to_datetime(uvz_raw['date'].values.astype(str), format="%Y%m%d"),
            'p': uvz_raw['pressure_level'].data,
            'lat': uvz_raw['latitude'].data,
            'lon': uvz_raw['longitude'].data
        }
    )

    uvz78 = uvz78.sel(time=uvz78['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    uvz78.to_netcdf(fr"{PYFILE}/p2/data/uvz_78.nc")

uvz78 = uvz78.sel(year=slice(TR_time[0], TR_time[1]))
uvz78 = uvz78.transpose('year', 'p', 'lat', 'lon')
uvz78 = uvz78.sortby('lat').sortby('lon')

if 200 not in uvz78['p'].values or 500 not in uvz78['p'].values:
    raise ValueError(f"uvz78 中必须同时包含 200 和 500 hPa。当前 p = {uvz78['p'].values}")

# ERA5 z 原始变量通常是 geopotential，单位 m2/s2；这里转为 gpm
if Z_TO_GPM:
    if float(np.nanmean(np.abs(uvz78['z'].values))) > 50000:
        uvz78['z'] = uvz78['z'] / G0


# 土壤湿度 7-8 月平均
sm78 = sm.sel(time=sm['time.month'].isin([7, 8])).groupby('time.year').mean('time')
sm78 = sm78.sel(year=slice(TR_time[0], TR_time[1]))
sm78 = sm78.transpose('year', 'lat', 'lon')
sm78 = sm78.sortby('lat').sortby('lon')


# SST 7-8 月平均
sst78 = sst.sel(time=sst['time.month'].isin([7, 8])).groupby('time.year').mean('time')
sst78 = sst78.sel(year=slice(TR_time[0], TR_time[1]))
sst78 = sst78.transpose('year', 'lat', 'lon')
sst78 = sst78.sortby('lat').sortby('lon')


# T2m 7-8 月平均
try:
    t2m78 = xr.open_dataset(fr"{PYFILE}/p2/data/t2m_78.nc")
    t2m78 = to_78_yearly(t2m78, varname='t2m', start=TR_time[0], end=TR_time[1])

except FileNotFoundError:
    t2m_raw = xr.open_dataset(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc")['t2m']
    t2m_raw = t2m_raw.sel(date=slice('1961-01-01', '2022-12-31'))

    t2m78 = xr.Dataset(
        {'t2m': (['time', 'lat', 'lon'], t2m_raw.data)},
        coords={
            'time': pd.to_datetime(t2m_raw['date'].values.astype(str), format="%Y%m%d"),
            'lat': t2m_raw['latitude'].data,
            'lon': t2m_raw['longitude'].data
        }
    )

    t2m78 = t2m78.sel(time=t2m78['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    t2m78.to_netcdf(fr"{PYFILE}/p2/data/t2m_78.nc")

t2m78 = t2m78.sel(year=slice(TR_time[0], TR_time[1]))
t2m78 = t2m78.transpose('year', 'lat', 'lon')
t2m78 = t2m78.sortby('lat').sortby('lon')


# omega 7-8 月平均
w78 = xr.open_dataset(fr"{PYFILE}/p2/data/W.nc")
w78 = rename_common(w78)

if 'p' in w78.coords:
    w78 = w78.sel(p=500)

if 'w' in w78.data_vars:
    w_name = 'w'
elif 'omega' in w78.data_vars:
    w_name = 'omega'
else:
    w_name = list(w78.data_vars)[0]

w78 = to_78_yearly(w78, varname=w_name, start=TR_time[0], end=TR_time[1])
w78 = w78.rename({w_name: 'omega'})
w78 = w78.transpose('year', 'lat', 'lon')
w78 = w78.sortby('lat').sortby('lon')


# 总云量 TCC 7-8 月平均
tcc78 = xr.open_dataset(fr"{PYFILE}/p2/data/TCC.nc")
tcc78 = rename_common(tcc78)

if 'tcc' in tcc78.data_vars:
    tcc_name = 'tcc'
elif 'TCC' in tcc78.data_vars:
    tcc_name = 'TCC'
else:
    tcc_name = list(tcc78.data_vars)[0]

tcc78 = to_78_yearly(tcc78, varname=tcc_name, start=TR_time[0], end=TR_time[1])
tcc78 = tcc78.rename({tcc_name: 'tcc'})
tcc78 = tcc78.transpose('year', 'lat', 'lon')
tcc78 = tcc78.sortby('lat').sortby('lon')


#————————————————————因子序列————————————————————

X1_idx = get_factor_da(X1, 'swvl').sel(year=slice(TR_time[0], TR_time[1]))
X2_idx = get_factor_da(X2, 'sst').sel(year=slice(TR_time[0], TR_time[1]))


#————————————————————回归 / 相关 / WAF 计算————————————————————

plot_data = {}

for fname, idx in [('X1', X1_idx), ('X2', X2_idx)]:

    # 200 hPa Z
    z200_reg, z200_corr = regcorr_2d(idx, uvz78['z'].sel(p=200), 'z200')

    # 500 hPa Z
    z500_reg, z500_corr = regcorr_2d(idx, uvz78['z'].sel(p=500), 'z500')

    # 7-8 月土壤湿度
    sm_reg, sm_corr = regcorr_2d(idx, sm78['swvl'], 'sm')

    # 7-8 月 SST
    sst_reg, sst_corr = regcorr_2d(idx, sst78['sst'], 'sst')

    # 7-8 月 T2m
    t2m_reg, t2m_corr = regcorr_2d(idx, t2m78['t2m'], 't2m')

    # 7-8 月 omega
    omega_reg, omega_corr = regcorr_2d(idx, w78['omega'], 'omega')

    # 7-8 月总云量
    tcc_reg, tcc_corr = regcorr_2d(idx, tcc78['tcc'], 'tcc')

    # 200 hPa WAF
    waf200 = make_waf_200(z200_reg, uvz78)

    plot_data[fname] = {
        'z200_reg': z200_reg,
        'z200_corr': z200_corr,
        'z500_reg': z500_reg,
        'z500_corr': z500_corr,
        'sm_reg': sm_reg,
        'sm_corr': sm_corr,
        'sst_reg': sst_reg,
        'sst_corr': sst_corr,
        't2m_reg': t2m_reg,
        't2m_corr': t2m_corr,
        'omega_reg': omega_reg,
        'omega_corr': omega_corr,
        'tcc_reg': tcc_reg,
        'tcc_corr': tcc_corr,
        'waf200': waf200
    }


#————————————————————绘图参数————————————————————

plot_extent = [-180, 180, -50, 80]

plot_geoticks = {
    'x': np.arange(-180, 181, 60),
    'y': yticks,
    'xminor': 10,
    'yminor': 10
}

# X1 第一行：土壤湿度填色，粉红-绿
sm_levels = np.array([-.04, -.03, -.02, -.01, .01, .02, .03, .04])
sm_cmap = ensure_cmap(
    cmaps.GreenMagenta16[8-5:8] + cmaps.GMT_red2green_r[11:11+4],
    name='sm_pink_green'
)

# X2 第一行：SST 填色，红-蓝
sst_levels = np.array([-.3, -.25, -.2, -.15, -.1, .1, .15, .2, .25, .3])
sst_cmap = ensure_cmap(
    cmaps.BlueWhiteOrangeRed[40:-40],
    name='sst_red_blue'
)

# 第二行：T2m 填色
t2m_levels = np.array([-1.2, -.9, -.6, -.3, .3, .6, .9, 1.2])
t2m_cmap = ensure_cmap(
    cmaps.BlueWhiteOrangeRed[40:-40],
    name='t2m_red_blue'
)

# 第三行：omega 填色
omega_levels = np.array([-.02, -.015, -.01, -.005, .005, .01, .015, .02])
omega_cmap = ensure_cmap(
    cmaps.BlueWhiteOrangeRed[40:-40],
    name='omega_red_blue'
)

# 200/500 hPa Z 等值线
z200_levels = [[-30, -20, -10], [10, 20, 30]]
z500_levels = [[-30, -20, -10], [10, 20, 30]]

# TCC 等值线
# 如果你的 TCC 是 0-100 百分比单位，可改为 [[-6, -4, -2], [2, 4, 6]]
tcc_levels = [[-.06, -.04, -.02], [.02, .04, .06]]

# WAF 矢量设置
waf_set = default_wind_2_set.copy()
waf_set.update({
    'regrid': 15,
    'arrowsize': 0.5,
    'scale': 5,
    'lw': 0.35,
    'color': 'purple',
    'thinning': ['40%', 'min'],
    'nanmax': 0.1,
    'MinDistance': [0.2, 0.1]
})

waf_key_set = default_wind_2_key_set.copy()
waf_key_set.update({
    'U': 0.03,
    'label': '0.03 m$^2$/s$^2$',
    'arrowsize': 0.5,
    'edgecolor': 'none',
    'lw': 0.5
})


#————————————————————正式绘图：3行 × 2列————————————————————

fig = plt.figure(figsize=np.array([8.5, 8.8]))
fig.subplots_adjust(hspace=0.28, wspace=0.18)

gs = gridspec.GridSpec(3, 2)
letters = ['a', 'b', 'c', 'd', 'e', 'f']

for col, fname in enumerate(['X1', 'X2']):

    pdata = plot_data[fname]
    draw_cbar = True if col == 1 else False

    #———————————————— 第一行 ————————————————
    # X1: 200Z 等值线 + WAF 矢量 + 土壤湿度填色
    # X2: 200Z 等值线 + WAF 矢量 + SST 填色
    ax = fig.add_subplot(gs[0, col], projection=ccrs.PlateCarree(central_longitude=180-70))

    if fname == 'X1':
        first_shading = pdata['sm_reg']
        first_corr = pdata['sm_corr']
        first_levels = sm_levels
        first_cmap = sm_cmap
        first_name = 'SM'
    else:
        first_shading = pdata['sst_reg']
        first_corr = pdata['sst_corr']
        first_levels = sst_levels
        first_cmap = sst_cmap
        first_name = 'SST'

    sub_pic(
        ax,
        title=f"({letters[col]}) 200Z&WAF&{first_name} reg onto {fname}",
        extent=plot_extent,
        geoticks=plot_geoticks,
        fontsize_times=default_fontsize_times,

        shading=None,
        shading_levels=first_levels,
        shading_cmap=first_cmap,
        shading_corr=None,
        cb_draw=False,
        p_test_drawSet={
            'N': TR_time[1] - TR_time[0] + 1,
            'alpha': 0.1,
            'lw': 0.2,
            'color': '#454545'
        },
        edgedraw=False,

        shading2=first_shading,
        shading2_levels=first_levels,
        shading2_cmap=first_cmap,
        shading2_corr=first_corr,
        p_test_drawSet2={
            'N': TR_time[1] - TR_time[0] + 1,
            'alpha': 0.1,
            'lw': 0.2,
            'color': '#454545'
        },
        edgedraw2=False,

        contour=pdata['z200_reg'],
        contour_levels=z200_levels,
        contour_cmap=default_contour_cmap,
        contour_corr=None,
        cb_draw2=draw_cbar,
        p_test_drawSet_corr={
            'N': TR_time[1] - TR_time[0] + 1,
            'alpha': 0.1
        },

        wind_1=default_wind_1, ###################pdata['waf200'],
        wind_1_set=waf_set,
        wind_1_key_set=waf_key_set,
        bbox_to_anchor_1=None,
        loc1='upper right',

        wind_2=default_wind_2,
        wind_2_set=default_wind_2_set,
        wind_2_key_set=default_wind_2_key_set,
        bbox_to_anchor_2=None,
        loc2='upper right',

        rec_Set=None
    )


    #———————————————— 第二行：500Z 等值线 + T2m 填色 ————————————————
    ax = fig.add_subplot(gs[1, col], projection=ccrs.PlateCarree(central_longitude=180-70))

    sub_pic(
        ax,
        title=f"({letters[col + 2]}) 500Z&T2m reg onto {fname}",
        extent=plot_extent,
        geoticks=plot_geoticks,
        fontsize_times=default_fontsize_times,

        shading=None,
        shading_levels=t2m_levels,
        shading_cmap=t2m_cmap,
        shading_corr=None,
        cb_draw=False,
        p_test_drawSet={
            'N': TR_time[1] - TR_time[0] + 1,
            'alpha': 0.1,
            'lw': 0.2,
            'color': '#454545'
        },
        edgedraw=False,

        shading2=pdata['t2m_reg'],
        shading2_levels=t2m_levels,
        shading2_cmap=t2m_cmap,
        shading2_corr=pdata['t2m_corr'],
        p_test_drawSet2={
            'N': TR_time[1] - TR_time[0] + 1,
            'alpha': 0.1,
            'lw': 0.2,
            'color': '#454545'
        },
        edgedraw2=False,

        contour=pdata['z500_reg'],
        contour_levels=z500_levels,
        contour_cmap=default_contour_cmap,
        contour_corr=None,
        cb_draw2=draw_cbar,
        p_test_drawSet_corr={
            'N': TR_time[1] - TR_time[0] + 1,
            'alpha': 0.1
        },

        wind_1=default_wind_1,
        wind_1_set=default_wind_1_set,
        wind_1_key_set=default_wind_1_key_set,
        bbox_to_anchor_1=None,
        loc1='upper right',

        wind_2=default_wind_2,
        wind_2_set=default_wind_2_set,
        wind_2_key_set=default_wind_2_key_set,
        bbox_to_anchor_2=None,
        loc2='upper right',

        rec_Set=None
    )


    #———————————————— 第三行：omega 填色 + TCC 等值线 ————————————————
    ax = fig.add_subplot(gs[2, col], projection=ccrs.PlateCarree(central_longitude=180-70))

    sub_pic(
        ax,
        title=f"({letters[col + 4]}) $\\omega$&TCC reg onto {fname}",
        extent=plot_extent,
        geoticks=plot_geoticks,
        fontsize_times=default_fontsize_times,

        shading=None,
        shading_levels=omega_levels,
        shading_cmap=omega_cmap,
        shading_corr=None,
        cb_draw=False,
        p_test_drawSet={
            'N': TR_time[1] - TR_time[0] + 1,
            'alpha': 0.1,
            'lw': 0.2,
            'color': '#454545'
        },
        edgedraw=False,

        shading2=pdata['omega_reg'],
        shading2_levels=omega_levels,
        shading2_cmap=omega_cmap,
        shading2_corr=pdata['omega_corr'],
        p_test_drawSet2={
            'N': TR_time[1] - TR_time[0] + 1,
            'alpha': 0.1,
            'lw': 0.2,
            'color': '#454545'
        },
        edgedraw2=False,

        contour=pdata['tcc_reg'],
        contour_levels=tcc_levels,
        contour_cmap=default_contour_cmap,
        contour_corr=None,
        cb_draw2=draw_cbar,
        p_test_drawSet_corr={
            'N': TR_time[1] - TR_time[0] + 1,
            'alpha': 0.1
        },

        wind_1=default_wind_1,
        wind_1_set=default_wind_1_set,
        wind_1_key_set=default_wind_1_key_set,
        bbox_to_anchor_1=None,
        loc1='upper right',

        wind_2=default_wind_2,
        wind_2_set=default_wind_2_set,
        wind_2_key_set=default_wind_2_key_set,
        bbox_to_anchor_2=None,
        loc2='upper right',

        rec_Set=None
    )


plt.savefig(f'{PYFILE}/p5/pic/因子发展_78平均.pdf', bbox_inches='tight')
plt.savefig(f'{PYFILE}/p5/pic/因子发展_78平均.png', dpi=600, bbox_inches='tight')
plt.show()
