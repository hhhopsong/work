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

# 屏蔽运行时警告 (主要解决 shapely 和 numpy 的除0/buffer 警告)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    def _np1(v):
        if hasattr(v, "data"):
            return np.asarray(v.data, dtype=float)
        return np.asarray(v, dtype=float)

    if shading_signal:
        shading_data, shading_lon = add_cyclic_point(shading, shading['lon'])
        shading_lon_np = _np1(shading_lon)
        shading_lat_np = _np1(shading['lat'])
        shading_data_np = _np1(shading_data)
        shading_draw = axes_sub.contourf(shading_lon_np, shading_lat_np, shading_data_np,
                                               levels=shading_levels,
                                               cmap=shading_cmap,
                                               extend='both', alpha=.75, norm=mcolors.BoundaryNorm(boundaries=shading_levels, ncolors=shading_cmap.N, clip=False),
                                               transform=ccrs.PlateCarree(central_longitude=0))
    else:
        shading_draw = False

    # 阴影图边缘绘制
    if shading_signal and edgedraw:
        axes_sub.contour(shading_lon_np, shading_lat_np, shading_data_np, colors='white', levels=shading_levels,
                                         linestyles='solid', linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))

    # 显著性检验
    if shading_corr_signal:
        # 去除白线
        shading_corr_data, shading_corr_lon = add_cyclic_point(shading_corr, shading_corr['lon'])
        shading_corr_lon_np = _np1(shading_corr_lon)
        shading_corr_lat_np = _np1(shading_corr['lat'])
        shading_corr_data_np = _np1(shading_corr_data)
        p_test = np.where(np.abs(shading_corr_data_np) > r_test(p_test_drawSet['N'], p_test_drawSet['alpha']), 0, np.nan)    # 显著性
        axes_sub.contourf(shading_corr_lon_np, shading_corr_lat_np, p_test, levels=[0, 1], hatches=['////////////', None],
                                  colors="none", add_colorbar=False, transform=ccrs.PlateCarree(central_longitude=0), edgecolor='none', linewidths=0)

    # 阴影2
    if shading2_signal:
        shading2_data, shading2_lon = add_cyclic_point(shading2, shading2['lon'])
        shading2_lon_np = _np1(shading2_lon)
        shading2_lat_np = _np1(shading2['lat'])
        shading2_data_np = _np1(shading2_data)
        shading2_draw = axes_sub.contourf(shading2_lon_np, shading2_lat_np, shading2_data_np,
                                               levels=shading2_levels,
                                               cmap=shading2_cmap,
                                               extend='both', alpha=.75, norm=mcolors.BoundaryNorm(boundaries=shading2_levels, ncolors=shading2_cmap.N, clip=False),
                                               transform=ccrs.PlateCarree(central_longitude=0))
    else:
        shading2_draw = False

    # 阴影2图边缘绘制
    if shading2_signal and edgedraw2:
        axes_sub.contour(shading2_lon_np, shading2_lat_np, shading2_data_np, colors='white', levels=shading2_levels,
                                            linestyles='solid', linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))

    # 显著性检验2
    if shading2_corr_signal:
        shading2_corr_data, shading2_corr_lon = add_cyclic_point(shading2_corr, shading2_corr['lon'])
        shading2_corr_lon_np = _np1(shading2_corr_lon)
        shading2_corr_lat_np = _np1(shading2_corr['lat'])
        shading2_corr_data_np = _np1(shading2_corr_data)
        p_test2 = np.where(np.abs(shading2_corr_data_np) > r_test(p_test_drawSet2['N'], p_test_drawSet2['alpha']), 0, np.nan)    # 显著性
        axes_sub.contourf(shading2_corr_lon_np, shading2_corr_lat_np, p_test2, levels=[0, 1], hatches=['////////////', None],
                                  colors="none", add_colorbar=False, transform=ccrs.PlateCarree(central_longitude=0), edgecolor='none', linewidths=0)

    # 等值线
    if contour_signal:
        # 去除白线
        contour_data, contour_lon = add_cyclic_point(contour, contour['lon'])
        contour_lon_np = _np1(contour_lon)
        contour_lat_np = _np1(contour['lat'])
        contour_data_np = _np1(contour_data)
        contour_low = axes_sub.contour(contour_lon_np, contour_lat_np, contour_data_np, colors=contour_cmap[0], linestyles='solid',
                                       levels=contour_levels[0], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
        contour_high = axes_sub.contour(contour_lon_np, contour_lat_np, contour_data_np, colors=contour_cmap[1], linestyles='solid',
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
        contour_corr_lon_np = _np1(contour_corr_lon)
        contour_corr_lat_np = _np1(contour_corr['lat'])
        contour_corr_data_np = _np1(contour_corr_data)
        p_test_corr = np.where(contour_corr_data_np > r_test(p_test_drawSet_corr['N'], p_test_drawSet_corr['alpha']), 0,
                           np.nan)  # 显著性 正
        axes_sub.quiver(contour_corr_lon_np, contour_corr_lat_np, p_test_corr, p_test_corr,
                        transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=40,
                        color=contour_cmap[1], scale=10, width=0.0025)
        p_test_corr = np.where(contour_corr_data_np < -r_test(p_test_drawSet_corr['N'], p_test_drawSet_corr['alpha']), 0, np.nan)  # 显著性 负
        axes_sub.quiver(contour_corr_lon_np, contour_corr_lat_np, p_test_corr, p_test_corr,
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


def plot_sea_ice(
    ax,
    title,
    lon,
    lat,
    ice,
    level,
    cmap=cmaps.BlueWhiteOrangeRed_r[40:-40],
    rec_Set=None,
    ice_corr=None,
    sig_draw_set=None,
):
    """
    lon, lat, ice can be 2D arrays of the same shape.
    Projection: North Polar Stereographic
    Central longitude: 110E
    Extent: 40N to 90N
    Thick border

    Optional significance stippling:
    - ice_corr: correlation field used for significance test; fallback to ice when None
    - sig_draw_set: {'N': sample_size, 'alpha': 0.1, 'color': '#303030', 's': 2, 'marker': '.', 'alpha_pt': 0.8, 'stride': 1}
    """
    import matplotlib.path as mpath

    def rec(ax_, point, color='blue', ls='--', lw=0.5):
        x1, x2 = point[:2]
        y1, y2 = point[2:]
        x = [x1, x2, x2, x1, x1]
        y = [y1, y1, y2, y2, y1]
        ax_.plot(x, y, color=color, linestyle=ls, lw=lw, transform=ccrs.PlateCarree(), zorder=100)

    proj = ccrs.NorthPolarStereo(central_longitude=110)

    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

    # 圆形边界
    theta = np.linspace(0, 2 * np.pi, 400)
    center = np.array([0.5, 0.5])
    radius = 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Optional base layers
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=2, color='#757575', alpha=0.75, zorder=99)
    ax.set_title(title, fontsize=10)
    ax.coastlines(linewidth=0.8, zorder=2)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        xlocs=np.array([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])-10,
        ylocs=[70, 80],
        linewidth=0.6,
        linestyle="--",
        alpha=0.7
    )

    gl.top_labels = True
    gl.bottom_labels = True
    gl.left_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 6}
    gl.ylabel_style = {"size": 6}


    # Plot sea ice field
    norm = mcolors.BoundaryNorm(level, ncolors=plt.get_cmap(cmap).N, clip=True)
    mesh = ax.pcolormesh(
        lon, lat, ice,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        cmap=cmap,
        zorder=3,
        norm=norm
    )

    # 90% significance stippling (|r| > r_crit), rendered as points instead of contourf
    if sig_draw_set is not None and sig_draw_set.get('N', None) is not None:
        corr_field = ice if ice_corr is None else ice_corr
        r_crit = r_test(sig_draw_set['N'], sig_draw_set.get('alpha', 0.1))
        sig_mask = np.abs(np.asarray(corr_field)) >= r_crit

        lon_np = np.asarray(lon)
        lat_np = np.asarray(lat)
        if lon_np.ndim == 1 and lat_np.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon_np, lat_np)
        else:
            lon2d, lat2d = lon_np, lat_np

        if lon2d.shape == sig_mask.shape and lat2d.shape == sig_mask.shape:
            stride = int(sig_draw_set.get('stride', 1))
            stride = 1 if stride < 1 else stride
            sig_lon = lon2d[sig_mask][::stride]
            sig_lat = lat2d[sig_mask][::stride]
            ax.scatter(
                sig_lon,
                sig_lat,
                s=sig_draw_set.get('s', 2),
                c=sig_draw_set.get('color', '#303030'),
                marker=sig_draw_set.get('marker', '.'),
                alpha=sig_draw_set.get('alpha_pt', 0.8),
                linewidths=0,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

    # 新增: 绘制因子框
    if rec_Set is not None:
        for rec_set in rec_Set:
            rec(
                ax,
                rec_set['point'],
                rec_set.get('color', 'blue'),
                rec_set.get('ls', '--'),
                rec_set.get('lw', 0.5)
            )

    ax_colorbar = inset_axes(ax, width="6%", height="100%", loc='lower left', bbox_to_anchor=(1.2, 0., 1, 1),
                              bbox_transform=ax.transAxes, borderpad=0)
    cb2 = plt.colorbar(mesh, cax=ax_colorbar, orientation='vertical', drawedges=True)
    cb2.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
    cb2.dividers.set_color('black')  # 将colorbar内间隔线调为黑色
    cb2.locator = ticker.FixedLocator(level)
    cb2.set_ticklabels([str(lev) for lev in level])
    cb2.ax.tick_params(length=0, labelsize=6)  # length为刻度线的长度
    ax.spines["geo"].set_linewidth(1.25)

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

plt.rcParams['font.family'] = ['AVHershey Simplex', 'AVHershey Duplex', 'Helvetica']    # 字体为Hershey (安装字体后，清除.matplotlib的字体缓存即可生效)
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
xticks = np.arange(-180, 181, 30)
yticks = np.arange(-30, 81, 30)

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
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
# 矩形框设置, 可为False
default_rec_Set = [{'point': [105, 120, 20, 30], 'color': 'blue', 'ls': '--', 'lw': 0.5}]

EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
EHCI30 = EHCI.apply(lambda x: (x > 0.6).sum())
EHCI30 = (EHCI30 - EHCI30.mean()) / EHCI30.std('year')
EHCI30 = EHCI30['EHCI']

# 2mT
t2m = era5_land(fr"{DATA}/ERA5/ERA5_land/uv_2mTTd_sfp_pre_0.nc", 1961, 2022, 't2m')
# SLP
slp = era5_s(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc", 1961, 2022, 'msl')
# sst
sst = ersst(fr"{DATA}/NOAA/ERSSTv5/sst.mnmean.nc", 1961, 2022)
# sic
sic = sic(fr"{DATA}/NOAA/HadISST/HadISST_ice.nc", 1961, 2022)

swvl1 = era5_land(fr"{DATA}/ERA5/ERA5_land/sm.nc", 1961, 2022, 'swvl1')
swvl2 = era5_land(fr"{DATA}/ERA5/ERA5_land/sm.nc", 1961, 2022, 'swvl2')
if isinstance(swvl1, xr.Dataset):
    swvl1_da = swvl1['swvl1']
else:
    swvl1_da = swvl1
if isinstance(swvl2, xr.Dataset):
    swvl2_da = swvl2['swvl2']
else: swvl2_da = swvl2
swvl = prepare_swvl_dataset((swvl1_da + swvl2_da).rename('swvl'))
# %%
# ============================================================
# 计算
# ============================================================
TR_time = [1962, 2004]  # 训练时间段
PR_time = [2005, 2022]

def detrend(data):
    return data - np.polyval(np.polyfit(range(len(data)), data, 1), range(len(data)))

timeSerie = EHCI30

# ============================================================
# predictor：加入 SWVL（土壤湿度）
# ============================================================
def predictor(timeSerie, TR_time, PR_time, month1, month2=None, predictor_zone=None, cross_month=9):

    if predictor_zone is None:
        predictor_zone = []

    def seasonal_mean_same_year(da, months, start_year, end_year):
        """不跨年的月份平均，例如 [3,4,5]"""
        return (
            da.sel(time=da["time.month"].isin(months))
              .sel(time=slice(f"{start_year}", f"{end_year}"))
              .groupby("time.year")
              .mean("time")
              .transpose("year", "lat", "lon")
        )

    def seasonal_mean_cross_year(da, months, start_year, end_year, cross_month=cross_month):
        """
        跨年月份平均，例如 [11,12,1,2]
        规则：
        - month >= cross_month 的月份，season_year = 原年份 + 1
        - month < cross_month 的月份，season_year = 原年份
        最终 season_year 作为输出 year
        """
        da_sel = da.sel(time=da["time.month"].isin(months)).sel(
            time=slice(f"{start_year-1}", f"{end_year}")
        )

        season_year = xr.where(
            da_sel["time.month"] >= cross_month,
            da_sel["time.year"] + 1,
            da_sel["time.year"]
        )

        da_sel = da_sel.assign_coords(season_year=("time", season_year.data))

        out = (
            da_sel.groupby("season_year")
                  .mean("time")
                  .rename({"season_year": "year"})
                  .transpose("year", "lat", "lon")
        )

        return out.sel(year=slice(start_year, end_year))

    def get_seasonal_mean(da, months, start_year, end_year, cross_month=cross_month):
        cross_year = any(m >= cross_month for m in months)
        if cross_year:
            return seasonal_mean_cross_year(da, months, start_year, end_year, cross_month)
        else:
            return seasonal_mean_same_year(da, months, start_year, end_year)

    # ============================================================
    # 计算 month1 / month2 季节平均
    # ============================================================
    if month2 is None:
        t2m_imonth = get_seasonal_mean(t2m, month1, TR_time[0], TR_time[1], cross_month)
        slp_imonth = get_seasonal_mean(slp, month1, TR_time[0], TR_time[1], cross_month)
        sst_imonth = get_seasonal_mean(sst, month1, TR_time[0], TR_time[1], cross_month)
        sic_imonth = get_seasonal_mean(sic, month1, TR_time[0], TR_time[1], cross_month)
        swvl_imonth = get_seasonal_mean(swvl, month1, TR_time[0], TR_time[1], cross_month)

        t2m_imonth_pre = get_seasonal_mean(t2m, month1, PR_time[0], PR_time[1], cross_month)
        slp_imonth_pre = get_seasonal_mean(slp, month1, PR_time[0], PR_time[1], cross_month)
        sst_imonth_pre = get_seasonal_mean(sst, month1, PR_time[0], PR_time[1], cross_month)
        sic_imonth_pre = get_seasonal_mean(sic, month1, PR_time[0], PR_time[1], cross_month)
        swvl_imonth_pre = get_seasonal_mean(swvl, month1, PR_time[0], PR_time[1], cross_month)

        t2m_imonth_all = get_seasonal_mean(t2m, month1, TR_time[0], PR_time[1], cross_month)
        slp_imonth_all = get_seasonal_mean(slp, month1, TR_time[0], PR_time[1], cross_month)
        sst_imonth_all = get_seasonal_mean(sst, month1, TR_time[0], PR_time[1], cross_month)
        sic_imonth_all = get_seasonal_mean(sic, month1, TR_time[0], PR_time[1], cross_month)
        swvl_imonth_all = get_seasonal_mean(swvl, month1, TR_time[0], PR_time[1], cross_month)

    else:
        # month1
        t2m_imonth_1 = get_seasonal_mean(t2m, month1, TR_time[0], TR_time[1], cross_month)
        slp_imonth_1 = get_seasonal_mean(slp, month1, TR_time[0], TR_time[1], cross_month)
        sst_imonth_1 = get_seasonal_mean(sst, month1, TR_time[0], TR_time[1], cross_month)
        sic_imonth_1 = get_seasonal_mean(sic, month1, TR_time[0], TR_time[1], cross_month)
        swvl_imonth_1 = get_seasonal_mean(swvl, month1, TR_time[0], TR_time[1], cross_month)

        t2m_imonth_pre_1 = get_seasonal_mean(t2m, month1, PR_time[0], PR_time[1], cross_month)
        slp_imonth_pre_1 = get_seasonal_mean(slp, month1, PR_time[0], PR_time[1], cross_month)
        sst_imonth_pre_1 = get_seasonal_mean(sst, month1, PR_time[0], PR_time[1], cross_month)
        sic_imonth_pre_1 = get_seasonal_mean(sic, month1, PR_time[0], PR_time[1], cross_month)
        swvl_imonth_pre_1 = get_seasonal_mean(swvl, month1, PR_time[0], PR_time[1], cross_month)

        t2m_imonth_all_1 = get_seasonal_mean(t2m, month1, TR_time[0], PR_time[1], cross_month)
        slp_imonth_all_1 = get_seasonal_mean(slp, month1, TR_time[0], PR_time[1], cross_month)
        sst_imonth_all_1 = get_seasonal_mean(sst, month1, TR_time[0], PR_time[1], cross_month)
        sic_imonth_all_1 = get_seasonal_mean(sic, month1, TR_time[0], PR_time[1], cross_month)
        swvl_imonth_all_1 = get_seasonal_mean(swvl, month1, TR_time[0], PR_time[1], cross_month)

        # month2
        t2m_imonth_2 = get_seasonal_mean(t2m, month2, TR_time[0], TR_time[1], cross_month)
        slp_imonth_2 = get_seasonal_mean(slp, month2, TR_time[0], TR_time[1], cross_month)
        sst_imonth_2 = get_seasonal_mean(sst, month2, TR_time[0], TR_time[1], cross_month)
        sic_imonth_2 = get_seasonal_mean(sic, month2, TR_time[0], TR_time[1], cross_month)
        swvl_imonth_2 = get_seasonal_mean(swvl, month2, TR_time[0], TR_time[1], cross_month)

        t2m_imonth_pre_2 = get_seasonal_mean(t2m, month2, PR_time[0], PR_time[1], cross_month)
        slp_imonth_pre_2 = get_seasonal_mean(slp, month2, PR_time[0], PR_time[1], cross_month)
        sst_imonth_pre_2 = get_seasonal_mean(sst, month2, PR_time[0], PR_time[1], cross_month)
        sic_imonth_pre_2 = get_seasonal_mean(sic, month2, PR_time[0], PR_time[1], cross_month)
        swvl_imonth_pre_2 = get_seasonal_mean(swvl, month2, PR_time[0], PR_time[1], cross_month)

        t2m_imonth_all_2 = get_seasonal_mean(t2m, month2, TR_time[0], PR_time[1], cross_month)
        slp_imonth_all_2 = get_seasonal_mean(slp, month2, TR_time[0], PR_time[1], cross_month)
        sst_imonth_all_2 = get_seasonal_mean(sst, month2, TR_time[0], PR_time[1], cross_month)
        sic_imonth_all_2 = get_seasonal_mean(sic, month2, TR_time[0], PR_time[1], cross_month)
        swvl_imonth_all_2 = get_seasonal_mean(swvl, month2, TR_time[0], PR_time[1], cross_month)

        # 差值
        t2m_imonth = t2m_imonth_1 - t2m_imonth_2
        slp_imonth = slp_imonth_1 - slp_imonth_2
        sst_imonth = sst_imonth_1 - sst_imonth_2
        sic_imonth = sic_imonth_1 - sic_imonth_2
        swvl_imonth = swvl_imonth_1 - swvl_imonth_2

        t2m_imonth_pre = t2m_imonth_pre_1 - t2m_imonth_pre_2
        slp_imonth_pre = slp_imonth_pre_1 - slp_imonth_pre_2
        sst_imonth_pre = sst_imonth_pre_1 - sst_imonth_pre_2
        sic_imonth_pre = sic_imonth_pre_1 - sic_imonth_pre_2
        swvl_imonth_pre = swvl_imonth_pre_1 - swvl_imonth_pre_2

        t2m_imonth_all = t2m_imonth_all_1 - t2m_imonth_all_2
        slp_imonth_all = slp_imonth_all_1 - slp_imonth_all_2
        sst_imonth_all = sst_imonth_all_1 - sst_imonth_all_2
        sic_imonth_all = sic_imonth_all_1 - sic_imonth_all_2
        swvl_imonth_all = swvl_imonth_all_1 - swvl_imonth_all_2

    # ============================================================
    # 训练/预测时间序列
    # ============================================================
    timeSerie_train = timeSerie.sel(year=slice(f'{TR_time[0]}', f'{TR_time[1]}')).data
    train_years = pd.to_datetime(np.arange(TR_time[0], TR_time[1] + 1), format='%Y')
    pre_years = pd.to_datetime(np.arange(PR_time[0], PR_time[1] + 1), format='%Y')

    # ============================================================
    # 回归 / 相关场
    # ============================================================
    t2mReg, t2mCorr = regress(timeSerie_train, t2m_imonth['t2m'].data), corr(timeSerie_train, t2m_imonth['t2m'].data)
    slpReg, slpCorr = regress(timeSerie_train, slp_imonth['msl'].data), corr(timeSerie_train, slp_imonth['msl'].data)
    sstReg, sstCorr = regress(timeSerie_train, sst_imonth['sst'].data), corr(timeSerie_train, sst_imonth['sst'].data)
    sicReg, sicCorr = regress(timeSerie_train, sic_imonth['sic'].data), corr(timeSerie_train, sic_imonth['sic'].data)
    swvlReg, swvlCorr = regress(timeSerie_train, swvl_imonth['swvl'].data), corr(timeSerie_train, swvl_imonth['swvl'].data)

    t2mReg = xr.DataArray(t2mReg, coords=[t2m_imonth['lat'], t2m_imonth['lon']],
                          dims=['lat', 'lon'], name='t2m_reg')
    slpReg = xr.DataArray(slpReg, coords=[slp_imonth['lat'], slp_imonth['lon']],
                          dims=['lat', 'lon'], name='slp_reg')
    sstReg = xr.DataArray(sstReg, coords=[sst_imonth['lat'], sst_imonth['lon']],
                          dims=['lat', 'lon'], name='sst_reg')
    sicReg = xr.DataArray(sicReg, coords=[sic_imonth['lat'], sic_imonth['lon']],
                          dims=['lat', 'lon'], name='sic_reg')
    swvlReg = xr.DataArray(swvlReg, coords=[swvl_imonth['lat'], swvl_imonth['lon']],
                           dims=['lat', 'lon'], name='swvl_reg')

    t2mCorr = xr.DataArray(t2mCorr, coords=[t2m_imonth['lat'], t2m_imonth['lon']],
                           dims=['lat', 'lon'], name='t2m_corr')
    slpCorr = xr.DataArray(slpCorr, coords=[slp_imonth['lat'], slp_imonth['lon']],
                           dims=['lat', 'lon'], name='slp_corr')
    sstCorr = xr.DataArray(sstCorr, coords=[sst_imonth['lat'], sst_imonth['lon']],
                           dims=['lat', 'lon'], name='sst_corr')
    sicCorr = xr.DataArray(sicCorr, coords=[sic_imonth['lat'], sic_imonth['lon']],
                           dims=['lat', 'lon'], name='sic_corr')
    swvlCorr = xr.DataArray(swvlCorr, coords=[swvl_imonth['lat'], swvl_imonth['lon']],
                            dims=['lat', 'lon'], name='swvl_corr')

    # ============================================================
    # 标准化目标序列
    # ============================================================
    nor_mean = np.mean(timeSerie_train)
    nor_std = np.std(timeSerie_train)
    timeSerie_train = (timeSerie_train - nor_mean) / nor_std
    TS = pd.Series(timeSerie_train, index=train_years, name='TS')

    timeSerie_pre = timeSerie.sel(year=slice(f'{PR_time[0]}', f'{PR_time[1]}'))
    timeSerie_pre = (timeSerie_pre - nor_mean) / nor_std
    TS_pre = pd.Series(timeSerie_pre, index=pre_years, name='TS_pre')

    timeSerie_all = timeSerie.sel(year=slice(f'{TR_time[0]}', f'{PR_time[1]}'))
    timeSerie_all = (timeSerie_all - nor_mean) / nor_std
    TS_all = pd.Series(timeSerie_all, index=np.arange(TR_time[0], PR_time[1] + 1), name='TS_all')

    # ============================================================
    # 场统一管理
    # ============================================================
    field_map = {
        'sst': {
            'corr': sstCorr,
            'train': sst_imonth,
            'pre': sst_imonth_pre,
            'all': sst_imonth_all,
            'var': 'sst',
        },
        'slp': {
            'corr': slpCorr,
            'train': slp_imonth,
            'pre': slp_imonth_pre,
            'all': slp_imonth_all,
            'var': 'msl',
        },
        't2m': {
            'corr': t2mCorr,
            'train': t2m_imonth,
            'pre': t2m_imonth_pre,
            'all': t2m_imonth_all,
            'var': 't2m',
        },
        'sic': {
            'corr': sicCorr,
            'train': sic_imonth,
            'pre': sic_imonth_pre,
            'all': sic_imonth_all,
            'var': 'sic',
        },
        'swvl': {
            'corr': swvlCorr,
            'train': swvl_imonth,
            'pre': swvl_imonth_pre,
            'all': swvl_imonth_all,
            'var': 'swvl',
        }
    }

    X_train_dict = {}
    X_pre_dict = {}
    X_rollingCorr_dict = {}

    index = 0
    for izone in predictor_zone:
        index += 1

        elem = izone[0].lower()
        X_zone = izone[1:]   # [lat1, lat2, lon1, lon2]

        if elem not in field_map:
            raise ValueError(
                f"Unsupported predictor element: {elem}. "
                f"Choose from ['sst', 'slp', 't2m', 'sic', 'swvl']."
            )

        corr_da = field_map[elem]['corr']
        train_da = field_map[elem]['train']
        pre_da = field_map[elem]['pre']
        all_da = field_map[elem]['all']
        var_name = field_map[elem]['var']

        # 经度统一
        if X_zone[2] < 0 or X_zone[3] < 0:
            corr_da_ = transform(corr_da, 'lon', '360->180')
            train_da_ = transform(train_da, 'lon', '360->180')
            pre_da_ = transform(pre_da, 'lon', '360->180')
            all_da_ = transform(all_da, 'lon', '360->180')
        else:
            corr_da_ = transform(corr_da, 'lon', '180->360')
            train_da_ = transform(train_da, 'lon', '180->360')
            pre_da_ = transform(pre_da, 'lon', '180->360')
            all_da_ = transform(all_da, 'lon', '180->360')

        # 权重场
        weight = corr_da_.sel(
            lat=slice(X_zone[0], X_zone[1]),
            lon=slice(X_zone[2], X_zone[3])
        )

        sig_mask = np.abs(weight) > r_test(TR_time[1] - TR_time[0] + 1, 0.1)

        # train
        X_train = train_da_[var_name].sel(
            lat=slice(X_zone[0], X_zone[1]),
            lon=slice(X_zone[2], X_zone[3])
        ) * xr.where(sig_mask, weight, np.nan)

        X_train = X_train.mean(['lat', 'lon'])
        X_mean, X_std = X_train.mean(), X_train.std()
        X_train = (X_train - X_mean) / X_std
        X_train = pd.Series(X_train.data, index=train_years, name=f'X{index}_train')

        # pre
        X_pre = pre_da_[var_name].sel(
            lat=slice(X_zone[0], X_zone[1]),
            lon=slice(X_zone[2], X_zone[3])
        ) * xr.where(sig_mask, weight, np.nan)

        X_pre = X_pre.mean(['lat', 'lon'])
        X_pre = (X_pre - X_mean) / X_std
        X_pre = pd.Series(X_pre.data, index=pre_years, name=f'X{index}_pre')

        # all / rolling corr
        X_all = all_da_[var_name].sel(
            lat=slice(X_zone[0], X_zone[1]),
            lon=slice(X_zone[2], X_zone[3])
        ) * xr.where(sig_mask, weight, np.nan)

        X_all = X_all.mean(['lat', 'lon'])
        X_all = (X_all - X_mean) / X_std
        X_all = pd.Series(X_all.data, index=np.arange(TR_time[0], PR_time[1] + 1), name=f'X{index}_all')

        X_rollingCorr = X_all.rolling(window=11).corr(TS_all)
        X_rollingCorr.name = f'X{index}_rollingCorr'

        X_train_dict[f'X{index}'] = X_train
        X_pre_dict[f'X{index}'] = X_pre
        X_rollingCorr_dict[f'X{index}'] = X_rollingCorr

    return (
        X_train_dict, X_pre_dict, X_rollingCorr_dict,
        TS, TS_pre, TS_all,
        t2mReg, t2mCorr,
        slpReg, slpCorr,
        sstReg, sstCorr,
        sicReg, sicCorr,
        swvlReg, swvlCorr
    )

# ============================================================
# 第一组预测因子
# ============================================================
X1 = ['sic', 76, 68, 35, 50]
X2 = ['sst', 70,	10,	-70,	-6]

X_train_dict, X_pre_dict, X_rollingCorr_dict, TS, TS_pre, TS_all, \
t2mReg, t2mCorr, slpReg, slpCorr, sstReg, sstCorr, sicReg, sicCorr, swvlReg, swvlCorr = predictor(
    timeSerie, [1962, 2004], [2005, 2022],
    month1=[3, 4], month2=[11, 12],
    predictor_zone=[],
    cross_month=9
)

# ============================================================
# 第二组预测因子
# ============================================================
X1_b = ['sst', 62.0,	52.0,	162.0,	188.0]
X2_b = ['slp', 34.0,	-20,	210,	290]
X3_b = ['sst', 65, 10, -65, -10]

X_train_dict2, X_pre_dict2, X_rollingCorr_dict2, _, _, _, \
t2mReg2, t2mCorr2, slpReg2, slpCorr2, sstReg2, sstCorr2, sicReg2, sicCorr2, swvlReg2, swvlCorr2 = predictor(
    timeSerie, [1962, 2004], [2005, 2022],
    month1=[5], month2=[12],
    predictor_zone=[X2_b],
    cross_month=9
)

# ============================================================
# 第三组预测因子
# ============================================================
X1_c = ['sst', 30.0, 10.0, 194.0, 242.0]
X2_c = ['sic', 85.5, 81.5, -165.5, -142.5]

X_train_dict3, X_pre_dict3, X_rollingCorr_dict3, _, _, _, \
t2mReg3, t2mCorr3, slpReg3, slpCorr3, sstReg3, sstCorr3, sicReg3, sicCorr3, swvlReg3, swvlCorr3 = predictor(
    timeSerie, [1962, 2004], [2005, 2022],
    month1=[4], month2=[2],
    predictor_zone=[X1_c],
    cross_month=9
)

# ============================================================
# 第四组预测因子：土壤湿度
# ============================================================
X4 = ['swvl', 30.500000000001000,	22.00000000000090,	104.49999999999800,	121.49999999999700]

X_train_dict4, X_pre_dict4, X_rollingCorr_dict4, _, _, _, \
t2mReg4, t2mCorr4, slpReg4, slpCorr4, sstReg4, sstCorr4, sicReg4, sicCorr4, swvlReg4, swvlCorr4 = predictor(
    timeSerie, [1962, 2004], [2005, 2022],
    month1=[3], month2=[12],
    predictor_zone=[],
    cross_month=9
)

# ============================================================
# 作图
# 7 行：SIC / SST / SLP / SST / SWVL / rollingCorr / forecast
# ============================================================
fig = plt.figure(figsize=(5, 16))
fig.subplots_adjust(hspace=0.4)
gs = gridspec.GridSpec(7, 1, height_ratios=[2, 1, 1, 1, 1, 1, 1])

# (a) SIC
ax_sic = fig.add_subplot(gs[0], projection=ccrs.NorthPolarStereo(central_longitude=110))
plot_sea_ice(
    ax_sic,
    "(a) 3+4_mean_SIC",
    sic.lon,
    sic.lat,
    sicCorr,
    np.array([-.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4]),
    rec_Set=[
        {'point': [X1[3], X1[4], X1[1], X1[2]], 'color': 'green', 'ls': (0, (1, 1)), 'lw': 1.6},
        {'point': [X2[3], X2[4], X2[1], X2[2]], 'color': 'brown', 'ls': (0, (1, 1)), 'lw': 1.6}
    ],
    ice_corr=sicCorr,
    sig_draw_set={'N': TR_time[1] - TR_time[0] + 1, 'alpha': 0.1, 'hatch': '..', 'lw': 0.2, 'color': '#303030'}
)

# (b) SST
ax = fig.add_subplot(gs[1], projection=ccrs.PlateCarree(central_longitude=180 - 70))
sub_pic(
    ax, title='(b) 6_minus_4_SST', extent=[-180, 180, -50, 80],
    geoticks={'x': np.arange(-180, 181, 30), 'y': yticks, 'xminor': 10, 'yminor': 10},
    fontsize_times=default_fontsize_times,
    shading=None,
    shading_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]),
    shading_cmap=cmaps.GreenMagenta16[8-5:8] + cmaps.GMT_red2green_r[11:11+4],
    shading_corr=None,
    p_test_drawSet={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
    edgedraw=False,
    cb_draw=True,
    shading2=sstCorr,
    shading2_levels=np.array([-.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4]),
    shading2_cmap=cmaps.BlueWhiteOrangeRed[40:-40],
    shading2_corr=sstCorr,
    p_test_drawSet2={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
    edgedraw2=False,
    cb_draw2=True,
    contour=None,
    contour_levels=np.array([[-50, -20], [20, 50]]) * 0.0005,
    contour_cmap=default_contour_cmap,
    contour_corr=None,
    p_test_drawSet_corr={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1},
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
    rec_Set=[{'point': [X2[3], X2[4], X2[1], X2[2]], 'color': '#e91e63', 'ls': (0, (1, 1)), 'lw': 1.6}]
)

# (c) SLP
ax = fig.add_subplot(gs[2], projection=ccrs.PlateCarree(central_longitude=180 - 70))
sub_pic(
    ax, title='(c) 5_minus_12_SLP', extent=[-180, 180, -50, 80],
    geoticks={'x': np.arange(-180, 181, 30), 'y': yticks, 'xminor': 10, 'yminor': 10},
    fontsize_times=default_fontsize_times,
    shading=None,
    shading_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]),
    shading_cmap=cmaps.GreenMagenta16[8-5:8] + cmaps.GMT_red2green_r[11:11+4],
    shading_corr=None,
    p_test_drawSet={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
    edgedraw=False,
    cb_draw=True,
    shading2=slpCorr2,
    shading2_levels=np.array([-.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4]),
    shading2_cmap=cmaps.BlueWhiteOrangeRed[40:-40],
    shading2_corr=slpCorr2,
    p_test_drawSet2={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
    edgedraw2=False,
    cb_draw2=True,
    contour=None,
    contour_levels=np.array([[-50, -20], [20, 50]]) * 0.0005,
    contour_cmap=default_contour_cmap,
    contour_corr=None,
    p_test_drawSet_corr={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1},
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
    rec_Set=[{'point': [X2_b[3], X2_b[4], X2_b[1], X2_b[2]], 'color': '#e91e63', 'ls': (0, (1, 1)), 'lw': 1.6}]
)

# (d) SST
ax = fig.add_subplot(gs[3], projection=ccrs.PlateCarree(central_longitude=180 - 70))
sub_pic(
    ax, title='(d) 4_minus_2_SST', extent=[-180, 180, -50, 80],
    geoticks={'x': np.arange(-180, 181, 30), 'y': yticks, 'xminor': 10, 'yminor': 10},
    fontsize_times=default_fontsize_times,
    shading=None,
    shading_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]),
    shading_cmap=cmaps.GreenMagenta16[8-5:8] + cmaps.GMT_red2green_r[11:11+4],
    shading_corr=None,
    p_test_drawSet={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
    edgedraw=False,
    cb_draw=True,
    shading2=sstCorr3,
    shading2_levels=np.array([-.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4]),
    shading2_cmap=cmaps.BlueWhiteOrangeRed[40:-40],
    shading2_corr=sstCorr3,
    p_test_drawSet2={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
    edgedraw2=False,
    cb_draw2=True,
    contour=None,
    contour_levels=np.array([[-50, -20], [20, 50]]) * 0.0005,
    contour_cmap=default_contour_cmap,
    contour_corr=None,
    p_test_drawSet_corr={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1},
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
    rec_Set=[{'point': [X1_c[3], X1_c[4], X1_c[1], X1_c[2]], 'color': 'darkgreen', 'ls': (0, (1, 1)), 'lw': 1.6}]
)

# (e) SWVL
ax = fig.add_subplot(gs[4], projection=ccrs.PlateCarree(central_longitude=180 - 70))
sub_pic(
    ax, title='(e) 5_minus_3_SWVL', extent=[-180, 180, -50, 80],
    geoticks={'x': np.arange(-180, 181, 30), 'y': yticks, 'xminor': 10, 'yminor': 10},
    fontsize_times=default_fontsize_times,
    shading=None,
    shading_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]),
    shading_cmap=cmaps.GreenMagenta16[8-5:8] + cmaps.GMT_red2green_r[11:11+4],
    shading_corr=None,
    p_test_drawSet={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
    edgedraw=False,
    cb_draw=True,
    shading2=swvlCorr4,
    shading2_levels=np.array([-.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4]),
    shading2_cmap=cmaps.BlueWhiteOrangeRed[40:-40],
    shading2_corr=swvlCorr4,
    p_test_drawSet2={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
    edgedraw2=False,
    cb_draw2=True,
    contour=None,
    contour_levels=np.array([[-50, -20], [20, 50]]) * 0.0005,
    contour_cmap=default_contour_cmap,
    contour_corr=None,
    p_test_drawSet_corr={'N': TR_time[1]-TR_time[0]+1, 'alpha': 0.1},
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
    rec_Set=[{'point': [X4[3], X4[4], X4[1], X4[2]], 'color': 'purple', 'ls': (0, (1, 1)), 'lw': 1.6}]
)

# ============================================================
# 自动因子池 + rolling corr + 逐步回归
# ============================================================
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

predictor_results = [
    (X_train_dict,  X_pre_dict,  X_rollingCorr_dict),
    (X_train_dict2, X_pre_dict2, X_rollingCorr_dict2),
    (X_train_dict3, X_pre_dict3, X_rollingCorr_dict3),
    (X_train_dict4, X_pre_dict4, X_rollingCorr_dict4),
]

# 构建因子池
all_X_train = {}
all_X_pre = {}
all_X_rollingCorr = {}

factor_count = 1
for train_dict, pre_dict, rolling_dict in predictor_results:
    for old_key in train_dict.keys():
        new_key = f'X{factor_count}'
        all_X_train[new_key] = train_dict[old_key].rename(new_key)
        all_X_pre[new_key] = pre_dict[old_key].rename(new_key)
        all_X_rollingCorr[new_key] = rolling_dict[old_key].rename(new_key)
        factor_count += 1

candidate_predictors = list(all_X_train.keys())
print("Available predictors:", candidate_predictors)

# 组装训练/预测数据
df_train = pd.concat(
    [TS.rename('TS')] + [all_X_train[x].rename(x) for x in candidate_predictors],
    axis=1
)

df_pre = pd.concat(
    [TS_pre.rename('TS')] + [all_X_pre[x].rename(x) for x in candidate_predictors],
    axis=1
)

df_train_step = df_train.iloc[1:].copy()

step_cols = ['TS'] + candidate_predictors
df_step = df_train_step[step_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

print("Training rows used for stepwise regression:", len(df_step))
print("Candidate predictors:", candidate_predictors)

def calc_vif(df, features):
    if len(features) <= 1:
        return pd.Series([1.0] * len(features), index=features, dtype=float)

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    if len(X) <= len(features):
        return pd.Series([1.0] * len(features), index=features, dtype=float)

    X_const = sm.add_constant(X, has_constant='add')
    vif_values = []
    for i in range(1, X_const.shape[1]):
        vif_values.append(variance_inflation_factor(X_const.values, i))
    return pd.Series(vif_values, index=features, dtype=float)

def stepwise_selection(
    df,
    response='TS',
    candidates=None,
    p_enter=0.05,
    p_remove=0.10,
    vif_thres=5.0,
    max_steps=100,
    verbose=True
):
    if candidates is None:
        candidates = [c for c in df.columns if c != response]

    selected = []
    remaining = candidates.copy()
    step = 0

    while step < max_steps:
        step += 1
        changed = False

        # forward
        best_feature = None
        best_pval = None

        for feature in remaining:
            trial_features = selected + [feature]
            formula = response + " ~ " + " + ".join(trial_features)
            try:
                model = smf.ols(formula=formula, data=df).fit()
                pval = model.pvalues.get(feature, np.nan)
                if np.isfinite(pval):
                    if (best_pval is None) or (pval < best_pval):
                        best_pval = pval
                        best_feature = feature
            except Exception:
                continue

        if (best_feature is not None) and (best_pval < p_enter):
            selected.append(best_feature)
            remaining.remove(best_feature)
            changed = True
            if verbose:
                print(f"[Forward] add {best_feature}, p={best_pval:.4f}")

        # backward by p
        while len(selected) > 0:
            formula = response + " ~ " + " + ".join(selected)
            model = smf.ols(formula=formula, data=df).fit()
            pvalues = model.pvalues.drop('Intercept', errors='ignore')

            if len(pvalues) == 0:
                break

            worst_feature = pvalues.idxmax()
            worst_pval = pvalues.max()

            if worst_pval > p_remove:
                selected.remove(worst_feature)
                if worst_feature not in remaining:
                    remaining.append(worst_feature)
                changed = True
                if verbose:
                    print(f"[Backward-p] remove {worst_feature}, p={worst_pval:.4f}")
            else:
                break

        # backward by vif
        while len(selected) >= 2:
            vif = calc_vif(df, selected)
            max_vif_feature = vif.idxmax()
            max_vif_value = vif.max()

            if max_vif_value > vif_thres:
                selected.remove(max_vif_feature)
                if max_vif_feature not in remaining:
                    remaining.append(max_vif_feature)
                changed = True
                if verbose:
                    print(f"[Backward-VIF] remove {max_vif_feature}, VIF={max_vif_value:.2f}")
            else:
                break

        if not changed:
            break

    cleaned = True
    while cleaned and len(selected) > 0:
        cleaned = False

        formula = response + " ~ " + " + ".join(selected)
        model = smf.ols(formula=formula, data=df).fit()
        pvalues = model.pvalues.drop('Intercept', errors='ignore')
        if len(pvalues) > 0 and pvalues.max() > p_remove:
            worst_feature = pvalues.idxmax()
            worst_pval = pvalues.max()
            selected.remove(worst_feature)
            if verbose:
                print(f"[Final-p-clean] remove {worst_feature}, p={worst_pval:.4f}")
            cleaned = True
            continue

        if len(selected) >= 2:
            vif = calc_vif(df, selected)
            if vif.max() > vif_thres:
                worst_feature = vif.idxmax()
                worst_vif = vif.max()
                selected.remove(worst_feature)
                if verbose:
                    print(f"[Final-VIF-clean] remove {worst_feature}, VIF={worst_vif:.2f}")
                cleaned = True

    return selected

model_predictors = stepwise_selection(
    df=df_step,
    response='TS',
    candidates=candidate_predictors,
    p_enter=0.05,
    p_remove=0.10,
    vif_thres=5.0,
    max_steps=100,
    verbose=True
)

if len(model_predictors) == 0:
    raise ValueError("No predictors survived stepwise selection. Please relax thresholds or check data quality.")

print("Selected predictors:", model_predictors)

# rollingCorr 图画全部候选因子；最终入模因子高亮
selected_predictors = candidate_predictors.copy()

# 重新组装
df_train = pd.concat(
    [TS.rename('TS')] + [all_X_train[x].rename(x) for x in selected_predictors],
    axis=1
)

df_pre = pd.concat(
    [TS_pre.rename('TS')] + [all_X_pre[x].rename(x) for x in selected_predictors],
    axis=1
)

# ============================================================
# (f) rolling correlation
# ============================================================
ax_rollingCorr = fig.add_subplot(gs[5])
ax_rollingCorr.set_ylim(-0.5, 1.0)

n_factor = len(selected_predictors)
try:
    cmap_amwg = plt.get_cmap(cmaps.amwg)
except Exception:
    cmap_amwg = cmaps.amwg
color_list = cmap_amwg(np.linspace(0, 1, max(n_factor, 1)))

line_handles = []
line_labels = []

for i, xname in enumerate(selected_predictors):
    is_selected = xname in model_predictors

    line, = ax_rollingCorr.plot(
        all_X_rollingCorr[xname].index,
        all_X_rollingCorr[xname].values,
        color=color_list[i],
        linewidth=1.5 if is_selected else 0.9,
        linestyle='-' if is_selected else '--',
        alpha=1.0,
        zorder=2 if is_selected else 1,
        label=xname
    )

    line_handles.append(line)
    line_labels.append(xname)

h1 = ax_rollingCorr.axhline(
    y=r_test(11, 0.1),
    color='black',
    linestyle='--',
    linewidth=1,
    label='90%',
    alpha=0.7
)
ax_rollingCorr.axhline(
    y=0,
    color='#999999',
    linestyle='-',
    linewidth=0.5,
    alpha=0.7
)

legend_ncol = min(4, max(1, int(np.ceil(n_factor / 2))))
leg = ax_rollingCorr.legend(
    handles=line_handles + [h1],
    labels=line_labels + ['90%'],
    loc='lower right',
    fontsize=6 * default_fontsize_times,
    ncol=legend_ncol,
    frameon=False
)

for txt in leg.get_texts()[:-1]:
    label = txt.get_text()
    if label in model_predictors:
        txt.set_fontweight('bold')
        txt.set_alpha(1.0)
    else:
        txt.set_fontweight('normal')
        txt.set_alpha(0.8)

leg.get_texts()[-1].set_fontweight('normal')
leg.get_texts()[-1].set_alpha(0.8)

ax_rollingCorr.set_title('(f) Rolling correlation', loc='left', fontsize=8)
ax_rollingCorr.tick_params(labelsize=6)

# ============================================================
# 最终回归建模
# ============================================================
formula = "TS ~ " + " + ".join(model_predictors)
print("Final formula:", formula)

model = smf.ols(formula=formula, data=df_train).fit()
print(model.summary())

intercept = model.params['Intercept']
coef_dict = {x: model.params[x] for x in model_predictors}

if len(model_predictors) >= 2:
    final_vif = calc_vif(df_step[['TS'] + model_predictors], model_predictors)
    print("Final VIF:")
    print(final_vif)

# 预测
df_train['predicted_TS'] = model.predict(df_train)
df_pre['inDependent_pre'] = model.predict(df_pre)
df_train['residuals'] = model.resid

TS_all_plot = pd.concat([TS.rename('TS'), TS_pre.rename('TS')])

# ============================================================
# (g) 预测图
# ============================================================
ax_predict = fig.add_subplot(gs[6])
ax_predict.set_ylim(-3, 3)

ax_predict.plot(TS_all_plot.index, TS_all_plot.values, color='black', linestyle='-', linewidth=1.5, label='Obs')
ax_predict.plot(df_train.index, df_train['predicted_TS'], color='blue', linestyle='--', linewidth=1.5, label='Reforecast')
ax_predict.plot(df_pre.index, df_pre['inDependent_pre'], color='red', linestyle=(0, (1, 1)), linewidth=1.5, label='Independent forecast')
ax_predict.axhline(y=0, color='#999999', linestyle='-', linewidth=0.5, alpha=0.5)
ax_predict.legend(loc='lower right', fontsize=6 * default_fontsize_times, ncol=3, frameon=False)
ax_predict.axvline(x=pd.to_datetime(f'{TR_time[1]}-6-30'), color='orange', linestyle='-', linewidth=1)

tcc_text_train = f"TCC={df_train['TS'].corr(df_train['predicted_TS']):.2f}"
rmse_text_train = f"RMSE={np.sqrt(np.mean((df_train['TS'] - df_train['predicted_TS'])**2)):.2f}"
ax_predict.text(
    0.08, 0.80, f'{tcc_text_train}\n{rmse_text_train}',
    transform=ax_predict.transAxes,
    ha='center', va='bottom', fontsize=6,
    bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='blue', alpha=0.6),
    zorder=10
)

tcc_text_pre = f"TCC={df_pre['TS'].corr(df_pre['inDependent_pre']):.2f}"
rmse_text_pre = f"RMSE={np.sqrt(np.mean((df_pre['TS'] - df_pre['inDependent_pre'])**2)):.2f}"
ax_predict.text(
    0.92, 0.80, f'{tcc_text_pre}\n{rmse_text_pre}',
    transform=ax_predict.transAxes,
    ha='center', va='bottom', fontsize=6,
    bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='red', alpha=0.6),
    zorder=10
)

equation_terms = [f'{intercept:.2f}']
for x in model_predictors:
    coef = coef_dict[x]
    sign = '+' if coef >= 0 else '-'
    equation_terms.append(f' {sign} {abs(coef):.2f}*{x}')

func = 'Days = ' + ''.join(equation_terms)

ax_predict.text(
    0.5, 0.88, func,
    transform=ax_predict.transAxes,
    ha='center', va='bottom', fontsize=8,
    bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='none', alpha=0.6),
    zorder=10
)

ax_predict.set_title('(g) Forecast', loc='left', fontsize=8)
ax_predict.tick_params(labelsize=6)

# ============================================================
# 输出结果
# ============================================================
print("Final selected predictors:", model_predictors)
print("Final equation:", func)

plt.savefig(fr'{PYFILE}/p5/pic/前期因子.pdf', bbox_inches='tight')
plt.savefig(fr'{PYFILE}/p5/pic/前期因子.png', dpi=600, bbox_inches='tight')
