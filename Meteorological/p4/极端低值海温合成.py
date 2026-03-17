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
import matplotlib.colors as mcolors
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
    spine_lw = 1.2
    def latlon_fmt(ax, xticks1, yticks1, xminorLocator, yminorLocator):
        ax.set_yticks(yticks1)
        ax.set_xticks(xticks1)
        ax.xlocator = FixedLocator(xticks1)
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_minor_locator(yminorLocator)
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.tick_params(axis='both', which='major', direction='out', length=4, width=spine_lw, color='black', bottom=True,
                       left=True)
        ax.tick_params(axis='both', which='minor', direction='out', length=2, width=.2, color='black', bottom=True,
                       left=True)
        ax.tick_params(axis='both', labelsize=6 * fontsize_times, colors='black')    # 统一加粗所有四个边框

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
    for spine in axes_sub.spines.values():
        spine.set_linewidth(spine_lw)  # 设置边框线宽
    latlon_fmt(axes_sub, geoticks['x'], geoticks['y'],  MultipleLocator(geoticks['xminor']),
               MultipleLocator(geoticks['yminor']))
    axes_sub.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.7, color='#757575', alpha=0.75)
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
    if abs(extent[1] - (extent[0] + 360)) <= 5:
        axes_sub.set_xlim(extent[0], extent[1])
        axes_sub.set_ylim(extent[2], extent[3])
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
            wind1.key(U=wind_1_key_set['U'], label=wind_1_key_set['label'], bbox_to_anchor=bbox_to_anchor_1, loc=loc1, edgewidth=spine_lw,
                      edgecolor=wind_1_key_set['edgecolor'], arrowsize=wind_1_key_set['arrowsize'], linewidth=wind_1_key_set['lw'])
        else:
            wind1.key(U=wind_1_key_set['U'], label=wind_1_key_set['label'], bbox_to_anchor=bbox_to_anchor_1, loc=loc1, edgewidth=spine_lw,
                      edgecolor=wind_1_key_set['edgecolor'], arrowsize=wind_1_key_set['arrowsize'], linewidth=wind_1_key_set['lw'])

    # 风矢量No.2
    if wind_2_signal:
        wind2 = axes_sub.Curlyquiver(wind_2['lon'], wind_2['lat'], wind_2['u'], wind_2['v'],
                            arrowsize=wind_2_set['arrowsize'], transform=ccrs.PlateCarree(central_longitude=0),
                            scale=wind_2_set['scale'], linewidth=wind_2_set['lw'], regrid=wind_2_set['regrid'],
                            color=wind_2_set['color'], thinning=wind_2_set['thinning'], nanmax=wind_2_set['nanmax'],
                            MinDistance=wind_2_set['MinDistance'])
        wind2.key(U=wind_2_key_set['U'], label=wind_2_key_set['label'], bbox_to_anchor=bbox_to_anchor_2, loc=loc2, edgewidth=spine_lw,
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


# 下列参数的默认值
xticks = np.arange(-180, 181, 30)
yticks = np.arange(-30, 81, 30)
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
default_shading_levels = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])
default_shading_cmap = cmaps.temp_diff_18lev[5:-5]
default_shading_corr = False
default_p_test_drawSet = {'N': 60, 'alpha': 0.1, 'lw': 0.2, 'color': '#FFFFFF'} # 显著性绘制设置, 可为False
default_edgedraw = False # 填色图边缘绘制
## 填色图2
default_extent2 = [-180, 180, -30, 80]  # 子图范围
default_geoticks2 = {'x': xticks, 'y': yticks,
                    'xminor': 10, 'yminor': 10}  # 地理坐标刻度
default_shading_levels2 = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])
default_shading_cmap2 = cmaps.temp_diff_18lev[5:-5]
default_shading_corr2 = False
default_p_test_drawSet2 = {'N': 60, 'alpha': 0.1, 'lw': 0.2, 'color': '#FFFFFF'} # 显著性绘制设置, 可为False
default_edgedraw2 = False # 填色图边缘绘制
## 等值线
default_contour_levels = [[-1, -0.5, -0.2], [0.2, 0.5, 1]]
default_contour_cmap = ['blue', 'red']
default_contour_corr = False # 等值线相关系数结果
default_p_test_drawSet_corr = {'N': 60, 'alpha': 0.1} # 显著
## 风矢量_1
default_wind_1_set = {'regrid': 15, 'arrowsize': 0.5, 'scale': 5, 'lw': 0.15,
                      'color': 'black', 'thinning': ['25%', 'min'], 'nanmax': 20/3, 'MinDistance': [0.2, 0.1]}
default_wind_1_key_set = {'U': 1, 'label': '1 m/s', 'ud': 7.7, 'lr': None, 'arrowsize': 0.5, 'edgecolor': 'none', 'lw': 0.5}
## 风矢量_2
default_wind_2_set = {'regrid': 12, 'arrowsize': 0.5, 'scale': 5, 'lw': 0.4,
                      'color': 'purple', 'thinning': ['40%', 'min'], 'nanmax': 0.1, 'MinDistance': [0.2, 0.1]}
default_wind_2_key_set = {'U': 0.03, 'label': '0.03 m$^2$/s$^2$', 'ud': 7.7, 'lr': 1.7, 'arrowsize': 0.5, 'edgecolor': 'none', 'lw': 0.5}
# 矩形框设置, 可为None
default_rec_Set = False


PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

# SST
sst = ersst(fr"{DATA}/NOAA/ERSSTv5/sst.mnmean.nc", 1961, 2023)
sst = sst.sel(time=slice('1961-01-01', '2023-12-31'))
sst = xr.Dataset(
    {'sst': (['time', 'lat', 'lon'], sst['sst'].data)},
    coords={'time': pd.to_datetime(sst['time'], format="%Y%m%d"),
            'lat': sst['lat'].data,
            'lon': sst['lon'].data})
sst = sst.sel(time=slice('1961-01-01', '2023-12-31'))
sst = sst.sel(time=sst['time.month'].isin([5, 6, 7, 8, 9]))
sst = sst.transpose('time', 'lat', 'lon')
sst_clim = sst.groupby('time.month').mean('time')  # 逐月气候态
sst_ano = sst.groupby('time.month') - sst_clim  # 逐月距平
sst_ano = sst_ano.assign_coords(year=sst_ano["time"].dt.year, month=sst_ano["time"].dt.month).set_index(time=["year", "month"]).unstack("time")

# UV
uvz = xr.open_dataset(fr"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc").sel(
    date=slice('1961-01-01', '2024-12-31'),
    pressure_level=[200, 500, 850],
    latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])
uvz = xr.Dataset(
    {'u': (['time', 'p', 'lat', 'lon'], uvz['u'].data),
     'v': (['time', 'p', 'lat', 'lon'], uvz['v'].data),
     'z': (['time', 'p', 'lat', 'lon'], uvz['z'].data)},
    coords={'time': pd.to_datetime(uvz['date'], format="%Y%m%d"),
            'p': uvz['pressure_level'].data,
            'lat': uvz['latitude'].data,
            'lon': uvz['longitude'].data})
uvz = uvz.sel(time=slice('1961-01-01', '2024-12-31'))
uvz = uvz.transpose('time', 'p', 'lat', 'lon')
uvz_clim = uvz.groupby('time.month').mean('time')  # 逐月气候态
uvz_ano = uvz.groupby('time.month') - uvz_clim  # 逐月距平
uvz_ano = uvz_ano.assign_coords(year=uvz_ano["time"].dt.year, month=uvz_ano["time"].dt.month).set_index(time=["year", "month"]).unstack("time")

# PRE
pre = prec(f"{DATA}/NOAA/PREC/precip.mon.anom.nc", 1961, 2023)
pre = transform(pre, type='180->360')
pre = xr.Dataset(
    {'pre': (['time', 'lat', 'lon'], pre['pre'].data)},
    coords={'time': pd.to_datetime(pre['time'], format="%Y%m%d"),
            'lat': pre['lat'].data,
            'lon': pre['lon'].data})
pre = pre.sel(time=slice('1961-01-01', '2023-12-31'))
pre = pre.transpose('time', 'lat', 'lon')
pre_clim = pre.groupby('time.month').mean('time')  # 逐月气候态
pre_ano = pre.groupby('time.month') - pre_clim  # 逐月距平
pre_ano = pre_ano.assign_coords(year=pre_ano["time"].dt.year, month=pre_ano["time"].dt.month).set_index(time=["year", "month"]).unstack("time")
#%%
print("######draw######")
fig = plt.figure(figsize=np.array([17, 6*3])*0.5)   # 创建画布
proj = ccrs.PlateCarree(central_longitude=110)  # 投影方式
spec = gridspec.GridSpec(ncols=3, nrows=2, wspace=0.16, hspace=0.13)  # 设置子图比例
plt.rcParams['font.family'] = ['AVHershey Simplex', 'AVHershey Duplex', 'Helvetica']    # 字体为Hershey (安装字体后，清除.matplotlib的字体缓存即可生效)
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

for i in [5, 6, 7, 8, 9]:
    year = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014, 2015]

    ax_spec = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=spec[i - 5], wspace=0, hspace=0)

    ax1 = fig.add_subplot(ax_spec[0], projection=proj)  # 添加子图
    ax1.set_aspect('auto')  # 设置长宽比
    sub_pic(ax1, title=f'({chr(ord('a')+i-5)}) {i}_200UVZ_500UV&SST_850UV&PRE', extent=[-180, 180, -30, 80],
            geoticks={'x': [], 'y': yticks, 'xminor': 10, 'yminor': 10},
            fontsize_times=1.0,

            shading=None, shading_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]),
            shading_cmap=cmaps.GreenMagenta16[8 - 5:8] + cmaps.GMT_red2green_r[11:11 + 4], cb_draw=True if i==7 else False,
            shading_corr=None,
            p_test_drawSet={'N': 2023-1961+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
            edgedraw=False,

            shading2=uvz_ano['z'].sel(year=year, month=i, p=200).mean('year'), shading2_levels=[-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600],
            shading2_cmap=cmaps.sunshine_diff_12lev, cb_draw2=True if i==7 else False,
            shading2_corr=None,
            p_test_drawSet2={'N': 2023-1961+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
            edgedraw2=False,

            contour=None, contour_levels=np.array([[-50, -20], [20, 50]]) * .005, contour_cmap=default_contour_cmap,

            contour_corr=None, p_test_drawSet_corr={'N': 2023-1961+1, 'alpha': 0.1},

            wind_1=uvz_ano.sel(year=year, month=i, p=200).mean('year'),
            wind_1_set={'regrid': 15, 'arrowsize': 0.5, 'scale': 10, 'lw': 0.4,
                      'color': 'black', 'thinning': ['30%', 'min'], 'nanmax': 20/3, 'MinDistance': [0.2, 0.5]},
            wind_1_key_set={'U': 5, 'label': '5 m/s', 'ud': 7.7, 'lr': None, 'arrowsize': 5, 'edgecolor': 'black', 'lw': 0.5},
            bbox_to_anchor_1=None, loc1="lower right",

            wind_2=None,
            wind_2_set={'regrid': 20, 'arrowsize': 0.5, 'scale': 5, 'lw': 0.4,
                      'color': 'purple', 'thinning': ['40%', 'min'], 'nanmax': 0.1, 'MinDistance': [0.2, 0.5]},
            wind_2_key_set={'U': 0.03, 'label': '0.03 m$^2$/s$^2$', 'ud': 7.7, 'lr': 1.7, 'arrowsize': 5, 'edgecolor': 'none', 'lw': 0.5},
            bbox_to_anchor_2=None, loc2="upper right",
            rec_Set=None)
    ax1.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax1.add_geometries(Reader(fr'{PYFILE}/map/地图线路数据/长江/长江.shp').geometries(), ccrs.PlateCarree(),
                       facecolor='none', edgecolor='blue', linewidth=0.2)
    ax1.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/长江区1：25万界线数据集（2002年）/长江区.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=.5)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ax2 = fig.add_subplot(ax_spec[1], projection=proj)  # 添加子图
    ax2.set_aspect('auto')  # 设置长宽比
    sub_pic(ax2, title=f'', extent=[-180, 180, -30, 80],
            geoticks={'x': [], 'y': yticks, 'xminor': 10, 'yminor': 10},
            fontsize_times=1.0,

            shading=None, shading_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]),
            shading_cmap=cmaps.GreenMagenta16[8 - 5:8] + cmaps.GMT_red2green_r[11:11 + 4],
            cb_draw=True if i == 7 else False,
            shading_corr=None,
            p_test_drawSet={'N': 2023-1961+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
            edgedraw=False,

            shading2=sst_ano.sel(year=year, month=i).mean('year').to_array()[0],
            shading2_levels=np.round(np.array([-1., -.8, -.6, -.4, -.2, -.1, .1, .2, .4, .6, .8, 1.]) * .5, 2),
            shading2_cmap=cmaps.BlueWhiteOrangeRed[40:-40], cb_draw2=True if i == 7 else False,
            shading2_corr=None,
            p_test_drawSet2={'N': 2023-1961+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
            edgedraw2=False,

            contour=None, contour_levels=np.array([[-50, -20], [20, 50]]) * .005, contour_cmap=default_contour_cmap,

            contour_corr=None, p_test_drawSet_corr={'N': 2023-1961+1, 'alpha': 0.1},

            wind_1=uvz_ano.sel(year=year, month=i, p=500).mean('year'),
            wind_1_set={'regrid': 15, 'arrowsize': 0.5, 'scale': 20, 'lw': 0.4,
                        'color': 'black', 'thinning': ['20%', 'min'], 'nanmax': 20 / 3, 'MinDistance': [0.2, 0.5]},
            wind_1_key_set={'U': 2.5, 'label': '2.5 m/s', 'ud': 7.7, 'lr': None, 'arrowsize': 2.5, 'edgecolor': 'k',
                            'lw': 0.5},
            bbox_to_anchor_1=None, loc1="lower right",

            wind_2=None,
            wind_2_set={'regrid': 20, 'arrowsize': 0.5, 'scale': 5, 'lw': 0.4,
                        'color': 'purple', 'thinning': ['40%', 'min'], 'nanmax': 0.1, 'MinDistance': [0.2, 0.5]},
            wind_2_key_set={'U': 0.03, 'label': '0.03 m$^2$/s$^2$', 'ud': 7.7, 'lr': 1.7, 'arrowsize': 2.5,
                            'edgecolor': 'none', 'lw': 0.5},
            bbox_to_anchor_2=None, loc2="upper right",
            rec_Set=None)
    ax2.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax2.add_geometries(Reader(fr'{PYFILE}/map/地图线路数据/长江/长江.shp').geometries(), ccrs.PlateCarree(),
                       facecolor='none', edgecolor='blue', linewidth=0.2)
    ax2.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/长江区1：25万界线数据集（2002年）/长江区.shp').geometries(),
                       ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=.5)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ax3 = fig.add_subplot(ax_spec[2], projection=proj)  # 添加子图
    ax3.set_aspect('auto')  # 设置长宽比
    sub_pic(ax3, title=f'', extent=[-180, 180, -30, 80],
            geoticks={'x': np.arange(-180, 181, 60), 'y': yticks, 'xminor': 10, 'yminor': 10},
            fontsize_times=1.0,

            shading=None, shading_levels=np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]),
            shading_cmap=cmaps.GreenMagenta16[8 - 5:8] + cmaps.GMT_red2green_r[11:11 + 4], cb_draw=True if i==7 else False,
            shading_corr=None,
            p_test_drawSet={'N': 2023-1961+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
            edgedraw=False,

            shading2=pre_ano['pre'].sel(year=year, month=i).mean('year'), shading2_levels=np.round(np.array([-1., -.8, -.6, -.4, -.2, -.1, .1, .2, .4, .6, .8, 1.])*5, 2),
            shading2_cmap=cmaps.MPL_RdYlGn[22+0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106-0], cb_draw2=True if i==7 else False,
            shading2_corr=None,
            p_test_drawSet2={'N': 2023-1961+1, 'alpha': 0.1, 'lw': 0.2, 'color': '#454545'},
            edgedraw2=False,

            contour=None, contour_levels=np.array([[-50, -20], [20, 50]]) * .005, contour_cmap=default_contour_cmap,

            contour_corr=None, p_test_drawSet_corr={'N': 2023-1961+1, 'alpha': 0.1},

            wind_1=uvz_ano.sel(year=year, month=i, p=850).mean('year'),
            wind_1_set={'regrid': 15, 'arrowsize': 0.5, 'scale': 33.3, 'lw': 0.4,
                      'color': 'black', 'thinning': ['20%', 'min'], 'nanmax': 20/3, 'MinDistance': [0.2, 0.5]},
            wind_1_key_set={'U': 1.5, 'label': '1.5 m/s', 'ud': 7.7, 'lr': None, 'arrowsize': 1.25, 'edgecolor': 'k', 'lw': 0.5},
            bbox_to_anchor_1=None, loc1="lower right",

            wind_2=None,
            wind_2_set={'regrid': 20, 'arrowsize': 0.5, 'scale': 5, 'lw': 0.4,
                      'color': 'purple', 'thinning': ['40%', 'min'], 'nanmax': 0.1, 'MinDistance': [0.2, 0.5]},
            wind_2_key_set={'U': 0.03, 'label': '0.03 m$^2$/s$^2$', 'ud': 7.7, 'lr': 1.7, 'arrowsize': 0.8, 'edgecolor': 'none', 'lw': 0.5},
            bbox_to_anchor_2=None, loc2="upper right",
            rec_Set=None)
    ax3.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax3.add_geometries(Reader(fr'{PYFILE}/map/地图线路数据/长江/长江.shp').geometries(), ccrs.PlateCarree(),
                       facecolor='none', edgecolor='blue', linewidth=0.2)
    ax3.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp').geometries(),
                       ccrs.PlateCarree(), facecolor='gray', edgecolor='gray', linewidth=.1, hatch='.', zorder=10)
    ax3.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/长江区1：25万界线数据集（2002年）/长江区.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=.5)

plt.savefig(fr'{PYFILE}/p4/pic/图2_低值合成.pdf', bbox_inches='tight')
plt.savefig(fr'{PYFILE}/p4/pic/图2_低值合成.png', dpi=600, bbox_inches='tight')