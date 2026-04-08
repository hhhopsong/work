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
from scipy import ndimage
from scipy.stats import ttest_ind

from climkit.Cquiver import *
from climkit.masked import masked
from climkit.significance_test import r_test
from climkit.lonlat_transform import *

from matplotlib import ticker
from metpy.calc import vertical_velocity
from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants


# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

def plot_text(ax, x, y, title, size, color):
    ax.text(x, y, title,
         transform=ccrs.PlateCarree(),
         ha='center',
         va='center',
         fontsize=size,
         fontweight='bold',
         color=color,
         fontname='Times New Roman',
         zorder=1000)
    return 0

def pic(fig, pic_loc, lat, lon, lev, corr_u, corr_v, corr_z, title):

    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')

    idx = int(str(pic_loc)[2]) - 3
    ax.set_title(title, loc='left', fontsize=12)

    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())


    # 显著性打点
    cont = ax.contour(lon, lat, corr_z, colors='red', levels=lev[1], linewidths=0.8, transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(lon, lat, corr_z, colors='blue', levels=lev[0], linestyles='--', linewidths=0.8,
                       transform=ccrs.PlateCarree(central_longitude=0))
    cont.clabel(inline=1, fontsize=4)
    cont_.clabel(inline=1, fontsize=4)
    #cont_clim = ax.contour(lon, lat, uvz_clim['z'], colors='k', levels=20, linewidths=0.6, transform=ccrs.PlateCarree(central_longitude=0))

    Cq = ax.Curlyquiver(lon, lat, corr_u, corr_v, center_lon=110, scale=5, linewidth=0.5, arrowsize=1., transform=ccrs.PlateCarree(central_longitude=0), MinDistance=[0.2, 0.5],
                     regrid=12, color='#454545', nanmax=5)

    Cq.key(U=2, label='2 m/s', color='k', fontproperties={'size': 8}, linewidth=.7, arrowsize=3.)
    nanmax = Cq.nanmax
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4)
    ax.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='gray', edgecolor='black', linewidth=.5)

    # 刻度线设置
    xticks1 = np.arange(60, 160, 20)
    yticks1 = np.arange(0, 60, 15)
    ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks1, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)

    ymajorLocator = MultipleLocator(15)  # 先定义xmajorLocator，再进行调用
    ax.yaxis.set_major_locator(ymajorLocator)  # x轴最大刻度
    yminorLocator = MultipleLocator(5)
    ax.yaxis.set_minor_locator(yminorLocator)  # x轴最小刻度
    xmajorLocator = MultipleLocator(20)  # 先定义xmajorLocator，再进行调用
    xminorLocator = MultipleLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
    ax.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
    # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
    ax.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax.tick_params(axis='both', labelsize=10, colors='black')

    return ax

import numpy as np
import xarray as xr
from scipy.stats import ttest_ind


def composite_analysis(year_list, data, years=None, equal_var=True, var_name=None):
    """
    YEAR合成分析 + 显著性检验

    Parameters
    ----------
    year_list : list
        合成年份
    data : np.ndarray | xr.DataArray | xr.Dataset
        如果是 ndarray，shape=(nyear, ...)
        如果是 DataArray，默认第一维为时间/年份维
        如果是 Dataset，需要指定 var_name
    years : array-like, optional
        与 data 第一维对应的年份
        若 data 为 DataArray/Dataset 且其第一维坐标就是年份，可不传
    equal_var : bool
        是否使用方差齐性假设
    var_name : str, optional
        当 data 为 xr.Dataset 时，要分析的变量名

    Returns
    -------
    comp_diff : np.ndarray | xr.DataArray
        合成差值
    p_val : np.ndarray | xr.DataArray
        t检验 p 值，不可检验处为 NaN
    """

    # -------------------------
    # 1. 统一输入类型
    # -------------------------
    original_type = None
    original_dims = None
    original_coords = None
    original_name = None

    if isinstance(data, xr.Dataset):
        if var_name is None:
            raise ValueError("当 data 为 xarray.Dataset 时，必须提供 var_name。")
        if var_name not in data:
            raise ValueError(f"var_name='{var_name}' 不在 Dataset 中。")
        data = data.to_array()
        original_type = "dataset"

    if isinstance(data, xr.DataArray):
        original_type = "dataarray" if original_type is None else original_type
        original_dims = data.dims
        original_coords = data.coords
        original_name = data.name

        if years is None:
            first_dim = data.dims[0]
            years = data[first_dim].values

        data_np = np.array(data)

    elif isinstance(data, np.ndarray):
        original_type = "ndarray"
        data_np = data
        if years is None:
            raise ValueError("当 data 为 np.ndarray 时，必须提供 years。")

    else:
        raise TypeError("data 必须是 np.ndarray、xarray.DataArray 或 xarray.Dataset。")

    # -------------------------
    # 2. 年份筛选
    # -------------------------
    years = np.array(years)
    year_list = np.array(year_list)

    if data_np.shape[0] != len(years):
        raise ValueError("years 长度必须与 data 第一维长度一致。")

    sel_mask = np.isin(years, year_list)
    oth_mask = ~sel_mask

    if sel_mask.sum() == 0:
        raise ValueError("YEAR 中没有匹配到任何年份。")
    if oth_mask.sum() == 0:
        raise ValueError("除 YEAR 外没有剩余年份，无法做显著性检验。")

    sample_sel = data_np[sel_mask]

    # -------------------------
    # 3. 合成差值
    # -------------------------
    comp_diff_np = np.nanmean(sample_sel, axis=0) - np.nanmean(data_np, axis=0)

    # -------------------------
    # 4. 显著性检验
    # -------------------------
    t_stat, p_val_np = ttest_ind(
        sample_sel,
        data_np,
        axis=0,
        equal_var=equal_var,
        nan_policy="omit"
    )

    # -------------------------
    # 5. 还原为 DataArray
    # -------------------------
    if original_type in ["dataarray", "dataset"]:
        out_dims = original_dims[1:]
        out_coords = {dim: original_coords[dim] for dim in out_dims if dim in original_coords}

        comp_diff = xr.DataArray(
            comp_diff_np,
            dims=out_dims,
            coords=out_coords,
            name=f"{original_name}" if original_name else "comp_diff"
        )

        p_val = xr.DataArray(
            p_val_np,
            dims=out_dims,
            coords=out_coords,
            name=f"{original_name}" if original_name else "p_val"
        )

        return comp_diff, p_val

    return comp_diff_np, p_val_np


try:
    uvz = xr.open_dataset(fr"{PYFILE}/p2/data/uvz_78.nc")
except:
    uvz = xr.open_dataset(fr"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc").sel(
        date=slice('1961-01-01', '2023-12-31'),
        pressure_level=[200, 500, 850],
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
    uvz.to_netcdf(fr"{PYFILE}/p2/data/uvz_78.nc")
uvz = uvz.sel(p=[200, 500, 850]).transpose('year', 'p', 'lat', 'lon')  # 500hPa
uvz_clim = uvz.mean('year')

#%%
# 去趋势处理
def detrend(obj, dim='year', deg=1):
    if isinstance(obj, xr.DataArray):
        coef = obj.polyfit(dim=dim, deg=deg, skipna=True)
        trend = xr.polyval(obj[dim], coef.polyfit_coefficients)
        return obj - trend

    elif isinstance(obj, xr.Dataset):
        out = xr.Dataset(coords=obj.coords, attrs=obj.attrs)
        for name, da in obj.data_vars.items():
            coef = da.polyfit(dim=dim, deg=deg, skipna=True)
            trend = xr.polyval(da[dim], coef.polyfit_coefficients)
            out[name] = da - trend
        return out

    else:
        raise TypeError("obj 必须是 xarray.DataArray 或 xarray.Dataset")

#%%

# YEAR = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014]
YEAR = [2015]

comp_u, _ = composite_analysis(YEAR, uvz['u'].data, uvz['year'].data)
comp_v, _ = composite_analysis(YEAR, uvz['v'].data, uvz['year'].data)
comp_z, _ = composite_analysis(YEAR, uvz['z'].data, uvz['year'].data)

#%%
fig = plt.figure(figsize=(11.5/3, 7))
plt.subplots_adjust(wspace=0.05, hspace=0.4)
title_head = '2015'

ax = fig.add_subplot(3, 1, 1)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.set_aspect('auto')


# 空间图
ax1 = pic(
    fig, 311,
    uvz['lat'], uvz['lon'],
    np.array([[-40, -20, -10], [40, 60, 80]])*8,
    comp_u[0], comp_v[0], comp_z[0], f'(a) {title_head} 200UVZ'
)

ax2 = pic(
    fig, 312,
    uvz['lat'], uvz['lon'],
    np.array([[-40, -20, -10], [40, 60, 80]])*4,
    comp_u[1], comp_v[1], comp_z[1], f'(b) {title_head} 500UVZ'
)

ax3 = pic(
    fig, 313,
    uvz['lat'], uvz['lon'],
    np.array([[-40, -20, -10], [40, 60, 80]])*4,
    comp_u[2], comp_v[2], comp_z[2], f'(c) {title_head} 850UVZ'
)
ax3.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp').geometries(),
                                   ccrs.PlateCarree(), facecolor='#909090', edgecolor='#909090', linewidth=.1, hatch='.', zorder=10)

for ax in fig.axes:
    # 遍历每个子图中的所有艺术家对象 (artist)
    for artist in ax.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)

plt.savefig(fr"{PYFILE}/p4/pic/三层环流_{title_head}.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p4/pic/三层环流_{title_head}.png", bbox_inches='tight', dpi=600)
plt.show()