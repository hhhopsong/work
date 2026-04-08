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
from climkit.filter import *


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

def pic(fig, pic_loc, lat, lon, lev, contf_var, title):

    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')

    ax.set_title(title, loc='left', fontsize=12)

    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    cont = ax.contourf(lon, lat, contf_var,  cmap=cmaps.CBR_wet[0] + cmaps.GMT_polar[11:-4], levels=lev, linewidths=0.8, transform=ccrs.PlateCarree(central_longitude=0))
    cont.clabel(inline=1, fontsize=4)




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

    return ax, cont

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


uvz6 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_500_201506_unzip.nc").sel(pressure_level=500)
uvz7 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_500_201507_unzip.nc").sel(pressure_level=500)
uvz8 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_500_201508_unzip.nc").sel(pressure_level=500)
uvz = xr.concat([uvz6, uvz7, uvz8], dim='valid_time')
uvz = uvz - uvz.mean('valid_time')  # 去时间平均

uvz = uvz.transpose('valid_time', 'latitude', 'longitude')  # 500hPa

#%%
u_10_hp    = LanczosFilter(uvz['u'], 'highpass', period=[10], nwts=9).filted()
v_10_hp    = LanczosFilter(uvz['v'], 'highpass', period=[10], nwts=9).filted()
z_10_hp    = LanczosFilter(uvz['z'], 'highpass', period=[10], nwts=9).filted()

u_10_30_bp = LanczosFilter(uvz['u'], 'bandpass', period=[10, 30], nwts=9).filted()
v_10_30_bp = LanczosFilter(uvz['v'], 'bandpass', period=[10, 30], nwts=9).filted()
z_10_30_bp = LanczosFilter(uvz['z'], 'bandpass', period=[10, 30], nwts=9).filted()

u_30_90_bp = LanczosFilter(uvz['u'], 'bandpass', period=[30, 90], nwts=9).filted()
v_30_90_bp = LanczosFilter(uvz['v'], 'bandpass', period=[30, 90], nwts=9).filted()
z_30_90_bp = LanczosFilter(uvz['z'], 'bandpass', period=[30, 90], nwts=9).filted()

u_90_lp    = LanczosFilter(uvz['u'], 'lowpass',  period=[90], nwts=9).filted()
v_90_lp    = LanczosFilter(uvz['v'], 'lowpass',  period=[90], nwts=9).filted()
z_90_lp    = LanczosFilter(uvz['z'], 'lowpass',  period=[90], nwts=9).filted()
#%%
fig = plt.figure(figsize=(7, 4))
plt.subplots_adjust(wspace=0.05, hspace=0.4)
title_head = '2015'

hp10_var_rt = ( z_10_hp.var(axis=0) /  uvz['z'].var(axis=0))
bp1030_var_rt = (z_10_30_bp.var(axis=0) / uvz['z'].var(axis=0))
bp3090_var_rt = (z_30_90_bp.var(axis=0) / uvz['z'].var(axis=0))
lp90_var_rt = ( z_90_lp.var(axis=0) / uvz['z'].var(axis=0))


ax = fig.add_subplot(2, 2, 1)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.set_aspect('auto')


# 空间图
lev = np.array([0., .08, .16, .24, .32, .4, .48, .56])
ax1, _ = pic(
    fig, 221,
    uvz['latitude'], uvz['longitude'], lev,
    hp10_var_rt, f'(a) {title_head} 0-10-day UV&EVR'
)

ax2, _ = pic(
    fig, 222,
    uvz['latitude'], uvz['longitude'], lev,
    bp1030_var_rt, f'(b) {title_head} 10-30-day UV&EVR'
)

ax3, _ = pic(
    fig, 223,
    uvz['latitude'], uvz['longitude'], lev,
    bp3090_var_rt, f'(c) {title_head} 30-90-day UV&EVR'
)

ax4, cont = pic(
    fig, 224,
    uvz['latitude'], uvz['longitude'], lev,
    lp90_var_rt, f'(d) {title_head} >90-day UV&EVR'
)

# 添加全局colorbar  # 为colorbar腾出空间
cbar_ax = inset_axes(ax4, width="4%", height="100%", loc='lower left', bbox_to_anchor=(1.025, 0., 1, 1),
                     bbox_transform=ax4.transAxes, borderpad=0)
cbar = fig.colorbar(cont, cax=cbar_ax, orientation='vertical', drawedges=True)
cbar.locator = ticker.FixedLocator(lev)
cbar.set_ticklabels([f"{i:.2f}" for i in lev])

for ax in fig.axes:
    # 遍历每个子图中的所有艺术家对象 (artist)
    for artist in ax.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)

plt.savefig(fr"{PYFILE}/p4/pic/多尺度分量方差占比_{title_head}.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p4/pic/多尺度分量方差占比_{title_head}.png", bbox_inches='tight', dpi=600)
plt.show()