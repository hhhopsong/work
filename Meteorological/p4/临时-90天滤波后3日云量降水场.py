import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.geometry import Point

from climkit.Cquiver import *
from climkit.masked import masked
from climkit.filter import *
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import dry_air_gas_constant as R
from metpy.constants import dry_air_spec_heat_press as cp


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

def pic(fig, pic_loc, lat, lon, cont_line, lev_line, lat_cf, lon_cf, contf_var, lev, title , lon_tick=np.arange(60, 160, 20), lat_tick=np.arange(0, 60, 15), key=True):

    ax = fig.add_subplot(*pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')

    ax.set_title(title, loc='left', fontsize=22)

    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    cont = ax.contourf(lon_cf, lat_cf, contf_var,  cmap=cmaps.sunshine_diff_12lev, levels=lev, linewidths=0, transform=ccrs.PlateCarree(central_longitude=0), extend='both', alpha=0.8)

    plt.rcParams['hatch.linewidth'] = 0.2
    plt.rcParams['hatch.color'] = 'green'
    ax.contourf(lon, lat, cont_line, levels=lev_line[1], hatches=['////////////', '////////////'], colors="none", add_colorbar=False, transform=ccrs.PlateCarree(central_longitude=0), edgecolor='none', linewidths=0)

    plt.rcParams['hatch.linewidth'] = 0.2
    plt.rcParams['hatch.color'] = 'brown'
    ax.contourf(lon, lat, cont_line, levels=lev_line[0], hatches=[r'\\\\\\\\\\\\', r'\\\\\\\\\\\\'], colors="none", add_colorbar=False, transform=ccrs.PlateCarree(central_longitude=0), edgecolor='none', linewidths=0)

    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4)
    ax.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='gray', edgecolor='black', linewidth=.5)

    # 刻度线设置
    xticks1 = lon_tick
    yticks1 = lat_tick
    if yticks1 is not None: ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
    if xticks1 is not None: ax.set_xticks(xticks1, crs=ccrs.PlateCarree())
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
    ax.tick_params(axis='both', labelsize=10, colors='black', length=0)

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

import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import dry_air_gas_constant as R
from metpy.constants import dry_air_spec_heat_press as cp

def region_mean_series(da, shp_path):
    vals = []
    for i in range(da.sizes['valid_time']):
        da_clip = masked(da.isel(valid_time=i), shp_path)
        vals.append(da_clip.mean(dim=('latitude', 'longitude'), skipna=True))
    return xr.concat(vals, dim='valid_time').assign_coords(valid_time=da['valid_time'])

tcc = xr.open_dataset(fr"{DATA}/ERA5/daily/2015060708_tcc_tpp/tcc.nc")
pre = xr.open_dataset(fr"{DATA}/ERA5/daily/2015060708_tcc_tpp/tpp.nc")*100 #mm
olr = xr.open_dataset(fr"{DATA}/NOAA/CPC/olr.cbo-1deg.day.mean.nc")
olr = olr.sel(time=slice('1961-01-01', '2022-12-31'))
olr = olr['olr'].sel(time=olr['time.month'].isin([6, 7, 8]))

tcc = tcc.transpose('valid_time', 'latitude', 'longitude')
pre = pre.transpose('valid_time', 'latitude', 'longitude')
olr = olr.transpose('time', 'lat', 'lon')
# olr逐日气候态

def get_typhoon_time_series(ds):
    """
    从台风 nc 中取出 obs 维时间，并转成 pandas.DatetimeIndex
    优先使用 CF 解码后的 time；
    若未解码成功，则尝试 time_str。
    """
    if 'time' in ds:
        try:
            t = pd.to_datetime(ds['time'].values)
            # 若能正常转 datetime，直接返回
            if not np.all(pd.isnull(t)):
                return pd.DatetimeIndex(t)
        except Exception:
            pass

    if 'time_str' in ds:
        # time_str 若是字符数组，需要拼接
        raw = ds['time_str'].values
        if raw.dtype.kind in ['S', 'U']:
            if raw.ndim == 2:
                t_str = [''.join(x.astype(str)).strip() for x in raw]
            else:
                t_str = [str(x).strip() for x in raw]
        else:
            t_str = [''.join([i.decode() if isinstance(i, bytes) else str(i) for i in row]).strip()
                     for row in raw]

        return pd.to_datetime(t_str, errors='coerce')

    raise ValueError("台风文件中既没有可识别的 time，也没有 time_str。")

def prepare_typhoon_tracks(ty_ds):
    """
    整理台风 obs 表，筛选出 2015-06-01 ~ 2015-08-31 期间出现过的台风。
    返回 DataFrame:
        storm_index, time, latitude, longitude
    """
    ty_time = get_typhoon_time_series(ty_ds)

    df = pd.DataFrame({
        'storm_index': ty_ds['storm_index'].values.astype(int),
        'time': ty_time,
        'latitude': ty_ds['latitude'].values,
        'longitude': ty_ds['longitude'].values,
    })

    # 去掉缺测
    df = df.dropna(subset=['time', 'latitude', 'longitude']).copy()

    # 只保留 2015 年 6-8 月的路径点
    summer_mask = (df['time'] >= pd.Timestamp('2015-06-01 00:00:00')) & \
                  (df['time'] <  pd.Timestamp('2015-09-01 00:00:00'))
    summer_df = df.loc[summer_mask].copy()

    # 选出“在 6-8 月出现过”的台风编号
    valid_storms = np.sort(summer_df['storm_index'].unique())

    # 保留这些台风的全部 2015 年路径点，方便画“从开始到截止日期”
    all_2015_mask = (df['time'] >= pd.Timestamp('2015-01-01 00:00:00')) & \
                    (df['time'] <  pd.Timestamp('2016-01-01 00:00:00'))
    df_2015 = df.loc[all_2015_mask & df['storm_index'].isin(valid_storms)].copy()

    # 排序
    df_2015 = df_2015.sort_values(['storm_index', 'time']).reset_index(drop=True)

    return df_2015, valid_storms

typhoon = xr.open_dataset(fr"{DATA}/Typhoon/CMABSTdata/CH2015BST.nc")

ty_df, valid_storms = prepare_typhoon_tracks(typhoon)
#%%
tcc_ano = tcc - tcc.mean('valid_time')
pre_ano = pre - pre.mean('valid_time')
olr_ano = olr.sel(time=slice('2015-01-01', '2015-12-31')) - olr.groupby(olr.time.dt.strftime('%m-%d')).mean('time').data
tcc_bp = LanczosFilter(tcc_ano['tcc'], 'bandpass', period=[10, 30], nwts=9).filted()
pre_bp = LanczosFilter(pre_ano['tp'], 'bandpass', period=[10, 30], nwts=9).filted()
olr_bp = LanczosFilter(olr_ano.data, 'bandpass', period=[10, 30], nwts=9).filted()

yangtze_shp = fr'{PYFILE}/map/self/长江_TP/长江_tp.shp'
#%%
fig = plt.figure(figsize=(12, 10))
plt.subplots_adjust(wspace=0, hspace=0)
title_head = '2015'

# 空间图
# lev = np.array([0., .08, .16, .24, .32, .4, .48, .56])
lev = np.array([-0.4, -0.35, -0.30, -0.25, -0.20, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4])
lev = np.array([-5, -4, -3, -2, -1, -.5, -.25, 0.25, 0.5, 1, 2, 3, 4, 5])*10
lev_pre = np.array([[-10, -3, -1], [1, 3, 10]])*0.01
index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
for i in range(29):
    iday=i*3+3
    if i==0:
        ax, cont = pic(
            fig, (6, 5, i+1),
            tcc['latitude'], tcc['longitude'], pre_bp[iday], lev_pre,
            olr['lat'], olr['lon'], olr_bp[iday], lev,
            f'2015 OLR&PRE',lat_tick=None, lon_tick=None, key=False
        )
    elif i==4:
        ax, cont = pic(
            fig, (6, 5, i + 1),
            tcc['latitude'], tcc['longitude'], pre_bp[iday], lev_pre,
            olr['lat'], olr['lon'], olr_bp[iday], lev,
            f'', lat_tick=None, lon_tick=None, key=True
        )
    else:
        ax, cont = pic(
            fig, (6, 5, i+1),
            tcc['latitude'], tcc['longitude'], pre_bp[iday], lev_pre,
            olr['lat'], olr['lon'], olr_bp[iday], lev,
            f'', key=False, lat_tick=None, lon_tick=None,
        )
    # ===== 当前子图日期 =====
    from matplotlib.patheffects import withStroke
    panel_time = pd.to_datetime(tcc['valid_time'].isel(valid_time=iday).values)
    date_text = panel_time.strftime('%m/%d')
    ax.text(0.05, 0.95, date_text, transform=ax.transAxes, fontsize=10, color='red',
            ha='left', va='top', bbox=dict(facecolor='none', edgecolor='none'),
            path_effects=[withStroke(linewidth=1.5, foreground='white')])

    # ===== 只叠加“2015年6-8月出现过的台风” =====
    # 对每个台风：画从开始到 panel_time 的轨迹
    # 再把当天位置打红点
    panel_day0 = panel_time.normalize()
    panel_day1 = panel_day0 + pd.Timedelta(days=1)

    for sid in valid_storms:
        storm_df = ty_df[ty_df['storm_index'] == sid].sort_values('time')

        if len(storm_df) == 0:
            continue

        storm_start = storm_df['time'].min().normalize()
        storm_end = storm_df['time'].max().normalize()

        # 只有当 panel_time 落在该台风生命期内，才显示这条台风
        if storm_start <= panel_day0 <= storm_end:
            # 直接画这条台风的完整轨迹（从开始到结束）
            ax.plot(
                storm_df['longitude'].values,
                storm_df['latitude'].values,
                color='#454545',
                linewidth=2,
                transform=ccrs.PlateCarree(),
                zorder=20,
                alpha=0.7
            )

            # 当天位置打红点
            storm_today = storm_df[(storm_df['time'] >= panel_day0) & (storm_df['time'] < panel_day1)]
            if len(storm_today) > 0:
                ax.plot(
                    storm_today['longitude'].values,
                    storm_today['latitude'].values,
                    color='red',
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                    zorder=21,
                    alpha=0.7
                )

# 添加全局colorbar  # 为colorbar腾出空间
cbar_ax = inset_axes(ax, width="4%", height="100%", loc='lower left', bbox_to_anchor=(1.025, 0., 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
cbar = fig.colorbar(cont, cax=cbar_ax, orientation='vertical', drawedges=True)
cbar.locator = ticker.FixedLocator(lev)
cbar.set_ticklabels([f"{i:.1f}" for i in lev])

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.set_aspect('auto')

for ax in fig.axes:
    # 遍历每个子图中的所有艺术家对象 (artist)
    for artist in ax.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)

plt.savefig(fr"{PYFILE}/p4/pic/30天滤波云量降水场_{title_head}.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p4/pic/30天滤波云量降水场_{title_head}.png", bbox_inches='tight', dpi=600)
plt.show()