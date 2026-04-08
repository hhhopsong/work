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

def pic(fig, pic_loc, lat, lon, u, v, lev, contf_var, title , lon_tick=np.arange(60, 160, 20), lat_tick=np.arange(0, 60, 15), key=True):

    ax = fig.add_subplot(*pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')

    ax.set_title(title, loc='left', fontsize=22)

    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    cont = ax.contourf(lon, lat, contf_var,  cmap=cmaps.BlueWhiteOrangeRed[10:-10], levels=lev, linewidths=0.8, transform=ccrs.PlateCarree(central_longitude=0), extend='both', alpha=0.8)

    Cq = ax.Curlyquiver(lon, lat, u, v, center_lon=110, scale=2, linewidth=0.5, arrowsize=1., transform=ccrs.PlateCarree(central_longitude=0), MinDistance=[0.2, 0.5],
                     regrid=12, color='#454545', nanmax=5)
    if key: Cq.key(U=5, label='5 m/s', color='k', fontproperties={'size': 14}, linewidth=.7, arrowsize=12., loc='upper right',
                   bbox_to_anchor=(0, 0.32, 1, 1), edgecolor='none', shrink=0.4, intetval=0.7)

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

def calc_daily_temperature_budget(t_da, u_da, v_da, w_da):
    """
    计算逐日温度方程各项：
    dTdt, adv_T, ver, Q

    Parameters
    ----------
    t_da, u_da, v_da, w_da : xr.DataArray
        维度必须为:
        (valid_time, pressure_level, latitude, longitude)

        t_da: 温度, 单位 K
        u_da: 纬向风, 单位 m/s
        v_da: 经向风, 单位 m/s
        w_da: 压力坐标垂直速度 omega, 单位 Pa/s

    Returns
    -------
    ds_out : xr.Dataset
        包含:
        dTdt  : 温度倾向, K/s
        adv_T : 水平温度平流, K/s
        ver   : 垂直运动项, K/s
        Q     : 非绝热加热项, K/s
        sigma : 静力稳定度项, K/Pa
    """

    # ----------------------------
    # 1. 复制并补单位
    # ----------------------------
    t_da = t_da.copy()
    u_da = u_da.copy()
    v_da = v_da.copy()
    w_da = w_da.copy()

    if 'units' not in t_da.attrs:
        t_da.attrs['units'] = 'K'
    if 'units' not in u_da.attrs:
        u_da.attrs['units'] = 'm/s'
    if 'units' not in v_da.attrs:
        v_da.attrs['units'] = 'm/s'
    if 'units' not in w_da.attrs:
        w_da.attrs['units'] = 'Pa/s'

    # 量纲化
    t_q = t_da.metpy.quantify()
    u_q = u_da.metpy.quantify()
    v_q = v_da.metpy.quantify()
    w_q = w_da.metpy.quantify()

    times = t_da['valid_time'].values
    levs_hpa = t_da['pressure_level'].values
    lats = t_da['latitude'].values
    lons = t_da['longitude'].values

    nt = len(times)
    nz = len(levs_hpa)
    ny = len(lats)
    nx = len(lons)

    # pressure_level: hPa -> Pa
    p_pa = (levs_hpa * 100.0) * units.Pa

    # 水平网格距离
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

    # 结果数组
    dTdt_arr = np.full((nt, nz, ny, nx), np.nan)
    adv_arr  = np.full((nt, nz, ny, nx), np.nan)
    ver_arr  = np.full((nt, nz, ny, nx), np.nan)
    Q_arr    = np.full((nt, nz, ny, nx), np.nan)
    sigma_arr = np.full((nt, nz, ny, nx), np.nan)

    # ----------------------------
    # 2. 逐日循环
    # ----------------------------
    for i in range(nt):

        # ===== (1) 时间差分: dTdt =====
        if i == 0:
            dt_seconds = (times[i + 1] - times[i]) / np.timedelta64(1, 's')
            dTdt = (t_q.isel(valid_time=i + 1) - t_q.isel(valid_time=i)) / (dt_seconds * units.s)
        elif i == nt - 1:
            dt_seconds = (times[i] - times[i - 1]) / np.timedelta64(1, 's')
            dTdt = (t_q.isel(valid_time=i) - t_q.isel(valid_time=i - 1)) / (dt_seconds * units.s)
        else:
            dt_seconds = (times[i + 1] - times[i - 1]) / np.timedelta64(1, 's')
            dTdt = (t_q.isel(valid_time=i + 1) - t_q.isel(valid_time=i - 1)) / (dt_seconds * units.s)

        # ===== (2) 水平温度平流 adv_T =====
        adv_each_level = []
        for k in range(nz):
            adv_k = mpcalc.advection(
                t_q.isel(valid_time=i, pressure_level=k),
                u=u_q.isel(valid_time=i, pressure_level=k),
                v=v_q.isel(valid_time=i, pressure_level=k),
                dx=dx,
                dy=dy,
                x_dim=-1,
                y_dim=-2
            )
            adv_each_level.append(adv_k)

        adv_T = xr.concat(
            [
                xr.DataArray(
                    a.metpy.dequantify().values,
                    coords={'latitude': lats, 'longitude': lons},
                    dims=('latitude', 'longitude')
                )
                for a in adv_each_level
            ],
            dim='pressure_level'
        )
        adv_T = adv_T.assign_coords(pressure_level=levs_hpa)
        adv_T = adv_T * units('K/s')

        # ===== (3) 静力稳定度 sigma =====
        T_now = t_q.isel(valid_time=i)   # (pressure_level, lat, lon)

        # dT/dp，注意 pressure_level 需要用 Pa
        T_vals = T_now.metpy.dequantify().values   # K
        dTdp_vals = np.gradient(T_vals, p_pa.magnitude, axis=0)  # K/Pa

        # R*T/(cp*p)
        p_4d = p_pa.magnitude[:, None, None]  # (lev,1,1)
        term1 = (R.magnitude * T_vals) / (cp.magnitude * p_4d)   # K/Pa

        sigma_vals = term1 - dTdp_vals  # K/Pa

        sigma = xr.DataArray(
            sigma_vals,
            coords=T_now.coords,
            dims=T_now.dims
        ) * units('K/Pa')

        # ===== (4) 垂直运动项 ver =====
        ver = w_q.isel(valid_time=i) * sigma   # Pa/s * K/Pa = K/s

        # ===== (5) 非绝热加热 Q =====
        Q = dTdt - adv_T - ver

        # 存储
        dTdt_arr[i] = dTdt.metpy.dequantify().values
        adv_arr[i]  = adv_T.metpy.dequantify().values
        ver_arr[i]  = ver.metpy.dequantify().values
        Q_arr[i]    = Q.metpy.dequantify().values
        sigma_arr[i] = sigma.metpy.dequantify().values

    # ----------------------------
    # 3. 输出为 Dataset
    # ----------------------------
    coords = {
        'valid_time': times,
        'pressure_level': levs_hpa,
        'latitude': lats,
        'longitude': lons
    }

    ds_out = xr.Dataset(
        {
            'dTdt': xr.DataArray(dTdt_arr, coords=coords, dims=('valid_time', 'pressure_level', 'latitude', 'longitude')),
            'adv_T': xr.DataArray(adv_arr, coords=coords, dims=('valid_time', 'pressure_level', 'latitude', 'longitude')),
            'ver': xr.DataArray(ver_arr, coords=coords, dims=('valid_time', 'pressure_level', 'latitude', 'longitude')),
            'Q': xr.DataArray(Q_arr, coords=coords, dims=('valid_time', 'pressure_level', 'latitude', 'longitude')),
            'sigma': xr.DataArray(sigma_arr, coords=coords, dims=('valid_time', 'pressure_level', 'latitude', 'longitude')),
        }
    )

    ds_out['dTdt'].attrs['units'] = 'K/s'
    ds_out['adv_T'].attrs['units'] = 'K/s'
    ds_out['ver'].attrs['units'] = 'K/s'
    ds_out['Q'].attrs['units'] = 'K/s'
    ds_out['sigma'].attrs['units'] = 'K/Pa'

    return ds_out

uvz6_200 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_200_201506_unzip.nc")
uvz6_500 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_500_201506_unzip.nc")
uvz6_850 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_850_201506_unzip.nc")
uvz6_925 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_925_201506_unzip.nc")
uvz6 = xr.concat([uvz6_200, uvz6_500, uvz6_850, uvz6_925], dim='pressure_level')
uvz7_200 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_200_201507_unzip.nc")
uvz7_500 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_500_201507_unzip.nc")
uvz7_850 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_850_201507_unzip.nc")
uvz7_925 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_925_201507_unzip.nc")
uvz7 = xr.concat([uvz7_200, uvz7_500, uvz7_850, uvz7_925], dim='pressure_level')
uvz8_200 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_200_201508_unzip.nc")
uvz8_500 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_500_201508_unzip.nc")
uvz8_850 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_850_201508_unzip.nc")
uvz8_925 = xr.open_dataset(fr"{DATA}/ERA5/daily/uvwztSh/ERA5_daily_uvwztSh_925_201508_unzip.nc")
uvz8 = xr.concat([uvz8_200, uvz8_500, uvz8_850, uvz8_925], dim='pressure_level')
uvz = xr.concat([uvz6, uvz7, uvz8], dim='valid_time')
uvz = uvz.transpose('valid_time', 'pressure_level', 'latitude', 'longitude')  # 500hPa

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
uvz_ano = uvz - uvz.mean('valid_time')
u_bp = LanczosFilter(uvz_ano['u'], 'bandpass', period=[10, 30], nwts=9).filted()
v_bp = LanczosFilter(uvz_ano['v'], 'bandpass', period=[10, 30], nwts=9).filted()
w_bp = LanczosFilter(uvz_ano['w'], 'bandpass', period=[10, 30], nwts=9).filted()
z_bp = LanczosFilter(uvz_ano['z'], 'bandpass', period=[10, 30], nwts=9).filted()
t_bp = LanczosFilter(uvz_ano['t'], 'bandpass', period=[10, 30], nwts=9).filted()

yangtze_shp = fr'{PYFILE}/map/self/长江_TP/长江_tp.shp'
budget_ds = calc_daily_temperature_budget(t_bp, u_bp, v_bp, w_bp)

dTdt_yz_500 = region_mean_series(budget_ds['dTdt'].sel(pressure_level=925), yangtze_shp)*86400
adv_yz_500 = region_mean_series(budget_ds['adv_T'].sel(pressure_level=925), yangtze_shp)*86400
ver_yz_500 = region_mean_series(budget_ds['ver'].sel(pressure_level=925), yangtze_shp)*86400
Q_yz_500 = region_mean_series(budget_ds['Q'].sel(pressure_level=925), yangtze_shp)*86400
#%%
fig = plt.figure(figsize=(12, 10))
plt.subplots_adjust(wspace=0, hspace=0)
title_head = '2015'

# 空间图
# lev = np.array([0., .08, .16, .24, .32, .4, .48, .56])
lev = np.array([-300, -250, -200, -150, -100, -50, 50, 50, 150, 200, 250, 300])
lev = np.array([-500, -400, -300, -200, -100, -50, 50, 100, 200, 300, 400, 500])
index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
for i in range(29):
    iday=i*3+3
    if i==0:
        ax, cont = pic(
            fig, (6, 5, i+1),
            uvz['latitude'], uvz['longitude'], u_bp[iday].sel(pressure_level=500), v_bp[iday].sel(pressure_level=500), lev,
            z_bp[iday].sel(pressure_level=500), f'2015 500UVZ',
            lat_tick=None, lon_tick=None, key=False
        )
    elif i==4:
        ax, cont = pic(
            fig, (6, 5, i + 1),
            uvz['latitude'], uvz['longitude'], u_bp[iday].sel(pressure_level=500), v_bp[iday].sel(pressure_level=500),
            lev,
            z_bp[iday].sel(pressure_level=500), f'',
            lat_tick=None, lon_tick=None, key=True
        )
    else:
        ax, cont = pic(
            fig, (6, 5, i+1),
            uvz['latitude'], uvz['longitude'], u_bp[iday].sel(pressure_level=500), v_bp[iday].sel(pressure_level=500), lev,
            z_bp[iday].sel(pressure_level=500), f'', key=False,
            lat_tick=None, lon_tick=None,
        )
    # ===== 当前子图日期 =====
    from matplotlib.patheffects import withStroke
    panel_time = pd.to_datetime(uvz['valid_time'].isel(valid_time=iday).values)
    date_text = panel_time.strftime('%m/%d')
    ax.text(0.05, 0.95, date_text, transform=ax.transAxes, fontsize=10, color='red',
            ha='left', va='top', bbox=dict(facecolor='none', edgecolor='none'),
            path_effects=[withStroke(linewidth=1.5, foreground='white')])

    # ===== 500 hPa 温度收支：MetPy + masked =====
    # 标在长江流域左侧
    def add_budget_inset(ax, dTdt_val, adv_val, ver_val, Q_val):
        """
        在主图左下角添加温度收支柱状图
        尺寸：主图的 15%
        范围：-2 ~ 2
        正值红色，负值蓝色，alpha=0.7，黑色描边
        左侧显示刻度轴，刻度朝内
        """
        ax_in = inset_axes(
            ax,
            width="20%",
            height="50%",
            loc='lower left',
            bbox_to_anchor=(0.00, 0.00, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        labels = ['dTdt', 'Adv', 'Ver', 'Q']
        vals = [dTdt_val, adv_val, ver_val, Q_val]
        colors = ['red' if v > 0 else 'blue' for v in vals]

        x = np.arange(len(labels))
        ax_in.bar(
            x, vals,
            width=0.97,
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.3,
            zorder=3
        )

        # y 轴范围和刻度
        ax_in.set_ylim(-1, 1)
        ax_in.set_yticks([-1, 0, 1])
        ax_in.set_yticks([0.5, 0.5], minor=True)
        ax_in.set_yticklabels(['', '', ''], fontsize=0)

        # x 轴标签
        ax_in.set_xticks([])

        # 左侧显示刻度轴，刻度朝内
        ax_in.tick_params(axis='y', which='major', direction='in', length=5, width=1.2, labelsize=6, pad=1)
        ax_in.tick_params(axis='y', which='minor', direction='in', length=3.5, width=0.8)
        ax_in.tick_params(axis='x', which='major', direction='in', length=0, width=0, labelsize=6, pad=1)

        # 只保留左和下边框
        ax_in.spines['top'].set_visible(False)
        ax_in.spines['right'].set_visible(False)
        ax_in.spines['left'].set_linewidth(0.8)
        ax_in.spines['bottom'].set_linewidth(0.8)

        # 0 线
        ax_in.axhline(0, color='black', linewidth=0.2, zorder=2)

        ax_in.set_facecolor('white')
        ax_in.patch.set_alpha(0.85)
        ax_in.set_facecolor('none')

        return ax_in

    # ===== 500 hPa 温度收支柱状图 =====
    add_budget_inset(
        ax,
        float(dTdt_yz_500[iday]),
        float(adv_yz_500[iday]),
        float(ver_yz_500[iday]),
        float(Q_yz_500[iday])
    )

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
cbar.set_ticklabels([f"{i:.0f}" for i in lev])

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.set_aspect('auto')

for ax in fig.axes:
    # 遍历每个子图中的所有艺术家对象 (artist)
    for artist in ax.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)

plt.savefig(fr"{PYFILE}/p4/pic/90天滤波环流场_{title_head}.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p4/pic/90天滤波环流场_{title_head}.png", bbox_inches='tight', dpi=600)
plt.show()