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
from matplotlib import ticker, gridspec
from matplotlib.ticker import MultipleLocator
from scipy import ndimage
from scipy.stats import ttest_ind

from climkit.Cquiver import *
from climkit.TN_WaveActivityFlux import TN_WAF_3D
from climkit.masked import masked
from climkit.significance_test import r_test
from climkit.lonlat_transform import *
from climkit.filter import *
from climkit.corr_reg import *

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

def pic(fig, pic_loc, lat, lon, lev, lev_t, corr_u, corr_v, corr_z, corr_t2m, title, nanmax=5):

    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')

    idx = int(str(pic_loc)[2]) - 3
    ax.set_title(title, loc='left', fontsize=10)

    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

    da_contour = xr.DataArray(
        corr_t2m,
        coords={'lat': lat.data, 'lon': lon.data},
        dims=('lat', 'lon')
    )
    roi_shape = ((60, 0), (160, 60))
    contf = ax.contourf(lon, lat, da_contour.salem.roi(corners=roi_shape), cmap=cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
                        levels=lev_t, extend='both', transform=ccrs.PlateCarree(central_longitude=0))

    cont = ax.contour(lon, lat, corr_z, colors='red', levels=lev[1], linewidths=0.8, transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(lon, lat, corr_z, colors='blue', levels=lev[0], linestyles='--', linewidths=0.8,
                       transform=ccrs.PlateCarree(central_longitude=0))

    #cont_clim = ax.contour(lon, lat, uvz_clim['z'], colors='k', levels=20, linewidths=0.6, transform=ccrs.PlateCarree(central_longitude=0))

    Cq = ax.Curlyquiver(lon, lat, corr_u, corr_v, center_lon=110, scale=10, linewidth=1, arrowsize=1., transform=ccrs.PlateCarree(central_longitude=0), MinDistance=[0.2, 0.5],
                     regrid=12, color='#454545', nanmax=nanmax, thinning=["10%", "min"])

    Cq.key(U=2, label='2 m/s', color='k', fontproperties={'size': 8}, linewidth=.7, arrowsize=3., facecolor='#FFFFFF')
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
    ax.tick_params(axis='both', labelsize=9, colors='black')

    return contf, ax

def ensure_celsius(da: xr.DataArray) -> xr.DataArray:
    """如果像 Kelvin，就转成 Celsius"""
    vmin = float(da.min().values)
    vmax = float(da.max().values)
    if vmin > 150 and vmax < 400:
        print("检测到温度可能为 Kelvin，自动转换为 Celsius。")
        da = da - 273.15
        da.attrs["units"] = "degC"
    return da

def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """统一经纬度坐标名为 lon/lat"""
    rename_dict = {}
    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if rename_dict:
        ds = ds.rename(rename_dict)
    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("数据中未找到 lon/lat 或 longitude/latitude 坐标。")
    return ds

def detect_main_var(ds: xr.Dataset) -> str:
    """自动识别主变量名"""
    preferred = ["t2m", "2m_temperature", "u"]
    for v in preferred:
        if v in ds.data_vars:
            return v
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"无法自动识别温度变量，请检查变量名：{list(ds.data_vars)}")

def daily_clim_by_mmdd(da: xr.DataArray, time_dim: str = "time", drop_feb29: bool = True) -> xr.DataArray:
    """按 MM-DD 计算逐日气候态（非 dayofyear，避免闰年错位）。"""
    if time_dim not in da.dims:
        raise ValueError(f"{da.name or 'DataArray'} 缺少时间维 {time_dim}。")

    out = da
    if drop_feb29:
        leap_mask = (out[time_dim].dt.month == 2) & (out[time_dim].dt.day == 29)
        out = out.where(~leap_mask, drop=True)

    mmdd = out[time_dim].dt.strftime("%m-%d")
    out = out.assign_coords(mmdd=(time_dim, mmdd.data))
    return out.groupby("mmdd").mean(time_dim)

def anomaly_by_mmdd(da, clim, start, end):
    sub = da.sel(time=slice(start, end))
    mmdd_indexer = xr.DataArray(
        sub.time.dt.strftime("%m-%d").values,
        coords={"time": sub.time},
        dims="time"
    )
    clim_on_time = clim.sel(mmdd=mmdd_indexer)
    return sub - clim_on_time

def region_mean_series(da, shp_path):
    vals = []
    for i in range(da.sizes['time']):
        da_clip = masked(da.isel(time=i), shp_path)
        vals.append(da_clip.mean(dim=('lat', 'lon'), skipna=True))
    return xr.concat(vals, dim='time').assign_coords(time=da['time'])

def calc_daily_temperature_budget(t_da, u_da, v_da, w_da):
    """
    计算逐日温度方程各项：
    dTdt, adv_T, ver, Q

    Parameters
    ----------
    t_da, u_da, v_da, w_da : xr.DataArray
        维度必须为:
        (time, level, lat, lon)

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
    from metpy.constants import dry_air_gas_constant as R
    from metpy.constants import dry_air_spec_heat_press as cp
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

    times = t_da['time'].values
    levs_hpa = t_da['level'].values
    lats = t_da['lat'].values
    lons = t_da['lon'].values

    nt = len(times)
    nz = len(levs_hpa)
    ny = len(lats)
    nx = len(lons)

    # level: hPa -> Pa
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
            dTdt = (t_q.isel(time=i + 1) - t_q.isel(time=i)) / (dt_seconds * units.s)
        elif i == nt - 1:
            dt_seconds = (times[i] - times[i - 1]) / np.timedelta64(1, 's')
            dTdt = (t_q.isel(time=i) - t_q.isel(time=i - 1)) / (dt_seconds * units.s)
        else:
            dt_seconds = (times[i + 1] - times[i - 1]) / np.timedelta64(1, 's')
            dTdt = (t_q.isel(time=i + 1) - t_q.isel(time=i - 1)) / (dt_seconds * units.s)

        # ===== (2) 水平温度平流 adv_T =====
        adv_each_level = []
        for k in range(nz):
            adv_k = mpcalc.advection(
                t_q.isel(time=i, level=k),
                u=u_q.isel(time=i, level=k),
                v=v_q.isel(time=i, level=k),
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
                    coords={'lat': lats, 'lon': lons},
                    dims=('lat', 'lon')
                )
                for a in adv_each_level
            ],
            dim='level'
        )
        adv_T = adv_T.assign_coords(level=levs_hpa)
        adv_T = adv_T * units('K/s')

        # ===== (3) 静力稳定度 sigma =====
        T_now = t_q.isel(time=i)   # (level, lat, lon)

        # dT/dp，注意 level 需要用 Pa
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
        ver = w_q.isel(time=i) * sigma   # Pa/s * K/Pa = K/s

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
        'time': times,
        'level': levs_hpa,
        'lat': lats,
        'lon': lons
    }

    ds_out = xr.Dataset(
        {
            'dTdt': xr.DataArray(dTdt_arr, coords=coords, dims=('time', 'level', 'lat', 'lon')),
            'adv_T': xr.DataArray(adv_arr, coords=coords, dims=('time', 'level', 'lat', 'lon')),
            'ver': xr.DataArray(ver_arr, coords=coords, dims=('time', 'level', 'lat', 'lon')),
            'Q': xr.DataArray(Q_arr, coords=coords, dims=('time', 'level', 'lat', 'lon')),
            'sigma': xr.DataArray(sigma_arr, coords=coords, dims=('time', 'level', 'lat', 'lon')),
        }
    )

    ds_out['dTdt'].attrs['units'] = 'K/s'
    ds_out['adv_T'].attrs['units'] = 'K/s'
    ds_out['ver'].attrs['units'] = 'K/s'
    ds_out['Q'].attrs['units'] = 'K/s'
    ds_out['sigma'].attrs['units'] = 'K/Pa'

    return ds_out

def clim_to_time(clim, time_coord):
    """
    把按 mmdd 的逐日气候态映射回具体 time 维
    """
    mmdd_indexer = xr.DataArray(
        pd.to_datetime(time_coord.values).strftime("%m-%d"),
        coords={"time": time_coord.values},
        dims="time"
    )

    out = clim.sel(mmdd=mmdd_indexer)

    # 用 time 覆盖 mmdd 选择后的时间坐标
    out = out.assign_coords(time=time_coord.values)

    # 若还残留 mmdd 坐标，就删掉
    if 'mmdd' in out.coords:
        out = out.drop_vars('mmdd')

    return out

def calc_horizontal_advection_2d(T2d, u2d, v2d):
    """
    计算单层二维水平温度平流
    返回值单位：K/s
    输入:
        T2d, u2d, v2d: dims = (lat, lon)
    """
    if 'units' not in T2d.attrs:
        T2d.attrs['units'] = 'K'
    if 'units' not in u2d.attrs:
        u2d.attrs['units'] = 'm/s'
    if 'units' not in v2d.attrs:
        v2d.attrs['units'] = 'm/s'

    Tq = T2d.metpy.quantify()
    uq = u2d.metpy.quantify()
    vq = v2d.metpy.quantify()

    lons = T2d['lon'].values
    lats = T2d['lat'].values
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

    adv = mpcalc.advection(
        Tq,
        u=uq,
        v=vq,
        dx=dx,
        dy=dy,
        x_dim=-1,
        y_dim=-2
    )

    out = xr.DataArray(
        adv.metpy.dequantify().values,
        coords=T2d.coords,
        dims=T2d.dims,
        name='adv'
    )
    out.attrs['units'] = 'K/s'
    return out

def decompose_temperature_advection(u, v, t, u_clim, v_clim, t_clim, start, end, level=None):
    """
    水平温度平流分解:
    total = -(u·∇T)

    分解为:
    adv_clim = -(uc · ∇Tc)
    adv_a_wind_on_climT = -(u' · ∇Tc)
    adv_clim_wind_on_aT = -(uc · ∇T')
    adv_nonlinear = -(u' · ∇T')

    返回:
        xr.Dataset
    """

    # 1. 选时间段
    u_sub = u.sel(time=slice(start, end))
    v_sub = v.sel(time=slice(start, end))
    t_sub = t.sel(time=slice(start, end))

    # 2. 若指定层次：先选层，再做 clim_to_time
    if level is not None:
        u_sub = u_sub.sel(level=level)
        v_sub = v_sub.sel(level=level)
        t_sub = t_sub.sel(level=level)

        uc_base = u_clim.sel(level=level)
        vc_base = v_clim.sel(level=level)
        Tc_base = t_clim.sel(level=level)
    else:
        uc_base = u_clim
        vc_base = v_clim
        Tc_base = t_clim

    # 3. 逐日气候态映射到 time
    uc = clim_to_time(uc_base, u_sub.time)
    vc = clim_to_time(vc_base, v_sub.time)
    Tc = clim_to_time(Tc_base, t_sub.time)

    # 4. 异常量
    ua = u_sub - uc
    va = v_sub - vc
    Ta = t_sub - Tc

    total_list = []
    clim_list = []
    awind_climT_list = []
    climwind_aT_list = []
    nl_list = []

    for i in range(u_sub.sizes['time']):
        total_i = calc_horizontal_advection_2d(
            t_sub.isel(time=i), u_sub.isel(time=i), v_sub.isel(time=i)
        )
        clim_i = calc_horizontal_advection_2d(
            Tc.isel(time=i), uc.isel(time=i), vc.isel(time=i)
        )
        awind_climT_i = calc_horizontal_advection_2d(
            Tc.isel(time=i), ua.isel(time=i), va.isel(time=i)
        )
        climwind_aT_i = calc_horizontal_advection_2d(
            Ta.isel(time=i), uc.isel(time=i), vc.isel(time=i)
        )
        nl_i = calc_horizontal_advection_2d(
            Ta.isel(time=i), ua.isel(time=i), va.isel(time=i)
        )

        total_list.append(total_i)
        clim_list.append(clim_i)
        awind_climT_list.append(awind_climT_i)
        climwind_aT_list.append(climwind_aT_i)
        nl_list.append(nl_i)

    adv_total = xr.concat(total_list, dim='time').assign_coords(time=u_sub.time)
    adv_clim = xr.concat(clim_list, dim='time').assign_coords(time=u_sub.time)
    adv_a_wind_on_climT = xr.concat(awind_climT_list, dim='time').assign_coords(time=u_sub.time)
    adv_clim_wind_on_aT = xr.concat(climwind_aT_list, dim='time').assign_coords(time=u_sub.time)
    adv_nonlinear = xr.concat(nl_list, dim='time').assign_coords(time=u_sub.time)

    adv_anom = adv_total - adv_clim

    ds_out = xr.Dataset({
        'adv_total': adv_total,
        'adv_clim': adv_clim,
        'adv_anom': adv_anom,
        'adv_a_wind_on_climT': adv_a_wind_on_climT,
        'adv_clim_wind_on_aT': adv_clim_wind_on_aT,
        'adv_nonlinear': adv_nonlinear
    })

    for vname in ds_out.data_vars:
        ds_out[vname].attrs['units'] = 'K/s'

    return ds_out


ds = xr.open_dataset(r"/Volumes/TiPlus7100/p4/data/ERA5_daily_uvwztq_sum.zarr")
ds = standardize_latlon(ds)

# 如果时间坐标叫 valid_time，就统一改成 time
if "time" not in ds.coords:
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    else:
        raise ValueError("数据中没有 time 或 valid_time 坐标。")

# 需要读取的变量
required_vars = ["u", "v", "z", "t", "w"]
missing_vars = [v for v in required_vars if v not in ds.data_vars]
if missing_vars:
    raise ValueError(f"数据中缺少变量: {missing_vars}；当前变量有: {list(ds.data_vars)}")

# 只保留 1961-2022 夏季
ds = ds[required_vars].sel(time=slice("1961-06-01", "2022-08-31"))

# 分别取出变量
u = ds["u"]
v = ds["v"]
z = ds["z"]
t = ds["t"]
w = ds["w"]


# 各要素按“月-日”逐日气候态（不是 dayofyear）
CLIM = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_clim_sum.nc")
u_clim = CLIM["u_clim"]
v_clim = CLIM["v_clim"]
z_clim = CLIM["z_clim"]
t_clim = CLIM["t_clim"]
w_clim = CLIM["w_clim"]
olr_clim = CLIM["olr_clim"]
t2m_clim = CLIM["t2m_clim"]

# YEAR = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014]
YEAR = [2015]

# =========================
# 为了适配 nwts=61 的 Lanczos 滤波，
# 先取 5–9 月做异常和滤波，
# 再裁剪回 6–8 月，保证 6–8 月结果完整
# =========================
filter_start = "2015-05-01"
filter_end   = "2015-09-30"

analysis_start = "2015-06-15"
analysis_end   = "2015-07-31"

# 1) 先在 5–9 月上计算逐日异常
ANO = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_ano_2015_MJJAS.nc")
u_ano_full = ANO['u_ano']
v_ano_full = ANO['v_ano']
z_ano_full = ANO['z_ano']
t_ano_full = ANO['t_ano']
w_ano_full = ANO['w_ano']
olr_ano_full = ANO['olr_ano']
t2m_ano_full = ANO['t2m_ano']

# 2) 在 5–9 月异常场上做带通滤波
BP = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_bp_2015_JJA_10-30d.nc")
u_bp_full = BP['u_bp']
v_bp_full = BP['v_bp']
z_bp_full = BP['z_bp']
t_bp_full = BP['t_bp']
w_bp_full = BP['w_bp']
olr_bp_full = BP['olr_bp']
t2m_bp_full = BP['t2m_bp']

# 3) 再裁回 6–8 月，供后续分析使用
u_ano = u_ano_full.sel(time=slice(analysis_start, analysis_end))
v_ano = v_ano_full.sel(time=slice(analysis_start, analysis_end))
z_ano = z_ano_full.sel(time=slice(analysis_start, analysis_end))
t_ano = t_ano_full.sel(time=slice(analysis_start, analysis_end))
w_ano = w_ano_full.sel(time=slice(analysis_start, analysis_end))
olr_ano = olr_ano_full.sel(time=slice(analysis_start, analysis_end))
t2m_ano = t2m_ano_full.sel(time=slice(analysis_start, analysis_end))

u_bp = u_bp_full.sel(time=slice(analysis_start, analysis_end))
v_bp = v_bp_full.sel(time=slice(analysis_start, analysis_end))
z_bp = z_bp_full.sel(time=slice(analysis_start, analysis_end))
t_bp = t_bp_full.sel(time=slice(analysis_start, analysis_end))
w_bp = w_bp_full.sel(time=slice(analysis_start, analysis_end))
olr_bp = olr_bp_full.sel(time=slice(analysis_start, analysis_end))
t2m_bp = t2m_bp_full.sel(time=slice(analysis_start, analysis_end))

#%%

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.util import add_cyclic_point

# =========================================================
# 0. 区域指数与回归场
# =========================================================
yangtze_shp = fr'{PYFILE}/map/self/长江_TP/长江_tp.shp'

# 回归指数：长江-青藏高原区域 10–30 天 T2m 负异常
t2m_index = region_mean_series(-t2m_bp, yangtze_shp)

u_reg = regress(t2m_index, u_bp)
v_reg = regress(t2m_index, v_bp)
w_reg = regress(t2m_index, w_bp)
z_reg = regress(t2m_index, z_bp)
t_reg = regress(t2m_index, t_bp)
t2m_reg = regress(t2m_index, t2m_bp)
olr_reg = regress(t2m_index, olr_bp)


# =========================================================
# 1. 工具函数：经度转换、DataArray 处理
# =========================================================
def to_lon180(da):
    """
    把 lon 从 0–360 转为 -180–180，并按 lon 排序。
    优先使用 climkit 里的 transform；若失败则使用 xarray 原生写法。
    """
    if "lon" not in da.coords:
        return da

    out = da

    if float(out.lon.max()) > 180:
        try:
            out = transform(out, "lon", "360->180")
        except Exception:
            lon_new = ((out.lon + 180) % 360) - 180
            out = out.assign_coords(lon=lon_new)

    out = out.sortby("lon")

    if "lat" in out.coords:
        out = out.sortby("lat")

    return out


def as_dataarray(arr, ref, name=None):
    """
    如果 WAF 返回 ndarray，则转成 xr.DataArray。
    如果已经是 DataArray，则直接返回。
    """
    if isinstance(arr, xr.DataArray):
        out = arr
    else:
        out = xr.DataArray(
            arr,
            coords={"lat": ref.lat, "lon": ref.lon},
            dims=("lat", "lon"),
            name=name
        )

    return to_lon180(out)


def cyclic_dataarray(da):
    """
    给全球场加 cyclic point，避免 -180/180 接缝白线。
    返回 data_cyc, lon_cyc, lat。
    """
    da = to_lon180(da)

    data_cyc, lon_cyc = add_cyclic_point(
        da.values,
        coord=da.lon.values,
        axis=da.get_axis_num("lon")
    )

    return data_cyc, lon_cyc, da.lat.values


# =========================================================
# 2. 计算 JJA 200 hPa WAF
# =========================================================
CLIM_JJA = xr.open_dataset(
    "/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_clim_sum.nc"
).sel(mmdd=slice("06-01", "08-31"))

u_clim_jja = CLIM_JJA["u_clim"].mean(dim="mmdd")
v_clim_jja = CLIM_JJA["v_clim"].mean(dim="mmdd")

z200 = to_lon180(z_reg.sel(level=200))
u200 = to_lon180(u_clim_jja.sel(level=200))
v200 = to_lon180(v_clim_jja.sel(level=200))

# 确保背景风和扰动场经纬度一致
u200 = u200.interp(lon=z200.lon, lat=z200.lat)
v200 = v200.interp(lon=z200.lon, lat=z200.lat)

WAF = TN_WAF_3D(
    u200,
    v200,
    z200,
    single_level=200
)

waf_x, waf_y = WAF
waf_x = as_dataarray(waf_x, z200, "waf_x")
waf_y = as_dataarray(waf_y, z200, "waf_y")

# 再次确保 WAF 与 u200/v200 坐标一致
waf_x = waf_x.interp(lon=u200.lon, lat=u200.lat)
waf_y = waf_y.interp(lon=u200.lon, lat=u200.lat)


# =========================================================
# 3. 准备 500 hPa / 850 hPa 回归场
# =========================================================
u500 = to_lon180(u_reg.sel(level=500))
v500 = to_lon180(v_reg.sel(level=500))
z500 = to_lon180(z_reg.sel(level=500))
olr_map = to_lon180(olr_reg)

u850 = to_lon180(u_reg.sel(level=850))
v850 = to_lon180(v_reg.sel(level=850))
z850 = to_lon180(z_reg.sel(level=850))
t2m_map = to_lon180(t2m_reg)


# =========================================================
# 4. 绘图函数
# =========================================================
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors


def _color_list(obj):
    """
    兼容 cmaps 的切片颜色、单个 RGBA 颜色、ListedColormap、颜色字符串等。

    关键修正：
    cmaps.CBR_wet[0] 这种单个 RGBA 数组不能直接 list()，
    否则会被拆成 [1.0, 1.0, 1.0, 1.0] 四个浮点数，
    Matplotlib 会报 Invalid RGBA argument: np.float64(1.0)。
    """

    # 如果是 ListedColormap 之类
    if hasattr(obj, "colors"):
        obj = obj.colors

    # 尝试按数值数组处理
    try:
        arr = np.asarray(obj, dtype=float)

        # 单个 RGB / RGBA，例如 cmaps.CBR_wet[0]
        if arr.ndim == 1 and arr.size in (3, 4):
            return [tuple(arr.tolist())]

        # 多个 RGB / RGBA，例如 cmaps.MPL_RdBu_r[35:64]
        if arr.ndim == 2 and arr.shape[1] in (3, 4):
            return [tuple(row) for row in arr.tolist()]

    except Exception:
        pass

    # 单个颜色字符串或 Matplotlib 可识别颜色
    if mcolors.is_color_like(obj):
        return [obj]

    # list / tuple 递归处理
    if isinstance(obj, (list, tuple, np.ndarray)):
        colors_out = []
        for item in obj:
            colors_out.extend(_color_list(item))
        return colors_out

    # 兜底
    return [mcolors.to_rgba(obj)]


def make_listed_cmap(colors, name):
    """
    把颜色列表强制转为 RGBA，再生成 ListedColormap。
    """
    rgba_colors = [tuple(mcolors.to_rgba(c)) for c in colors]
    return ListedColormap(rgba_colors, name=name)


# 图 a：200Z 填色 cmap
cmap_200z = make_listed_cmap(
    _color_list(cmaps.MPL_RdBu_r[35:64])
    + _color_list(cmaps.CBR_wet[0]) * 4
    + _color_list(cmaps.MPL_RdBu_r[64:-35]),
    "custom_200z_RdBu"
)

# 图 b：OLR 填色 cmap
cmap_olr = make_listed_cmap(
    _color_list(cmaps.MPL_BrBG_r[:64])
    + _color_list(cmaps.CBR_wet[0]) * 4
    + _color_list(cmaps.MPL_BrBG_r[64:]),
    "custom_olr_BrBG"
)

# 图 c：T2m 填色 cmap
cmap_t2m = make_listed_cmap(
    _color_list(cmaps.GMT_polar[3:10])
    + _color_list(cmaps.CBR_wet[0])
    + _color_list(cmaps.GMT_polar[10:-3]),
    "custom_t2m_GMT_polar"
)


def remove_contourf_white_lines(cf):
    """
    去除 contourf 填色块之间可能出现的白线，
    尤其是全球场在 0° / 180° 附近的接缝白线。
    """
    if hasattr(cf, "collections"):
        for c in cf.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.0)
            c.set_antialiased(False)
    return cf


def setup_global_ax(fig, loc, title):
    ax = fig.add_subplot(
        loc,
        projection=ccrs.PlateCarree(central_longitude=110)
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_extent([-180, 180, -30, 80], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")
    ax.set_title(title, loc="left", fontsize=10)

    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=1., color='#AAAAAA')
    ax.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='gray', edgecolor='black', linewidth=.5)

    xticks = np.arange(-180, 181, 60)
    yticks = np.arange(-30, 81, 30)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.xaxis.set_minor_locator(MultipleLocator(30))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    ax.tick_params(which="major", length=4, width=0.5, color="black")
    ax.tick_params(which="minor", length=2, width=0.2, color="black")
    ax.tick_params(which="both", bottom=True, top=False, left=True, right=False)
    ax.tick_params(axis="both", labelsize=8, colors="black")

    return ax


def add_right_colorbar(ax, mappable, ticks=None, labelsize=8):
    cax = inset_axes(
        ax,
        width="2.5%",
        height="85%",
        loc="center right",
        bbox_to_anchor=(0.055, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    cb = plt.colorbar(
        mappable,
        cax=cax,
        orientation="vertical",
        drawedges=True,
        ticks=ticks
    )

    cb.ax.tick_params(length=0, labelsize=labelsize, direction="in")
    cb.dividers.set_linewidth(1.0)
    cb.outline.set_linewidth(1.0)

    return cb


def plot_200z_waf(fig, loc, z, u200, waf_x, waf_y, title):
    ax1 = setup_global_ax(fig, loc, title)
    data_crs = ccrs.PlateCarree()

    # -------------------------
    # 200 hPa Z 填色
    # 图 a：删除等值线，填色透明度 0.7
    # -------------------------
    z_levels = np.arange(-240, 241, 40)
    z_data, z_lon, z_lat = cyclic_dataarray(z)

    cf = ax1.contourf(
        z_lon,
        z_lat,
        z_data,
        levels=z_levels,
        cmap=cmap_200z,
        extend="both",
        alpha=0.95,
        transform=data_crs,
        antialiased=False
    )
    cf = remove_contourf_white_lines(cf)

    # -------------------------
    # WAF 矢量
    # -------------------------
    q = ax1.Curlyquiver(
        u200.lon,
        u200.lat,
        waf_x,
        waf_y,
        arrowsize=1.3,
        transform=data_crs,
        scale=10,
        linewidth=1.2,
        regrid=13,
        color="purple",
        thinning=["50%", "min"],
        nanmax=2,
        MinDistance=[0.2, 0.4]
    )

    q.key(
        U=2,
        label="2",
        color="purple",
        fontproperties={"size": 8},
        linewidth=0.8,
        arrowsize=1.8,
        facecolor="#FFFFFF"
    )

    add_right_colorbar(
        ax1,
        cf,
        ticks=np.arange(-240, 241, 120),
        labelsize=8
    )

    return ax1, cf


def plot_uvz_scalar(
    fig,
    loc,
    u,
    v,
    z,
    scalar,
    title,
    scalar_levels,
    scalar_ticks,
    scalar_cmap,
    scalar_alpha=0.75,
    remove_white_line=True
):
    ax1 = setup_global_ax(fig, loc, title)
    data_crs = ccrs.PlateCarree()

    # -------------------------
    # scalar 填色：OLR 或 T2m
    # -------------------------
    scalar_data, scalar_lon, scalar_lat = cyclic_dataarray(scalar)

    cf = ax1.contourf(
        scalar_lon,
        scalar_lat,
        scalar_data,
        levels=scalar_levels,
        cmap=scalar_cmap,
        extend="both",
        alpha=scalar_alpha,
        transform=data_crs,
        antialiased=False
    )

    if remove_white_line:
        cf = remove_contourf_white_lines(cf)

    # -------------------------
    # Z 等值线
    # -------------------------
    z_data, z_lon, z_lat = cyclic_dataarray(z)

    z_cont_pos = [60, 120, 240]
    z_cont_neg = [-240, -120, -60]

    cont_p = ax1.contour(
        z_lon,
        z_lat,
        z_data,
        levels=z_cont_pos,
        colors="red",
        linewidths=0.8,
        transform=data_crs
    )

    cont_n = ax1.contour(
        z_lon,
        z_lat,
        z_data,
        levels=z_cont_neg,
        colors="blue",
        linestyles="--",
        linewidths=0.8,
        transform=data_crs
    )


    # -------------------------
    # UV 矢量
    # -------------------------
    q = ax1.Curlyquiver(
        u.lon,
        u.lat,
        u,
        v,
        arrowsize=1,
        transform=data_crs,
        scale=5,
        linewidth=0.8,
        regrid=16,
        color="#454545",
        thinning=["15%", "min"],
        nanmax=2,
        MinDistance=[0.2, 0.4]
    )

    q.key(
        U=2,
        label="2 m/s",
        color="k",
        fontproperties={"size": 8},
        linewidth=0.8,
        arrowsize=1.8,
        facecolor="#FFFFFF"
    )

    add_right_colorbar(
        ax1,
        cf,
        ticks=scalar_ticks,
        labelsize=8
    )

    return ax1, cf


# =========================================================
# 5. 作图：三张图
# =========================================================
fig = plt.figure(figsize=(7.2, 8.8))
plt.subplots_adjust(wspace=0.2, hspace=0.28)

title_head = "JJA"

# -------------------------
# 第一张图：200Z & WAF
# 图 a：
# 1. 删除 Z 等值线
# 2. 填色 alpha = 0.7
# 3. cmap 改为指定 RdBu 拼接色表
# -------------------------
ax1, cf1 = plot_200z_waf(
    fig,
    311,
    z200,
    u200,
    waf_x,
    waf_y,
    f"(a) {title_head} 200Z & WAF"
)

# -------------------------
# 第二张图：500UVZ & OLR
# 图 b：
# 1. 去除填色 0° 接缝白线
# 2. 填色 alpha = 0.75
# 3. cmap 改为指定 BrBG 拼接色表
# -------------------------
olr_levels = np.array([-12, -9, -6, -3, -1, 1, 3, 6, 9, 12])
olr_ticks = [-12, -6, -1, 1, 6, 12]

ax2, cf2 = plot_uvz_scalar(
    fig,
    312,
    u500,
    v500,
    z500,
    olr_map,
    f"(b) {title_head} 500UVZ & OLR",
    olr_levels,
    olr_ticks,
    scalar_cmap=cmap_olr,
    scalar_alpha=0.75,
    remove_white_line=True
)

# -------------------------
# 第三张图：850UVZ & T2m
# 图 c：
# 1. 填色 cmap 改为指定 GMT_polar 拼接色表
# 2. 填色 alpha = 0.75
# -------------------------
t2m_levels = np.array([-1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5])
t2m_ticks = [-1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5]

ax3, cf3 = plot_uvz_scalar(
    fig,
    313,
    u850,
    v850,
    z850,
    t2m_map,
    f"(c) {title_head} 850UVZ & T2m",
    t2m_levels,
    t2m_ticks,
    scalar_cmap=cmap_t2m,
    scalar_alpha=0.75,
    remove_white_line=True
)

ax3.add_geometries(Reader(f'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp').geometries(),
        ccrs.PlateCarree(), facecolor='#909090', edgecolor='#909090', linewidth=0, hatch='.', zorder=10)

# =========================================================
# 6. 强制裁剪 & 保存
# =========================================================
for ax_ in fig.axes:
    for artist in ax_.get_children():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(True)

plt.savefig(
    fr"{PYFILE}/p4/pic/2015_JJA_200WAF_500OLR_850T2m_reg.pdf",
    bbox_inches="tight"
)

plt.savefig(
    fr"{PYFILE}/p4/pic/2015_JJA_200WAF_500OLR_850T2m_reg.png",
    bbox_inches="tight",
    dpi=600
)

plt.show()
