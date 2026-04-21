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
from climkit.masked import masked
from climkit.significance_test import r_test
from climkit.lonlat_transform import *
from climkit.filter import *

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
    ax.set_title(title, loc='left', fontsize=12)

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
    cont.clabel(inline=1, fontsize=4)
    cont_.clabel(inline=1, fontsize=4)
    #cont_clim = ax.contour(lon, lat, uvz_clim['z'], colors='k', levels=20, linewidths=0.6, transform=ccrs.PlateCarree(central_longitude=0))

    Cq = ax.Curlyquiver(lon, lat, corr_u, corr_v, center_lon=110, scale=5, linewidth=1, arrowsize=1., transform=ccrs.PlateCarree(central_longitude=0), MinDistance=[0.2, 0.5],
                     regrid=12, color='#454545', nanmax=nanmax)

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
    ax.tick_params(axis='both', labelsize=10, colors='black')

    return contf, ax

def pic2(fig, pic_loc, lat, lon, lat_f, lon_f, contour_1, contourf_1, lev, lev_f, color, clabel_tf, cmap, title):
    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')
    ax.set_title(f'{title}', loc='left', fontsize=12)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

    da_contour = xr.DataArray(
        contourf_1,
        coords={'lat': lat_f, 'lon': lon_f},
        dims=('lat', 'lon')
    )
    roi_shape = ((60, 0), (160, 60))
    contf = ax.contourf(lon_f, lat_f, da_contour.salem.roi(corners=roi_shape), cmap=cmap,
                        levels=lev_f, extend='both', transform=ccrs.PlateCarree(central_longitude=0))

    cont = ax.contour(lon, lat, contour_1, colors=color[1], levels=lev[1], linestyles='--', linewidths=0.8,
                      transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(lon, lat, contour_1, colors=color[0], levels=lev[0], linestyles='solid', linewidths=0.8,
                       transform=ccrs.PlateCarree(central_longitude=0))
    if clabel_tf:
        cont.clabel(inline=1, fontsize=8)
        cont_.clabel(inline=1, fontsize=8)


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


ds = xr.open_dataset(r"/Volumes/TiPlus7100/p4/data/ERA5_daily_uvwztq_sum.nc")
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

olr = xr.open_dataset(fr"{DATA}/NOAA/CPC/olr.cbo-1deg.day.mean.nc")['olr']
t2m = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_daily_t2m_sum.nc")['t2m']
t2m = ensure_celsius(t2m)
# 如果时间坐标叫 valid_time，就统一改成 time
if "time" not in t2m.coords:
    if "valid_time" in t2m.coords:
        t2m = t2m.rename({"valid_time": "time"})
    else:
        raise ValueError("数据中没有 time 或 valid_time 坐标。")

# 各要素按“月-日”逐日气候态（不是 dayofyear）
u_clim = daily_clim_by_mmdd(u)
v_clim = daily_clim_by_mmdd(v)
z_clim = daily_clim_by_mmdd(z)
t_clim = daily_clim_by_mmdd(t)
w_clim = daily_clim_by_mmdd(w)
olr_clim = daily_clim_by_mmdd(olr)
t2m_clim = daily_clim_by_mmdd(t2m)

#%%

time = [1961, 2022]
# YEAR = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014]
YEAR = [2015]

u_ano = anomaly_by_mmdd(u, u_clim, "2015-06-01", "2015-08-31")
v_ano = anomaly_by_mmdd(v, v_clim, "2015-06-01", "2015-08-31")
z_ano = anomaly_by_mmdd(z, z_clim, "2015-06-01", "2015-08-31")
t_ano = anomaly_by_mmdd(t, t_clim, "2015-06-01", "2015-08-31")
w_ano = anomaly_by_mmdd(w, w_clim, "2015-06-01", "2015-08-31")
olr_ano = anomaly_by_mmdd(olr, olr_clim, "2015-06-01", "2015-08-31")
t2m_ano = anomaly_by_mmdd(t2m, t2m_clim, "2015-06-01", "2015-08-31")

u_bp = LanczosFilter(u_ano, 'bandpass', period=[10, 30], nwts=9).filted()
v_bp = LanczosFilter(v_ano, 'bandpass', period=[10, 30], nwts=9).filted()
z_bp = LanczosFilter(z_ano, 'bandpass', period=[10, 30], nwts=9).filted()
t_bp = LanczosFilter(t_ano, 'bandpass', period=[10, 30], nwts=9).filted()
w_bp = LanczosFilter(w_ano, 'bandpass', period=[10, 30], nwts=9).filted()
olr_bp = LanczosFilter(olr_ano, 'bandpass', period=[10, 30], nwts=9).filted()
t2m_bp = LanczosFilter(t2m_ano, 'bandpass', period=[10, 30], nwts=9).filted()
#%%
yangtze_shp = fr'{PYFILE}/map/self/长江_TP/长江_tp.shp'
budget_ds_ori = calc_daily_temperature_budget(t.sel(time=slice("2015-06-01", "2015-08-31")),
                                          u.sel(time=slice("2015-06-01", "2015-08-31")),
                                          v.sel(time=slice("2015-06-01", "2015-08-31")),
                                          w.sel(time=slice("2015-06-01", "2015-08-31")))

# 2015 年对应的真实时间轴
target_time = t.sel(time=slice("2015-06-01", "2015-08-31")).time

t_clim_2015 = clim_to_time(t_clim, target_time)
u_clim_2015 = clim_to_time(u_clim, target_time)
v_clim_2015 = clim_to_time(v_clim, target_time)
w_clim_2015 = clim_to_time(w_clim, target_time)

budget_ds_clim = calc_daily_temperature_budget(
    t_clim_2015,
    u_clim_2015,
    v_clim_2015,
    w_clim_2015
)

dTdt = budget_ds_ori['dTdt'].sel(level=925) - budget_ds_clim['dTdt'].sel(level=925)
adv  = budget_ds_ori['adv_T'].sel(level=925) - budget_ds_clim['adv_T'].sel(level=925)
ver  = budget_ds_ori['ver'].sel(level=925) - budget_ds_clim['ver'].sel(level=925)
Q    = budget_ds_ori['Q'].sel(level=925) - budget_ds_clim['Q'].sel(level=925)
dTdt_yz_925 = region_mean_series(dTdt, yangtze_shp)*86400
adv_yz_925 = region_mean_series(adv, yangtze_shp)*86400
ver_yz_925 = region_mean_series(ver, yangtze_shp)*86400
Q_yz_925 = region_mean_series(Q, yangtze_shp)*86400

event_start = "2015-07-15"
event_end   = "2015-07-26"

adv925 = decompose_temperature_advection(
    u=u.sel(time=slice("2015-06-01", "2015-08-31")),
    v=v.sel(time=slice("2015-06-01", "2015-08-31")),
    t=t.sel(time=slice("2015-06-01", "2015-08-31")),
    u_clim=u_clim, v_clim=v_clim, t_clim=t_clim,
    start=event_start, end=event_end,
    level=925
)

adv925 *= 86400

adv925_series = xr.Dataset({
    'adv_clim': region_mean_series(adv925['adv_clim'], yangtze_shp),
    'adv_a_wind_on_climT': region_mean_series(adv925['adv_a_wind_on_climT'], yangtze_shp),
    'adv_clim_wind_on_aT': region_mean_series(adv925['adv_clim_wind_on_aT'], yangtze_shp),
    'adv_nonlinear': region_mean_series(adv925['adv_nonlinear'], yangtze_shp),
    'adv_ano_out_subseason': region_mean_series(adv925['adv_anom']-adv925['adv_a_wind_on_climT'].data-adv925['adv_clim_wind_on_aT'].data-adv925['adv_nonlinear'].data, yangtze_shp),
    'adv_anom': region_mean_series(adv925['adv_total']-adv925['adv_clim'].data, yangtze_shp)
})
#%%
fig = plt.figure(figsize=(8/2, 7/1.5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
title_head = '2015'

# 柱状图
ax = fig.add_subplot(211)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)


ax.set_aspect('auto')
ax.set_title(f'(a) {title_head} Temp_budget', fontsize=12, loc='left')
ax.grid(True, linestyle='--', zorder=0, axis='y')

time_range = ["07-15", "07-26"]
adv_X_dTdt = np.nanmean(adv_yz_925.sel(time=slice(f"2015-{time_range[0]}", f"2015-{time_range[1]}")))
ver_X_dTdt = np.nanmean(ver_yz_925.sel(time=slice(f"2015-{time_range[0]}", f"2015-{time_range[1]}")))
Q_X_dTdt = np.nanmean(Q_yz_925.sel(time=slice(f"2015-{time_range[0]}", f"2015-{time_range[1]}")))
adv_clim = np.nanmean(adv925_series['adv_clim'].sel(time=slice(f"2015-{time_range[0]}", f"2015-{time_range[1]}")))
adv_a_wind_on_climT = np.nanmean(adv925_series['adv_a_wind_on_climT'].sel(time=slice(f"2015-{time_range[0]}", f"2015-{time_range[1]}")))
adv_clim_wind_on_aT = np.nanmean(adv925_series['adv_clim_wind_on_aT'].sel(time=slice(f"2015-{time_range[0]}", f"2015-{time_range[1]}")))
adv_nonlinear = np.nanmean(adv925_series['adv_nonlinear'].sel(time=slice(f"2015-{time_range[0]}", f"2015-{time_range[1]}")))
adv_outofsubseason = np.nanmean(adv925_series['adv_ano_out_subseason'].sel(time=slice(f"2015-{time_range[0]}", f"2015-{time_range[1]}")))

values = [adv_X_dTdt, ver_X_dTdt, Q_X_dTdt, adv_a_wind_on_climT, adv_clim_wind_on_aT, adv_nonlinear]
colors = ['#ff7373' if val > 0 else '#7373ff' for val in values]

bars = ax.bar(range(6), values, width=0.3, color=colors, edgecolor='black', zorder=2)

ax.set_xticks(range(6))
ax.set_xticklabels([
    r'$-(\mathbf{V} \cdot \nabla T)^{\prime}$',
    r'$(\omega \sigma)^{\prime}$',
    r'${Q}^{\prime}$',
    #r'Nonsub Adv.',     #adv_anom_outofsubseason
    r'$-({u}^{\prime} \cdot \nabla \overline{T})$',     #adv_a_wind_on_climT = -(u' · ∇Tc)
    r'$-(\overline{u} \cdot \nabla {T}^{\prime})$',     #adv_clim_wind_on_aT = -(uc · ∇T')
    r'$-({u}^{\prime} \cdot \nabla {T}^{\prime})$'      #adv_nonlinear = -(u' · ∇T')
], fontsize=9.5)

# 上下交错排列
for i, tick in enumerate(ax.xaxis.get_major_ticks()):
    label_y =  0 if i % 2 == 0 else -0.12
    tick_len = 6 if i % 2 == 0 else 14

    tick.label1.set_y(label_y)
    tick.tick1line.set_markersize(tick_len)
    tick.tick1line.set_markeredgewidth(1.0)

ax.axvspan(2.5, 6.5, color="deepskyblue", alpha=0.15, zorder=0)
ax.set_ylim(-1, 1)
ax.set_xlim(-.5, 5.5)
ax.axhline(0, color='#454545', lw=0.7)

ax.set_yticks(np.arange(-2, 2.1, .4))
ax.set_yticklabels(['-2.0', '-1.6', '-1.2', '-0.8', '-0.4', '0', '0.4', '0.8', '1.2', '1.6', '2.0'], fontsize=12, color='#000000')
ax.tick_params(axis='y', labelsize=12, color='#000000')

level=925
# 空间图
contourfs1, ax1 = pic(
    fig, 212,
    u['lat'], u['lon'],
    np.array([[-40, -20, -10], [10, 20, 40]])*4 * 2,
    np.array([-3, -2, -1, -.5, .5, 1, 2, 3]),
    u_ano.sel(time=slice("2015-07-15", "2015-07-26"), level=500).mean('time'),
    v_ano.sel(time=slice("2015-07-15", "2015-07-26"), level=500).mean('time'),
    z_ano.sel(time=slice("2015-07-15", "2015-07-26"), level=500).mean('time'),
    t2m_ano.sel(time=slice("2015-07-15", "2015-07-26")).mean('time'),
f'(b) {title_head} 500UVZ&T2M',
    10
)
# dTdt_bp = LanczosFilter(dTdt, 'bandpass', period=[10, 30], nwts=9).filted()
# for i in range(5):
#     day = i*2+16
#     contourfs2, ax2 = pic(
#         fig, 322+i,
#         u['lat'], u['lon'],
#         np.array([[-40, -20, -10], [10, 20, 40]])*4*2,
#         np.array([-3, -2, -1, -.5, .5, 1, 2, 3]),
#         u_bp.sel(time=slice(f"2015-07-{day:>02}", f"2015-07-{day:>02}"), level=level).mean('time'),
#         v_bp.sel(time=slice(f"2015-07-{day:>02}", f"2015-07-{day:>02}"), level=level).mean('time'),
#         z_bp.sel(time=slice(f"2015-07-{day:>02}", f"2015-07-{day:>02}"), level=level).mean('time'),
#         t2m_bp.sel(time=slice(f"2015-07-{day:>02}", f"2015-07-{day:>02}")).mean('time'),
#     f'({["b", "c", "d", "e", "f"][i]}) {title_head}/07/{i*2+2:>02} 925UVZ&T2M',
#         5
#     )
#     import matplotlib.patheffects as path_effects
#     ax2.text(0.02, 0.95, f"Adv={region_mean_series(dTdt.sel(time=slice(f"2015-07-{day:>02}", f"2015-07-{day:>02}")), yangtze_shp)[0]*86400:.2f}",
#              color='red', va='top', ha='left', fontsize=10, transform=ax2.transAxes, path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
#     ax2.add_geometries(
#         Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp').geometries(),
#         ccrs.PlateCarree(), facecolor='gray', edgecolor='gray', linewidth=.1, hatch='.', zorder=10)

# plot_text(ax1, 124.3, 35.3, 'C', 12, 'red')

# 添加全局colorbar  # 为colorbar腾出空间
ax_colorbar = inset_axes(ax1, width="4%", height="150%", loc='center right', bbox_to_anchor=(0.1, 0.7, 1, 1),
                         bbox_transform=ax1.transAxes, borderpad=0)
cb1 = plt.colorbar(contourfs1, cax=ax_colorbar, orientation='vertical', drawedges=True)
cb1.locator = ticker.FixedLocator(np.array([-3, -2, -1, -.5, .5, 1, 2, 3]))
cb1.set_ticklabels(['-3.0', '-2.0', '-1.0', '-0.5', '0.5', ' 1.0', ' 2.0', ' 3.0'])
cb1.ax.tick_params(length=0, labelsize=12, direction='in')  # length为刻度线的长度
cb1.dividers.set_linewidth(1.25)  # 设置分割线宽度
cb1.outline.set_linewidth(1.25)  # 设置色标轮廓宽度


for ax in fig.axes:
    # 遍历每个子图中的所有艺术家对象 (artist)
    for artist in ax.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)

plt.savefig(fr"{PYFILE}/p4/pic/局地环流_2015次季节中旬.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p4/pic/局地环流_2015次季节中旬.png", bbox_inches='tight', dpi=600)
plt.show()