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

from climkit.masked import masked

from metpy.units import units
import metpy.calc as mpcalc


# =========================
# 字体设置
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"


# =========================================================
# 基础函数
# =========================================================
def standardize_latlon(ds):
    """统一经纬度坐标名为 lon/lat"""
    rename_dict = {}

    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"

    if "longitude" in ds.dims:
        rename_dict["longitude"] = "lon"
    if "latitude" in ds.dims:
        rename_dict["latitude"] = "lat"

    if rename_dict:
        ds = ds.rename(rename_dict)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("数据中未找到 lon/lat 或 longitude/latitude 坐标。")

    return ds


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
    out = out.assign_coords(time=time_coord.values)

    if 'mmdd' in out.coords:
        out = out.drop_vars('mmdd')

    return out


def calc_daily_temperature_budget(t_da, u_da, v_da, w_da):
    """
    计算逐日温度方程各项：
    dTdt, adv_T, ver, Q

    Parameters
    ----------
    t_da, u_da, v_da, w_da : xr.DataArray
        dims 必须为: (time, level, lat, lon)

        t_da: 温度, 单位 K
        u_da: 纬向风, 单位 m/s
        v_da: 经向风, 单位 m/s
        w_da: 压力坐标垂直速度 omega, 单位 Pa/s

    Returns
    -------
    ds_out : xr.Dataset
        dTdt  : 温度倾向, K/s
        adv_T : 水平温度平流, K/s
        ver   : 垂直运动项, K/s
        Q     : 非绝热加热项, K/s
        sigma : 静力稳定度项, K/Pa
    """
    from metpy.constants import dry_air_gas_constant as R
    from metpy.constants import dry_air_spec_heat_press as cp

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

    p_pa = (levs_hpa * 100.0) * units.Pa
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

    dTdt_arr = np.full((nt, nz, ny, nx), np.nan)
    adv_arr  = np.full((nt, nz, ny, nx), np.nan)
    ver_arr  = np.full((nt, nz, ny, nx), np.nan)
    Q_arr    = np.full((nt, nz, ny, nx), np.nan)
    sigma_arr = np.full((nt, nz, ny, nx), np.nan)

    for i in range(nt):

        # ----------------------------
        # (1) 时间差分 dTdt
        # ----------------------------
        if i == 0:
            dt_seconds = (times[i + 1] - times[i]) / np.timedelta64(1, 's')
            dTdt = (t_q.isel(time=i + 1) - t_q.isel(time=i)) / (dt_seconds * units.s)
        elif i == nt - 1:
            dt_seconds = (times[i] - times[i - 1]) / np.timedelta64(1, 's')
            dTdt = (t_q.isel(time=i) - t_q.isel(time=i - 1)) / (dt_seconds * units.s)
        else:
            dt_seconds = (times[i + 1] - times[i - 1]) / np.timedelta64(1, 's')
            dTdt = (t_q.isel(time=i + 1) - t_q.isel(time=i - 1)) / (dt_seconds * units.s)

        # ----------------------------
        # (2) 水平温度平流 adv_T
        # ----------------------------
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

        # ----------------------------
        # (3) 静力稳定度 sigma
        # ----------------------------
        T_now = t_q.isel(time=i)
        T_vals = T_now.metpy.dequantify().values
        dTdp_vals = np.gradient(T_vals, p_pa.magnitude, axis=0)

        p_3d = p_pa.magnitude[:, None, None]
        term1 = (R.magnitude * T_vals) / (cp.magnitude * p_3d)

        sigma_vals = term1 - dTdp_vals

        sigma = xr.DataArray(
            sigma_vals,
            coords=T_now.coords,
            dims=T_now.dims
        ) * units('K/Pa')

        # ----------------------------
        # (4) 垂直运动项 ver
        # ----------------------------
        ver = w_q.isel(time=i) * sigma

        # ----------------------------
        # (5) 非绝热加热项 Q
        # ----------------------------
        Q = dTdt - adv_T - ver

        dTdt_arr[i] = dTdt.metpy.dequantify().values
        adv_arr[i]  = adv_T.metpy.dequantify().values
        ver_arr[i]  = ver.metpy.dequantify().values
        Q_arr[i]    = Q.metpy.dequantify().values
        sigma_arr[i] = sigma.metpy.dequantify().values

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


def plot_scalar_map(
    fig,
    pic_loc,
    lon,
    lat,
    scalar_plot,
    scalar_levels,
    title,
    cbar_label,
    cmap_use,
    add_tp_hatch=True
):
    """
    仅绘制填色场，不再叠加 UV / Z 等其他要素
    """
    ax = fig.add_subplot(
        pic_loc,
        projection=ccrs.PlateCarree(central_longitude=180 - 70)
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_aspect("auto")
    ax.set_title(title, loc='left', fontsize=10)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

    # -------------------------
    # 填色区域
    # -------------------------
    contf = ax.contourf(
        lon,
        lat,
        scalar_plot,
        levels=scalar_levels,
        cmap=cmap_use,
        extend='both',
        transform=ccrs.PlateCarree()
    )

    # -------------------------
    # 底图
    # -------------------------
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4)

    ax.add_geometries(
        Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black',
        linewidth=0.5
    )

    ax.add_geometries(
        Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='gray',
        edgecolor='black',
        linewidth=0.5
    )

    # a、b 画青藏高原阴影；c 不画
    if add_tp_hatch:
        ax.add_geometries(
            Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp').geometries(),
            ccrs.PlateCarree(),
            facecolor='#909090',
            edgecolor='#909090',
            linewidth=0,
            hatch='.',
            zorder=10
        )

    # -------------------------
    # 刻度
    # -------------------------
    xticks1 = np.arange(60, 161, 20)
    yticks1 = np.arange(0, 61, 15)

    ax.set_xticks(xticks1, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks1, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.tick_params(which='major', length=4, width=0.5, color='black')
    ax.tick_params(which='minor', length=2, width=0.2, color='black')
    ax.tick_params(
        which='both',
        bottom=True,
        top=False,
        left=True,
        labelbottom=True,
        labeltop=False
    )
    ax.tick_params(axis='both', labelsize=8, colors='black')

    # -------------------------
    # colorbar
    # -------------------------
    ax_colorbar = inset_axes(
        ax,
        width="3.5%",
        height="92%",
        loc='center right',
        bbox_to_anchor=(0.075, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    cb = plt.colorbar(
        contf,
        cax=ax_colorbar,
        orientation='vertical',
        drawedges=True
    )

    cb.locator = ticker.FixedLocator(scalar_levels)
    cb.update_ticks()
    cb.ax.tick_params(length=0, labelsize=8, direction='in')
    cb.dividers.set_linewidth(0.8)
    cb.outline.set_linewidth(1.0)
    cb.set_label(cbar_label, fontsize=8)

    return contf, ax


# =========================================================
# 主程序
# =========================================================

# -------------------------
# 分析时段与层次
# -------------------------
analysis_start = "2015-06-15"
analysis_end = "2015-07-31"
level_plot = 925

# -------------------------
# 读取原始数据
# -------------------------
ds = xr.open_dataset(r"/Volumes/TiPlus7100/p4/data/ERA5_daily_uvwztq_sum.zarr")
ds = standardize_latlon(ds)

if "time" not in ds.coords:
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    else:
        raise ValueError("数据中没有 time 或 valid_time 坐标。")

required_vars = ["u", "v", "t", "w"]
missing_vars = [v for v in required_vars if v not in ds.data_vars]
if missing_vars:
    raise ValueError(f"数据中缺少变量: {missing_vars}；当前变量有: {list(ds.data_vars)}")

ds = ds[required_vars].sel(time=slice("2015-05-01", "2015-09-30"))

u = ds["u"]
v = ds["v"]
t = ds["t"]
w = ds["w"]

# -------------------------
# 读取逐日气候态
# -------------------------
CLIM = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_clim_sum.nc")
CLIM = standardize_latlon(CLIM)

u_clim = CLIM["u_clim"]
v_clim = CLIM["v_clim"]
t_clim = CLIM["t_clim"]
w_clim = CLIM["w_clim"]

# -------------------------
# 气候态映射到 2015 真实时间轴
# -------------------------
target_time = t.sel(time=slice("2015-05-01", "2015-09-30")).time

t_clim_2015 = clim_to_time(t_clim, target_time)
u_clim_2015 = clim_to_time(u_clim, target_time)
v_clim_2015 = clim_to_time(v_clim, target_time)
w_clim_2015 = clim_to_time(w_clim, target_time)

# -------------------------
# 计算温度诊断方程各项
# -------------------------
budget_ds_ori = calc_daily_temperature_budget(
    t.sel(time=slice("2015-05-01", "2015-09-30")),
    u.sel(time=slice("2015-05-01", "2015-09-30")),
    v.sel(time=slice("2015-05-01", "2015-09-30")),
    w.sel(time=slice("2015-05-01", "2015-09-30"))
)

budget_ds_clim = calc_daily_temperature_budget(
    t_clim_2015,
    u_clim_2015,
    v_clim_2015,
    w_clim_2015
)

# -------------------------
# 异常项（原始 - 气候态）
# 取 925 hPa
# 并转换为 K/day 便于作图
# -------------------------
adv_anom = (
    budget_ds_ori['adv_T'].sel(level=level_plot)
    - budget_ds_clim['adv_T'].sel(level=level_plot)
) * 86400.0

ver_anom = (
    budget_ds_ori['ver'].sel(level=level_plot)
    - budget_ds_clim['ver'].sel(level=level_plot)
) * 86400.0

Q_anom = (
    budget_ds_ori['Q'].sel(level=level_plot)
    - budget_ds_clim['Q'].sel(level=level_plot)
) * 86400.0

adv_anom.attrs['units'] = 'K/day'
ver_anom.attrs['units'] = 'K/day'
Q_anom.attrs['units'] = 'K/day'

# -------------------------
# 取分析时段平均
# -------------------------
adv_plot = adv_anom.sel(time=slice(analysis_start, analysis_end)).mean('time', skipna=True)
ver_plot = ver_anom.sel(time=slice(analysis_start, analysis_end)).mean('time', skipna=True)
Q_plot   = Q_anom.sel(time=slice(analysis_start, analysis_end)).mean('time', skipna=True)

# -------------------------
# 坐标
# -------------------------
lon = adv_plot['lon']
lat = adv_plot['lat']

# -------------------------
# 色标范围
# 单位：K/day
# 你可以根据结果大小再调整
# -------------------------
lev_budget = np.array([
    -1.6, -1.2, -0.8, -0.4, -0.2,
     0.2,  0.4,  0.8,  1.2,  1.6
])

# 配色
cmap_a = plt.get_cmap("RdBu_r")   # 平流项
cmap_b = plt.get_cmap("PuOr")     # 垂直运动项
cmap_c = plt.get_cmap("BrBG")     # 非绝热加热项

# =========================================================
# 作图
# =========================================================
fig = plt.figure(figsize=(5.2, 8.4))
plt.subplots_adjust(wspace=0.15, hspace=0.28)

title_head = "2015"

# -------------------------
# (a) 平流项
# 画青藏高原
# -------------------------
contf1, ax1 = plot_scalar_map(
    fig=fig,
    pic_loc=311,
    lon=lon,
    lat=lat,
    scalar_plot=adv_plot,
    scalar_levels=lev_budget,
    title=f"(a) {title_head} horizontal advection term at {level_plot} hPa",
    cbar_label="K/day",
    cmap_use=cmap_a,
    add_tp_hatch=True
)

# -------------------------
# (b) 垂直运动项
# 画青藏高原
# -------------------------
contf2, ax2 = plot_scalar_map(
    fig=fig,
    pic_loc=312,
    lon=lon,
    lat=lat,
    scalar_plot=ver_plot,
    scalar_levels=lev_budget,
    title=f"(b) {title_head} vertical motion term at {level_plot} hPa",
    cbar_label="K/day",
    cmap_use=cmap_b,
    add_tp_hatch=True
)

# -------------------------
# (c) 非绝热加热项
# 不画青藏高原
# -------------------------
contf3, ax3 = plot_scalar_map(
    fig=fig,
    pic_loc=313,
    lon=lon,
    lat=lat,
    scalar_plot=Q_plot,
    scalar_levels=lev_budget,
    title=f"(c) {title_head} diabatic heating term at {level_plot} hPa",
    cbar_label="K/day",
    cmap_use=cmap_c,
    add_tp_hatch=False
)

# -------------------------
# 强制裁剪
# -------------------------
for ax_ in fig.axes:
    for artist in ax_.get_children():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(True)

# -------------------------
# 保存
# -------------------------
plt.savefig(
    fr"{PYFILE}/p4/pic/2015温度诊断方程_925hPa_平流_垂直运动_非绝热加热.pdf",
    bbox_inches='tight'
)

plt.savefig(
    fr"{PYFILE}/p4/pic/2015温度诊断方程_925hPa_平流_垂直运动_非绝热加热.png",
    bbox_inches='tight',
    dpi=600
)

plt.show()
