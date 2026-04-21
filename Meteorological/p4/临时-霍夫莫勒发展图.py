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

    ax.set_title(title, loc='left', fontsize=18)

    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    cont = ax.contourf(lon, lat, contf_var,  cmap=cmaps.BlueWhiteOrangeRed[10:-10], levels=lev, linewidths=0.8, transform=ccrs.PlateCarree(central_longitude=0), extend='both', alpha=0.8)

    if u is not None or v is not None:
        Cq = ax.Curlyquiver(lon, lat, u, v, center_lon=110, scale=2, linewidth=0.9, arrowsize=1., transform=ccrs.PlateCarree(central_longitude=0), MinDistance=[0.2, 0.5],
                         regrid=12, color='#454545', nanmax=2.5)
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

def ensure_celsius(da: xr.DataArray) -> xr.DataArray:
    """如果像 Kelvin，就转成 Celsius"""
    vmin = float(da.min().values)
    vmax = float(da.max().values)
    if vmin > 150 and vmax < 400:
        print("检测到温度可能为 Kelvin，自动转换为 Celsius。")
        da = da - 273.15
        da.attrs["units"] = "degC"
    return da

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
if "time" not in t2m.coords:
    if "valid_time" in t2m.coords:
        t2m = t2m.rename({"valid_time": "time"})
    else:
        raise ValueError("数据中没有 time 或 valid_time 坐标。")
# 各要素按“月-日”逐日气候态（不是 dayofyear）
CLIM = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_clim_sum.nc")
u_clim = CLIM["u_clim"]
v_clim = CLIM["v_clim"]
z_clim = CLIM["z_clim"]
t_clim = CLIM["t_clim"]
w_clim = CLIM["w_clim"]
olr_clim = CLIM["olr_clim"]
t2m_clim = CLIM["t2m_clim"]

time = [1961, 2022]
# YEAR = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014]
YEAR = [2015]

# =========================
# 为了适配 nwts=61 的 Lanczos 滤波，
# 先取 5–9 月做异常和滤波，
# 再裁剪回 6–8 月，保证 6–8 月结果完整
# =========================
filter_start = "2015-05-01"
filter_end   = "2015-09-30"

analysis_start = "2015-06-01"
analysis_end   = "2015-08-31"

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
import os

g = 9.80665  # m s^-2

def get_level_dim(da):
    """自动识别气压层维度名"""
    for name in ["level", "plev", "isobaricInhPa", "pressure_level"]:
        if name in da.dims or name in da.coords:
            return name
    raise ValueError(f"{da.name or 'DataArray'} 中未找到气压层坐标名。当前 coords: {list(da.coords)}")

def normalize_lon(da):
    """统一经度到 0~360"""
    if "lon" not in da.coords:
        raise ValueError("缺少 lon 坐标。")
    lon = da["lon"]
    if float(lon.min()) < 0:
        da = da.assign_coords(lon=((lon + 360) % 360)).sortby("lon")
    return da

def lon_mean_100_120(da):
    """取 100E-120E 的经向平均"""
    da = normalize_lon(da)
    return da.sel(lon=slice(100, 125)).mean("lon")

def calc_temp_advection_925(t_da, u_da, v_da):
    """
    计算 925 hPa 温度平流
    返回单位：K/day
    """
    # 保证经纬度顺序正确
    t_da = normalize_lon(t_da).sortby("lat").sortby("lon")
    u_da = normalize_lon(u_da).sortby("lat").sortby("lon")
    v_da = normalize_lon(v_da).sortby("lat").sortby("lon")

    # metpy 计算 dx, dy
    lon2d, lat2d = np.meshgrid(t_da.lon.values, t_da.lat.values)
    dx, dy = mpcalc.lat_lon_grid_deltas(lon2d, lat2d)

    adv_list = []
    for it in range(t_da.sizes["time"]):
        T = t_da.isel(time=it).values * units.degC
        U = u_da.isel(time=it).values * units("m/s")
        V = v_da.isel(time=it).values * units("m/s")

        # 温度平流：- (u*dT/dx + v*dT/dy)
        adv = mpcalc.advection(
            scalar=T,
            u=U,
            v=V,
            dx=dx,
            dy=dy
        )

        # 转为 K/day
        adv = adv.to("kelvin/day").magnitude
        adv_list.append(adv)

    adv_da = xr.DataArray(
        np.array(adv_list),
        coords={"time": t_da.time, "lat": t_da.lat, "lon": t_da.lon},
        dims=("time", "lat", "lon"),
        name="tadv925"
    )
    adv_da.attrs["units"] = "K/day"
    return adv_da

# ---------------------------
# 1. 选层
# ---------------------------
lev_dim_u = get_level_dim(u_bp)
lev_dim_v = get_level_dim(v_bp)
lev_dim_z = get_level_dim(z_bp)
lev_dim_t = get_level_dim(t_bp)

u850 = u_bp.sel({lev_dim_u: 850})
v850 = v_bp.sel({lev_dim_v: 850})
z500 = z_bp.sel({lev_dim_z: 500})
t925 = t_bp.sel({lev_dim_t: 925})
u925 = u_bp.sel({lev_dim_u: 925})
v925 = v_bp.sel({lev_dim_v: 925})

# ERA5 z 一般是 geopotential (m^2/s^2)，转成位势高度 gpm
z500_hgt = z500 / g
z500_hgt.attrs["units"] = "gpm"

# ---------------------------
# 2. 计算 925 hPa 温度平流
# ---------------------------
tadv925 = calc_temp_advection_925(t925, u925, v925)

# ---------------------------
# 3. 100°–120°E 平均
# ---------------------------
z500_hov = lon_mean_100_120(z500_hgt)
v850_hov = lon_mean_100_120(v850)
tadv925_hov = lon_mean_100_120(tadv925)

# 若纬度是降序，这里统一成升序，方便作图
z500_hov = z500_hov.sortby("lat")
v850_hov = v850_hov.sortby("lat")
tadv925_hov = tadv925_hov.sortby("lat")

# ---------------------------
# 4. 画 Hovmöller 图
# ---------------------------
title_head = "2015夏季_10to30d_Hovmoller_100E120E"

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)

# 填色：500 hPa 高度异常
cf_lev = np.arange(-20, 22, 2)
cf = ax.contourf(
    z500_hov["time"].values,
    z500_hov["lat"].values,
    z500_hov.transpose("lat", "time").values,
    levels=cf_lev,
    cmap=cmaps.BlueWhiteOrangeRed[10:-10],
    extend="both"
)

# 矢量箭头：850 hPa 经向风异常
# 在 time-lat 平面上，用 U=0, V=v850 表示南北向异常
time_vals = pd.to_datetime(v850_hov.time.values)
lat_vals = v850_hov.lat.values

# 抽稀，避免太密
t_skip = 3
y_skip = 2

Tq, Yq = np.meshgrid(time_vals[::t_skip], lat_vals[::y_skip])
Vq = v850_hov.isel(time=slice(None, None, t_skip), lat=slice(None, None, y_skip)).transpose("lat", "time").values
Uq = np.zeros_like(Vq)

# 正异常红色
Vq_pos = np.where(Vq > 0, Vq, np.nan)
ax.quiver(
    Tq, Yq, Uq, Vq_pos,
    angles="xy", scale_units="height", scale=50,
    color="red", width=0.0022,
    headwidth=3.5, headlength=4.5, headaxislength=4.2,
    pivot="middle", zorder=4
)

# 负异常蓝色
Vq_neg = np.where(Vq < 0, Vq, np.nan)
ax.quiver(
    Tq, Yq, Uq, Vq_neg,
    angles="xy", scale_units="height", scale=50,
    color="blue", width=0.0022,
    headwidth=3.5, headlength=4.5, headaxislength=4.2,
    pivot="middle", zorder=4
)

# 坐标轴与标题
ax.set_title("(a) 500Z&850V over 100°–125°E", loc="left", fontsize=18)

ax.set_ylim(0, 60)
ax.set_yticks(np.arange(0, 61, 10))
ax.set_yticklabels(['0°', '10°N', '20°N', '30°N', '40°N', '50°N', '60°N'])
ax.tick_params(axis="both", labelsize=12)

ax.axhline(35, color='purple', linestyle='-', linewidth=1)
ax.axhline(25, color='purple', linestyle='-', linewidth=1)

# x 轴日期格式
import matplotlib.dates as mdates

ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 色标
cbar = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, aspect=28, shrink=0.95)
cbar.ax.tick_params(labelsize=11)

plt.tight_layout()

# ---------------------------
# 5. 保存
# ---------------------------
outdir = fr"{PYFILE}/p4/pic"
os.makedirs(outdir, exist_ok=True)

plt.savefig(fr"{outdir}/100_120E平均Hovmoller_{title_head}.pdf", bbox_inches='tight')
plt.savefig(fr"{outdir}/100_120E平均Hovmoller_{title_head}.png", bbox_inches='tight', dpi=600)
plt.show()
