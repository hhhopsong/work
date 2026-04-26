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
from climkit.TN_WaveActivityFlux import *
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
    ax.text(
        x, y, title,
        transform=ccrs.PlateCarree(),
        ha='center',
        va='center',
        fontsize=size,
        fontweight='bold',
        color=color,
        fontname='Times New Roman',
        zorder=1000
    )
    return 0


def pic(fig, pic_loc, lat, lon, u, v, lev, contf_var, title,
        lon_tick=np.arange(60, 160, 20), lat_tick=np.arange(0, 60, 15), key=True):

    ax = fig.add_subplot(*pic_loc, projection=ccrs.PlateCarree(central_longitude=180 - 70))

    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_aspect('auto')
    ax.set_title(title, loc='left', fontsize=18)
    ax.set_extent([-180, 180, -20, 80], crs=ccrs.PlateCarree())

    cont = ax.contourf(
        lon, lat, contf_var,
        cmap=cmaps.BlueWhiteOrangeRed[10:-10],
        levels=lev,
        linewidths=0.8,
        transform=ccrs.PlateCarree(central_longitude=0),
        extend='both',
        alpha=0.8
    )

    if u is not None or v is not None:
        Cq = ax.Curlyquiver(
            lon, lat, u, v,
            scale=2,
            linewidth=0.9,
            arrowsize=1.,
            transform=ccrs.PlateCarree(central_longitude=0),
            MinDistance=[0.2, 0.5],
            regrid=12,
            color='#454545',
            nanmax=5,
            thinning=[['20%', '100%'], 'range']
        )
        if key:
            Cq.key(
                U=5,
                label='5 $m/s$',
                color='k',
                fontproperties={'size': 10},
                linewidth=.7,
                arrowsize=12.,
                loc='upper right',
                bbox_to_anchor=(0, 0.32, 1, 1),
                edgecolor='none',
                shrink=0.4,
                intetval=0.7
            )

    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4)
    ax.add_geometries(
        Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black',
        linewidth=.5
    )
    ax.add_geometries(
        Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='gray',
        edgecolor='black',
        linewidth=.5
    )

    # 刻度线设置
    xticks1 = lon_tick
    yticks1 = lat_tick
    if yticks1 is not None:
        ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
    if xticks1 is not None:
        ax.set_xticks(xticks1, crs=ccrs.PlateCarree())

    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)

    ymajorLocator = MultipleLocator(15)
    ax.yaxis.set_major_locator(ymajorLocator)
    yminorLocator = MultipleLocator(5)
    ax.yaxis.set_minor_locator(yminorLocator)

    xmajorLocator = MultipleLocator(20)
    xminorLocator = MultipleLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)

    ax.tick_params(which='major', length=4, width=.5, color='black')
    ax.tick_params(which='minor', length=2, width=.2, color='black')
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'
    ax.tick_params(axis='both', labelsize=10, colors='black', length=0)

    return ax, cont


def daily_clim_by_mmdd(da: xr.DataArray, time_dim: str = "time", drop_feb29: bool = True) -> xr.DataArray:
    """按 MM-DD 计算逐日气候态（非 dayofyear，避免闰年错位）"""
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


# =========================
# 参数区
# =========================
year = 2015

# 为 61 点滤波留足两端缓冲
clim_start = "1961-05-01"
clim_end   = "2022-09-30"

filter_start = f"{year}-05-01"
filter_end   = f"{year}-09-30"

plot_start = f"{year}-06-01"
plot_end   = f"{year}-08-31"


# =========================
# 读取数据
# =========================
ds = xr.open_zarr(r"/Volumes/TiPlus7100/p4/data/ERA5_daily_uvwztq_sum.zarr")
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

# 改成 5-9 月，避免 61 点滤波后 6-8 月两端不完整
ds = ds[required_vars].sel(time=slice(clim_start, clim_end))

# 分别取出变量
u = ds["u"]
v = ds["v"]
z = ds["z"]
t = ds["t"]
w = ds["w"]

olr = xr.open_dataset(fr"{DATA}/NOAA/CPC/olr.cbo-1deg.day.mean.nc")['olr']
olr = olr.sel(time=slice(clim_start, clim_end))

t2m = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_daily_t2m_sum.nc")['t2m']
t2m = ensure_celsius(t2m)
if "time" not in t2m.coords:
    if "valid_time" in t2m.coords:
        t2m = t2m.rename({"valid_time": "time"})
    else:
        raise ValueError("数据中没有 time 或 valid_time 坐标。")
t2m = t2m.sel(time=slice(clim_start, clim_end))


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
# =========================
# 作图
# =========================
title_head = '850UVZ'

# lev = np.array([0., .08, .16, .24, .32, .4, .48, .56])
# lev = np.array([-3, -2.5, -2, -1.5, -1, -0.5, .5, 1, 1.5, 2, 2.5, 3])*10   # olr
lev = np.array([-500, -400, -300, -200, -100, -50, 50, 100, 200, 300, 400, 500])    # 500z
# lev = np.array([-500, -400, -300, -200, -100, -50, 50, 100, 200, 300, 400, 500])*2  # 200z
# lev = np.array([-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5]) * 2  # 200v
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter, FFMpegWriter
from matplotlib.patheffects import withStroke

# =========================
# 动画参数
# =========================

center_day = pd.Timestamp(f"{year}-07-05")   # 这里改成你的 Day 0
day_offsets = np.arange(-8, 9, 1)
date_list = [center_day + pd.Timedelta(days=int(d)) for d in day_offsets]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180 - 70))

global_cbar = {"obj": None}

def draw_base(ax, title=''):
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    ax.set_aspect('auto')
    ax.set_title(title, loc='left', fontsize=18)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4)

    ax.add_geometries(
        Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black',
        linewidth=.5
    )
    ax.add_geometries(
        Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='gray',
        edgecolor='black',
        linewidth=.5
    )

    lon_tick = np.arange(60, 160, 20)
    lat_tick = np.arange(0, 60, 15)
    ax.set_xticks(lon_tick, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_tick, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.tick_params(which='major', length=4, width=.5, color='black')
    ax.tick_params(which='minor', length=2, width=.2, color='black')
    ax.tick_params(axis='both', labelsize=10, colors='black')
    ax.tick_params(which='both', bottom=True, top=False, left=True,
                   labelbottom=True, labeltop=False)

def update(frame):
    ax.clear()
    draw_base(ax, title=f'{year} {title_head}')

    current_day = date_list[frame]
    iday = current_day.strftime("%Y-%m-%d")
    day = day_offsets[frame]
    day_str = f"+{day}" if day > 0 else f"{day}"

    cont = ax.contourf(
        z_bp['lon'],
        z_bp['lat'],
        z_bp.sel(time=iday, level=850),
        cmap=cmaps.BlueWhiteOrangeRed[10:-10],
        levels=lev,
        transform=ccrs.PlateCarree(central_longitude=0),
        extend='both',
        alpha=0.8
    )

    vec =   ax.Curlyquiver(
            u_bp['lon'], u_bp['lat'], u_bp.sel(time=iday, level=850), v_bp.sel(time=iday, level=850),
            scale=2,
            linewidth=0.9,
            arrowsize=1.8,
            transform=ccrs.PlateCarree(central_longitude=0),
            MinDistance=[0.2, 0.5],
            regrid=16,
            color='#454545',
            nanmax=5,
            thinning=[['20%', '100%'], 'range']
        )

    ax.text(
        0.03, 0.96, f"Day {day_str}",
        transform=ax.transAxes,
        fontsize=13,
        color='red',
        ha='left',
        va='top',
        path_effects=[withStroke(linewidth=2, foreground='white')]
    )

    ax.text(
        0.97, 0.96, current_day.strftime('%Y/%m/%d'),
        transform=ax.transAxes,
        fontsize=12,
        color='black',
        ha='right',
        va='top',
        path_effects=[withStroke(linewidth=2, foreground='white')]
    )

    if global_cbar["obj"] is None:
        cbar_ax = inset_axes(
            ax,
            width="4%",
            height="100%",
            loc='lower left',
            bbox_to_anchor=(1.025, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        cb = fig.colorbar(cont, cax=cbar_ax, orientation='vertical', drawedges=True)
        cb.locator = ticker.FixedLocator(lev)
        cb.set_ticklabels([f"{i:.0f}" for i in lev])
        global_cbar["obj"] = cb

    # blit=False 时这里返回值不会被使用
    return []

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(date_list),
    interval=600,
    blit=False,
    repeat=True
)

mp4_path = fr"{PYFILE}/p4/pic/{title_head}_daily_evolution_day-8_to_8.mp4"
gif_path = fr"{PYFILE}/p4/pic/{title_head}_daily_evolution_day-8_to_8.gif"

# 先判断 ffmpeg 是否存在
import shutil
if shutil.which("ffmpeg") is not None:
    writer_mp4 = FFMpegWriter(fps=2, bitrate=6000)
    ani.save(mp4_path, writer=writer_mp4, dpi=600)
    print(f"MP4 saved: {mp4_path}")
else:
    print("ffmpeg not found, skip MP4 export.")

# GIF 不依赖 ffmpeg
writer_gif = PillowWriter(fps=2)
ani.save(gif_path, writer=writer_gif, dpi=600)
print(f"GIF saved: {gif_path}")

plt.show()