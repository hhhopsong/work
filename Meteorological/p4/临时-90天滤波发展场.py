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
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

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
fig = plt.figure(figsize=(8, 5))
plt.subplots_adjust(wspace=0, hspace=0)
title_head = 'OLR'

# lev = np.array([0., .08, .16, .24, .32, .4, .48, .56])
lev = np.array([-3, -2.5, -2, -1.5, -1, -0.5, .5, 1, 1.5, 2, 2.5, 3])*10   # olr
# lev = np.array([-500, -400, -300, -200, -100, -50, 50, 100, 200, 300, 400, 500])    # 500z
# lev = np.array([-500, -400, -300, -200, -100, -50, 50, 100, 200, 300, 400, 500])*2  # 200z
# lev = np.array([-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5]) * 2  # 200v

index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']

for i in range(9):
    iday = f"{year}-06-{27 + 2*i}" if i < 2 else f"{year}-07-{-1 + 2*i}"
    mmdd = pd.Timestamp(iday).strftime("%m-%d")

    Uc_day = u_clim.sel(mmdd=mmdd).transpose("level", "lat", "lon")
    Vc_day = v_clim.sel(mmdd=mmdd).transpose("level", "lat", "lon")
    Tc_day = t_clim.sel(mmdd=mmdd).transpose("level", "lat", "lon")

    GEOa_day = z_bp.sel(time=iday).transpose("level", "lat", "lon")

    WAF = TN_WAF_3D(Uc_day, Vc_day, GEOa_day, Tc_day)

    if i == 0:
        ax, cont = pic(
            fig, (3, 3, i + 1),
            z_bp['lat'], z_bp['lon'],
            None, None,
            lev,
            olr_bp.sel(time=iday),
            f'{year} {title_head}',
            lat_tick=None, lon_tick=None, key=False
        )
    elif i == 2:
        ax, cont = pic(
            fig, (3, 3, i + 1),
            z_bp['lat'], z_bp['lon'],
            None, None,
            lev,
            olr_bp.sel(time=iday),
            '',
            lat_tick=None, lon_tick=None, key=True
        )
    else:
        ax, cont = pic(
            fig, (3, 3, i + 1),
            z_bp['lat'], z_bp['lon'],
            None, None,
            lev,
            olr_bp.sel(time=iday),
            '',
            key=False,
            lat_tick=None, lon_tick=None,
        )

    # ===== 当前子图日期 =====
    from matplotlib.patheffects import withStroke
    date_text = u_bp['time'].sel(time=iday).dt.strftime('%m/%d').item()
    day = i * 2 - 8
    day_str = f"+{day:.0f}" if day > 0 else f"{day:.0f}"

    ax.text(
        0.05, 0.95, f"Day {day_str}",
        transform=ax.transAxes,
        fontsize=10,
        color='red',
        ha='left',
        va='top',
        bbox=dict(facecolor='none', edgecolor='none'),
        path_effects=[withStroke(linewidth=1.5, foreground='white')]
    )

# =========================
# 全局 colorbar
# =========================
cbar_ax = inset_axes(
    ax,
    width="4%",
    height="100%",
    loc='lower left',
    bbox_to_anchor=(1.025, 0., 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0
)
cbar = fig.colorbar(cont, cax=cbar_ax, orientation='vertical', drawedges=True)
cbar.locator = ticker.FixedLocator(lev)
cbar.set_ticklabels([f"{i:.0f}" for i in lev])

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.set_aspect('auto')

for ax in fig.axes:
    for artist in ax.get_children():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(True)

plt.savefig(fr"{PYFILE}/p4/pic/90天滤波环流场_{title_head}.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p4/pic/90天滤波环流场_{title_head}.png", bbox_inches='tight', dpi=600)
plt.show()
