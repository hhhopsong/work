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
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy import ndimage
from scipy.stats import ttest_ind, pearsonr
import matplotlib.path as mpath
import matplotlib.patches as patches

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

import xeofs

def regress_map(field, pc):
    """field: time, lat, lon; pc: time"""
    pc = (pc - pc.mean("time")) / pc.std("time")
    return xr.cov(field, pc, dim="time") / pc.var("time")


# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

CLIM = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_clim_sum.nc").sel(mmdd=slice("06-01", "08-31"))
ANO = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_ano_2015_MJJAS.nc").sel(time=slice("2015-06-01", "2015-08-31"))
BP = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_bp_2015_JJA_10-30d.nc")
budget_ds_ori = xr.open_dataset("/Volumes/TiPlus7100/p4/data/budget_ori_2015_JJA.nc")
budget_ds_clim = xr.open_dataset("/Volumes/TiPlus7100/p4/data/budget_clim_2015_JJA.nc")
adv  = budget_ds_ori['adv_T'].sel(level=925) - budget_ds_clim['adv_T'].sel(level=925)

def region_mean_series(da, shp_path):
    vals = []
    for i in range(da.sizes['time']):
        da_clip = masked(da.isel(time=i), shp_path)
        vals.append(da_clip.mean(dim=('lat', 'lon'), skipna=True))
    return xr.concat(vals, dim='time').assign_coords(time=da['time'])
adv_yz_925 = region_mean_series(adv, fr'{PYFILE}/map/self/长江_TP/长江_tp.shp')*86400

adv_yz_bp = LanczosFilter(adv_yz_925, 'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice("2015-06-01", "2015-08-31"))
u_bp = BP['u_bp']
v_bp = BP['v_bp']
z_bp = BP['z_bp']
t_bp = BP['t_bp']
w_bp = BP['w_bp']
olr_bp = BP['olr_bp']
t2m_bp = BP['t2m_bp']

u_clim = CLIM['u_clim'].mean(dim='mmdd')
v_clim = CLIM['v_clim'].mean(dim='mmdd')
u_ano = ANO['u_ano'].mean(dim='time')
v_ano = ANO['v_ano'].mean(dim='time')

t2m_bp_all = t2m_bp
t2m_bp = masked(t2m_bp, fr"{PYFILE}/map/self/长江_TP/长江_tp.shp") # 取反位向


# =========================
# Lead-lag regression: fixed T2m filtered index, lag = -8 ... +8
# =========================

lags = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
proj = ccrs.PlateCarree(central_longitude=110)
data_crs = ccrs.PlateCarree(central_longitude=0)
# 确保 6-9 月
t2m_bp_all = t2m_bp_all.sel(time=slice("2015-06-01", "2015-09-30"))
t2m_bp_69 = t2m_bp.sel(time=slice("2015-06-01", "2015-09-30"))
u_bp_69   = u_bp.sel(time=t2m_bp_69.time)
v_bp_69   = v_bp.sel(time=t2m_bp_69.time)
z_bp_69   = z_bp.sel(time=t2m_bp_69.time)
olr_bp_69 = olr_bp.sel(time=t2m_bp_69.time)

# t2m 滤波序列不变：这里用长江-TP区域平均作为固定指数
t2m_index = t2m_bp_69.mean(dim=("lat", "lon"), skipna=True)*-1
t2m_index = (t2m_index - t2m_index.mean("time")) / t2m_index.std("time")


def lag_regress(field, index, lag):
    """
    lag > 0: 要素场滞后 t2m index lag 天
    lag < 0: 要素场超前 t2m index |lag| 天
    """
    f_lag = field.shift(time=-lag)
    valid = np.isfinite(index) & np.isfinite(f_lag.mean(
        dim=[d for d in f_lag.dims if d != "time"], skipna=True
    ))

    idx = index.where(valid, drop=True)
    fld = f_lag.where(valid, drop=True)

    idx = (idx - idx.mean("time")) / idx.std("time")
    return xr.cov(fld, idx, dim="time") / idx.var("time")

def add_base_map(ax, extent):
    ax.set_extent(extent, crs=data_crs)

    ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                   linewidth=1.2, color="#BBBBBB")

    ax.add_geometries(
        Reader(fr"{PYFILE}/map/self/长江_TP/长江_tp.shp").geometries(),
        data_crs, facecolor="none", edgecolor="black", linewidth=0.4
    )

    ax.add_geometries(
        Reader(fr"{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp").geometries(),
        data_crs, facecolor="gray", edgecolor="black", linewidth=0.3
    )

    ax.set_xticks([], crs=data_crs)
    ax.set_yticks([], crs=data_crs)



clevs_z200 = np.array([-200, -100, -50, 50, 100, 200])
clevs_t2m  = np.array([-1.5, -1.2, -0.9, -0.6, -0.3, 0.3, 0.6, 0.9, 1.2, 1.5])
clevs_olr  = np.array([-20, -15, -10, -5, 5, 10, 15, 20])

cmap_z   = cmaps.MPL_RdBu_r[35:64] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.MPL_RdBu_r[64:-35]
cmap_t2m = cmaps.GMT_polar[3:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-3]
cmap_olr = cmaps.MPL_BrBG_r[:64] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.MPL_BrBG_r[64:]

fig = plt.figure(figsize=(18, 8.8))

outer = gridspec.GridSpec(
    2, 6,
    height_ratios=[1, 1],
    hspace=0.1,
    wspace=0
)

big_slots = []

# 第一行：4列，占第 1,2,3,4 列
for col in [1, 2, 3, 4]:
    big_slots.append(outer[0, col])

# 第二行：5列，占第 0,1,2,3,4 列
for col in [0, 1, 2, 3, 4]:
    big_slots.append(outer[1, col])


cf_z200_last = None
cf_t2m_last = None
cf_olr_last = None

for n, lag in enumerate(lags):

    inner = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=big_slots[n],
        hspace=0
    )

    # ---------- lag regression ----------
    z200 = transform(lag_regress(z_bp_69.sel(level=200), t2m_index, lag), "lon", "360->180")
    u200 = transform(lag_regress(u_bp_69.sel(level=200), t2m_index, lag), "lon", "360->180")
    v200 = transform(lag_regress(v_bp_69.sel(level=200), t2m_index, lag), "lon", "360->180")

    z500 = transform(lag_regress(z_bp_69.sel(level=500), t2m_index, lag), "lon", "360->180")
    u500 = transform(lag_regress(u_bp_69.sel(level=500), t2m_index, lag), "lon", "360->180")
    v500 = transform(lag_regress(v_bp_69.sel(level=500), t2m_index, lag), "lon", "360->180")
    t2m  = transform(lag_regress(t2m_bp_all, t2m_index, lag), "lon", "360->180")

    z850 = transform(lag_regress(z_bp_69.sel(level=850), t2m_index, lag), "lon", "360->180")
    u850 = transform(lag_regress(u_bp_69.sel(level=850), t2m_index, lag), "lon", "360->180")
    v850 = transform(lag_regress(v_bp_69.sel(level=850), t2m_index, lag), "lon", "360->180")
    olr  = transform(lag_regress(olr_bp_69, t2m_index, lag), "lon", "360->180")
    adv = lag_regress(adv_yz_bp, t2m_index, lag)

    # ================= 200 hPa: UVZ, Z填色 =================
    ax1 = fig.add_subplot(inner[0], projection=proj)
    ax1.set_aspect('auto')
    add_base_map(ax1, [-180, 180, -15, 80])

    WAF = TN_WAF_3D(
        u_clim.sel(level=200),
        v_clim.sel(level=200),
        z200,
        single_level=200
    )

    waf_x, waf_y = WAF

    cf_z200_last = ax1.contourf(
        z200.lon, z200.lat, z200,
        levels=clevs_z200,
        cmap=cmap_z,
        extend="both",
        transform=data_crs
    )

    q = ax1.Curlyquiver(
        u200.lon, u200.lat, waf_x, waf_y,
        arrowsize=1,
        transform=data_crs,
        scale=10,
        linewidth=0.8,
        regrid=13,
        color="purple",
        thinning=["50%", "min"],
        nanmax=2,
        MinDistance=[0.2, 0.4]
    )

    ax1.text(
        0.02, 0.94,
        f"{lag:+d} d" if lag > 0 else f"{lag:d} d",
        transform=ax1.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
    )

    ax1.text(
        0.9, 0.94,
        f"{adv:.2f}",
        transform=ax1.transAxes,
        fontsize=11,
        fontweight="bold",
        color='blue' if adv <= 0 else 'red',
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
    )

    ax1.set_title("", loc="left", fontsize=9)

    # ================= 500 hPa: UVZ & T2m, T2m填色, Z等值线无clabel =================
    ax2 = fig.add_subplot(inner[1], projection=proj)
    ax2.set_aspect('auto')
    add_base_map(ax2, [-180, 180, -15, 80])

    cf_t2m_last = ax2.contourf(
        t2m.lon, t2m.lat, t2m,
        levels=clevs_t2m,
        cmap=cmap_t2m,
        extend="both",
        transform=data_crs
    )

    ax2.contour(
        z500.lon, z500.lat, z500,
        levels=np.array([-200]),
        colors="blue",
        linewidths=1,
        transform=data_crs
    )

    ax2.contour(
        z500.lon, z500.lat, z500,
        levels=np.array([200]),
        colors="red",
        linewidths=1,
        transform=data_crs
    )

    q = ax2.Curlyquiver(
        u500.lon, u500.lat, u500, v500,
        arrowsize=1,
        transform=data_crs,
        scale=3,
        linewidth=0.8,
        regrid=13,
        color="#444444",
        thinning=["40%", "min"],
        nanmax=2,
        MinDistance=[0.2, 0.4]
    )

    ax2.set_title("", loc="left", fontsize=9)

    # ================= 850 hPa: UVZ & OLR, OLR填色, Z等值线无clabel =================
    ax3 = fig.add_subplot(inner[2], projection=proj)
    ax3.set_aspect('auto')
    add_base_map(ax3, [-180, 180, -15, 80])

    cf_olr_last = ax3.contourf(
        olr.lon, olr.lat, olr,
        levels=clevs_olr,
        cmap=cmap_olr,
        extend="both",
        transform=data_crs
    )

    ax3.contour(
        z850.lon, z850.lat, z850,
        levels=np.array([-100]),
        colors="blue",
        linewidths=1,
        transform=data_crs
    )

    ax3.contour(
        z850.lon, z850.lat, z850,
        levels=np.array([100]),
        colors="red",
        linewidths=1,
        transform=data_crs
    )

    q = ax3.Curlyquiver(
        u850.lon, u850.lat, u850, v850,
        arrowsize=1,
        transform=data_crs,
        scale=8,
        linewidth=0.8,
        regrid=13,
        color="#444444",
        thinning=["40%", "min"],
        nanmax=2,
        MinDistance=[0.2, 0.4]
    )

    ax3.set_title("", loc="left", fontsize=9)

# ================= 三个 colorbar：底部从左到右 =================
cax1 = fig.add_axes([0.12-0.065, 0.045, 0.22, 0.018])
cb1 = fig.colorbar(cf_z200_last, cax=cax1, orientation="horizontal")
cb1.set_label("Z200 regression", fontsize=10)
cb1.ax.tick_params(labelsize=8, length=0)

cax2 = fig.add_axes([0.39-0.065, 0.045, 0.22, 0.018])
cb2 = fig.colorbar(cf_t2m_last, cax=cax2, orientation="horizontal")
cb2.set_label("T2m regression", fontsize=10)
cb2.ax.tick_params(labelsize=8, length=0)

cax3 = fig.add_axes([0.66-0.065, 0.045, 0.22, 0.018])
cb3 = fig.colorbar(cf_olr_last, cax=cax3, orientation="horizontal")
cb3.set_label("OLR regression", fontsize=10)
cb3.ax.tick_params(labelsize=8, length=0)

# 边框
for ax_ in fig.axes:
    for spine in ax_.spines.values():
        spine.set_linewidth(2)

plt.subplots_adjust(bottom=0.1, top=0.96, left=0.04, right=0.98)

plt.savefig(fr"{PYFILE}/p4/pic/Leadlag_UVZ_T2m_OLR_6-9.png",
            bbox_inches="tight", dpi=600)
plt.savefig(fr"{PYFILE}/p4/pic/Leadlag_UVZ_T2m_OLR_6-9.pdf",
            bbox_inches="tight")
plt.show()

