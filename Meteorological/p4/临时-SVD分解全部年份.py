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
from scipy.stats import t as student_t
import matplotlib.path as mpath
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from climkit.Cquiver import *
from climkit.TN_WaveActivityFlux import TN_WAF_3D
from climkit.masked import masked
from climkit.significance_test import r_test
from climkit.lonlat_transform import *
from climkit.filter import *
from climkit.corr_reg import *

from metpy.calc import vertical_velocity
from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants

import xeofs


# =========================
# Basic settings
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

TIME_SLICE = slice("1961-01-01", "2022-12-31")
LEVELS_USE = [200, 850]
LAT_SLICE = slice(-20, 80)
REGION_180 = {"lon": slice(-90, 180), "lat": slice(-20, 80)}


# =========================
# Utility functions
# =========================
def regress_map(field, pc):
    """field: time, lat, lon; pc: time"""
    pc = (pc - pc.mean("time")) / pc.std("time")
    return xr.cov(field, pc, dim="time") / pc.var("time")


def select_jja_1961_2022(da):
    """
    Select 1961-2022 and keep only Jun 15 - Jul 31.
    """
    da = da.sel(time=TIME_SLICE)

    if "time" in da.coords and np.issubdtype(da.time.dtype, np.datetime64):
        month = da.time.dt.month
        day = da.time.dt.day
        in_window = ((month == 7) & (day >= 1)) | ((month == 7) & (day <= 31))
        da = da.where(in_window, drop=True)

    return da


def align_on_common_time(*arrays):
    """
    Align multiple xarray objects only on their common time coordinate.
    This avoids accidental alignment on lat/lon/level.
    """
    common_time = arrays[0].time.values

    for da in arrays[1:]:
        common_time = np.intersect1d(common_time, da.time.values)

    common_time = np.sort(common_time)

    return tuple(da.sel(time=common_time) for da in arrays)


def build_compact_summer_axis(time_values, tick_step=10):
    """
    Compress all JJA days into one continuous axis.

    Example:
    1961-06-01 ... 1961-08-31, 1962-06-01 ... 1962-08-31
    will be displayed as adjacent points without gaps from Sep-May.
    """
    dates = pd.to_datetime(time_values)
    xpos = np.arange(len(dates))

    years = dates.year.values
    unique_years = np.unique(years)

    year_starts = []
    year_centers = []

    for yy in unique_years:
        idx = np.where(years == yy)[0]
        year_starts.append(idx[0])
        year_centers.append((idx[0] + idx[-1]) / 2)

    year_starts = np.asarray(year_starts)
    year_centers = np.asarray(year_centers)

    tick_years = unique_years[::tick_step]
    tick_pos = []

    for yy in tick_years:
        j = np.where(unique_years == yy)[0][0]
        tick_pos.append(year_centers[j])

    tick_pos = np.asarray(tick_pos)

    return xpos, unique_years, year_starts, year_centers, tick_pos, tick_years


def effective_sample_size(x, y):
    """
    Estimate effective sample size using lag-1 autocorrelation:

    neff = n * (1 - r1x*r1y) / (1 + r1x*r1y)

    This is commonly used for correlation significance correction
    when serial autocorrelation exists.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)

    if n <= 3:
        return np.nan

    if n < 5:
        return float(n)

    if np.nanstd(x[:-1]) == 0 or np.nanstd(x[1:]) == 0:
        r1x = 0.0
    else:
        r1x = pearsonr(x[:-1], x[1:])[0]

    if np.nanstd(y[:-1]) == 0 or np.nanstd(y[1:]) == 0:
        r1y = 0.0
    else:
        r1y = pearsonr(y[:-1], y[1:])[0]

    if not np.isfinite(r1x):
        r1x = 0.0

    if not np.isfinite(r1y):
        r1y = 0.0

    denom = 1 + r1x * r1y

    if np.abs(denom) < 1e-6:
        neff = n
    else:
        neff = n * (1 - r1x * r1y) / denom

    neff = np.clip(neff, 3, n)

    return float(neff)


def corr_threshold_95(neff, alpha=0.05):
    """
    Two-sided 95% significance threshold for correlation coefficient.
    """
    if (not np.isfinite(neff)) or neff <= 2:
        return np.nan

    df = neff - 2
    tcrit = student_t.ppf(1 - alpha / 2, df)
    rcrit = np.sqrt(tcrit**2 / (df + tcrit**2))

    return rcrit


def _as_1d_dataarray(x):
    """
    Convert input to 1D DataArray while preserving xarray time coordinate if possible.
    """
    if isinstance(x, xr.DataArray):
        return x

    return xr.DataArray(np.asarray(x), dims=["time"])


def lag_corr(x, y=None, max_lag=30, use_neff=True, within_same_summer=True):
    """
    lag > 0: x leads y
    lag < 0: x lags y

    Returns:
    lags, r, r95

    r95 is calculated separately for each lag.

    within_same_summer=True:
    only pairs within the same year are used. This prevents artificial correlation
    between Aug 31 of one year and Jun 1 of the next year.
    """
    xda = _as_1d_dataarray(x)

    if y is None:
        yda = xda.copy()
    else:
        yda = _as_1d_dataarray(y)

    if "time" in xda.coords and "time" in yda.coords:
        xda, yda = xr.align(xda, yda, join="inner")

    xval = xda.values.astype(float)
    yval = yda.values.astype(float)

    xval = (xval - np.nanmean(xval)) / np.nanstd(xval)
    yval = (yval - np.nanmean(yval)) / np.nanstd(yval)

    has_datetime = (
        within_same_summer
        and "time" in xda.coords
        and np.issubdtype(xda.time.dtype, np.datetime64)
    )

    if has_datetime:
        years = pd.to_datetime(xda.time.values).year.values
        unique_years = np.unique(years)
    else:
        years = None
        unique_years = None

    lags = np.arange(-max_lag, max_lag + 1)
    rs = []
    r95s = []

    for lag in lags:
        xx_list = []
        yy_list = []

        if has_datetime:
            for yy in unique_years:
                idx = np.where(years == yy)[0]

                xv = xval[idx]
                yv = yval[idx]

                if len(xv) <= abs(lag):
                    continue

                if lag < 0:
                    xx = xv[-lag:]
                    yyv = yv[:lag]
                elif lag > 0:
                    xx = xv[:-lag]
                    yyv = yv[lag:]
                else:
                    xx = xv
                    yyv = yv

                xx_list.append(xx)
                yy_list.append(yyv)

            if len(xx_list) == 0:
                rs.append(np.nan)
                r95s.append(np.nan)
                continue

            xx_all = np.concatenate(xx_list)
            yy_all = np.concatenate(yy_list)

        else:
            if lag < 0:
                xx_all = xval[-lag:]
                yy_all = yval[:lag]
            elif lag > 0:
                xx_all = xval[:-lag]
                yy_all = yval[lag:]
            else:
                xx_all = xval
                yy_all = yval

        mask = np.isfinite(xx_all) & np.isfinite(yy_all)

        if mask.sum() > 3:
            xx_use = xx_all[mask]
            yy_use = yy_all[mask]

            r = pearsonr(xx_use, yy_use)[0]

            if use_neff:
                neff = effective_sample_size(xx_use, yy_use)
            else:
                neff = len(xx_use)

            r95 = corr_threshold_95(neff)

        else:
            r = np.nan
            r95 = np.nan

        rs.append(r)
        r95s.append(r95)

    return lags, np.asarray(rs), np.asarray(r95s)


def open_var(path, var_name, *, time_slice=None, level=None, lat_slice=None, lon_slice=None,
             scale=1.0, dtype="float32"):
    """Open and subset a DataArray early to reduce peak memory."""
    da = xr.open_dataset(path)[var_name]

    if time_slice is not None and "time" in da.coords:
        da = da.sel(time=time_slice)

    if level is not None and "level" in da.coords:
        da = da.sel(level=level)

    if lat_slice is not None and "lat" in da.coords:
        da = da.sel(lat=lat_slice)

    if lon_slice is not None and "lon" in da.coords:
        da = da.sel(lon=lon_slice)

    if scale != 1.0:
        da = da * scale

    if dtype:
        da = da.astype(dtype)

    return da


# =========================
# Read data: 1961-2022 JJA
# =========================
CLIM = xr.open_dataset(
    "/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_clim_1991_2020.nc"
).sel(mmdd=slice("06-01", "08-31"))

u_clim = CLIM["u_clim"].sel(level=LEVELS_USE).mean(dim="mmdd").astype("float32")
v_clim = CLIM["v_clim"].sel(level=LEVELS_USE).mean(dim="mmdd").astype("float32")

u_clim = transform(u_clim, "lon", "360->180").sel(**REGION_180)
v_clim = transform(v_clim, "lon", "360->180").sel(**REGION_180)

u_bp = open_var(
    "/Volumes/TiPlus7100/p4/data/u_bp_JJA_10-30d.nc",
    "u_bp",
    time_slice=TIME_SLICE,
    level=LEVELS_USE,
    lat_slice=LAT_SLICE
)

v_bp = open_var(
    "/Volumes/TiPlus7100/p4/data/v_bp_JJA_10-30d.nc",
    "v_bp",
    time_slice=TIME_SLICE,
    level=LEVELS_USE,
    lat_slice=LAT_SLICE
)

z_bp = open_var(
    "/Volumes/TiPlus7100/p4/data/z_bp_JJA_10-30d.nc",
    "z_bp",
    time_slice=TIME_SLICE,
    level=LEVELS_USE,
    lat_slice=LAT_SLICE
)

olr_bp = open_var(
    "/Volumes/TiPlus7100/p4/data/ttr_bp_JJA_10-30d.nc",
    "ttr_bp",
    time_slice=TIME_SLICE,
    lat_slice=slice(-20, 30),
    lon_slice=slice(40, 180),
    scale=1 / 86400
)

t2m_bp = open_var(
    "/Volumes/TiPlus7100/p4/data/t2m_bp_JJA_10-30d.nc",
    "t2m_bp",
    time_slice=TIME_SLICE
)

# TIME_SLICE = u_bp.time.dt.year.isin([1965, 1966, 1968, 1974, 1976, 1980, 1982, 1983, 1986, 1987, 1989, 1992, 1993, 1997, 1999, 2004, 2008, 2014, 2015])

u_bp = select_jja_1961_2022(u_bp)
v_bp = select_jja_1961_2022(v_bp)
z_bp = select_jja_1961_2022(z_bp)
olr_bp = select_jja_1961_2022(olr_bp)
t2m_bp = select_jja_1961_2022(t2m_bp)

u_bp, v_bp, z_bp, olr_bp, t2m_bp = align_on_common_time(
    u_bp, v_bp, z_bp, olr_bp, t2m_bp
)

t2m_bp = masked(
    t2m_bp,
    fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"
)


# =========================
# MCA / SVD
# =========================
svd_t2m_v200 = xeofs.cross.MCA(
    n_modes=4,
    standardize=False,
    use_coslat=True
).fit(
    t2m_bp,
    transform(v_bp, "lon", "360->180").sel(level=200, lon=slice(40, 130), lat=slice(36, 65)),
    dim="time"
)

svd_t2m_olr = xeofs.cross.MCA(
    n_modes=4,
    standardize=False,
    use_coslat=True
).fit(
    t2m_bp,
    transform(olr_bp, "lon", "360->180").sel(lon=slice(90, 180), lat=slice(-10, 24)),
    dim="time"
)


# =========================
# Pre-transform longitude once
# =========================
z_bp_180 = transform(z_bp, "lon", "360->180").sel(**REGION_180)
u_bp_180 = transform(u_bp, "lon", "360->180").sel(**REGION_180)
v_bp_180 = transform(v_bp, "lon", "360->180").sel(**REGION_180)
olr_bp_180 = transform(olr_bp, "lon", "360->180")


# =========================
# Draw figure
# =========================
proj = ccrs.PlateCarree(central_longitude=0)

fig = plt.figure(figsize=(18, 6.5))

gs = gridspec.GridSpec(
    nrows=2,
    ncols=3,
    width_ratios=[1.05, 1.05, 0.8],
    height_ratios=[1, 1],
    wspace=0.25,
    hspace=0.3
)

clevs_t2m = np.array([
    -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4
]) * 5

cmap_t2m = cmaps.GMT_polar[2:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-2]

# 右场模态填色使用另一套发散色表，并设透明度，避免完全盖住 T2m 模态
cmap_right = (
    cmaps.MPL_PuOr[10:64-10:5]
    + cmaps.CBR_wet[0]
    + cmaps.CBR_wet[0]
    + cmaps.CBR_wet[0]
    + cmaps.MPL_PuOr[64+10:-10:5]
)

# 右场模态共用 levels，后面用于共享竖直 colorbar
right_levels = np.array([
    -0.4, -0.3, -0.2, -0.1, -0.05,
     0.05, 0.1, 0.2, 0.3, 0.4
])

cf_last_t2m = None


def draw_one(row, level, svd_obj, left_score_name, right_score_name,
             map_extent, box_blue, box_orange, title_left, title_right,
             vector_label, score_label):

    global cf_last_t2m

    # ================= MCA mode and score =================
    if level <= 500:
        t2m_mode = svd_obj.components()[0].sel(mode=1)
        right_mode = svd_obj.components()[1].sel(mode=1)

        atm_score = svd_obj.scores()[1].sel(mode=1)
        t2m_score = svd_obj.scores()[0].sel(mode=1)

        # 保持你原来 200 hPa 的符号和幅度设置
        SIGN = 1 if t2m_mode.mean() < 0 else -1
        t2m_mode = t2m_mode * 10 * SIGN
        right_mode = right_mode * 10 * SIGN

        t2m_score = t2m_score / 10 * SIGN
        atm_score = atm_score / 10 * SIGN
        corvar = float(svd_obj.squared_covariance_fraction().sel(mode=1) * 100)

    elif level == 850:
        t2m_mode = svd_obj.components()[0].sel(mode=1)
        right_mode = svd_obj.components()[1].sel(mode=1)

        atm_score = svd_obj.scores()[1].sel(mode=1)
        t2m_score = svd_obj.scores()[0].sel(mode=1)
        corvar = float(svd_obj.squared_covariance_fraction().sel(mode=1) * 100)

        # 保持你原来 850 hPa / OLR 的符号和幅度设置
        SIGN = 1 if t2m_mode.mean() < 0 else -1
        t2m_mode = t2m_mode * 10 * SIGN
        right_mode = right_mode * 10 * SIGN

        t2m_score = t2m_score / 10 * SIGN
        atm_score = atm_score / 10 * SIGN

    else:
        raise ValueError("Only level <= 500 or level == 850 is supported.")

    t2m_score_raw = t2m_score
    atm_score_raw = atm_score

    # 右场模态只画在右场 MCA 分析区域 / 当前地图范围内
    right_mode = right_mode.where(
        (right_mode.lon >= map_extent[0]) & (right_mode.lon <= map_extent[1]) &
        (right_mode.lat >= map_extent[2]) & (right_mode.lat <= map_extent[3])
    )

    # ================= regression fields =================
    z_reg = regress(atm_score, z_bp_180.sel(level=level))
    u_reg = regress(atm_score, u_bp_180.sel(level=level))
    v_reg = regress(atm_score, v_bp_180.sel(level=level))

    if level <= 500:
        olr_reg = regress(atm_score, olr_bp_180)
        olr_reg = olr_reg.where((z_reg.lat >= -15) & (z_reg.lat <= 15))

        WAF = TN_WAF_3D(
            u_clim.sel(level=level),
            v_clim.sel(level=level),
            z_reg,
            single_level=200
        )

        waf_x, waf_y = WAF

        waf_x = waf_x.where(
            (u_reg.lon >= map_extent[0]) & (u_reg.lon <= map_extent[1]) &
            (u_reg.lat >= map_extent[2]) & (u_reg.lat <= map_extent[3])
        )

        waf_y = waf_y.where(
            (u_reg.lon >= map_extent[0]) & (u_reg.lon <= map_extent[1]) &
            (u_reg.lat >= map_extent[2]) & (u_reg.lat <= map_extent[3])
        )

    z_reg = z_reg.where(
        (z_reg.lon >= map_extent[0]) & (z_reg.lon <= map_extent[1]) &
        (z_reg.lat >= map_extent[2]) & (z_reg.lat <= map_extent[3])
    )

    u_reg = u_reg.where(
        (u_reg.lon >= map_extent[0]) & (u_reg.lon <= map_extent[1]) &
        (u_reg.lat >= map_extent[2]) & (u_reg.lat <= map_extent[3])
    )

    v_reg = v_reg.where(
        (v_reg.lon >= map_extent[0]) & (v_reg.lon <= map_extent[1]) &
        (v_reg.lat >= map_extent[2]) & (v_reg.lat <= map_extent[3])
    )

    # ================= left map =================
    ax = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree(central_longitude=110))
    ax.set_aspect("auto")

    if level == 200:
        extent = [10, 160, 10, 80]
        extent_crs = ccrs.PlateCarree(central_longitude=0)
    else:
        extent = [20, 200, -20, 50]
        extent_crs = ccrs.PlateCarree(central_longitude=0)

    ax.set_extent(extent, crs=extent_crs)

    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=1.5,
        color="#BBBBBB"
    )

    ax.add_geometries(
        Reader(fr"{PYFILE}/map/self/长江_TP/长江_tp.shp").geometries(),
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=0.5
    )

    ax.add_geometries(
        Reader(fr"{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp").geometries(),
        ccrs.PlateCarree(),
        facecolor="gray",
        edgecolor="black",
        linewidth=0.5
    )

    if level >= 600:
        ax.add_geometries(
            Reader(f"{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp").geometries(),
            ccrs.PlateCarree(),
            facecolor="#909090",
            edgecolor="#909090",
            linewidth=0,
            hatch=".",
            zorder=10
        )

    if level <= 500:
        ax.set_xticks(np.arange(extent[0], extent[1], 40), crs=proj)
        ax.set_yticks(np.arange(extent[2], extent[3] + 1, 15), crs=proj)
    elif level == 850:
        ax.set_xticks(np.arange(extent[0], extent[1], 30), crs=proj)
        ax.set_yticks(np.arange(extent[2], extent[3] + 1, 10), crs=proj)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=12)

    # ---- 左场 T2m 模态填色 ----
    cf_t2m = ax.contourf(
        t2m_mode.lon,
        t2m_mode.lat,
        t2m_mode,
        levels=clevs_t2m,
        cmap=cmap_t2m,
        extend="both",
        transform=proj,
        zorder=1
    )

    cf_last_t2m = cf_t2m

    # ---- 右场模态填色：row=0 为 V200 mode，row=1 为 OLR mode ----
    cf_right = ax.contourf(
        right_mode.lon,
        right_mode.lat,
        right_mode,
        levels=right_levels,
        cmap=cmap_right,
        extend="both",
        alpha=0.62,
        transform=proj,
        zorder=2
    )

    # 注意：这里已经删除图内右场 colorbar，改为外部共享竖直 colorbar

    # ---- Z regression contours ----
    zlev_pos = [6, 18] if level == 200 else [17]
    zlev_neg = [-18, -6] if level == 200 else [-30]

    ax.contour(
        z_reg.lon,
        z_reg.lat,
        z_reg,
        levels=zlev_pos,
        colors="red",
        linewidths=1.2,
        transform=proj,
        zorder=4
    )

    ax.contour(
        z_reg.lon,
        z_reg.lat,
        z_reg,
        levels=zlev_neg,
        colors="blue",
        linewidths=1.2,
        linestyles="--",
        transform=proj,
        zorder=4
    )

    z_reg.to_netcdf(fr"/Volumes/TiPlus7100/p4/data/SVD{level}V_{level}Z.nc")

    # ---- Wind vectors and WAF ----
    if level <= 500:
        q = ax.Curlyquiver(
            u_reg.lon,
            u_reg.lat,
            u_reg,
            v_reg,
            arrowsize=1,
            transform=ccrs.PlateCarree(central_longitude=0),
            scale=20,
            linewidth=1,
            regrid=18,
            color="#555555",
            thinning=["50%", "min"],
            nanmax=2,
            MinDistance=[0.2, 0.4]
        )

        q.key(
            U=0.5,
            label=r"0.5 $m/s$",
            fontproperties={"size": 8},
            facecolor="none",
            bbox_to_anchor=(0, 0.14, 1, 1),
            edgecolor="none",
            arrowsize=1,
            linewidth=1,
            intetval=0.8
        )

        q_waf = ax.Curlyquiver(
            u_reg.lon,
            u_reg.lat,
            waf_x,
            waf_y,
            arrowsize=1.5,
            transform=ccrs.PlateCarree(central_longitude=0),
            scale=1e3,
            linewidth=2,
            regrid=12,
            color="purple",
            thinning=["70%", "min"],
            nanmax=2,
            MinDistance=[0.2, 0.4]
        )

        q_waf.key(
            U=0.02,
            label=r"0.02 ${m}^{2}/{s}^{2}$",
            fontproperties={"size": 8},
            facecolor="none",
            bbox_to_anchor=(-0.15, 0.14, 1, 1),
            edgecolor="none",
            arrowsize=0.01,
            linewidth=1,
            intetval=0.8
        )

    elif level == 850:
        q = ax.Curlyquiver(
            u_reg.lon,
            u_reg.lat,
            u_reg,
            v_reg,
            arrowsize=1,
            transform=ccrs.PlateCarree(central_longitude=0),
            scale=10,
            linewidth=1,
            regrid=18,
            color="#555555",
            thinning=["25%", "min"],
            nanmax=2,
            MinDistance=[0.2, 0.4]
        )

        q.key(
            U=0.5,
            label=r"0.5 $m/s$",
            fontproperties={"size": 8},
            facecolor="none",
            bbox_to_anchor=(0, 0.14, 1, 1),
            edgecolor="none",
            arrowsize=1,
            linewidth=1,
            intetval=0.8
        )

    ax.add_patch(
        patches.Rectangle(
            (box_blue[0], box_blue[2]),
            box_blue[1] - box_blue[0],
            box_blue[3] - box_blue[2],
            linewidth=1.8,
            edgecolor="royalblue",
            facecolor="none",
            linestyle="--",
            transform=proj,
            zorder=5
        )
    )

    ax.set_title(title_left, loc="left", fontsize=16)

    # ================= right time series =================
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.set_aspect("auto")

    t2m_std = t2m_score.groupby("time.year").std("time", skipna=True)
    atm_std = atm_score.groupby("time.year").std("time", skipna=True)
    t2m_std, atm_std = xr.align(t2m_std, atm_std, join="inner")

    years = t2m_std["year"].values

    ax2.bar(
        years,
        t2m_std.values,
        color="red",
        alpha=0.7,
        width=0.6,
        label="T2m",
        zorder=0
    )

    ax2.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax2.axhline(0, color="0.75", lw=0.7)

    xticks = years[::10] if len(years) > 10 else years
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, fontsize=10)

    ax2.tick_params(labelsize=12)
    ax2.set_ylabel("", fontsize=11, color="red")
    ax2.set_ylim(0, 2)
    ax2.spines["left"].set_color("red")
    ax2.yaxis.label.set_color("red")

    ax2b = ax2.twinx()

    ax2b.bar(
        years,
        atm_std.values,
        facecolor="none",
        edgecolor="black",
        alpha=1.,
        width=0.6,
        label=f"{score_label}",
        zorder=1000
    )

    if 2015 in years:
        i2015 = int(np.where(years == 2015)[0][0])
        ax2b.scatter(
            years[i2015],
            atm_std.values[i2015] + 3
            if atm_std.values[i2015] > 2
            else atm_std.values[i2015] + .8,
            color="blue",
            s=90,
            marker="*",
            zorder=1200
        )

    ax2b.tick_params(labelsize=12)
    ax2b.set_ylabel("", fontsize=11)
    ax2b.spines["left"].set_color("red")
    ax2b.set_ylim(0, 20 if row == 0 else 2)
    ax2b.set_xlim(years[0] - 0.5, years[-1] + 0.5)

    handles_a, labels_a = ax2.get_legend_handles_labels()
    handles_b, labels_b = ax2b.get_legend_handles_labels()

    ax2.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(-0.01, 1.03),
        loc="upper left"
    )


    t2m_score_raw, atm_score_raw = xr.align(t2m_score_raw, atm_score_raw, join="inner")
    score_mask = np.isfinite(t2m_score_raw.values) & np.isfinite(atm_score_raw.values)

    if score_mask.sum() > 3:
        r, p = pearsonr(t2m_score_raw.values[score_mask], atm_score_raw.values[score_mask])
    else:
        r, p = np.nan, np.nan

    if np.isfinite(p):
        ptxt = "p<0.01" if p < 0.01 else f"p={p:.2f}"
    else:
        ptxt = "p=nan"

    ax2.text(
        0.43,
        0.92,
        f"CorVar={corvar:.0f}%",
        transform=ax2.transAxes,
        fontsize=12
    )

    ax2.text(
        0.75,
        0.92,
        f"r={r:.2f} ({ptxt})",
        transform=ax2.transAxes,
        fontsize=12
    )

    ax2.set_title(title_right, loc="left", fontsize=16)

    return ax, ax2, t2m_score_raw, atm_score_raw, cf_right


# ================= first row: 200 hPa =================
ax200, ax200_ts, t2m200_score, v200_score, cf_v200_mode = draw_one(
    row=0,
    level=200,
    svd_obj=svd_t2m_v200,
    left_score_name="T2m",
    right_score_name="V200",
    map_extent=[10, 160, 20, 80],
    box_blue=[40, 130, 36, 65],
    box_orange=[-70, 0, -20, 45],
    title_left="(a) SVD (T2m&V200)",
    title_right="(b) Annual STD",
    vector_label="1.5 m/s",
    score_label="V200"
)


# ================= second row: 850 hPa / OLR score =================
ax850, ax850_ts, t2m850_score, olr_score, cf_olr_mode = draw_one(
    row=1,
    level=850,
    svd_obj=svd_t2m_olr,
    left_score_name="T2m",
    right_score_name="OLR",
    map_extent=[30, 190, -20, 50],
    box_blue=[120, 170, -10, 30],
    box_orange=[90, 122, 20, 35],
    title_left="(c) SVD (T2m&OLR)",
    title_right="(d) Annual STD",
    vector_label="1.5 m/s",
    score_label="OLR"
)


# ================= shared vertical colorbar for right-field modes =================
# 放在 (a)(c) 右侧，不写 title
fig.canvas.draw()

pos_a = ax200.get_position()
pos_c = ax850.get_position()
pos_b = ax200_ts.get_position()

map_right = max(pos_a.x1, pos_c.x1)
gap = pos_b.x0 - map_right

cbar_width = min(0.006, gap * 0.12)
cbar_x = map_right + gap * 0.08
cbar_y = pos_c.y0
cbar_h = pos_a.y1 - pos_c.y0

cax_right_shared = fig.add_axes([
    cbar_x,
    cbar_y,
    cbar_width,
    cbar_h
])

cb_right_shared = fig.colorbar(
    cf_v200_mode,
    cax=cax_right_shared,
    orientation="vertical",
    drawedges=True
)

cb_right_shared.set_ticks(right_levels)
cb_right_shared.ax.tick_params(length=0, labelsize=10, pad=2)
cb_right_shared.dividers.set_linewidth(2.5)
cb_right_shared.outline.set_linewidth(2.5)


# ================= third column: lag correlations =================
gs_right = gridspec.GridSpecFromSubplotSpec(
    3,
    1,
    subplot_spec=gs[:, 2],
    hspace=0.25
)

max_lag = 30

lag_v200, r_v200, sig_v200 = lag_corr(
    v200_score,
    max_lag=max_lag,
    use_neff=True,
    within_same_summer=True
)

lag_cross, r_cross, sig_cross = lag_corr(
    v200_score,
    olr_score,
    max_lag=max_lag,
    use_neff=True,
    within_same_summer=True
)

lag_olr, r_olr, sig_olr = lag_corr(
    olr_score,
    max_lag=max_lag,
    use_neff=True,
    within_same_summer=True
)

lag_panels = [
    (lag_v200, r_v200, sig_v200, "(e) Auto-V200"),
    (lag_cross, r_cross, sig_cross, "(f) V200&OLR"),
    (lag_olr, r_olr, sig_olr, "(g) Auto-OLR"),
]

for i, (lags, rs, rsig, title) in enumerate(lag_panels):
    ax_lag = fig.add_subplot(gs_right[i, 0])

    ax_lag.plot(lags, rs, color="black", lw=2.2)

    ax_lag.axhline(0, color="black", lw=1.0, ls="--", alpha=0.8)
    ax_lag.axvline(0, color="black", lw=1.0, ls="--", alpha=0.8)

    # 95% significance threshold recalculated for each lag
    ax_lag.plot(lags, rsig, color="red", lw=1.2, ls=(0, (8, 5)), alpha=0.8)
    ax_lag.plot(lags, -rsig, color="red", lw=1.2, ls=(0, (8, 5)), alpha=0.8)

    ax_lag.set_xlim(-max_lag, max_lag)

    max_y = np.nanmax(np.abs(rs))
    if (not np.isfinite(max_y)) or max_y == 0:
        max_y = 0.1

    ax_lag.set_ylim(-(max_y + 0.1), (max_y + 0.1))

    ax_lag.set_xticks(np.arange(-30, 31, 10))

    ytick_min = -np.round((max_y + 0.1) * 5) / 5
    ytick_max = np.round((max_y + 0.1) * 5) / 5
    ax_lag.set_yticks(np.arange(ytick_min, ytick_max + 1e-9, 0.4))

    ax_lag.xaxis.set_minor_locator(MultipleLocator(2))
    ax_lag.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax_lag.tick_params(labelsize=11, direction="out", length=4, width=1.2)
    ax_lag.tick_params(which="minor", length=2, width=1)

    if i < 2:
        ax_lag.set_xticklabels([])

    ax_lag.set_title(title, loc="left", fontsize=16)

    for spine in ax_lag.spines.values():
        spine.set_linewidth(2.0)


# ================= T2m colorbar =================
cax = inset_axes(
    ax850,
    width="100%",
    height="5%",
    loc="lower left",
    bbox_to_anchor=(0, -0.18, 1, 1),
    bbox_transform=ax850.transAxes,
    borderpad=0
)

cb = fig.colorbar(
    cf_last_t2m,
    cax=cax,
    orientation="horizontal",
    drawedges=True
)

cb.set_ticks(clevs_t2m)
cb.ax.tick_params(length=0, labelsize=10)
cb.dividers.set_linewidth(2.5)
cb.outline.set_linewidth(2.5)
cb.ax.set_title("", fontsize=9, pad=3)


# ================= borders =================
for ax_ in fig.axes:
    for spine in ax_.spines.values():
        spine.set_linewidth(2.5)


# ================= save =================
plt.savefig(
    fr"{PYFILE}/p4/pic/SVD_UVZ_all_1961_2022.png",
    bbox_inches="tight",
    dpi=600
)

plt.savefig(
    fr"{PYFILE}/p4/pic/SVD_UVZ_all_1961_2022.pdf",
    bbox_inches="tight"
)

plt.show()
