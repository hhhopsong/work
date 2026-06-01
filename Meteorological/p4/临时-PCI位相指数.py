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
        in_window = ((month == 6) & (day >= 15)) | ((month == 7) & (day <= 31))
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
    lon_slice=slice(40, 160),
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

# ============================================================
# Step 3 | Annual STD indices, amplitude-limited cold/hot PCI,
#          and variance decomposition
#
# Modified:
# 1. I_mid(y) and I_trop(y) are annual STD indices.
# 2. Cold-favorable phase bins are diagnosed without amplitude threshold.
# 3. Hot-favorable phase bins are diagnosed without amplitude threshold.
# 4. Cold PCI includes amplitude restriction:
#    C(t)=1 only when both modes are in cold-favorable phase bins
#    and both amplitudes exceed their own 1991-2020 1-sigma thresholds.
# 5. Hot phase index counts days when both modes are in hot-favorable
#    phase bins and both amplitudes exceed their own 1991-2020
#    1-sigma thresholds.
# 6. Figure panel (c) shows:
#    filled gray/blue bars = cold coupled strong-amplitude days;
#    red outline bars = hot coupled strong-amplitude days.
# ============================================================

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.signal import hilbert
from scipy.sparse.linalg import LinearOperator, svds
from scipy.stats import t as student_t
from matplotlib.ticker import MultipleLocator


# =========================
# Step-3 settings
# =========================
STEP3_OUT = f"{PYFILE}/p4/pic/"
os.makedirs(STEP3_OUT, exist_ok=True)

TARGET_YEAR = 2015

N_SVD_MODES = 2
MODE_TO_USE = 1

MID_DOMAIN = {
    "lon": (-90, 180),
    "lat": (20, 80)
}

TROP_DOMAIN = {
    "lon": (40, 160),
    "lat": (-20, 30)
}

N_PHASE = 8

COLD_ALPHA = 0.10
HOT_ALPHA = 0.10

MIN_BIN_DAYS = 30
FALLBACK_COLD_BINS = 2
FALLBACK_HOT_BINS = 2

# Amplitude threshold reference period
AMP_REF_START = 1991
AMP_REF_END = 2020
AMP_SIGMA_MULT = 1.0

COMPUTE_SCF = True


# =========================
# SCI plotting style
# =========================
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix"
})


# =========================
# Basic utility functions
# =========================
def to_lon180(da):
    """Convert longitude from 0-360 to -180-180 if needed."""
    if "lon" not in da.coords:
        return da

    lon = da["lon"]

    if float(lon.max()) > 180:
        new_lon = ((lon + 180) % 360) - 180
        da = da.assign_coords(lon=new_lon).sortby("lon")

    return da


def sel_latlon_box(da, lon=None, lat=None):
    """Select lat-lon box regardless of coordinate ascending or descending order."""
    da = to_lon180(da)

    if lon is not None and "lon" in da.coords:
        lon1, lon2 = lon
        lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)

        if da.lon.values[0] <= da.lon.values[-1]:
            da = da.sel(lon=slice(lon_min, lon_max))
        else:
            da = da.sel(lon=slice(lon_max, lon_min))

    if lat is not None and "lat" in da.coords:
        lat1, lat2 = lat
        lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)

        if da.lat.values[0] <= da.lat.values[-1]:
            da = da.sel(lat=slice(lat_min, lat_max))
        else:
            da = da.sel(lat=slice(lat_max, lat_min))

    return da


def align_time_only(*arrays):
    """Align xarray objects only by common time coordinate."""
    common_time = arrays[0].time.values

    for da in arrays[1:]:
        common_time = np.intersect1d(common_time, da.time.values)

    common_time = np.sort(common_time)

    return tuple(da.sel(time=common_time) for da in arrays)


def area_weighted_mean(da):
    """Area-weighted spatial mean over lat-lon."""
    spatial_dims = [d for d in ["lat", "lon"] if d in da.dims]

    if "lat" in da.coords:
        w = np.cos(np.deg2rad(da["lat"]))
        w = xr.where(w > 0, w, 0)
        return da.weighted(w).mean(dim=spatial_dims, skipna=True)

    return da.mean(dim=spatial_dims, skipna=True)


def zscore_1d(da, dim):
    return (da - da.mean(dim, skipna=True)) / da.std(dim, skipna=True)


def percentile_rank(da, year=2015):
    """Percentile rank of a given year in annual distribution."""
    if year not in da["year"].values:
        return np.nan

    vals = da.values.astype(float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return np.nan

    target = float(da.sel(year=year))

    return 100.0 * np.sum(vals <= target) / len(vals)


def select_year_range(da, start_year=1991, end_year=2020):
    """Select data within a given year range."""
    years = da.time.dt.year
    return da.where((years >= start_year) & (years <= end_year), drop=True)


# =========================
# MCA / SVD functions
# =========================
def to_weighted_matrix(da, min_valid_fraction=0.99):
    """
    Convert DataArray(time, lat, lon) to weighted matrix(time, space).

    Weight = sqrt(cos(lat)).
    Missing grid cells are removed.
    """
    if "time" not in da.dims:
        raise ValueError("Input DataArray must have a time dimension.")

    da = to_lon180(da)

    spatial_dims = [d for d in da.dims if d != "time"]
    da = da.transpose("time", *spatial_dims)

    da_anom = da - da.mean("time", skipna=True)

    w = xr.ones_like(da.isel(time=0)) * 1.0

    if "lat" in da.coords:
        w_lat = np.sqrt(np.cos(np.deg2rad(da["lat"])))
        w_lat = xr.where(np.isfinite(w_lat) & (w_lat > 0), w_lat, 0)
        w = w * w_lat

    A = da_anom.stack(space=spatial_dims)
    W = w.stack(space=spatial_dims)

    valid_fraction = np.isfinite(A).mean("time")
    std_space = A.std("time", skipna=True)

    valid = (
        (valid_fraction >= min_valid_fraction)
        & np.isfinite(std_space)
        & (std_space > 1.0e-12)
        & np.isfinite(W)
        & (W > 0)
    )

    valid_bool = np.asarray(valid.values, dtype=bool)

    if valid_bool.sum() < 2:
        raise ValueError("Too few valid spatial grid points for SVD.")

    A = A.isel(space=valid_bool).fillna(0.0)
    W = W.isel(space=valid_bool).fillna(0.0)

    mat = A.values.astype(np.float64) * W.values.astype(np.float64)[None, :]

    return mat, A["space"], W


def standardize_columns(mat):
    """Standardize each column."""
    mat = mat - np.nanmean(mat, axis=0, keepdims=True)

    std = np.nanstd(mat, axis=0, ddof=1, keepdims=True)
    std[std == 0] = np.nan

    return mat / std


def mca_svd_pair(x_da, y_da, n_modes=2, tag=""):
    """
    Maximum covariance analysis / SVD between two fields.

    x_da: left field, usually Yangtze T2m.
    y_da: right field, V200 or OLR.

    Return standardized left and right PCs.
    """
    x_da, y_da = align_time_only(x_da, y_da)

    X, x_space, x_w = to_weighted_matrix(x_da)
    Y, y_space, y_w = to_weighted_matrix(y_da)

    ntime, nx = X.shape
    _, ny = Y.shape

    if ntime <= 3:
        raise ValueError("Too few time samples for SVD.")

    k = min(n_modes, nx - 1, ny - 1)

    if k < 1:
        raise ValueError("Too few spatial degrees of freedom for SVD.")

    scale = 1.0 / (ntime - 1)

    def matvec(v):
        return X.T @ (Y @ v) * scale

    def rmatvec(u):
        return Y.T @ (X @ u) * scale

    Aop = LinearOperator(
        shape=(nx, ny),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64
    )

    U, s, Vt = svds(
        Aop,
        k=k,
        which="LM",
        tol=1.0e-6,
        maxiter=3000
    )

    order = np.argsort(s)[::-1]
    s = s[order]
    U = U[:, order]
    Vt = Vt[order, :]
    V = Vt.T

    pc_x = X @ U
    pc_y = Y @ V

    pc_x = standardize_columns(pc_x)
    pc_y = standardize_columns(pc_y)

    modes = np.arange(1, k + 1)

    pc_x_da = xr.DataArray(
        pc_x,
        dims=["time", "mode"],
        coords={
            "time": x_da.time,
            "mode": modes
        },
        name=f"{tag}_pc_left"
    )

    pc_y_da = xr.DataArray(
        pc_y,
        dims=["time", "mode"],
        coords={
            "time": x_da.time,
            "mode": modes
        },
        name=f"{tag}_pc_right"
    )

    s_da = xr.DataArray(
        s,
        dims=["mode"],
        coords={"mode": modes},
        name=f"{tag}_singular_value"
    )

    if COMPUTE_SCF:
        XXt = X @ X.T
        YYt = Y @ Y.T
        total_cov2 = np.sum(XXt * YYt) * scale**2
        scf = (s**2) / total_cov2
    else:
        scf = np.full_like(s, np.nan)

    scf_da = xr.DataArray(
        scf,
        dims=["mode"],
        coords={"mode": modes},
        name=f"{tag}_SCF"
    )

    return {
        "pc_left": pc_x_da,
        "pc_right": pc_y_da,
        "singular_values": s_da,
        "scf": scf_da
    }


def orient_pc_cold_positive(pc, t2m_ts, name):
    """
    Orient PC so that positive PC tends to correspond to cold Yangtze T2m anomaly.
    This only changes sign.
    """
    pc, t2m_ts = align_time_only(pc, t2m_ts)

    r = xr.corr(pc, t2m_ts, dim="time")

    if np.isfinite(float(r)) and float(r) > 0:
        pc = -pc

    pc.name = name

    return pc


def ensure_1d_time_pc(pc, mode_to_use=1):
    """Ensure PC is a 1D DataArray with only the time dimension."""
    if not isinstance(pc, xr.DataArray):
        raise TypeError("pc must be an xarray.DataArray with a time coordinate.")

    if "mode" in pc.dims:
        if "mode" in pc.coords and mode_to_use in pc["mode"].values:
            pc = pc.sel(mode=mode_to_use)
        else:
            pc = pc.isel(mode=0)

    extra_dims = [d for d in pc.dims if d != "time"]

    if len(extra_dims) > 0:
        pc = pc.squeeze(drop=True)

    if pc.ndim != 1 or "time" not in pc.dims:
        raise ValueError(
            f"PC must be 1D with only time dimension, "
            f"but got dims={pc.dims}, shape={pc.shape}"
        )

    return pc.transpose("time").astype("float64")


# =========================
# 3.1 Prepare SVD input fields
# =========================
t2m_yangtze_field = to_lon180(t2m_bp)

v200_bp = v_bp.sel(level=200)
v200_bp = sel_latlon_box(
    v200_bp,
    lon=MID_DOMAIN["lon"],
    lat=MID_DOMAIN["lat"]
)

olr_trop_bp = sel_latlon_box(
    olr_bp,
    lon=TROP_DOMAIN["lon"],
    lat=TROP_DOMAIN["lat"]
)

t2m_yangtze_ts = area_weighted_mean(t2m_yangtze_field)
t2m_yangtze_ts.name = "Yangtze_T2m_bp"


# =========================
# 3.2 SVD / MCA
# =========================
print("Running MCA/SVD: Yangtze T2m vs V200 ...")

svd_mid = mca_svd_pair(
    t2m_yangtze_field,
    v200_bp,
    n_modes=N_SVD_MODES,
    tag="MID"
)

print("Running MCA/SVD: Yangtze T2m vs OLR ...")

svd_trop = mca_svd_pair(
    t2m_yangtze_field,
    olr_trop_bp,
    n_modes=N_SVD_MODES,
    tag="TROP"
)

# Default:
# MID  uses V200-side PC.
# TROP uses OLR-side PC.
pc_mid = svd_mid["pc_right"].sel(mode=MODE_TO_USE)
pc_trop = svd_trop["pc_right"].sel(mode=MODE_TO_USE)

pc_mid = orient_pc_cold_positive(pc_mid, t2m_yangtze_ts, "PC_mid")
pc_trop = orient_pc_cold_positive(pc_trop, t2m_yangtze_ts, "PC_trop")

pc_mid, pc_trop, t2m_yangtze_ts = align_time_only(
    pc_mid,
    pc_trop,
    t2m_yangtze_ts
)

pc_mid = ensure_1d_time_pc(pc_mid, mode_to_use=MODE_TO_USE)
pc_trop = ensure_1d_time_pc(pc_trop, mode_to_use=MODE_TO_USE)

print("pc_mid dims:", pc_mid.dims, "shape:", pc_mid.shape)
print("pc_trop dims:", pc_trop.dims, "shape:", pc_trop.shape)


# =========================
# 3.3 Annual STD activity indices
# =========================
I_mid = pc_mid.groupby("time.year").std("time", skipna=True, ddof=1)
I_trop = pc_trop.groupby("time.year").std("time", skipna=True, ddof=1)

I_mid.name = "I_mid_STD"
I_trop.name = "I_trop_STD"

I_mid_z = zscore_1d(I_mid, "year")
I_trop_z = zscore_1d(I_trop, "year")

I_mid_z.name = "I_mid_STD_z"
I_trop_z.name = "I_trop_STD_z"


# =========================
# 3.4 Hilbert phase and amplitude
# =========================
def hilbert_phase_amp_by_year(pc, remove_year_mean=True, mode_to_use=1):
    """
    Compute Hilbert phase and amplitude year by year.

    The Hilbert transform is applied separately to each Jun15-Jul31 segment.
    This avoids artificial phase connection between adjacent years.
    """
    pc = ensure_1d_time_pc(pc, mode_to_use=mode_to_use)

    values = np.asarray(pc.values, dtype=float)
    times = pd.to_datetime(pc.time.values)
    years = times.year.values

    amp = np.full(values.shape, np.nan, dtype=float)
    phase = np.full(values.shape, np.nan, dtype=float)

    for yy in np.unique(years):
        idx = np.where(years == yy)[0]
        x = values[idx].astype(float)

        if np.isfinite(x).sum() < 8:
            continue

        x_fill = pd.Series(x).interpolate(limit_direction="both").values
        x_fill = np.asarray(x_fill, dtype=float).reshape(-1)

        if remove_year_mean:
            x_fill = x_fill - np.nanmean(x_fill)

        analytic = hilbert(x_fill)

        if analytic.shape[0] != idx.shape[0]:
            raise ValueError(
                f"Hilbert length mismatch in year {yy}: "
                f"analytic length={analytic.shape[0]}, year length={idx.shape[0]}."
            )

        amp[idx] = np.abs(analytic)
        phase[idx] = (np.angle(analytic) + 2 * np.pi) % (2 * np.pi)

    amp_da = xr.DataArray(
        amp,
        dims=["time"],
        coords={"time": pc.time},
        name=f"{pc.name}_amp" if pc.name is not None else "pc_amp"
    )

    phase_da = xr.DataArray(
        phase,
        dims=["time"],
        coords={"time": pc.time},
        name=f"{pc.name}_phase" if pc.name is not None else "pc_phase"
    )

    return phase_da, amp_da


def make_phase_bin(phase, n_phase=8, name="phase_bin"):
    """Convert phase angle into discrete phase bins."""
    bin_width = 2 * np.pi / n_phase

    vals = phase.values.astype(float)
    out = np.full(vals.shape, -1, dtype=int)

    finite = np.isfinite(vals)
    out[finite] = (np.floor(vals[finite] / bin_width).astype(int)) % n_phase

    return xr.DataArray(
        out,
        dims=["time"],
        coords={"time": phase.time},
        name=name
    )


def diagnose_cold_phase_bins(
    phase,
    t2m_ts,
    n_phase=8,
    alpha=0.10,
    min_days=30,
    fallback_n=2,
    tag=""
):
    """
    Diagnose cold-favorable phase bins.

    Important:
    No amplitude threshold is used here.

    A bin is selected if:
    1. composite Yangtze T2m anomaly is negative;
    2. one-sided t test against zero passes alpha;
    3. sample size is not too small.

    If no bin passes, use the fallback_n coldest bins.
    """
    phase, t2m_ts = align_time_only(phase, t2m_ts)

    phase_bin = make_phase_bin(
        phase,
        n_phase=n_phase,
        name=f"{tag}_phase_bin"
    )

    bins = phase_bin.values
    y_vals = t2m_ts.values.astype(float)

    rows = []

    for b in range(n_phase):
        mask = (
            (bins == b)
            & np.isfinite(y_vals)
        )

        vals = y_vals[mask]
        n = len(vals)

        if n >= 2:
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals, ddof=1)

            if std_val > 0:
                tval = mean_val / (std_val / np.sqrt(n))
                p_less = student_t.cdf(tval, df=n - 1)
            else:
                tval = np.nan
                p_less = np.nan
        else:
            mean_val = np.nan
            std_val = np.nan
            tval = np.nan
            p_less = np.nan

        rows.append({
            "phase_bin": b,
            "n_days": n,
            "mean_T2m": mean_val,
            "std_T2m": std_val,
            "t_value": tval,
            "p_less_than_0": p_less
        })

    df = pd.DataFrame(rows)

    selected = df[
        (df["n_days"] >= min_days)
        & (df["mean_T2m"] < 0)
        & (df["p_less_than_0"] < alpha)
    ]["phase_bin"].values.astype(int)

    method = "significant_negative_bins"

    if len(selected) == 0:
        selected = (
            df.sort_values("mean_T2m")
            .head(fallback_n)["phase_bin"]
            .values.astype(int)
        )
        method = "fallback_coldest_bins"

    df["selected_as_cold"] = df["phase_bin"].isin(selected)
    df["selection_method"] = method

    return selected, df, phase_bin


def diagnose_hot_phase_bins(
    phase,
    t2m_ts,
    n_phase=8,
    alpha=0.10,
    min_days=30,
    fallback_n=2,
    tag=""
):
    """
    Diagnose hot-favorable phase bins.

    Important:
    No amplitude threshold is used here.

    A bin is selected if:
    1. composite Yangtze T2m anomaly is positive;
    2. one-sided t test against zero passes alpha;
    3. sample size is not too small.

    If no bin passes, use the fallback_n warmest bins.
    """
    phase, t2m_ts = align_time_only(phase, t2m_ts)

    phase_bin = make_phase_bin(
        phase,
        n_phase=n_phase,
        name=f"{tag}_phase_bin"
    )

    bins = phase_bin.values
    y_vals = t2m_ts.values.astype(float)

    rows = []

    for b in range(n_phase):
        mask = (
            (bins == b)
            & np.isfinite(y_vals)
        )

        vals = y_vals[mask]
        n = len(vals)

        if n >= 2:
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals, ddof=1)

            if std_val > 0:
                tval = mean_val / (std_val / np.sqrt(n))
                p_greater = 1.0 - student_t.cdf(tval, df=n - 1)
            else:
                tval = np.nan
                p_greater = np.nan
        else:
            mean_val = np.nan
            std_val = np.nan
            tval = np.nan
            p_greater = np.nan

        rows.append({
            "phase_bin": b,
            "n_days": n,
            "mean_T2m": mean_val,
            "std_T2m": std_val,
            "t_value": tval,
            "p_greater_than_0": p_greater
        })

    df = pd.DataFrame(rows)

    selected = df[
        (df["n_days"] >= min_days)
        & (df["mean_T2m"] > 0)
        & (df["p_greater_than_0"] < alpha)
    ]["phase_bin"].values.astype(int)

    method = "significant_positive_bins"

    if len(selected) == 0:
        selected = (
            df.sort_values("mean_T2m", ascending=False)
            .head(fallback_n)["phase_bin"]
            .values.astype(int)
        )
        method = "fallback_warmest_bins"

    df["selected_as_hot"] = df["phase_bin"].isin(selected)
    df["selection_method"] = method

    return selected, df, phase_bin


phase_mid, amp_mid = hilbert_phase_amp_by_year(
    pc_mid,
    mode_to_use=MODE_TO_USE
)

phase_trop, amp_trop = hilbert_phase_amp_by_year(
    pc_trop,
    mode_to_use=MODE_TO_USE
)

# Cold-favorable phase bins
cold_bins_mid, cold_diag_mid, phase_bin_mid = diagnose_cold_phase_bins(
    phase_mid,
    t2m_yangtze_ts,
    n_phase=N_PHASE,
    alpha=COLD_ALPHA,
    min_days=MIN_BIN_DAYS,
    fallback_n=FALLBACK_COLD_BINS,
    tag="MID_COLD"
)

cold_bins_trop, cold_diag_trop, phase_bin_trop = diagnose_cold_phase_bins(
    phase_trop,
    t2m_yangtze_ts,
    n_phase=N_PHASE,
    alpha=COLD_ALPHA,
    min_days=MIN_BIN_DAYS,
    fallback_n=FALLBACK_COLD_BINS,
    tag="TROP_COLD"
)

# Hot-favorable phase bins
hot_bins_mid, hot_diag_mid, _ = diagnose_hot_phase_bins(
    phase_mid,
    t2m_yangtze_ts,
    n_phase=N_PHASE,
    alpha=HOT_ALPHA,
    min_days=MIN_BIN_DAYS,
    fallback_n=FALLBACK_HOT_BINS,
    tag="MID_HOT"
)

hot_bins_trop, hot_diag_trop, _ = diagnose_hot_phase_bins(
    phase_trop,
    t2m_yangtze_ts,
    n_phase=N_PHASE,
    alpha=HOT_ALPHA,
    min_days=MIN_BIN_DAYS,
    fallback_n=FALLBACK_HOT_BINS,
    tag="TROP_HOT"
)

phase_mid, amp_mid, phase_trop, amp_trop, phase_bin_mid, phase_bin_trop = align_time_only(
    phase_mid,
    amp_mid,
    phase_trop,
    amp_trop,
    phase_bin_mid,
    phase_bin_trop
)


# =========================
# 3.5 Amplitude thresholds from 1991-2020
# =========================
amp_mid_ref = select_year_range(
    amp_mid,
    AMP_REF_START,
    AMP_REF_END
)

amp_trop_ref = select_year_range(
    amp_trop,
    AMP_REF_START,
    AMP_REF_END
)

amp_mid_sigma = float(amp_mid_ref.std("time", skipna=True, ddof=1))
amp_trop_sigma = float(amp_trop_ref.std("time", skipna=True, ddof=1))

amp_mid_thr = AMP_SIGMA_MULT * amp_mid_sigma
amp_trop_thr = AMP_SIGMA_MULT * amp_trop_sigma

if not np.isfinite(amp_mid_thr) or amp_mid_thr <= 0:
    raise ValueError("Invalid MID amplitude threshold. Please check amp_mid during 1991-2020.")

if not np.isfinite(amp_trop_thr) or amp_trop_thr <= 0:
    raise ValueError("Invalid TROP amplitude threshold. Please check amp_trop during 1991-2020.")

amp_mid_norm = amp_mid / amp_mid_thr
amp_trop_norm = amp_trop / amp_trop_thr

amp_mid_norm.name = "AMP_mid_over_1sigma_1991_2020"
amp_trop_norm.name = "AMP_trop_over_1sigma_1991_2020"

strong_amp_mid = amp_mid > amp_mid_thr
strong_amp_trop = amp_trop > amp_trop_thr

strong_amp_mid.name = "strong_amp_mid"
strong_amp_trop.name = "strong_amp_trop"

amp_threshold_df = pd.DataFrame({
    "mode": ["MID", "TROP"],
    "reference_period": [
        f"{AMP_REF_START}-{AMP_REF_END}",
        f"{AMP_REF_START}-{AMP_REF_END}"
    ],
    "sigma_multiplier": [
        AMP_SIGMA_MULT,
        AMP_SIGMA_MULT
    ],
    "amp_sigma": [
        amp_mid_sigma,
        amp_trop_sigma
    ],
    "amp_threshold": [
        amp_mid_thr,
        amp_trop_thr
    ]
})

amp_threshold_csv = os.path.join(
    STEP3_OUT,
    "step3_amplitude_thresholds_1991_2020.csv"
)

amp_threshold_df.to_csv(amp_threshold_csv, index=False)


# ------------------------------------------------------------
# Cold PCI definition:
# C(t) = 1 only if:
# 1. MID phase is in its cold-favorable phase bins;
# 2. TROP phase is in its cold-favorable phase bins;
# 3. MID Hilbert amplitude exceeds its 1991-2020 1-sigma threshold;
# 4. TROP Hilbert amplitude exceeds its 1991-2020 1-sigma threshold.
# ------------------------------------------------------------
cold_phase_mid_flag = np.isin(phase_bin_mid.values, cold_bins_mid)
cold_phase_trop_flag = np.isin(phase_bin_trop.values, cold_bins_trop)

strong_amp_mid_flag = amp_mid.values > amp_mid_thr
strong_amp_trop_flag = amp_trop.values > amp_trop_thr

C_vals = (
    cold_phase_mid_flag
    & cold_phase_trop_flag
    & strong_amp_mid_flag
    & strong_amp_trop_flag
)

C = xr.DataArray(
    C_vals.astype(float),
    dims=["time"],
    coords={"time": phase_mid.time},
    name="cold_phase_strong_amp_coupling_flag"
)

PCI = C.groupby("time.year").mean("time", skipna=True)
PCI.name = "PCI"

PCI_z = zscore_1d(PCI, "year")
PCI_z.name = "PCI_z"

N_coupled_days = C.groupby("time.year").sum("time", skipna=True)
N_coupled_days.name = "N_cold_coupled_days"


# ------------------------------------------------------------
# Hot phase index definition:
# H(t) = 1 only if:
# 1. MID phase is in its hot-favorable phase bins;
# 2. TROP phase is in its hot-favorable phase bins;
# 3. MID Hilbert amplitude exceeds its 1991-2020 1-sigma threshold;
# 4. TROP Hilbert amplitude exceeds its 1991-2020 1-sigma threshold.
# ------------------------------------------------------------
hot_phase_mid_flag = np.isin(phase_bin_mid.values, hot_bins_mid)
hot_phase_trop_flag = np.isin(phase_bin_trop.values, hot_bins_trop)

H_vals = (
    hot_phase_mid_flag
    & hot_phase_trop_flag
    & strong_amp_mid_flag
    & strong_amp_trop_flag
)

H = xr.DataArray(
    H_vals.astype(float),
    dims=["time"],
    coords={"time": phase_mid.time},
    name="hot_phase_strong_amp_coupling_flag"
)

HCI = H.groupby("time.year").mean("time", skipna=True)
HCI.name = "HCI"

HCI_z = zscore_1d(HCI, "year")
HCI_z.name = "HCI_z"

N_hot_coupled_days = H.groupby("time.year").sum("time", skipna=True)
N_hot_coupled_days.name = "N_hot_coupled_days"


# Other diagnostic day counts
N_strong_mid = strong_amp_mid.groupby("time.year").sum("time", skipna=True)
N_strong_trop = strong_amp_trop.groupby("time.year").sum("time", skipna=True)

N_strong_mid.name = "N_strong_amp_mid_days"
N_strong_trop.name = "N_strong_amp_trop_days"

cold_phase_mid_da = xr.DataArray(
    cold_phase_mid_flag.astype(float),
    dims=["time"],
    coords={"time": phase_mid.time},
    name="cold_phase_mid_flag"
)

cold_phase_trop_da = xr.DataArray(
    cold_phase_trop_flag.astype(float),
    dims=["time"],
    coords={"time": phase_trop.time},
    name="cold_phase_trop_flag"
)

hot_phase_mid_da = xr.DataArray(
    hot_phase_mid_flag.astype(float),
    dims=["time"],
    coords={"time": phase_mid.time},
    name="hot_phase_mid_flag"
)

hot_phase_trop_da = xr.DataArray(
    hot_phase_trop_flag.astype(float),
    dims=["time"],
    coords={"time": phase_trop.time},
    name="hot_phase_trop_flag"
)

N_cold_phase_mid = cold_phase_mid_da.groupby("time.year").sum("time", skipna=True)
N_cold_phase_trop = cold_phase_trop_da.groupby("time.year").sum("time", skipna=True)

N_hot_phase_mid = hot_phase_mid_da.groupby("time.year").sum("time", skipna=True)
N_hot_phase_trop = hot_phase_trop_da.groupby("time.year").sum("time", skipna=True)

N_cold_phase_mid.name = "N_cold_phase_mid_days"
N_cold_phase_trop.name = "N_cold_phase_trop_days"

N_hot_phase_mid.name = "N_hot_phase_mid_days"
N_hot_phase_trop.name = "N_hot_phase_trop_days"


# =========================
# 3.6 Variance decomposition
# =========================
def variance_decomposition_t2m(y, x1, x2):
    """
    Decompose annual variance of Yangtze T2m explained by two standardized PCs.

    y = a0 + a1*x1 + a2*x2 + residual

    Annual fitted variance is approximated by:
    Var_fit = a1^2 Var(x1) + a2^2 Var(x2) + 2 a1 a2 Cov(x1, x2)
    """
    y, x1, x2 = align_time_only(y, x1, x2)

    yv = y.values.astype(float)
    x1v = x1.values.astype(float)
    x2v = x2.values.astype(float)

    valid = np.isfinite(yv) & np.isfinite(x1v) & np.isfinite(x2v)

    Xmat = np.column_stack([
        np.ones(valid.sum()),
        x1v[valid],
        x2v[valid]
    ])

    coef = np.linalg.lstsq(Xmat, yv[valid], rcond=None)[0]

    a0, a1, a2 = coef

    times = pd.to_datetime(y.time.values)
    years = times.year.values

    rows = []

    for yy in np.unique(years):
        m = (years == yy) & valid

        if m.sum() < 5:
            continue

        yyv = yv[m]
        x1y = x1v[m]
        x2y = x2v[m]

        var_obs = np.nanvar(yyv, ddof=1)
        var_x1 = np.nanvar(x1y, ddof=1)
        var_x2 = np.nanvar(x2y, ddof=1)
        cov_x1x2 = np.cov(x1y, x2y, ddof=1)[0, 1]

        term_mid = a1**2 * var_x1
        term_trop = a2**2 * var_x2
        term_cov = 2 * a1 * a2 * cov_x1x2

        fit_y = a0 + a1 * x1y + a2 * x2y
        resid_y = yyv - fit_y

        var_fit = np.nanvar(fit_y, ddof=1)
        var_resid = np.nanvar(resid_y, ddof=1)

        if var_obs > 0:
            frac_mid = 100 * term_mid / var_obs
            frac_trop = 100 * term_trop / var_obs
            frac_cov = 100 * term_cov / var_obs
            frac_fit = 100 * var_fit / var_obs
            frac_resid = 100 * var_resid / var_obs
        else:
            frac_mid = np.nan
            frac_trop = np.nan
            frac_cov = np.nan
            frac_fit = np.nan
            frac_resid = np.nan

        rows.append({
            "year": yy,
            "a0": a0,
            "a_mid": a1,
            "a_trop": a2,
            "Var_obs": var_obs,
            "Var_mid_term": term_mid,
            "Var_trop_term": term_trop,
            "Var_cov_term": term_cov,
            "Var_fit_direct": var_fit,
            "Var_residual": var_resid,
            "Frac_mid_percent": frac_mid,
            "Frac_trop_percent": frac_trop,
            "Frac_cov_percent": frac_cov,
            "Frac_fit_percent": frac_fit,
            "Frac_residual_percent": frac_resid
        })

    df = pd.DataFrame(rows)
    ds = df.set_index("year").to_xarray()

    return ds, coef


var_ds, reg_coef = variance_decomposition_t2m(
    t2m_yangtze_ts,
    pc_mid,
    pc_trop
)


# =========================
# 3.7 Collect annual results
# =========================
annual_ds = xr.Dataset({
    "I_mid_STD": I_mid,
    "I_trop_STD": I_trop,
    "I_mid_STD_z": I_mid_z,
    "I_trop_STD_z": I_trop_z,

    "PCI": PCI,
    "PCI_z": PCI_z,
    "HCI": HCI,
    "HCI_z": HCI_z,

    "N_cold_coupled_days": N_coupled_days,
    "N_hot_coupled_days": N_hot_coupled_days,

    "N_strong_amp_mid_days": N_strong_mid,
    "N_strong_amp_trop_days": N_strong_trop,

    "N_cold_phase_mid_days": N_cold_phase_mid,
    "N_cold_phase_trop_days": N_cold_phase_trop,
    "N_hot_phase_mid_days": N_hot_phase_mid,
    "N_hot_phase_trop_days": N_hot_phase_trop
})

annual_ds = xr.merge([annual_ds, var_ds], compat="override")

annual_df = annual_ds.to_dataframe().reset_index()

annual_csv = os.path.join(
    STEP3_OUT,
    "step3_annual_indices_variance_cold_hot.csv"
)

annual_df.to_csv(annual_csv, index=False)

daily_df = pd.DataFrame({
    "time": pd.to_datetime(pc_mid.time.values),
    "PC_mid": pc_mid.values,
    "PC_trop": pc_trop.values,

    "AMP_mid": amp_mid.values,
    "AMP_trop": amp_trop.values,
    "AMP_mid_over_1sigma_1991_2020": amp_mid_norm.values,
    "AMP_trop_over_1sigma_1991_2020": amp_trop_norm.values,
    "AMP_mid_threshold_1991_2020": amp_mid_thr,
    "AMP_trop_threshold_1991_2020": amp_trop_thr,

    "STRONG_AMP_mid": strong_amp_mid_flag.astype(int),
    "STRONG_AMP_trop": strong_amp_trop_flag.astype(int),

    "PHASE_mid": phase_mid.values,
    "PHASE_trop": phase_trop.values,
    "PHASE_BIN_mid": phase_bin_mid.values,
    "PHASE_BIN_trop": phase_bin_trop.values,

    "COLD_PHASE_mid": cold_phase_mid_flag.astype(int),
    "COLD_PHASE_trop": cold_phase_trop_flag.astype(int),
    "HOT_PHASE_mid": hot_phase_mid_flag.astype(int),
    "HOT_PHASE_trop": hot_phase_trop_flag.astype(int),

    "C_flag": C.values,
    "H_flag": H.values
})

daily_csv = os.path.join(
    STEP3_OUT,
    "step3_daily_pcs_phase_pci_cold_hot.csv"
)

daily_df.to_csv(daily_csv, index=False)

cold_diag_mid.to_csv(
    os.path.join(STEP3_OUT, "step3_cold_phase_bins_mid.csv"),
    index=False
)

cold_diag_trop.to_csv(
    os.path.join(STEP3_OUT, "step3_cold_phase_bins_trop.csv"),
    index=False
)

hot_diag_mid.to_csv(
    os.path.join(STEP3_OUT, "step3_hot_phase_bins_mid.csv"),
    index=False
)

hot_diag_trop.to_csv(
    os.path.join(STEP3_OUT, "step3_hot_phase_bins_trop.csv"),
    index=False
)

svd_info = pd.DataFrame({
    "mode": svd_mid["singular_values"].mode.values,
    "MID_singular_value": svd_mid["singular_values"].values,
    "MID_SCF": svd_mid["scf"].values,
    "TROP_singular_value": svd_trop["singular_values"].values,
    "TROP_SCF": svd_trop["scf"].values
})

svd_info.to_csv(
    os.path.join(STEP3_OUT, "step3_svd_info.csv"),
    index=False
)


# =========================
# 3.8 Print 2015 diagnosis
# =========================
print("\n================ Step 3 Summary ================")
print(f"Output directory: {STEP3_OUT}")

print("\nCold-favorable phase bins:")
print(f"MID  cold bins: {cold_bins_mid.tolist()}")
print(f"TROP cold bins: {cold_bins_trop.tolist()}")

print("\nHot-favorable phase bins:")
print(f"MID  hot bins: {hot_bins_mid.tolist()}")
print(f"TROP hot bins: {hot_bins_trop.tolist()}")

print("\nAmplitude thresholds:")
print(
    f"MID  amplitude threshold = {amp_mid_thr:.4f} "
    f"({AMP_SIGMA_MULT:.1f} sigma, {AMP_REF_START}-{AMP_REF_END})"
)
print(
    f"TROP amplitude threshold = {amp_trop_thr:.4f} "
    f"({AMP_SIGMA_MULT:.1f} sigma, {AMP_REF_START}-{AMP_REF_END})"
)

print("\nRegression coefficients for variance decomposition:")
print(
    f"T2m = {reg_coef[0]:.4f} "
    f"+ {reg_coef[1]:.4f} * PC_mid "
    f"+ {reg_coef[2]:.4f} * PC_trop"
)

if TARGET_YEAR in annual_ds.year.values:
    p_mid = percentile_rank(annual_ds["I_mid_STD"], TARGET_YEAR)
    p_trop = percentile_rank(annual_ds["I_trop_STD"], TARGET_YEAR)
    p_pci = percentile_rank(annual_ds["PCI"], TARGET_YEAR)
    p_hci = percentile_rank(annual_ds["HCI"], TARGET_YEAR)
    p_cold_days = percentile_rank(annual_ds["N_cold_coupled_days"], TARGET_YEAR)
    p_hot_days = percentile_rank(annual_ds["N_hot_coupled_days"], TARGET_YEAR)

    summary_2015 = pd.DataFrame({
        "Metric": [
            "I_mid_STD",
            "I_trop_STD",
            "PCI",
            "HCI",
            "N_cold_coupled_days",
            "N_hot_coupled_days",
            "N_strong_amp_mid_days",
            "N_strong_amp_trop_days",
            "N_cold_phase_mid_days",
            "N_cold_phase_trop_days",
            "N_hot_phase_mid_days",
            "N_hot_phase_trop_days",
            "Frac_mid_percent",
            "Frac_trop_percent",
            "Frac_cov_percent",
            "Frac_fit_percent"
        ],
        "Value": [
            float(annual_ds["I_mid_STD"].sel(year=TARGET_YEAR)),
            float(annual_ds["I_trop_STD"].sel(year=TARGET_YEAR)),
            float(annual_ds["PCI"].sel(year=TARGET_YEAR)),
            float(annual_ds["HCI"].sel(year=TARGET_YEAR)),
            float(annual_ds["N_cold_coupled_days"].sel(year=TARGET_YEAR)),
            float(annual_ds["N_hot_coupled_days"].sel(year=TARGET_YEAR)),
            float(annual_ds["N_strong_amp_mid_days"].sel(year=TARGET_YEAR)),
            float(annual_ds["N_strong_amp_trop_days"].sel(year=TARGET_YEAR)),
            float(annual_ds["N_cold_phase_mid_days"].sel(year=TARGET_YEAR)),
            float(annual_ds["N_cold_phase_trop_days"].sel(year=TARGET_YEAR)),
            float(annual_ds["N_hot_phase_mid_days"].sel(year=TARGET_YEAR)),
            float(annual_ds["N_hot_phase_trop_days"].sel(year=TARGET_YEAR)),
            float(annual_ds["Frac_mid_percent"].sel(year=TARGET_YEAR)),
            float(annual_ds["Frac_trop_percent"].sel(year=TARGET_YEAR)),
            float(annual_ds["Frac_cov_percent"].sel(year=TARGET_YEAR)),
            float(annual_ds["Frac_fit_percent"].sel(year=TARGET_YEAR))
        ],
        "Percentile": [
            p_mid,
            p_trop,
            p_pci,
            p_hci,
            p_cold_days,
            p_hot_days,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ]
    })

    print(f"\n{TARGET_YEAR} diagnosis:")
    print(summary_2015.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    summary_2015.to_csv(
        os.path.join(STEP3_OUT, f"step3_{TARGET_YEAR}_diagnosis_cold_hot.csv"),
        index=False
    )


# =========================
# 3.9 SCI-style figure
# Three panels:
# (a) I_mid
# (b) I_trop
# (c) Cold/hot coupled strong-amplitude days
# =========================
def set_sci_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", which="both")
    ax.grid(False)


def add_panel_label(ax, label):
    ax.text(
        0.01,
        0.93,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left"
    )


years = annual_df["year"].values

fig, axes = plt.subplots(
    3,
    1,
    figsize=(7.2, 6.0),
    sharex=True,
    constrained_layout=True
)

# ---------- (a) I_mid ----------
ax = axes[0]

ax.plot(
    years,
    annual_df["I_mid_STD_z"],
    lw=1.4,
    color="#4C72B0",
    label=r"$I_{\mathrm{mid}}$"
)

ax.axhline(0, color="0.35", lw=0.7)
ax.axvline(TARGET_YEAR, color="0.15", lw=0.9, ls="--")

if TARGET_YEAR in years:
    y_target = float(annual_ds["I_mid_STD_z"].sel(year=TARGET_YEAR))
    ax.scatter(
        TARGET_YEAR,
        y_target,
        s=28,
        color="#0072B2",
        zorder=5
    )

ax.set_ylabel("STD index\n(z-score)")
ax.legend(frameon=False, loc="upper right")
add_panel_label(ax, "(a)")
set_sci_axis(ax)


# ---------- (b) I_trop ----------
ax = axes[1]

ax.plot(
    years,
    annual_df["I_trop_STD_z"],
    lw=1.4,
    color="#55A868",
    label=r"$I_{\mathrm{trop}}$"
)

ax.axhline(0, color="0.35", lw=0.7)
ax.axvline(TARGET_YEAR, color="0.15", lw=0.9, ls="--")

if TARGET_YEAR in years:
    y_target = float(annual_ds["I_trop_STD_z"].sel(year=TARGET_YEAR))
    ax.scatter(
        TARGET_YEAR,
        y_target,
        s=28,
        color="#0072B2",
        zorder=5
    )

ax.set_ylabel("STD index\n(z-score)")
ax.legend(frameon=False, loc="upper right")
add_panel_label(ax, "(b)")
set_sci_axis(ax)


# ---------- (c) Cold/hot coupled strong-amplitude days ----------
ax = axes[2]

cold_days = annual_df["N_cold_coupled_days"].values
hot_days = annual_df["N_hot_coupled_days"].values

# Cold coupled days: filled gray bars
ax.bar(
    years,
    cold_days,
    width=0.80,
    color="0.70",
    edgecolor="none",
    label="Cold phase + strong amp."
)

# 2015 cold coupled days: blue filled bar
if TARGET_YEAR in years:
    ax.bar(
        TARGET_YEAR,
        float(annual_ds["N_cold_coupled_days"].sel(year=TARGET_YEAR)),
        width=0.80,
        color="#0072B2",
        edgecolor="none",
        zorder=4,
        label=f"{TARGET_YEAR} cold"
    )

# Hot coupled days: red outline bars
ax.bar(
    years,
    hot_days,
    width=0.80,
    facecolor="none",
    edgecolor="#C44E52",
    linewidth=1.1,
    zorder=5,
    label="Hot phase + strong amp."
)

# Mean line for cold coupled days
ax.axhline(
    float(annual_ds["N_cold_coupled_days"].mean("year")),
    color="0.20",
    lw=0.8,
    label="Cold mean"
)

ax.axvline(TARGET_YEAR, color="0.15", lw=0.9, ls="--")

ymax = np.nanmax([
    np.nanmax(cold_days),
    np.nanmax(hot_days),
    float(annual_ds["N_cold_coupled_days"].mean("year"))
])

if np.isfinite(ymax) and ymax > 0:
    ax.set_ylim(0, ymax * 1.18)

ax.set_ylabel("Coupled days")
ax.set_xlabel("Year")
ax.legend(frameon=False, loc="upper right", ncol=1)

add_panel_label(ax, "(c)")
set_sci_axis(ax)

axes[-1].xaxis.set_major_locator(MultipleLocator(10))
axes[-1].xaxis.set_minor_locator(MultipleLocator(5))

fig.savefig(
    os.path.join(STEP3_OUT, "Fig_step3_annual_indices_cold_hot_days_amp_limited.pdf"),
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(STEP3_OUT, "Fig_step3_annual_indices_cold_hot_days_amp_limited.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()

###################################################################################################################################################################################################
# ============================================================
# Plot cold-favorable phases as one standalone SCI-style figure
# ============================================================

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# =========================
# SCI plotting style
# =========================
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix"
})


def plot_cold_phase_single_figure(
    cold_diag_mid,
    cold_diag_trop,
    cold_bins_mid,
    cold_bins_trop,
    out_dir,
    filename_prefix="Fig_cold_favorable_phase"
):
    """
    Plot cold-favorable phase diagnosis for the middle-high-latitude
    and tropical modes in one standalone SCI-style figure.

    Required columns in cold_diag_mid / cold_diag_trop:
        phase_bin
        n_days
        mean_T2m
        std_T2m
        p_less_than_0
        selected_as_cold
    """

    def _prepare_df(df, cold_bins):
        df = df.copy()

        # Convert phase bin from 0-7 to 1-8 for plotting
        df["phase_plot"] = df["phase_bin"].astype(int) + 1

        # Ensure selected cold phase is consistent with cold_bins
        cold_bins = np.asarray(cold_bins, dtype=int)
        df["selected_as_cold"] = df["phase_bin"].astype(int).isin(cold_bins)

        # 95% confidence interval
        df["se"] = df["std_T2m"] / np.sqrt(df["n_days"])
        df["ci95"] = 1.96 * df["se"]

        return df

    mid_df = _prepare_df(cold_diag_mid, cold_bins_mid)
    trop_df = _prepare_df(cold_diag_trop, cold_bins_trop)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.2, 3.2),
        dpi=300,
        constrained_layout=False
    )

    plot_info = [
        {
            "ax": axes[0],
            "df": mid_df,
            "title": "(a) Middle-high-latitude mode",
            "cold_label": "Cold-favorable phase"
        },
        {
            "ax": axes[1],
            "df": trop_df,
            "title": "(b) Tropical mode",
            "cold_label": "Cold-favorable phase"
        }
    ]

    for item in plot_info:
        ax = item["ax"]
        df = item["df"]

        x = df["phase_plot"].values
        y = df["mean_T2m"].values
        yerr = df["ci95"].values
        n_days = df["n_days"].values
        pvals = df["p_less_than_0"].values
        is_cold = df["selected_as_cold"].values

        # Base bars
        bars = ax.bar(
            x,
            y,
            width=0.68,
            color="0.78",
            edgecolor="0.25",
            linewidth=0.7,
            zorder=2
        )

        # Highlight cold-favorable phases
        for b, flag in zip(bars, is_cold):
            if flag:
                b.set_color("0.25")
                b.set_edgecolor("0.05")
                b.set_linewidth(0.9)
                b.set_hatch("///")

        # 95% CI error bars
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="0.15",
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=3
        )

        # Zero line
        ax.axhline(
            0,
            color="0.15",
            linewidth=0.8,
            linestyle="-",
            zorder=1
        )

        # Significance marks: one-sided cold anomaly test
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin

        for xi, yi, pv in zip(x, y, pvals):
            if np.isfinite(pv) and pv < 0.10 and yi < 0:
                ax.text(
                    xi,
                    yi - 0.07 * yrange,
                    "*",
                    ha="center",
                    va="top",
                    fontsize=10,
                    color="0.05"
                )

        # Sample size above/below bars
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin

        for xi, yi, ni in zip(x, y, n_days):
            if yi >= 0:
                y_text = yi + 0.05 * yrange
                va = "bottom"
            else:
                y_text = yi - 0.05 * yrange
                va = "top"

            ax.text(
                xi,
                y_text,
                f"{int(ni)}",
                ha="center",
                va=va,
                fontsize=6.5,
                color="0.25"
            )

        ax.set_title(item["title"], loc="left", pad=4)
        ax.set_xlabel("Phase")
        ax.set_xticks(np.arange(1, 9, 1))
        ax.set_xlim(0.35, 8.65)

        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="both", which="both", direction="out")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Yangtze T2m anomaly (K)")

    # Unified y-axis range
    all_y = np.concatenate([
        mid_df["mean_T2m"].values + mid_df["ci95"].values,
        mid_df["mean_T2m"].values - mid_df["ci95"].values,
        trop_df["mean_T2m"].values + trop_df["ci95"].values,
        trop_df["mean_T2m"].values - trop_df["ci95"].values
    ])

    y_abs = np.nanmax(np.abs(all_y))
    y_lim = np.ceil(y_abs * 10) / 10 + 0.1

    for ax in axes:
        ax.set_ylim(-y_lim, y_lim)

    # Legend
    cold_patch = mpl.patches.Patch(
        facecolor="0.25",
        edgecolor="0.05",
        hatch="///",
        label="Cold-favorable phase"
    )

    normal_patch = mpl.patches.Patch(
        facecolor="0.78",
        edgecolor="0.25",
        label="Other phase"
    )

    axes[1].legend(
        handles=[cold_patch, normal_patch],
        frameon=False,
        loc="upper right",
        handlelength=1.8,
        borderaxespad=0.2
    )

    fig.text(
        0.5,
        0.015,
        "Numbers indicate sample size. Asterisks denote one-sided cold-anomaly test at p < 0.10.",
        ha="center",
        va="bottom",
        fontsize=7.5
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.20,
        top=0.90,
        wspace=0.24
    )

    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{filename_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{filename_prefix}.pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.show()

    print("Saved:")
    print(png_path)
    print(pdf_path)


# =========================
# Run
# =========================
plot_cold_phase_single_figure(
    cold_diag_mid=cold_diag_mid,
    cold_diag_trop=cold_diag_trop,
    cold_bins_mid=cold_bins_mid,
    cold_bins_trop=cold_bins_trop,
    out_dir=STEP3_OUT,
    filename_prefix="Fig_cold_favorable_phase"
)
