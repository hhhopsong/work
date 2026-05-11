# -*- coding: utf-8 -*-
"""
Task 3:
Project 2015 JJA 10-30-day filtered fields onto climatological MCA/SVD modes.

Purpose:
1. Refit climatological MCA modes using 1961-2022 JJA data.
2. Use MCA-A: Yangtze T2m & northern V200.
3. Use MCA-B: Yangtze T2m & southern OLR.
4. Project 2015 fields onto these fixed MCA modes.
5. Diagnose whether the northern wave-train mode and southern OLR/BSISO-like mode
   are jointly favorable for Yangtze River basin cooling in 2015.

Author: replace with your name
"""

# ============================================================
# Imports
# ============================================================
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.stats import pearsonr
from scipy.stats import t as student_t

import xeofs

from climkit.masked import masked
from climkit.lonlat_transform import transform


# ============================================================
# User settings
# ============================================================

# -------------------------
# Paths
# -------------------------
PYFILE = r"/Volumes/TiPlus7100/PyFile"
DATA_DIR = r"/Volumes/TiPlus7100/p4/data"

YANGTZE_SHP = fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"

V_BP_FILE = fr"{DATA_DIR}/v_bp_JJA_10-30d.nc"
OLR_BP_FILE = fr"{DATA_DIR}/ttr_bp_JJA_10-30d.nc"
T2M_BP_FILE = fr"{DATA_DIR}/t2m_bp_JJA_10-30d.nc"

OUT_DIR_FIG = fr"{PYFILE}/p4/pic"
OUT_DIR_DATA = fr"{PYFILE}/p4/data"

Path(OUT_DIR_FIG).mkdir(parents=True, exist_ok=True)
Path(OUT_DIR_DATA).mkdir(parents=True, exist_ok=True)

SAVE_FIG = fr"{OUT_DIR_FIG}/2015‍投影历史SVD.png"
SAVE_PDF = fr"{OUT_DIR_FIG}/2015‍投影历史SVD.pdf"
SAVE_NC = fr"{OUT_DIR_DATA}/2015‍投影历史SVD.nc"


# -------------------------
# Time settings
# -------------------------
TIME_SLICE = slice("1961-01-01", "2022-12-31")

TARGET_YEAR = 2015
TARGET_START = f"{TARGET_YEAR}-06-01"
TARGET_END = f"{TARGET_YEAR}-08-31"

# Whether to remove 2015 when fitting MCA.
# False: fit MCA using 1961-2022, same as your current all-year figure.
# True : leave-one-year-out test. This is more rigorous for final paper.
EXCLUDE_TARGET_YEAR_FROM_MCA = False


# -------------------------
# MCA settings
# -------------------------
N_MODES = 4

# MCA-A: Yangtze T2m & northern V200
V200_LEVEL = 200
V200_REGION = {
    "lon_min": -80,
    "lon_max": 160,
    "lat_min": 36,
    "lat_max": 75,
}

# MCA-B: Yangtze T2m & southern OLR
OLR_REGION = {
    "lon_min": 40,
    "lon_max": 160,
    "lat_min": -10,
    "lat_max": 24,
}

# Reduce target T2m domain before masking.
# This only improves efficiency; the actual target area is still controlled by the Yangtze shapefile.
T2M_TARGET_BOX = {
    "lon_min": 90,
    "lon_max": 125,
    "lat_min": 20,
    "lat_max": 40,
}


# -------------------------
# Plot settings
# -------------------------
COLD_PHASE_THRESHOLD = 1.0

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["mathtext.fontset"] = "stix"


# ============================================================
# Utility functions
# ============================================================

def select_latlon_box(da, lon_min, lon_max, lat_min, lat_max):
    """
    Select a lon-lat box safely regardless of whether latitude is ascending or descending.
    """
    if "lon" not in da.coords or "lat" not in da.coords:
        raise ValueError("Input DataArray must contain lon and lat coordinates.")

    # Make sure lon is monotonically increasing if possible.
    if da.lon.size > 1:
        da = da.sortby("lon")

    lat0 = float(da.lat.values[0])
    lat1 = float(da.lat.values[-1])

    if lat0 > lat1:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    return da.sel(
        lon=slice(lon_min, lon_max),
        lat=lat_slice
    )


def align_on_common_time(*arrays):
    """
    Align multiple xarray objects only by common time coordinate.
    This avoids accidental alignment along lat/lon/level.
    """
    common_time = arrays[0].time.values

    for da in arrays[1:]:
        common_time = np.intersect1d(common_time, da.time.values)

    common_time = np.sort(common_time)

    return tuple(da.sel(time=common_time) for da in arrays)


def area_mean_latlon(da):
    """
    Cosine-latitude weighted area mean over lat-lon.
    """
    w = np.cos(np.deg2rad(da.lat))
    return da.weighted(w).mean(("lat", "lon"))


def zscore_by_reference(da, ref):
    """
    Standardize da using the mean and std of ref.
    This is better than standardizing only within 2015 because it preserves
    whether 2015 is strong or weak relative to the climatological distribution.
    """
    return (da - ref.mean("time")) / ref.std("time")


def get_cold_phase_sign(t2m_mode):
    """
    Choose MCA sign so that positive MCA score corresponds to cold T2m phase
    over the Yangtze River basin.

    If the T2m spatial mode is already negative on average, keep sign.
    If the T2m spatial mode is positive on average, multiply both left and right modes by -1.
    """
    mean_val = float(t2m_mode.mean(skipna=True))

    if mean_val < 0:
        return 1.0
    else:
        return -1.0


def weighted_spatial_projection(field, pattern, use_coslat=True):
    """
    Project field(time, lat, lon) onto fixed pattern(lat, lon).

    Parameters
    ----------
    field : xr.DataArray
        time x lat x lon field.
    pattern : xr.DataArray
        lat x lon MCA spatial mode.
    use_coslat : bool
        If True, use cos(lat) as spatial area weight.

    Returns
    -------
    pc : xr.DataArray
        Projected time coefficient.
    """
    field, pattern = xr.align(field, pattern, join="inner")

    valid = np.isfinite(pattern)

    if use_coslat:
        w = np.cos(np.deg2rad(field.lat))
    else:
        w = xr.ones_like(field.lat)

    numerator = (
        field.where(valid)
        * pattern.where(valid)
        * w
    ).sum(("lat", "lon"), skipna=True)

    denominator = (
        pattern.where(valid) ** 2
        * w
    ).sum(("lat", "lon"), skipna=True)

    pc = numerator / denominator

    return pc


def effective_sample_size(x, y):
    """
    Effective sample size correction based on lag-1 autocorrelation:

        Neff = N * (1 - r1x*r1y) / (1 + r1x*r1y)

    This is used for correlation significance with serially correlated data.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)

    if n <= 3:
        return np.nan

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

    if np.abs(denom) < 1.0e-8:
        neff = n
    else:
        neff = n * (1 - r1x * r1y) / denom

    neff = np.clip(neff, 3, n)

    return float(neff)


def corr_eff_p(x, y):
    """
    Pearson correlation and p value using effective sample size correction.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) <= 5:
        return np.nan, np.nan, np.nan

    r = pearsonr(x, y)[0]
    neff = effective_sample_size(x, y)

    if not np.isfinite(neff) or neff <= 2 or np.abs(r) >= 1:
        return r, neff, np.nan

    df = neff - 2
    tval = r * np.sqrt(df / (1 - r ** 2))
    p = 2 * student_t.sf(np.abs(tval), df)

    return r, neff, p


def p_text(p):
    """
    Format p value for plotting.
    """
    if not np.isfinite(p):
        return "p=nan"
    elif p < 0.01:
        return "p<0.01"
    elif p < 0.05:
        return "p<0.05"
    else:
        return f"p={p:.2f}"


def contiguous_segments(mask):
    """
    Convert a Boolean mask into continuous index segments.
    """
    mask = np.asarray(mask).astype(bool)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return []

    split_pos = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, split_pos)

    return [(int(g[0]), int(g[-1])) for g in groups]


def setup_axis_style(ax):
    """
    SCI-style axis appearance.
    """
    ax.tick_params(
        direction="out",
        length=4.5,
        width=1.2,
        labelsize=11
    )

    ax.tick_params(
        which="minor",
        direction="out",
        length=2.5,
        width=1.0
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)


def get_scf_percent(mca_obj, mode=1):
    """
    Get squared covariance fraction in percent.
    """
    try:
        val = float(mca_obj.squared_covariance_fraction().sel(mode=mode).values * 100.0)
    except Exception:
        try:
            val = float(mca_obj.explained_variance_ratio().sel(mode=mode).values * 100.0)
        except Exception:
            val = np.nan

    return val


# ============================================================
# 1. Read data
# ============================================================

print("Reading data...")

v_bp = xr.open_dataset(V_BP_FILE)["v_bp"]
olr_bp = xr.open_dataset(OLR_BP_FILE)["ttr_bp"] / 86400.0
t2m_bp = xr.open_dataset(T2M_BP_FILE)["t2m_bp"]

# Select time
v_bp = v_bp.sel(time=TIME_SLICE)
olr_bp = olr_bp.sel(time=TIME_SLICE)
t2m_bp = t2m_bp.sel(time=TIME_SLICE)

# Longitude transform for right fields
v_bp_180 = transform(v_bp, "lon", "360->180").sortby("lon")
olr_bp_180 = transform(olr_bp, "lon", "360->180").sortby("lon")

# Target T2m field: select broad Yangtze box, then mask with shapefile
t2m_bp = select_latlon_box(
    t2m_bp,
    **T2M_TARGET_BOX
)

t2m_bp = masked(
    t2m_bp,
    YANGTZE_SHP
)

# Right fields
v200_bp = select_latlon_box(
    v_bp_180.sel(level=V200_LEVEL),
    **V200_REGION
)

olr_south_bp = select_latlon_box(
    olr_bp_180,
    **OLR_REGION
)

# Time alignment
t2m_bp, v200_bp, olr_south_bp = align_on_common_time(
    t2m_bp,
    v200_bp,
    olr_south_bp
)

print("Data loaded.")
print(f"T2m shape       : {t2m_bp.shape}")
print(f"V200 shape      : {v200_bp.shape}")
print(f"OLR south shape : {olr_south_bp.shape}")


# ============================================================
# 2. Fit climatological MCA modes
# ============================================================

print("Fitting MCA modes...")

if EXCLUDE_TARGET_YEAR_FROM_MCA:
    print(f"Leave-one-year-out MCA: excluding {TARGET_YEAR}.")
    fit_mask = t2m_bp.time.dt.year != TARGET_YEAR

    t2m_fit = t2m_bp.where(fit_mask, drop=True)
    v200_fit = v200_bp.where(fit_mask, drop=True)
    olr_fit = olr_south_bp.where(fit_mask, drop=True)

else:
    print("All-year MCA: using 1961-2022 including 2015.")
    t2m_fit = t2m_bp
    v200_fit = v200_bp
    olr_fit = olr_south_bp


# MCA-A: T2m & V200
mca_t2m_v200 = xeofs.cross.MCA(
    n_modes=N_MODES,
    standardize=False,
    use_coslat=True
).fit(
    t2m_fit,
    v200_fit,
    dim="time"
)

# MCA-B: T2m & OLR
mca_t2m_olr = xeofs.cross.MCA(
    n_modes=N_MODES,
    standardize=False,
    use_coslat=True
).fit(
    t2m_fit,
    olr_fit,
    dim="time"
)

scf_north = get_scf_percent(mca_t2m_v200, mode=1)
scf_south = get_scf_percent(mca_t2m_olr, mode=1)

print("MCA fitted.")
print(f"North MCA SCF mode 1: {scf_north:.2f}%")
print(f"South MCA SCF mode 1: {scf_south:.2f}%")


# ============================================================
# 3. Extract MCA spatial modes and set cold phase positive
# ============================================================

print("Extracting MCA modes...")

# North MCA
t2m_mode_north = mca_t2m_v200.components()[0].sel(mode=1)
v200_mode = mca_t2m_v200.components()[1].sel(mode=1)

sign_north = get_cold_phase_sign(t2m_mode_north)

t2m_mode_north = t2m_mode_north * sign_north
v200_mode = v200_mode * sign_north


# South MCA
t2m_mode_south = mca_t2m_olr.components()[0].sel(mode=1)
olr_mode = mca_t2m_olr.components()[1].sel(mode=1)

sign_south = get_cold_phase_sign(t2m_mode_south)

t2m_mode_south = t2m_mode_south * sign_south
olr_mode = olr_mode * sign_south

print(f"North MCA sign: {sign_north:+.0f}")
print(f"South MCA sign: {sign_south:+.0f}")


# ============================================================
# 4. Project all years onto fixed MCA modes
# ============================================================

print("Projecting fields onto fixed MCA modes...")

# Project T2m side and right-field side
pc_t2m_north_all = weighted_spatial_projection(
    t2m_bp,
    t2m_mode_north,
    use_coslat=True
).rename("north_t2m_side_pc")

pc_v200_all = weighted_spatial_projection(
    v200_bp,
    v200_mode,
    use_coslat=True
).rename("north_v200_pc")

pc_t2m_south_all = weighted_spatial_projection(
    t2m_bp,
    t2m_mode_south,
    use_coslat=True
).rename("south_t2m_side_pc")

pc_olr_all = weighted_spatial_projection(
    olr_south_bp,
    olr_mode,
    use_coslat=True
).rename("south_olr_pc")


# Yangtze regional cold T2m index.
# Since lower T2m means colder, multiply by -1.
t2m_yangtze_all = area_mean_latlon(t2m_bp).rename("yangtze_t2m_bp")
cold_t2m_raw_all = (-1.0 * t2m_yangtze_all).rename("cold_t2m_raw")


# Standardize all indices using the full 1961-2022 reference distribution.
cold_t2m_index_all = zscore_by_reference(
    cold_t2m_raw_all,
    cold_t2m_raw_all
).rename("cold_t2m_index")

north_t2m_index_all = zscore_by_reference(
    pc_t2m_north_all,
    pc_t2m_north_all
).rename("north_t2m_side_index")

north_v200_index_all = zscore_by_reference(
    pc_v200_all,
    pc_v200_all
).rename("north_v200_mca_index")

south_t2m_index_all = zscore_by_reference(
    pc_t2m_south_all,
    pc_t2m_south_all
).rename("south_t2m_side_index")

south_olr_index_all = zscore_by_reference(
    pc_olr_all,
    pc_olr_all
).rename("south_olr_mca_index")


# Align before constructing joint index
(
    cold_t2m_index_all,
    north_t2m_index_all,
    north_v200_index_all,
    south_t2m_index_all,
    south_olr_index_all
) = align_on_common_time(
    cold_t2m_index_all,
    north_t2m_index_all,
    north_v200_index_all,
    south_t2m_index_all,
    south_olr_index_all
)

joint_raw_all = (
    north_v200_index_all + south_olr_index_all
).rename("joint_raw_index")

joint_index_all = zscore_by_reference(
    joint_raw_all,
    joint_raw_all
).rename("joint_north_south_index")


# ============================================================
# 5. Select 2015 indices
# ============================================================

ds_index_all = xr.Dataset(
    {
        "cold_t2m_index": cold_t2m_index_all,
        "north_t2m_side_index": north_t2m_index_all,
        "north_v200_mca_index": north_v200_index_all,
        "south_t2m_side_index": south_t2m_index_all,
        "south_olr_mca_index": south_olr_index_all,
        "joint_north_south_index": joint_index_all,
    }
)

ds_index_2015 = ds_index_all.sel(time=slice(TARGET_START, TARGET_END))

ds_index_2015.attrs["description"] = (
    "2015 JJA indices projected onto climatological MCA modes. "
    "Positive cold_t2m_index denotes colder-than-normal 10-30-day T2m phase over the Yangtze River basin. "
    "Positive north_v200_mca_index denotes the cold-favorable phase of the northern V200 MCA mode. "
    "Positive south_olr_mca_index denotes the cold-favorable phase of the southern OLR MCA mode."
)

ds_index_2015.attrs["mca_period"] = "1961-2022 JJA"
ds_index_2015.attrs["target_year"] = str(TARGET_YEAR)
ds_index_2015.attrs["exclude_target_year_from_mca"] = str(EXCLUDE_TARGET_YEAR_FROM_MCA)
ds_index_2015.attrs["north_mca_scf_mode1_percent"] = f"{scf_north:.3f}"
ds_index_2015.attrs["south_mca_scf_mode1_percent"] = f"{scf_south:.3f}"

ds_index_2015.to_netcdf(SAVE_NC)

print(f"Saved 2015 indices to: {SAVE_NC}")


# Extract variables for plotting
cold_t2m_index = ds_index_2015["cold_t2m_index"]
north_t2m_index = ds_index_2015["north_t2m_side_index"]
north_v200_index = ds_index_2015["north_v200_mca_index"]
south_t2m_index = ds_index_2015["south_t2m_side_index"]
south_olr_index = ds_index_2015["south_olr_mca_index"]
joint_index = ds_index_2015["joint_north_south_index"]


# ============================================================
# 6. Correlation diagnostics
# ============================================================

r_n_cold, neff_n_cold, p_n_cold = corr_eff_p(
    north_v200_index.values,
    cold_t2m_index.values
)

r_s_cold, neff_s_cold, p_s_cold = corr_eff_p(
    south_olr_index.values,
    cold_t2m_index.values
)

r_joint_cold, neff_joint_cold, p_joint_cold = corr_eff_p(
    joint_index.values,
    cold_t2m_index.values
)

r_n_pair, neff_n_pair, p_n_pair = corr_eff_p(
    north_t2m_index.values,
    north_v200_index.values
)

r_s_pair, neff_s_pair, p_s_pair = corr_eff_p(
    south_t2m_index.values,
    south_olr_index.values
)

r_ns, neff_ns, p_ns = corr_eff_p(
    north_v200_index.values,
    south_olr_index.values
)

print("")
print("========== 2015 projection correlations ==========")
print(f"Cold T2m vs North V200 index : r={r_n_cold:.2f}, Neff={neff_n_cold:.1f}, {p_text(p_n_cold)}")
print(f"Cold T2m vs South OLR index  : r={r_s_cold:.2f}, Neff={neff_s_cold:.1f}, {p_text(p_s_cold)}")
print(f"Cold T2m vs Joint index      : r={r_joint_cold:.2f}, Neff={neff_joint_cold:.1f}, {p_text(p_joint_cold)}")
print(f"North MCA pair consistency   : r={r_n_pair:.2f}, Neff={neff_n_pair:.1f}, {p_text(p_n_pair)}")
print(f"South MCA pair consistency   : r={r_s_pair:.2f}, Neff={neff_s_pair:.1f}, {p_text(p_s_pair)}")
print(f"North vs South index         : r={r_ns:.2f}, Neff={neff_ns:.1f}, {p_text(p_ns)}")


# ============================================================
# 7. Plot SCI-style figure
# ============================================================

print("Drawing figure...")

dates = pd.to_datetime(cold_t2m_index.time.values)

cold_mask = np.asarray(cold_t2m_index.values >= COLD_PHASE_THRESHOLD)
cold_segments = contiguous_segments(cold_mask)

joint_favorable_mask = np.asarray(
    (north_v200_index.values > 0) &
    (south_olr_index.values > 0)
)

fig = plt.figure(figsize=(13.6, 8.4))

gs = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    width_ratios=[1.48, 1.0],
    height_ratios=[1.0, 1.0],
    wspace=0.28,
    hspace=0.34
)


# ------------------------------------------------------------
# (a) Main projected indices
# ------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 1])

ax1.plot(
    dates,
    cold_t2m_index,
    color="black",
    lw=2.4,
    label=r"$-T2m$"
)

ax1.plot(
    dates,
    joint_index,
    color="#8172B2",
    lw=1.65,
    ls="--",
    label="Joint index"
)


ax1.axhline(0, color="0.35", lw=1.0, ls="--")
ax1.axhline(COLD_PHASE_THRESHOLD, color="0.6", lw=0.9, ls=":")

ax1.set_ylim(-4.0, 4.0)
ax1.set_ylabel("Standardized index", fontsize=12)

ax1.xaxis.set_major_locator(mdates.DayLocator(interval=15))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=5))

ax1.legend(
    frameon=False,
    fontsize=10.2,
    ncol=2,
    loc="upper left",
    columnspacing=1.1,
    handlelength=2.6
)

ax1.text(
    0.01,
    1.045,
    "(c) Projected MCA indices in 2015",
    transform=ax1.transAxes,
    fontsize=15,
    fontweight="bold"
)

ax1.text(
    0.83,
    0.045,
    f"r = {r_joint_cold:.2f}",
    transform=ax1.transAxes,
    fontsize=10.5,
    color="0.20"
)

setup_axis_style(ax1)


# ------------------------------------------------------------
# (b) North MCA pair consistency
# ------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 0])

ax2.plot(
    dates,
    north_t2m_index,
    color="black",
    lw=2.2,
    label="-T2m"
)

ax2.plot(
    dates,
    north_v200_index,
    color="#C44E52",
    lw=1.9,
    label="V200"
)

ax2.axhline(0, color="0.35", lw=1.0, ls="--")

ax2.set_ylim(-4.0, 4.0)
ax2.set_ylabel("Standardized index", fontsize=12)

ax2.xaxis.set_major_locator(mdates.DayLocator(interval=15))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=5))

ax2.legend(
    frameon=False,
    fontsize=10.5,
    loc="upper left"
)

ax2.text(
    0.01,
    1.045,
    "(a) North MCA pair consistency",
    transform=ax2.transAxes,
    fontsize=15,
    fontweight="bold"
)

ax2.text(
    0.97,
    0.96,
    f"SCF = {scf_north:.1f}%\n"
    f"r = {r_n_pair:.2f}",
    transform=ax2.transAxes,
    fontsize=11.2,
    va="top",
    ha="right"
)

setup_axis_style(ax2)


# ------------------------------------------------------------
# (c) South MCA pair consistency
# ------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])

ax3.plot(
    dates,
    south_t2m_index,
    color="black",
    lw=2.2,
    label="-T2m"
)

ax3.plot(
    dates,
    south_olr_index,
    color="#4C72B0",
    lw=1.9,
    label="OLR"
)

ax3.axhline(0, color="0.35", lw=1.0, ls="--")

ax3.set_ylim(-4.0, 4.0)
ax3.set_ylabel("Standardized index", fontsize=12)

ax3.xaxis.set_major_locator(mdates.DayLocator(interval=15))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax3.xaxis.set_minor_locator(mdates.DayLocator(interval=5))

ax3.legend(
    frameon=False,
    fontsize=10.2,
    loc="upper left"
)

ax3.text(
    0.01,
    1.045,
    "(b) South MCA pair consistency",
    transform=ax3.transAxes,
    fontsize=15,
    fontweight="bold"
)

ax3.text(
    0.97,
    0.96,
    f"SCF = {scf_south:.1f}%\n"
    f"r = {r_s_pair:.2f}",
    transform=ax3.transAxes,
    fontsize=11.2,
    va="top",
    ha="right"
)

setup_axis_style(ax3)


# ------------------------------------------------------------
# (d) North-south phase-space relationship
# ------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])

x_phase = north_v200_index.values
y_phase = south_olr_index.values
c_phase = cold_t2m_index.values

levels = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
cmap = mpl.cm.get_cmap("RdBu", len(levels) - 1)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

sc = ax4.scatter(
    x_phase,
    y_phase,
    c=c_phase,
    s=45,
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    linewidth=0.35,
    zorder=3
)


# Highlight joint-favorable quadrant background
xlim_tmp = (-3.1, 3.1)
ylim_tmp = (-3.1, 3.1)

ax4.axvspan(
    0,
    xlim_tmp[1],
    ymin=0.5,
    ymax=1.0,
    color="#8172B2",
    alpha=0.08,
    lw=0
)

ax4.axvline(0, color="0.35", lw=1.1, ls="--")
ax4.axhline(0, color="0.35", lw=1.1, ls="--")

ax4.set_xlim(xlim_tmp)
ax4.set_ylim(ylim_tmp)

ax4.set_xlabel("North V200 MCA index", fontsize=12)
ax4.set_ylabel("South OLR MCA index", fontsize=12)

ax4.xaxis.set_minor_locator(MultipleLocator(0.5))
ax4.yaxis.set_minor_locator(MultipleLocator(0.5))

ax4.text(
    0.01,
    1.045,
    "(d) North-South phase relationship",
    transform=ax4.transAxes,
    fontsize=15,
    fontweight="bold"
)

ax4.text(
    0.97,
    0.96,
    f"r(North, South) = {r_ns:.2f}",
    transform=ax4.transAxes,
    fontsize=11,
    ha="right",
    va="top"
)


setup_axis_style(ax4)

cax = inset_axes(
    ax4,
    width="4.2%",
    height="82%",
    loc="center right",
    bbox_to_anchor=(0.13, 0.0, 1, 1),
    bbox_transform=ax4.transAxes,
    borderpad=0
)

cb = fig.colorbar(sc, cax=cax, orientation="vertical", boundaries=levels, ticks=levels)
cb.ax.tick_params(labelsize=9, length=3)
cb.set_label(r"$-T2m$ index", fontsize=10)
cb.outline.set_linewidth(1.3)


# ------------------------------------------------------------
# Figure title and saving
# ------------------------------------------------------------
mca_note = "leave-one-year-out MCA" if EXCLUDE_TARGET_YEAR_FROM_MCA else "all-year MCA"

plt.savefig(SAVE_FIG, dpi=600, bbox_inches="tight")
plt.savefig(SAVE_PDF, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {SAVE_FIG}")
print(f"Saved PDF to   : {SAVE_PDF}")
print("Task 3 finished.")
