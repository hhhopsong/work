# -*- coding: utf-8 -*-
"""
Task 7:
Compare 2015 Yangtze River basin cold summer with other cold summers
based on projections onto climatological MCA/SVD modes.

Purpose:
1. Refit climatological MCA modes using 1961-2022 JJA 10-30-day filtered data.
2. Use MCA-A: Yangtze T2m & northern V200.
3. Use MCA-B: Yangtze T2m & southern OLR.
4. Project all years onto fixed MCA modes.
5. Compare 2015 with other cold summers.
6. Diagnose whether 2015 shows unusually strong joint northern-southern cold-favorable forcing.

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

SAVE_FIG = fr"{OUT_DIR_FIG}/SVD_2015_vs_cold_summers.png"
SAVE_PDF = fr"{OUT_DIR_FIG}/SVD_2015_vs_cold_summers.pdf"


# -------------------------
# Time settings
# -------------------------
TIME_SLICE = slice("1961-01-01", "2022-12-31")

TARGET_YEAR = 2015
TARGET_START = f"{TARGET_YEAR}-06-01"
TARGET_END = f"{TARGET_YEAR}-08-31"

# For final paper, True is recommended.
# This avoids using 2015 itself to define the MCA modes used to diagnose 2015.
EXCLUDE_TARGET_YEAR_FROM_MCA = True


# -------------------------
# Cold / warm summer lists
# -------------------------
# IMPORTANT:
# Replace this list with your objectively selected cold summers
# based on Yangtze JJA or midsummer T2m anomaly <= -0.5 sigma.
#
# The current list is only a placeholder.
COLD_YEARS = [1965, 1966, 1968, 1974, 1976, 1980, 1982, 1983, 1986, 1987, 1989, 1992, 1993, 1997, 1999, 2004, 2008, 2014]

# Optional.
# Replace this list with warm summers based on Yangtze T2m anomaly >= +0.5 sigma.
# If you do not want to show warm summers separately, leave it as an empty list.
WARM_YEARS = [
    # 1961, 1967, 1978, 1994, 2006, 2013, 2017, 2022
]


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
# The actual target area is still controlled by the Yangtze shapefile.
T2M_TARGET_BOX = {
    "lon_min": 90,
    "lon_max": 125,
    "lat_min": 20,
    "lat_max": 40,
}


# -------------------------
# Diagnostic settings
# -------------------------
COLD_PHASE_THRESHOLD = 1.0

# Joint index definition:
# "sum"     : standardized north + standardized south
# "mean"    : 0.5 * north + 0.5 * south
# "product" : north * south, emphasizing simultaneous same-phase occurrence
JOINT_METHOD = "mean"


# -------------------------
# Plot settings
# -------------------------
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.linewidth": 1.2,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
    }
)


# ============================================================
# Utility functions
# ============================================================

def select_latlon_box(da, lon_min, lon_max, lat_min, lat_max):
    """
    Select a lon-lat box safely regardless of whether latitude is ascending or descending.
    """
    if "lon" not in da.coords or "lat" not in da.coords:
        raise ValueError("Input DataArray must contain lon and lat coordinates.")

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
    """
    return (da - ref.mean("time")) / ref.std("time")


def zscore_by_reference_1d(da, dim_name):
    """
    Standardize one-dimensional DataArray along a given dimension.
    """
    return (da - da.mean(dim_name)) / da.std(dim_name)


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
    Effective sample size correction based on lag-1 autocorrelation.

    Neff = N * (1 - r1x*r1y) / (1 + r1x*r1y)
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

    denom = 1.0 + r1x * r1y

    if np.abs(denom) < 1.0e-8:
        neff = n
    else:
        neff = n * (1.0 - r1x * r1y) / denom

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

    df = neff - 2.0
    tval = r * np.sqrt(df / (1.0 - r ** 2))
    p = 2.0 * student_t.sf(np.abs(tval), df)

    return r, neff, p


def p_text(p):
    """
    Format p value.
    """
    if not np.isfinite(p):
        return "p=nan"
    elif p < 0.01:
        return "p<0.01"
    elif p < 0.05:
        return "p<0.05"
    else:
        return f"p={p:.2f}"


def setup_axis_style(ax):
    """
    SCI-style axis appearance.
    """
    ax.tick_params(
        direction="out",
        length=4.5,
        width=1.2,
        labelsize=10.5
    )

    ax.tick_params(
        which="minor",
        direction="out",
        length=2.5,
        width=1.0
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


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


def annual_rms(da):
    """
    Annual RMS amplitude of daily index.
    """
    return da.groupby("time.year").map(lambda x: np.sqrt((x ** 2).mean("time")))


def annual_extreme_frequency(da, threshold=1.0):
    """
    Annual percentage/fraction of days above threshold.
    """
    return da.groupby("time.year").map(lambda x: (x >= threshold).mean("time"))


def annual_joint_favorable_frequency(north, south):
    """
    Annual fraction of days when both northern and southern indices are in
    cold-favorable positive phases.
    """
    joint_bool = xr.where((north > 0.0) & (south > 0.0), 1.0, 0.0)
    return joint_bool.groupby("time.year").mean("time")


def annual_corr(x, y):
    """
    Compute year-by-year daily correlation between two indices.
    """
    years = np.unique(x.time.dt.year.values)
    vals = []

    for yy in years:
        xs = x.sel(time=x.time.dt.year == yy).values
        ys = y.sel(time=y.time.dt.year == yy).values

        mask = np.isfinite(xs) & np.isfinite(ys)

        if mask.sum() >= 10:
            vals.append(pearsonr(xs[mask], ys[mask])[0])
        else:
            vals.append(np.nan)

    return xr.DataArray(
        vals,
        coords={"year": years},
        dims=["year"],
        name="annual_corr"
    )


def percentile_rank(sample, value):
    """
    Percentile rank of value within sample.
    """
    sample = np.asarray(sample)
    sample = sample[np.isfinite(sample)]

    if sample.size == 0 or not np.isfinite(value):
        return np.nan

    return 100.0 * np.sum(sample <= value) / sample.size


def safe_pearsonr(x, y):
    """
    Pearson correlation with finite mask.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 5:
        return np.nan, np.nan

    return pearsonr(x[mask], y[mask])


# ============================================================
# 1. Read data
# ============================================================

print("Reading data...")

v_bp = xr.open_dataset(V_BP_FILE)["v_bp"]
olr_bp = xr.open_dataset(OLR_BP_FILE)["ttr_bp"] / 86400.0
t2m_bp = xr.open_dataset(T2M_BP_FILE)["t2m_bp"]

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


# Standardize all indices using full reference distribution.
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


# Joint index
if JOINT_METHOD == "sum":
    joint_raw_all = (
        north_v200_index_all + south_olr_index_all
    ).rename("joint_raw_index")

elif JOINT_METHOD == "mean":
    joint_raw_all = (
        0.5 * north_v200_index_all + 0.5 * south_olr_index_all
    ).rename("joint_raw_index")

elif JOINT_METHOD == "product":
    joint_raw_all = (
        north_v200_index_all * south_olr_index_all
    ).rename("joint_raw_index")

else:
    raise ValueError("JOINT_METHOD must be one of: 'sum', 'mean', 'product'.")

joint_index_all = zscore_by_reference(
    joint_raw_all,
    joint_raw_all
).rename("joint_north_south_index")


# ============================================================
# 5. Build daily index datasets
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

ds_index_all.attrs["description"] = (
    "Daily JJA indices projected onto climatological MCA modes. "
    "Positive cold_t2m_index denotes colder-than-normal 10-30-day T2m phase over the Yangtze River basin. "
    "Positive northern and southern MCA indices denote cold-favorable phases."
)

ds_index_all.attrs["mca_period"] = "1961-2022 JJA"
ds_index_all.attrs["target_year"] = str(TARGET_YEAR)
ds_index_all.attrs["exclude_target_year_from_mca"] = str(EXCLUDE_TARGET_YEAR_FROM_MCA)
ds_index_all.attrs["joint_method"] = JOINT_METHOD
ds_index_all.attrs["north_mca_scf_mode1_percent"] = f"{scf_north:.3f}"
ds_index_all.attrs["south_mca_scf_mode1_percent"] = f"{scf_south:.3f}"

ds_index_2015 = ds_index_all.sel(time=slice(TARGET_START, TARGET_END))




# ============================================================
# 6. Annual diagnostics for comparison
# ============================================================

print("Computing annual diagnostics...")

annual_north_amp = annual_rms(north_v200_index_all).rename("north_amp")
annual_south_amp = annual_rms(south_olr_index_all).rename("south_amp")
annual_joint_amp = annual_rms(joint_index_all).rename("joint_amp")

annual_cold_freq = annual_extreme_frequency(
    cold_t2m_index_all,
    threshold=COLD_PHASE_THRESHOLD
).rename("cold_event_frequency")

annual_joint_freq = annual_joint_favorable_frequency(
    north_v200_index_all,
    south_olr_index_all
).rename("joint_favorable_frequency")

annual_joint_cold_corr = annual_corr(
    joint_index_all,
    cold_t2m_index_all
).rename("joint_cold_corr")

annual_north_south_corr = annual_corr(
    north_v200_index_all,
    south_olr_index_all
).rename("north_south_corr")

annual_north_cold_corr = annual_corr(
    north_v200_index_all,
    cold_t2m_index_all
).rename("north_cold_corr")

annual_south_cold_corr = annual_corr(
    south_olr_index_all,
    cold_t2m_index_all
).rename("south_cold_corr")


ds_annual = xr.Dataset(
    {
        "north_amp": annual_north_amp,
        "south_amp": annual_south_amp,
        "joint_amp": annual_joint_amp,
        "cold_event_frequency": annual_cold_freq,
        "joint_favorable_frequency": annual_joint_freq,
        "joint_cold_corr": annual_joint_cold_corr,
        "north_south_corr": annual_north_south_corr,
        "north_cold_corr": annual_north_cold_corr,
        "south_cold_corr": annual_south_cold_corr,
    }
)

ds_annual.attrs["description"] = (
    "Annual comparison of projected MCA indices. "
    "Amplitude is defined as the annual RMS of daily standardized MCA indices. "
    "Joint-favorable frequency is the fraction of JJA days when both northern and southern MCA indices are positive."
)

ds_annual.attrs["cold_years"] = ",".join([str(y) for y in COLD_YEARS])
ds_annual.attrs["warm_years"] = ",".join([str(y) for y in WARM_YEARS])
ds_annual.attrs["cold_phase_threshold"] = str(COLD_PHASE_THRESHOLD)
ds_annual.attrs["joint_method"] = JOINT_METHOD
ds_annual.attrs["exclude_target_year_from_mca"] = str(EXCLUDE_TARGET_YEAR_FROM_MCA)


# ============================================================
# 7. 2015 daily correlation diagnostics
# ============================================================

cold_t2m_index_2015 = ds_index_2015["cold_t2m_index"]
north_t2m_index_2015 = ds_index_2015["north_t2m_side_index"]
north_v200_index_2015 = ds_index_2015["north_v200_mca_index"]
south_t2m_index_2015 = ds_index_2015["south_t2m_side_index"]
south_olr_index_2015 = ds_index_2015["south_olr_mca_index"]
joint_index_2015 = ds_index_2015["joint_north_south_index"]

r_n_cold, neff_n_cold, p_n_cold = corr_eff_p(
    north_v200_index_2015.values,
    cold_t2m_index_2015.values
)

r_s_cold, neff_s_cold, p_s_cold = corr_eff_p(
    south_olr_index_2015.values,
    cold_t2m_index_2015.values
)

r_joint_cold, neff_joint_cold, p_joint_cold = corr_eff_p(
    joint_index_2015.values,
    cold_t2m_index_2015.values
)

r_n_pair, neff_n_pair, p_n_pair = corr_eff_p(
    north_t2m_index_2015.values,
    north_v200_index_2015.values
)

r_s_pair, neff_s_pair, p_s_pair = corr_eff_p(
    south_t2m_index_2015.values,
    south_olr_index_2015.values
)

r_ns, neff_ns, p_ns = corr_eff_p(
    north_v200_index_2015.values,
    south_olr_index_2015.values
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
# 8. Plot SCI-style annual comparison figure
# ============================================================

print("Drawing SCI-style annual comparison figure...")

years = ds_annual.year.values

north_amp = ds_annual["north_amp"].values
south_amp = ds_annual["south_amp"].values
joint_amp = ds_annual["joint_amp"].values

joint_freq = ds_annual["joint_favorable_frequency"].values * 100.0
cold_freq = ds_annual["cold_event_frequency"].values * 100.0

joint_cold_corr_annual = ds_annual["joint_cold_corr"].values
north_cold_corr_annual = ds_annual["north_cold_corr"].values
south_cold_corr_annual = ds_annual["south_cold_corr"].values

cold_mask_year = np.isin(years, COLD_YEARS)
warm_mask_year = np.isin(years, WARM_YEARS)
target_mask_year = years == TARGET_YEAR
neutral_mask_year = ~(cold_mask_year | warm_mask_year)

if not np.any(target_mask_year):
    raise ValueError(f"TARGET_YEAR {TARGET_YEAR} is not found in annual diagnostics.")

# 2015 values
north_2015 = north_amp[target_mask_year][0]
south_2015 = south_amp[target_mask_year][0]
joint_2015 = joint_amp[target_mask_year][0]
joint_freq_2015 = joint_freq[target_mask_year][0]
cold_freq_2015 = cold_freq[target_mask_year][0]
corr_2015 = joint_cold_corr_annual[target_mask_year][0]

joint_pct_all = percentile_rank(joint_amp, joint_2015)
joint_pct_cold = percentile_rank(joint_amp[cold_mask_year], joint_2015)

freq_pct_all = percentile_rank(joint_freq, joint_freq_2015)
freq_pct_cold = percentile_rank(joint_freq[cold_mask_year], joint_freq_2015)

corr_pct_all = percentile_rank(joint_cold_corr_annual, corr_2015)
corr_pct_cold = percentile_rank(joint_cold_corr_annual[cold_mask_year], corr_2015)

# Overall relationship
r_freq, p_freq = safe_pearsonr(joint_freq, cold_freq)
r_amp_cold, p_amp_cold = safe_pearsonr(joint_amp, cold_freq)

# Colorblind-friendly colors
color_neutral = "0.70"
color_cold = "#0072B2"
color_warm = "#D55E00"
color_2015 = "#CC3311"


fig = plt.figure(figsize=(12.8, 9.2))

gs = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    figure=fig,
    wspace=0.28,
    hspace=0.35
)


# ============================================================
# (a) Phase-space of annual northern and southern MCA amplitudes
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])

ax1.scatter(
    north_amp[neutral_mask_year],
    south_amp[neutral_mask_year],
    s=38,
    c=color_neutral,
    edgecolor="white",
    linewidth=0.4,
    alpha=0.85,
    label="Other years",
    zorder=2
)

if np.any(warm_mask_year):
    ax1.scatter(
        north_amp[warm_mask_year],
        south_amp[warm_mask_year],
        s=48,
        c=color_warm,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.90,
        label="Warm summers",
        zorder=3
    )

ax1.scatter(
    north_amp[cold_mask_year],
    south_amp[cold_mask_year],
    s=54,
    c=color_cold,
    edgecolor="black",
    linewidth=0.45,
    alpha=0.95,
    label="Cold summers",
    zorder=4
)

ax1.scatter(
    north_2015,
    south_2015,
    s=190,
    marker="*",
    c=color_2015,
    edgecolor="black",
    linewidth=0.8,
    label="2015",
    zorder=6
)

ax1.axvline(
    np.nanmedian(north_amp),
    color="0.35",
    lw=1.0,
    ls="--",
    zorder=1
)

ax1.axhline(
    np.nanmedian(south_amp),
    color="0.35",
    lw=1.0,
    ls="--",
    zorder=1
)

ax1.set_xlabel("Northern MCA amplitude")
ax1.set_ylabel("Southern MCA amplitude")

ax1.set_title(
    "(a) Annual MCA amplitude phase space",
    loc="left",
    fontweight="bold"
)

ax1.text(
    0.03,
    0.96,
    f"2015: North={north_2015:.2f}, South={south_2015:.2f}",
    transform=ax1.transAxes,
    ha="left",
    va="top"
)

ax1.legend(
    frameon=False,
    loc="lower right",
    handletextpad=0.4,
    borderpad=0.2
)

setup_axis_style(ax1)


# ============================================================
# (b) Joint MCA amplitude distribution
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])

box_data = [
    joint_amp[~target_mask_year],
    joint_amp[cold_mask_year & ~target_mask_year],
]

box = ax2.boxplot(
    box_data,
    widths=0.52,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2)
)

box["boxes"][0].set_facecolor("0.85")
box["boxes"][1].set_facecolor(color_cold)
box["boxes"][1].set_alpha(0.55)

rng = np.random.default_rng(42)

for i, vals in enumerate(box_data, start=1):
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    xj = rng.normal(i, 0.045, size=len(vals))

    ax2.scatter(
        xj,
        vals,
        s=25,
        c="white",
        edgecolor="0.35",
        linewidth=0.6,
        alpha=0.85,
        zorder=3
    )

ax2.scatter(
    1,
    joint_2015,
    s=170,
    marker="*",
    c=color_2015,
    edgecolor="black",
    linewidth=0.8,
    zorder=5
)

ax2.scatter(
    2,
    joint_2015,
    s=170,
    marker="*",
    c=color_2015,
    edgecolor="black",
    linewidth=0.8,
    zorder=5
)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(["All years\nexcept 2015", "Cold summers\nexcept 2015"])
ax2.set_ylabel("Joint MCA amplitude")

ax2.set_title(
    "(b) Strength of joint northern-southern forcing",
    loc="left",
    fontweight="bold"
)

ax2.text(
    0.05,
    0.96,
    f"2015 percentile:\nAll years = {joint_pct_all:.0f}%\nCold summers = {joint_pct_cold:.0f}%",
    transform=ax2.transAxes,
    ha="left",
    va="top"
)

setup_axis_style(ax2)


# ============================================================
# (c) Joint-favorable days versus cold-event days
# ============================================================
ax3 = fig.add_subplot(gs[1, 0])

ax3.scatter(
    joint_freq[neutral_mask_year],
    cold_freq[neutral_mask_year],
    s=38,
    c=color_neutral,
    edgecolor="white",
    linewidth=0.4,
    alpha=0.85,
    label="Other years",
    zorder=2
)

if np.any(warm_mask_year):
    ax3.scatter(
        joint_freq[warm_mask_year],
        cold_freq[warm_mask_year],
        s=48,
        c=color_warm,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.90,
        label="Warm summers",
        zorder=3
    )

ax3.scatter(
    joint_freq[cold_mask_year],
    cold_freq[cold_mask_year],
    s=54,
    c=color_cold,
    edgecolor="black",
    linewidth=0.45,
    alpha=0.95,
    label="Cold summers",
    zorder=4
)

ax3.scatter(
    joint_freq_2015,
    cold_freq_2015,
    s=190,
    marker="*",
    c=color_2015,
    edgecolor="black",
    linewidth=0.8,
    label="2015",
    zorder=6
)

mask_fit = np.isfinite(joint_freq) & np.isfinite(cold_freq)

if mask_fit.sum() >= 5:
    coef = np.polyfit(joint_freq[mask_fit], cold_freq[mask_fit], 1)
    xx = np.linspace(
        np.nanmin(joint_freq[mask_fit]),
        np.nanmax(joint_freq[mask_fit]),
        100
    )
    yy = coef[0] * xx + coef[1]

    ax3.plot(
        xx,
        yy,
        color="0.20",
        lw=1.4,
        ls="-",
        zorder=1
    )

ax3.set_xlabel("Joint-favorable days (%)")
ax3.set_ylabel(f"Cold days with $-T2m$ index ≥ {COLD_PHASE_THRESHOLD:.1f} (%)")

ax3.set_title(
    "(c) Co-occurrence of joint forcing and cold events",
    loc="left",
    fontweight="bold"
)

ax3.text(
    0.04,
    0.96,
    f"r = {r_freq:.2f}\n2015 percentile = {freq_pct_all:.0f}%",
    transform=ax3.transAxes,
    ha="left",
    va="top"
)

setup_axis_style(ax3)


# ============================================================
# (d) Annual correlation between joint index and cold T2m
# ============================================================
ax4 = fig.add_subplot(gs[1, 1])

corr_vals = joint_cold_corr_annual

ax4.axhline(
    0,
    color="0.40",
    lw=1.0,
    ls="--",
    zorder=1
)

ax4.bar(
    years,
    corr_vals,
    width=0.78,
    color="0.72",
    edgecolor="none",
    zorder=2,
    label="Other years"
)

if np.any(cold_mask_year):
    ax4.bar(
        years[cold_mask_year],
        corr_vals[cold_mask_year],
        width=0.78,
        color=color_cold,
        edgecolor="none",
        alpha=0.85,
        zorder=3,
        label="Cold summers"
    )

ax4.bar(
    TARGET_YEAR,
    corr_2015,
    width=0.78,
    color=color_2015,
    edgecolor="black",
    linewidth=0.7,
    zorder=5,
    label="2015"
)

ax4.set_xlim(years.min() - 1, years.max() + 1)
ax4.set_ylim(-1.0, 1.0)

ax4.set_xlabel("Year")
ax4.set_ylabel("Correlation: joint index vs. cold T2m index")

ax4.xaxis.set_major_locator(MultipleLocator(10))
ax4.xaxis.set_minor_locator(MultipleLocator(5))
ax4.yaxis.set_major_locator(MultipleLocator(0.25))

ax4.set_title(
    "(d) Annual linkage to Yangtze cooling",
    loc="left",
    fontweight="bold"
)

ax4.text(
    0.04,
    0.96,
    f"2015: r = {corr_2015:.2f}\nPercentile = {corr_pct_all:.0f}%",
    transform=ax4.transAxes,
    ha="left",
    va="top"
)

setup_axis_style(ax4)


# ============================================================
# Figure note and saving
# ============================================================
mca_note = "leave-one-year-out MCA" if EXCLUDE_TARGET_YEAR_FROM_MCA else "all-year MCA"

fig.text(
    0.5,
    0.018,
    f"MCA modes are derived from 1961–2022 JJA 10–30-day filtered fields "
    f"({mca_note}; joint method: {JOINT_METHOD}). Positive indices denote cold-favorable phases.",
    ha="center",
    va="bottom",
    fontsize=10.2
)

plt.savefig(SAVE_FIG, dpi=600, bbox_inches="tight")
plt.savefig(SAVE_PDF, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {SAVE_FIG}")
print(f"Saved PDF to   : {SAVE_PDF}")
print("Task 7 finished.")
