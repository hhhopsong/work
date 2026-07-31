import os

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
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

PYFILE = r"/Volumes/TiPlus7100/PyFile"
DATA = r"/Volumes/TiPlus7100/data"

TIME_SLICE = slice("1961-01-01", "2022-12-31")
LEVELS_USE = [200, 850]
LAT_SLICE = slice(-20, 80)

REGION_180 = {
    "lon": slice(-90, 180),
    "lat": slice(-20, 80),
}


# =========================
# Utility functions
# =========================
def regress_map(field, pc):
    """
    Linear regression of field onto a standardized principal component.
    """
    pc = (pc - pc.mean("time")) / pc.std("time")
    return xr.cov(field, pc, dim="time") / pc.var("time")


def select_july_1961_2022(da):
    """
    Select 1961-2022 and retain only July.
    """
    da = da.sel(time=TIME_SLICE)

    if (
        "time" in da.coords
        and np.issubdtype(da.time.dtype, np.datetime64)
    ):
        da = da.where(da.time.dt.month == 7, drop=True)

    return da


def align_on_common_time(*arrays):
    """
    Align xarray objects only on their common time coordinate.
    """
    common_time = arrays[0].time.values

    for da in arrays[1:]:
        common_time = np.intersect1d(common_time, da.time.values)

    common_time = np.sort(common_time)

    return tuple(da.sel(time=common_time) for da in arrays)


def build_compact_july_axis(time_values, tick_step=10):
    """
    Compress all July days into a continuous plotting coordinate.
    """
    dates = pd.to_datetime(time_values)
    xpos = np.arange(len(dates), dtype=float)

    years = dates.year.to_numpy()
    unique_years = np.unique(years)

    year_starts = []
    year_ends = []
    year_centers = []

    for year in unique_years:
        indices = np.where(years == year)[0]
        year_starts.append(indices[0])
        year_ends.append(indices[-1])
        year_centers.append((indices[0] + indices[-1]) / 2)

    year_starts = np.asarray(year_starts, dtype=float)
    year_ends = np.asarray(year_ends, dtype=float)
    year_centers = np.asarray(year_centers, dtype=float)

    tick_years = unique_years[::tick_step]
    tick_pos = []

    for year in tick_years:
        year_index = np.where(unique_years == year)[0][0]
        tick_pos.append(year_centers[year_index])

    tick_pos = np.asarray(tick_pos, dtype=float)

    return (
        xpos,
        unique_years,
        year_starts,
        year_ends,
        year_centers,
        tick_pos,
        tick_years,
    )


def symmetric_axis_limit(values, minimum=0.1, padding=0.10):
    """
    Return a symmetric y-axis limit around zero.
    """
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return minimum

    maximum = np.nanmax(np.abs(finite_values))

    if not np.isfinite(maximum) or maximum == 0:
        maximum = minimum

    return max(maximum * (1 + padding), minimum)


def positive_axis_limit(*arrays, minimum=1.5, padding=0.15):
    """
    Return a positive y-axis upper limit.
    """
    finite_values = []

    for values in arrays:
        values = np.asarray(values, dtype=float)
        valid = values[np.isfinite(values)]

        if valid.size > 0:
            finite_values.append(valid)

    if len(finite_values) == 0:
        return minimum

    maximum = np.nanmax(np.concatenate(finite_values))

    if not np.isfinite(maximum) or maximum <= 0:
        return minimum

    upper = maximum * (1 + padding)
    upper = np.ceil(upper / 0.2) * 0.2

    return max(float(upper), minimum)


def standardize_over_time(da, *, name=None, eps=1e-12):
    """
    Standardize a DataArray along time.

    For spatial data, every grid point is standardized independently.

    z = (x - time_mean) / time_std

    The reference period consists of all available July daily values
    during 1961-2022.
    """
    if "time" not in da.dims:
        raise ValueError(
            "standardize_over_time requires a 'time' dimension."
        )

    da64 = da.astype("float64")

    time_mean = da64.mean(dim="time", skipna=True)
    time_variance = da64.var(
        dim="time",
        skipna=True,
        ddof=0,
    )

    time_variance = time_variance.clip(min=0)
    time_std = np.sqrt(time_variance)

    valid_std = np.isfinite(time_std) & (time_std > eps)

    standardized = xr.where(
        valid_std,
        (da64 - time_mean) / time_std,
        np.nan,
    ).astype("float32")

    if name is not None:
        standardized = standardized.rename(name)

    standardized.attrs.update(
        {
            "standardization": "z-score",
            "standardization_dimension": "time",
            "standardization_reference_period": (
                "All July daily values during 1961-2022"
            ),
            "standardization_formula": (
                "(x - time_mean) / time_standard_deviation"
            ),
            "standardization_ddof": 0,
        }
    )

    return standardized


def annual_std_from_standardized(standardized_da, *, name):
    """
    Calculate annual July STD from standardized daily data.
    """
    if "time" not in standardized_da.dims:
        raise ValueError(
            "annual_std_from_standardized requires a 'time' dimension."
        )

    annual_variance = (
        standardized_da
        .groupby("time.year")
        .var(
            dim="time",
            skipna=True,
            ddof=0,
        )
    )

    annual_variance = annual_variance.clip(min=0)

    annual_std = (
        np.sqrt(annual_variance)
        .astype("float32")
        .rename(name)
    )

    annual_std.attrs.update(
        {
            "calculation": (
                "Population standard deviation of standardized "
                "daily values within July of each year"
            ),
            "annual_std_ddof": 0,
            "negative_value_policy": (
                "Annual variance was clipped at zero before square root"
            ),
        }
    )

    return annual_std


def validate_nonnegative_std(obj, *, tolerance=1e-7):
    """
    Verify that every variable ending in '_std' is nonnegative.
    """
    if isinstance(obj, xr.DataArray):
        variables = {obj.name or "unnamed_std": obj}

    elif isinstance(obj, xr.Dataset):
        variables = {
            name: da
            for name, da in obj.data_vars.items()
            if name.endswith("_std")
        }

    else:
        raise TypeError(
            "obj must be an xarray DataArray or Dataset."
        )

    diagnostics = {}

    for variable_name, da in variables.items():
        values = np.asarray(da.values, dtype=float)
        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            diagnostics[variable_name] = np.nan
            continue

        minimum = float(np.nanmin(finite_values))
        diagnostics[variable_name] = minimum

        if minimum < -tolerance:
            raise ValueError(
                f"Negative STD detected in {variable_name}: "
                f"minimum={minimum:.8e}"
            )

    return diagnostics


def print_standardization_diagnostic(name, da):
    """
    Print mean and STD of a standardized one-dimensional series.
    """
    mean_value = float(
        da.mean("time", skipna=True).values
    )

    std_value = float(
        da.std(
            "time",
            skipna=True,
            ddof=0,
        ).values
    )

    print(
        f"{name:<28s} "
        f"mean={mean_value: .8f}, "
        f"std={std_value: .8f}"
    )


def effective_sample_size(x, y):
    """
    Estimate effective sample size using lag-1 autocorrelation.
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

    denominator = 1 + r1x * r1y

    if np.abs(denominator) < 1e-6:
        neff = n
    else:
        neff = (
            n
            * (1 - r1x * r1y)
            / denominator
        )

    neff = np.clip(neff, 3, n)

    return float(neff)


def corr_threshold_95(neff, alpha=0.05):
    """
    Two-sided 95% significance threshold for a correlation.
    """
    if not np.isfinite(neff) or neff <= 2:
        return np.nan

    df = neff - 2

    tcrit = student_t.ppf(
        1 - alpha / 2,
        df,
    )

    rcrit = np.sqrt(
        tcrit**2
        / (df + tcrit**2)
    )

    return rcrit


def _as_1d_dataarray(x):
    """
    Convert input to a one-dimensional DataArray.
    """
    if isinstance(x, xr.DataArray):
        return x

    return xr.DataArray(
        np.asarray(x),
        dims=["time"],
    )


def lag_corr(
    x,
    y=None,
    max_lag=30,
    use_neff=True,
    within_same_summer=True,
):
    """
    Calculate lag correlation.

    lag > 0:
        x leads y.

    lag < 0:
        x lags y.
    """
    xda = _as_1d_dataarray(x)

    if y is None:
        yda = xda.copy()
    else:
        yda = _as_1d_dataarray(y)

    if "time" in xda.coords and "time" in yda.coords:
        xda, yda = xr.align(
            xda,
            yda,
            join="inner",
        )

    xval = xda.values.astype(float)
    yval = yda.values.astype(float)

    xstd = np.nanstd(xval)
    ystd = np.nanstd(yval)

    if xstd == 0 or ystd == 0:
        lags = np.arange(
            -max_lag,
            max_lag + 1,
        )

        return (
            lags,
            np.full_like(
                lags,
                np.nan,
                dtype=float,
            ),
            np.full_like(
                lags,
                np.nan,
                dtype=float,
            ),
        )

    xval = (xval - np.nanmean(xval)) / xstd
    yval = (yval - np.nanmean(yval)) / ystd

    has_datetime = (
        within_same_summer
        and "time" in xda.coords
        and np.issubdtype(
            xda.time.dtype,
            np.datetime64,
        )
    )

    if has_datetime:
        years = pd.to_datetime(
            xda.time.values
        ).year.to_numpy()

        unique_years = np.unique(years)

    else:
        years = None
        unique_years = None

    lags = np.arange(
        -max_lag,
        max_lag + 1,
    )

    rs = []
    r95s = []

    for lag in lags:
        xx_list = []
        yy_list = []

        if has_datetime:
            for year in unique_years:
                indices = np.where(
                    years == year
                )[0]

                xv = xval[indices]
                yv = yval[indices]

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

        valid_mask = (
            np.isfinite(xx_all)
            & np.isfinite(yy_all)
        )

        if valid_mask.sum() > 3:
            xx_use = xx_all[valid_mask]
            yy_use = yy_all[valid_mask]

            r = pearsonr(
                xx_use,
                yy_use,
            )[0]

            if use_neff:
                neff = effective_sample_size(
                    xx_use,
                    yy_use,
                )
            else:
                neff = len(xx_use)

            r95 = corr_threshold_95(neff)

        else:
            r = np.nan
            r95 = np.nan

        rs.append(r)
        r95s.append(r95)

    return (
        lags,
        np.asarray(rs),
        np.asarray(r95s),
    )


def open_var(
    path,
    var_name,
    *,
    time_slice=None,
    level=None,
    lat_slice=None,
    lon_slice=None,
    scale=1.0,
    dtype="float32",
):
    """
    Open and subset a DataArray early to reduce peak memory.
    """
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
# Read data
# =========================
CLIM = xr.open_dataset(
    "/Volumes/TiPlus7100/p4/data/"
    "ERA5_CPC_daily_clim_1991_2020.nc"
).sel(
    mmdd=slice("06-01", "08-31")
)

u_clim = (
    CLIM["u_clim"]
    .sel(level=LEVELS_USE)
    .mean(dim="mmdd")
    .astype("float32")
)

v_clim = (
    CLIM["v_clim"]
    .sel(level=LEVELS_USE)
    .mean(dim="mmdd")
    .astype("float32")
)

u_clim = transform(
    u_clim,
    "lon",
    "360->180",
).sel(**REGION_180)

v_clim = transform(
    v_clim,
    "lon",
    "360->180",
).sel(**REGION_180)


u_bp = open_var(
    "/Volumes/TiPlus7100/p4/data/"
    "u_bp_JJA_10-30d.nc",
    "u_bp",
    time_slice=TIME_SLICE,
    level=LEVELS_USE,
    lat_slice=LAT_SLICE,
)

v_bp = open_var(
    "/Volumes/TiPlus7100/p4/data/"
    "v_bp_JJA_10-30d.nc",
    "v_bp",
    time_slice=TIME_SLICE,
    level=LEVELS_USE,
    lat_slice=LAT_SLICE,
)

z_bp = open_var(
    "/Volumes/TiPlus7100/p4/data/"
    "z_bp_JJA_10-30d.nc",
    "z_bp",
    time_slice=TIME_SLICE,
    level=LEVELS_USE,
    lat_slice=LAT_SLICE,
)

olr_bp = open_var(
    "/Volumes/TiPlus7100/p4/data/"
    "ttr_bp_JJA_10-30d.nc",
    "ttr_bp",
    time_slice=TIME_SLICE,
    lat_slice=slice(-20, 30),
    lon_slice=slice(40, 180),
    scale=-1 / 86400,
)

t2m_bp = open_var(
    "/Volumes/TiPlus7100/p4/data/"
    "t2m_bp_JJA_10-30d.nc",
    "t2m_bp",
    time_slice=TIME_SLICE,
)


# Retain only July 1-July 31.
u_bp = select_july_1961_2022(u_bp)
v_bp = select_july_1961_2022(v_bp)
z_bp = select_july_1961_2022(z_bp)
olr_bp = select_july_1961_2022(olr_bp)
t2m_bp = select_july_1961_2022(t2m_bp)


u_bp, v_bp, z_bp, olr_bp, t2m_bp = align_on_common_time(
    u_bp,
    v_bp,
    z_bp,
    olr_bp,
    t2m_bp,
)


t2m_bp = masked(
    t2m_bp,
    fr"{PYFILE}/map/self/"
    "长江_TP/长江_tp.shp",
)


# =========================
# MCA / SVD
# =========================
svd_t2m_v200 = xeofs.cross.MCA(
    n_modes=4,
    standardize=False,
    use_coslat=True,
).fit(
    t2m_bp,
    transform(
        v_bp,
        "lon",
        "360->180",
    ).sel(
        level=200,
        lon=slice(40, 130),
        lat=slice(36, 65),
    ),
    dim="time",
)


svd_t2m_olr = xeofs.cross.MCA(
    n_modes=4,
    standardize=False,
    use_coslat=True,
).fit(
    t2m_bp,
    transform(
        olr_bp,
        "lon",
        "360->180",
    ).sel(
        lon=slice(50, 170),
        lat=slice(-10, 24),
    ),
    dim="time",
)


# =========================
# Transform longitude once
# =========================
z_bp_180 = transform(
    z_bp,
    "lon",
    "360->180",
).sel(**REGION_180)

u_bp_180 = transform(
    u_bp,
    "lon",
    "360->180",
).sel(**REGION_180)

v_bp_180 = transform(
    v_bp,
    "lon",
    "360->180",
).sel(**REGION_180)

olr_bp_180 = transform(
    olr_bp,
    "lon",
    "360->180",
)


# =========================
# Draw figure
# =========================
proj = ccrs.PlateCarree(
    central_longitude=0
)

fig = plt.figure(
    figsize=(18, 7.6)
)

gs = gridspec.GridSpec(
    nrows=2,
    ncols=3,
    width_ratios=[
        1.05,
        1.05,
        0.8,
    ],
    height_ratios=[
        1,
        1,
    ],
    wspace=0.25,
    hspace=0.30,
)


clevs_t2m = np.array(
    [
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.1,
        0.2,
        0.3,
        0.4,
    ],
    dtype=float,
) * 5.0


cmap_t2m = (
    cmaps.GMT_polar[2:10]
    + cmaps.CBR_wet[0]
    + cmaps.GMT_polar[10:-2]
)


cmap_right = (
    cmaps.MPL_PuOr[10:64 - 10:5]
    + cmaps.CBR_wet[0]
    + cmaps.CBR_wet[0]
    + cmaps.CBR_wet[0]
    + cmaps.MPL_PuOr[64 + 10:-10:5]
)


# Lower OLR mode levels.
right_levels_base = np.array(
    [
        -0.3,
        -0.2,
        -0.1,
        -0.05,
        0.05,
        0.1,
        0.2,
        0.3,
    ],
    dtype=float,
)


# Upper V200 mode levels: ten times the original levels.
right_levels_v200 = right_levels_base * 10.0

# Lower OLR mode retains the original levels.
right_levels_olr = right_levels_base.copy()


cf_last_t2m = None


def draw_one(
    row,
    level,
    svd_obj,
    left_score_name,
    right_score_name,
    map_extent,
    box_blue,
    box_orange,
    title_left,
    title_right,
    vector_label,
    score_label,
    right_levels_use,
):
    """
    Draw one MCA map, standardized daily scores,
    and annual STD of standardized daily scores.

    Daily score standardization:

        score_z = (score - score_mean) / score_std

    Corresponding mode rescaling:

        mode_z = mode * score_std

    Thus:

        mode_z * score_z
        = mode * (score - score_mean)

    Regression fields are calculated against the standardized
    atmospheric score. Their amplitudes therefore represent
    anomalies corresponding to a one-standard-deviation change
    in the atmospheric MCA score.
    """
    global cf_last_t2m

    right_levels_use = np.asarray(
        right_levels_use,
        dtype=float,
    )

    # =========================
    # MCA modes and scores
    # =========================
    t2m_mode_original = (
        svd_obj.components()[0]
        .sel(mode=1)
    )

    right_mode_original = (
        svd_obj.components()[1]
        .sel(mode=1)
    )

    atm_score_original = (
        svd_obj.scores()[1]
        .sel(mode=1)
    )

    t2m_score_original = (
        svd_obj.scores()[0]
        .sel(mode=1)
    )

    corvar = float(
        svd_obj
        .squared_covariance_fraction()
        .sel(mode=1)
        * 100
    )

    # =========================
    # Unify MCA sign
    # =========================
    t2m_mode_mean = float(
        t2m_mode_original.mean(
            skipna=True
        ).values
    )

    SIGN = 1 if t2m_mode_mean < 0 else -1

    # Keep the original factor-of-ten transformation.
    # Mode x 10 and score / 10 preserve the MCA product.
    t2m_mode_scaled = (
        t2m_mode_original
        * 10.0
        * SIGN
    )

    right_mode_scaled = (
        right_mode_original
        * 10.0
        * SIGN
    )

    t2m_score_raw = (
        t2m_score_original
        / 10.0
        * SIGN
    )

    atm_score_raw = (
        atm_score_original
        / 10.0
        * SIGN
    )

    # =========================
    # Align and retain July
    # =========================
    t2m_daily, atm_daily = xr.align(
        t2m_score_raw,
        atm_score_raw,
        join="inner",
    )

    if np.issubdtype(
        t2m_daily.time.dtype,
        np.datetime64,
    ):
        july_mask = (
            t2m_daily.time.dt.month == 7
        )

        t2m_daily = t2m_daily.where(
            july_mask,
            drop=True,
        )

        atm_daily = atm_daily.where(
            july_mask,
            drop=True,
        )

    # =========================
    # Variable names
    # =========================
    if row == 0:
        t2m_z_name = "b_t2m_score_z"
        atm_z_name = "b_v200_score_z"
        t2m_std_name = "b_t2m_score_z_std"
        atm_std_name = "b_v200_score_z_std"

    else:
        t2m_z_name = "d_t2m_score_z"
        atm_z_name = "d_olr_score_z"
        t2m_std_name = "d_t2m_score_z_std"
        atm_std_name = "d_olr_score_z_std"

    # =========================
    # Standardize daily scores
    # =========================
    t2m_daily_z = standardize_over_time(
        t2m_daily,
        name=t2m_z_name,
    )

    atm_daily_z = standardize_over_time(
        atm_daily,
        name=atm_z_name,
    )

    # Population STD used during score standardization.
    t2m_score_std_value = float(
        t2m_daily.astype("float64").std(
            dim="time",
            skipna=True,
            ddof=0,
        ).values
    )

    atm_score_std_value = float(
        atm_daily.astype("float64").std(
            dim="time",
            skipna=True,
            ddof=0,
        ).values
    )

    if (
        not np.isfinite(t2m_score_std_value)
        or t2m_score_std_value <= 1e-12
    ):
        raise ValueError(
            f"Invalid T2m score STD in row {row}: "
            f"{t2m_score_std_value}"
        )

    if (
        not np.isfinite(atm_score_std_value)
        or atm_score_std_value <= 1e-12
    ):
        raise ValueError(
            f"Invalid {score_label} score STD in row {row}: "
            f"{atm_score_std_value}"
        )

    # =========================
    # Rescale spatial modes
    # =========================
    t2m_mode = (
        t2m_mode_scaled
        * t2m_score_std_value
    )

    right_mode = (
        right_mode_scaled
        * atm_score_std_value
    )

    t2m_mode.attrs.update(
        {
            "mode_scaling": (
                "MCA mode multiplied by the population STD "
                "of the corresponding signed and scaled score"
            ),
            "score_standard_deviation": t2m_score_std_value,
            "score_standardization": (
                "(score - time_mean) / time_std"
            ),
        }
    )

    right_mode.attrs.update(
        {
            "mode_scaling": (
                "MCA mode multiplied by the population STD "
                "of the corresponding signed and scaled score"
            ),
            "score_standard_deviation": atm_score_std_value,
            "score_standardization": (
                "(score - time_mean) / time_std"
            ),
        }
    )

    right_mode = right_mode.where(
        (
            (right_mode.lon >= map_extent[0])
            & (right_mode.lon <= map_extent[1])
            & (right_mode.lat >= map_extent[2])
            & (right_mode.lat <= map_extent[3])
        )
    )

    # =========================
    # Regression fields
    # =========================
    z_reg = regress(
        atm_daily_z,
        z_bp_180.sel(level=level),
    )

    u_reg = regress(
        atm_daily_z,
        u_bp_180.sel(level=level),
    )

    v_reg = regress(
        atm_daily_z,
        v_bp_180.sel(level=level),
    )

    if level <= 500:
        olr_reg = regress(
            atm_daily_z,
            olr_bp_180,
        )


        WAF = TN_WAF_3D(
            u_clim.sel(level=level),
            v_clim.sel(level=level),
            z_reg,
            single_level=200,
        )

        waf_x, waf_y = WAF

        waf_x = waf_x.where(
            (
                (waf_x.lon >= map_extent[0])
                & (waf_x.lon <= map_extent[1])
                & (waf_x.lat >= map_extent[2])
                & (waf_x.lat <= map_extent[3])
            )
        )

        waf_y = waf_y.where(
            (
                (waf_y.lon >= map_extent[0])
                & (waf_y.lon <= map_extent[1])
                & (waf_y.lat >= map_extent[2])
                & (waf_y.lat <= map_extent[3])
            )
        )

    z_reg = z_reg.where(
        (
            (z_reg.lon >= map_extent[0])
            & (z_reg.lon <= map_extent[1])
            & (z_reg.lat >= map_extent[2])
            & (z_reg.lat <= map_extent[3])
        )
    )

    u_reg = u_reg.where(
        (
            (u_reg.lon >= map_extent[0])
            & (u_reg.lon <= map_extent[1])
            & (u_reg.lat >= map_extent[2])
            & (u_reg.lat <= map_extent[3])
        )
    )

    v_reg = v_reg.where(
        (
            (v_reg.lon >= map_extent[0])
            & (v_reg.lon <= map_extent[1])
            & (v_reg.lat >= map_extent[2])
            & (v_reg.lat <= map_extent[3])
        )
    )

    # =========================
    # Left map
    # =========================
    ax = fig.add_subplot(
        gs[row, 0],
        projection=ccrs.PlateCarree(
            central_longitude=110
        ),
    )

    ax.set_aspect("auto")

    if level == 200:
        extent = [10, 160, 10, 80]
    else:
        extent = [20, 200, -20, 50]

    ax.set_extent(
        extent,
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=1.5,
        color="#BBBBBB",
    )

    ax.add_geometries(
        Reader(
            fr"{PYFILE}/map/self/"
            "长江_TP/长江_tp.shp"
        ).geometries(),
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.add_geometries(
        Reader(
            fr"{PYFILE}/map/地图边界数据/"
            "青藏高原边界数据总集/"
            "TPBoundary2500m_长江流域/"
            "TPBoundary2500m_长江流域.shp"
        ).geometries(),
        ccrs.PlateCarree(),
        facecolor="gray",
        edgecolor="black",
        linewidth=0.5,
    )

    if level >= 600:
        ax.add_geometries(
            Reader(
                fr"{PYFILE}/map/地图边界数据/"
                "青藏高原边界数据总集/"
                "TPBoundary_2500m/"
                "TPBoundary_2500m.shp"
            ).geometries(),
            ccrs.PlateCarree(),
            facecolor="#909090",
            edgecolor="#909090",
            linewidth=0,
            hatch=".",
            zorder=10,
        )

    if level <= 500:
        ax.set_xticks(
            np.arange(
                extent[0],
                extent[1],
                40,
            ),
            crs=proj,
        )

        ax.set_yticks(
            np.arange(
                extent[2],
                extent[3] + 1,
                15,
            ),
            crs=proj,
        )

    elif level == 850:
        ax.set_xticks(
            np.arange(
                extent[0],
                extent[1],
                30,
            ),
            crs=proj,
        )

        ax.set_yticks(
            np.arange(
                extent[2],
                extent[3] + 1,
                10,
            ),
            crs=proj,
        )

    ax.xaxis.set_major_formatter(
        LongitudeFormatter()
    )

    ax.yaxis.set_major_formatter(
        LatitudeFormatter()
    )

    ax.tick_params(labelsize=12)

    cf_t2m = ax.contourf(
        t2m_mode.lon,
        t2m_mode.lat,
        t2m_mode,
        levels=clevs_t2m,
        cmap=cmap_t2m,
        extend="both",
        transform=proj,
        zorder=1,
    )

    cf_last_t2m = cf_t2m

    cf_right = ax.contourf(
        right_mode.lon,
        right_mode.lat,
        right_mode,
        levels=right_levels_use,
        cmap=cmap_right,
        extend="both",
        alpha=0.62,
        transform=proj,
        zorder=2,
    )

    # Panel (a): geopotential-height contour levels are multiplied by 10.
    if level == 200:
        zlev_pos = [60, 180]
        zlev_neg = [-180, -60]

    else:
        zlev_pos = [17]
        zlev_neg = [-30]

    ax.contour(
        z_reg.lon,
        z_reg.lat,
        z_reg,
        levels=zlev_pos,
        colors="red",
        linewidths=1.2,
        transform=proj,
        zorder=4,
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
        zorder=4,
    )

    z_reg.attrs.update(
        {
            "regression_index": (
                f"standardized {score_label} MCA score"
            ),
            "regression_interpretation": (
                "Anomaly associated with a one-standard-"
                f"deviation increase in the {score_label} score"
            ),
        }
    )

    z_reg.to_netcdf(
        fr"/Volumes/TiPlus7100/p4/data/"
        f"SVD{level}V_{level}Z.nc"
    )

    # =========================
    # Wind vectors and WAF
    # =========================
    if level <= 500:
        q = ax.Curlyquiver(
            u_reg.lon,
            u_reg.lat,
            u_reg,
            v_reg,
            arrowsize=1,
            transform=ccrs.PlateCarree(
                central_longitude=0
            ),
            scale=2,
            linewidth=1,
            regrid=18,
            color="#555555",
            thinning=[
                "50%",
                "min",
            ],
            nanmax=2,
            MinDistance=[
                0.2,
                0.4,
            ],
        )

        # Panel (a): wind reference-vector magnitude is multiplied by 10.
        q.key(
            U=5.0,
            label=r"5 $m/s$",
            fontproperties={
                "size": 8
            },
            facecolor="none",
            bbox_to_anchor=(
                0,
                0.14,
                1,
                1,
            ),
            edgecolor="none",
            arrowsize=1,
            linewidth=1,
            intetval=0.8,
        )

        q_waf = ax.Curlyquiver(
            u_reg.lon,
            u_reg.lat,
            waf_x,
            waf_y,
            arrowsize=1.5,
            transform=ccrs.PlateCarree(
                central_longitude=0
            ),
            scale=1e1,
            linewidth=2,
            regrid=12,
            color="purple",
            thinning=[
                "70%",
                "min",
            ],
            nanmax=2,
            MinDistance=[
                0.2,
                0.4,
            ],
        )

        q_waf.key(
            U=0.2,
            label=r"0.2 ${m}^{2}/{s}^{2}$",
            fontproperties={
                "size": 8
            },
            facecolor="none",
            bbox_to_anchor=(
                -0.15,
                0.14,
                1,
                1,
            ),
            edgecolor="none",
            arrowsize=0.01,
            linewidth=1,
            intetval=0.8,
        )

    elif level == 850:
        q = ax.Curlyquiver(
            u_reg.lon,
            u_reg.lat,
            u_reg,
            v_reg,
            arrowsize=1,
            transform=ccrs.PlateCarree(
                central_longitude=0
            ),
            scale=10,
            linewidth=1,
            regrid=18,
            color="#555555",
            thinning=[
                "25%",
                "min",
            ],
            nanmax=2,
            MinDistance=[
                0.2,
                0.4,
            ],
        )

        q.key(
            U=0.5,
            label=r"0.5 $m/s$",
            fontproperties={
                "size": 8
            },
            facecolor="none",
            bbox_to_anchor=(
                0,
                0.14,
                1,
                1,
            ),
            edgecolor="none",
            arrowsize=1,
            linewidth=1,
            intetval=0.8,
        )

    ax.add_patch(
        patches.Rectangle(
            (
                box_blue[0],
                box_blue[2],
            ),
            box_blue[1] - box_blue[0],
            box_blue[3] - box_blue[2],
            linewidth=1.8,
            edgecolor="royalblue",
            facecolor="none",
            linestyle="--",
            transform=proj,
            zorder=5,
        )
    )

    ax.set_title(
        title_left,
        loc="left",
        fontsize=16,
    )

    # =========================
    # Middle column
    # =========================
    gs_middle = gridspec.GridSpecFromSubplotSpec(
        nrows=2,
        ncols=1,
        subplot_spec=gs[row, 1],
        height_ratios=[
            3.2,
            1.20,
        ],
        hspace=0.04,
    )

    (
        daily_x,
        daily_years,
        year_starts,
        year_ends,
        year_centers,
        tick_positions,
        tick_years,
    ) = build_compact_july_axis(
        t2m_daily_z.time.values,
        tick_step=10,
    )

    # =========================
    # Upper: standardized scores
    # =========================
    ax2 = fig.add_subplot(
        gs_middle[0, 0]
    )

    ax2b = ax2.twinx()

    line_t2m = ax2.plot(
        daily_x,
        t2m_daily_z.values,
        color="red",
        linewidth=0.75,
        alpha=0.85,
        label="T2m",
        zorder=3,
    )[0]

    line_atm = ax2b.plot(
        daily_x,
        atm_daily_z.values,
        color="black",
        linewidth=0.75,
        alpha=0.82,
        label=score_label,
        zorder=2,
    )[0]

    ax2.axhline(
        0,
        color="0.65",
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )

    for year_index in range(
        5,
        len(year_starts),
        5,
    ):
        ax2.axvline(
            year_starts[year_index] - 0.5,
            color="0.82",
            linewidth=0.55,
            linestyle="-",
            zorder=0,
        )

    ax2.set_xlim(
        daily_x[0] - 0.5,
        daily_x[-1] + 0.5,
    )

    standardized_limit = symmetric_axis_limit(
        np.concatenate(
            [
                np.asarray(
                    t2m_daily_z.values,
                    dtype=float,
                ).ravel(),
                np.asarray(
                    atm_daily_z.values,
                    dtype=float,
                ).ravel(),
            ]
        ),
        minimum=1.0,
        padding=0.08,
    )

    ax2.set_ylim(
        -standardized_limit,
        standardized_limit,
    )

    ax2b.set_ylim(
        -standardized_limit,
        standardized_limit,
    )

    ax2.tick_params(
        axis="y",
        labelsize=10,
        colors="red",
        direction="out",
        length=3,
        width=1,
    )

    ax2b.tick_params(
        axis="y",
        labelsize=10,
        colors="black",
        direction="out",
        length=3,
        width=1,
    )

    ax2.spines["left"].set_color("red")
    ax2b.spines["right"].set_color("black")

    ax2.set_ylabel(
        "T2m (z)",
        fontsize=10,
        color="red",
    )

    ax2b.set_ylabel(
        f"{score_label} (z)",
        fontsize=10,
        color="black",
    )

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([])

    ax2.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )

    score_mask = (
        np.isfinite(t2m_daily_z.values)
        & np.isfinite(atm_daily_z.values)
    )

    if score_mask.sum() > 3:
        r, p = pearsonr(
            t2m_daily_z.values[score_mask],
            atm_daily_z.values[score_mask],
        )

    else:
        r = np.nan
        p = np.nan

    title_right_final = (
        f"{title_right} "
        f"(CorVar={corvar:.0f}%, r={r:.2f})"
    )

    ax2.legend(
        handles=[
            line_t2m,
            line_atm,
        ],
        labels=[
            "T2m",
            score_label,
        ],
        frameon=False,
        fontsize=10,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(
            -0.01,
            1.02,
        ),
        handlelength=2.2,
        columnspacing=0.9,
    )

    ax2.set_title(
        title_right_final,
        loc="left",
        fontsize=16,
    )

    # =========================
    # Lower: annual STD
    # =========================
    ax_std = fig.add_subplot(
        gs_middle[1, 0],
        sharex=ax2,
    )

    ax_stdb = ax_std.twinx()

    t2m_std = annual_std_from_standardized(
        t2m_daily_z,
        name=t2m_std_name,
    )

    atm_std = annual_std_from_standardized(
        atm_daily_z,
        name=atm_std_name,
    )

    t2m_std, atm_std = xr.align(
        t2m_std,
        atm_std,
        join="inner",
    )

    validate_nonnegative_std(
        xr.Dataset(
            {
                t2m_std.name: t2m_std,
                atm_std.name: atm_std,
            }
        )
    )

    std_years = t2m_std["year"].values

    year_to_center = {
        int(year): center
        for year, center in zip(
            daily_years,
            year_centers,
        )
    }

    std_x = np.asarray(
        [
            year_to_center[int(year)]
            for year in std_years
        ],
        dtype=float,
    )

    std_bar_width = 18.0

    ax_std.bar(
        std_x,
        t2m_std.values,
        width=std_bar_width,
        color="red",
        alpha=0.70,
        label="T2m standardized STD",
        zorder=2,
    )

    ax_stdb.bar(
        std_x,
        atm_std.values,
        width=std_bar_width,
        facecolor="none",
        edgecolor="black",
        linewidth=0.75,
        alpha=1.0,
        label=(
            f"{score_label} standardized STD"
        ),
        zorder=3,
    )

    std_upper_limit = positive_axis_limit(
        t2m_std.values,
        atm_std.values,
        minimum=1.5,
        padding=0.18,
    )

    ax_std.set_ylim(0, std_upper_limit)
    ax_stdb.set_ylim(0, std_upper_limit)

    ax_std.set_ylabel(
        "STD",
        fontsize=9,
        color="red",
        labelpad=2,
    )

    ax_stdb.set_ylabel(
        "STD",
        fontsize=9,
        color="black",
        labelpad=2,
    )

    ax_std.tick_params(
        axis="y",
        labelsize=8,
        colors="red",
        direction="out",
        length=2.5,
        width=0.9,
        pad=2,
    )

    ax_stdb.tick_params(
        axis="y",
        labelsize=8,
        colors="black",
        direction="out",
        length=2.5,
        width=0.9,
        pad=2,
    )

    ax_std.spines["left"].set_color("red")
    ax_stdb.spines["right"].set_color("black")

    ax_std.axhline(
        0,
        color="0.5",
        linewidth=0.7,
        zorder=1,
    )

    if 2015 in std_years:
        i2015 = int(
            np.where(
                std_years == 2015
            )[0][0]
        )


        ax_stdb.scatter(
            std_x[i2015],
            2,
            color="blue",
            s=65,
            marker="*",
            zorder=10,
            clip_on=False,
        )

    ax_std.set_xticks(tick_positions)

    ax_std.set_xticklabels(
        tick_years.astype(int),
        fontsize=9,
    )

    ax_std.tick_params(
        axis="x",
        direction="out",
        length=3,
        width=1,
        pad=2,
    )

    ax_std.set_xlabel(
        "",
        fontsize=10,
        labelpad=2,
    )

    ax_std.text(
        0.01,
        0.90,
        "Annual STD",
        transform=ax_std.transAxes,
        fontsize=9,
        ha="left",
        va="top",
    )

    return (
        ax,
        ax2,
        ax_std,
        t2m_score_raw,
        atm_score_raw,
        t2m_daily_z,
        atm_daily_z,
        cf_right,
        t2m_std,
        atm_std,
    )


# =========================
# First row: 200 hPa
# =========================
(
    ax200,
    ax200_daily,
    ax200_std,
    t2m200_score,
    v200_score,
    b_t2m_score_z,
    b_v200_score_z,
    cf_v200_mode,
    b_t2m_std,
    b_v200_std,
) = draw_one(
    row=0,
    level=200,
    svd_obj=svd_t2m_v200,
    left_score_name="T2m",
    right_score_name="V200",
    map_extent=[
        10,
        160,
        20,
        80,
    ],
    box_blue=[
        40,
        130,
        36,
        65,
    ],
    box_orange=[
        -70,
        0,
        -20,
        45,
    ],
    title_left="(a) SVD (T2m&V200)",
    title_right="(b) TC",
    vector_label="5 m/s",
    score_label="V200",
    right_levels_use=right_levels_v200,
)


# =========================
# Second row: 850 hPa / OLR
# =========================
(
    ax850,
    ax850_daily,
    ax850_std,
    t2m850_score,
    olr_score,
    d_t2m_score_z,
    d_olr_score_z,
    cf_olr_mode,
    d_t2m_std,
    d_olr_std,
) = draw_one(
    row=1,
    level=850,
    svd_obj=svd_t2m_olr,
    left_score_name="T2m",
    right_score_name="OLR",
    map_extent=[
        30,
        190,
        -20,
        50,
    ],
    box_blue=[
        50,
        170,
        -10,
        24,
    ],
    box_orange=[
        90,
        122,
        20,
        35,
    ],
    title_left="(c) SVD (T2m&OLR)",
    title_right="(d) TC",
    vector_label="0.5 m/s",
    score_label="OLR",
    right_levels_use=right_levels_olr,
)


# =========================
# Standardize original T2m
# and calculate annual STD
# =========================
raw_t2m = t2m_bp.astype("float32")

if "time" not in raw_t2m.dims:
    raise ValueError(
        "The original T2m field has no time dimension."
    )

if not np.issubdtype(
    raw_t2m.time.dtype,
    np.datetime64,
):
    raise TypeError(
        "The original T2m time coordinate is not datetime64."
    )

raw_t2m = raw_t2m.where(
    raw_t2m.time.dt.month == 7,
    drop=True,
)


raw_t2m_grid_z = standardize_over_time(
    raw_t2m,
    name="raw_t2m_grid_z",
)

raw_t2m_grid_z_std = annual_std_from_standardized(
    raw_t2m_grid_z,
    name="raw_t2m_grid_z_std",
)


spatial_dims = [
    dim
    for dim in ["lat", "lon"]
    if dim in raw_t2m.dims
]

if "lat" not in raw_t2m.coords:
    raise ValueError(
        "Cannot calculate area-weighted T2m because 'lat' is missing."
    )

if len(spatial_dims) == 0:
    raise ValueError(
        "No spatial dimension was found in the original T2m field."
    )

latitude_weights = np.cos(
    np.deg2rad(
        raw_t2m["lat"]
    )
).clip(min=0)

latitude_weights = latitude_weights.rename(
    "latitude_weights"
)

raw_t2m_area_mean_daily = (
    raw_t2m
    .weighted(latitude_weights)
    .mean(
        dim=spatial_dims,
        skipna=True,
    )
    .astype("float32")
    .rename(
        "raw_t2m_area_mean_daily"
    )
)

raw_t2m_area_mean_z = standardize_over_time(
    raw_t2m_area_mean_daily,
    name="raw_t2m_area_mean_z",
)

raw_t2m_area_mean_z_std = annual_std_from_standardized(
    raw_t2m_area_mean_z,
    name="raw_t2m_area_mean_z_std",
)


# =========================
# Align annual variables
# =========================
(
    b_t2m_std,
    b_v200_std,
    d_t2m_std,
    d_olr_std,
    raw_t2m_area_mean_z_std,
) = xr.align(
    b_t2m_std,
    b_v200_std,
    d_t2m_std,
    d_olr_std,
    raw_t2m_area_mean_z_std,
    join="inner",
)

common_years = b_t2m_std["year"].values

raw_t2m_grid_z_std = (
    raw_t2m_grid_z_std
    .sel(year=common_years)
)


# =========================
# Construct NetCDF Dataset
# =========================
std_ds = xr.Dataset(
    {
        "b_t2m_score_z_std": b_t2m_std.rename(
            "b_t2m_score_z_std"
        ),
        "b_v200_score_z_std": b_v200_std.rename(
            "b_v200_score_z_std"
        ),
        "d_t2m_score_z_std": d_t2m_std.rename(
            "d_t2m_score_z_std"
        ),
        "d_olr_score_z_std": d_olr_std.rename(
            "d_olr_score_z_std"
        ),
        "raw_t2m_grid_z_std": raw_t2m_grid_z_std,
        "raw_t2m_area_mean_z_std": raw_t2m_area_mean_z_std,
    }
)

std_ds = std_ds.assign_coords(
    year=std_ds["year"].astype("int32")
)


# =========================
# Coordinate attributes
# =========================
std_ds["year"].attrs.update(
    {
        "long_name": "calendar year",
        "standard_name": "year",
        "description": (
            "Calendar year corresponding to the standard "
            "deviation of standardized daily July values."
        ),
    }
)

if "lat" in std_ds.coords:
    std_ds["lat"].attrs.update(
        {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        }
    )

if "lon" in std_ds.coords:
    std_ds["lon"].attrs.update(
        {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        }
    )


# =========================
# Variable attributes
# =========================
std_ds["b_t2m_score_z_std"].attrs.update(
    {
        "long_name": (
            "Annual July STD of standardized T2m MCA score "
            "from the T2m-V200 MCA"
        ),
        "panel": "b",
        "paired_field": "V200",
        "units": "1",
        "standardization_reference_period": (
            "All July daily values during 1961-2022"
        ),
        "annual_calculation": (
            "Population STD of standardized daily values "
            "within July of each year"
        ),
        "ddof": 0,
    }
)

std_ds["b_v200_score_z_std"].attrs.update(
    {
        "long_name": (
            "Annual July STD of standardized V200 MCA score"
        ),
        "panel": "b",
        "paired_field": "T2m",
        "units": "1",
        "standardization_reference_period": (
            "All July daily values during 1961-2022"
        ),
        "annual_calculation": (
            "Population STD of standardized daily values "
            "within July of each year"
        ),
        "ddof": 0,
    }
)

std_ds["d_t2m_score_z_std"].attrs.update(
    {
        "long_name": (
            "Annual July STD of standardized T2m MCA score "
            "from the T2m-OLR MCA"
        ),
        "panel": "d",
        "paired_field": "OLR",
        "units": "1",
        "standardization_reference_period": (
            "All July daily values during 1961-2022"
        ),
        "annual_calculation": (
            "Population STD of standardized daily values "
            "within July of each year"
        ),
        "ddof": 0,
    }
)

std_ds["d_olr_score_z_std"].attrs.update(
    {
        "long_name": (
            "Annual July STD of standardized OLR MCA score"
        ),
        "panel": "d",
        "paired_field": "T2m",
        "units": "1",
        "standardization_reference_period": (
            "All July daily values during 1961-2022"
        ),
        "annual_calculation": (
            "Population STD of standardized daily values "
            "within July of each year"
        ),
        "ddof": 0,
    }
)

std_ds["raw_t2m_grid_z_std"].attrs.update(
    {
        "long_name": (
            "Grid-point annual July STD of standardized "
            "original T2m"
        ),
        "source_variable": "t2m_bp",
        "units": "1",
        "standardization": (
            "Original T2m was standardized independently "
            "at every grid point along time"
        ),
        "standardization_reference_period": (
            "All July daily values during 1961-2022"
        ),
        "annual_calculation": (
            "Population STD of standardized daily values "
            "within July of each year"
        ),
        "mask": (
            "The spatial mask applied to t2m_bp before MCA"
        ),
        "ddof": 0,
    }
)

std_ds["raw_t2m_area_mean_z_std"].attrs.update(
    {
        "long_name": (
            "Annual July STD of standardized area-mean "
            "original T2m"
        ),
        "source_variable": "t2m_bp",
        "units": "1",
        "spatial_average": (
            "Cosine-latitude-weighted spatial mean over "
            "valid masked grid points"
        ),
        "standardization_reference_period": (
            "All July daily values during 1961-2022"
        ),
        "annual_calculation": (
            "Population STD of standardized daily values "
            "within July of each year"
        ),
        "ddof": 0,
    }
)


# =========================
# Global attributes
# =========================
std_ds.attrs.update(
    {
        "title": (
            "Annual July standard deviations after "
            "z-score standardization"
        ),
        "time_period": "1961-2022",
        "month": "July",
        "daily_period": "July 1-July 31",
        "standardization_reference": (
            "All available July daily values during 1961-2022"
        ),
        "standardization_formula": (
            "z = (x - full-period July mean) / "
            "full-period July standard deviation"
        ),
        "annual_std_definition": (
            "Population standard deviation of standardized "
            "daily values within July of each year"
        ),
        "annual_std_ddof": 0,
        "negative_std_note": (
            "Standard deviation cannot be negative. Annual "
            "variance was clipped to zero before square root, "
            "and all output variables were checked before and "
            "after writing."
        ),
    }
)


# =========================
# Diagnostics before export
# =========================
print("=" * 84)
print("Standardization diagnostics")
print("-" * 84)

print_standardization_diagnostic(
    "b_t2m_score_z",
    b_t2m_score_z,
)

print_standardization_diagnostic(
    "b_v200_score_z",
    b_v200_score_z,
)

print_standardization_diagnostic(
    "d_t2m_score_z",
    d_t2m_score_z,
)

print_standardization_diagnostic(
    "d_olr_score_z",
    d_olr_score_z,
)

print_standardization_diagnostic(
    "raw_t2m_area_mean_z",
    raw_t2m_area_mean_z,
)


minimum_std_values = validate_nonnegative_std(
    std_ds,
    tolerance=1e-7,
)

print("-" * 84)
print("Minimum value of every STD variable")

for variable_name, minimum_value in minimum_std_values.items():
    print(
        f"{variable_name:<30s}: "
        f"{minimum_value:.10f}"
    )

print("\nAll finite STD values are nonnegative.")
print("=" * 84)


# =========================
# Export NetCDF
# =========================
std_output_dir = fr"{PYFILE}/p4/data"

os.makedirs(
    std_output_dir,
    exist_ok=True,
)

std_nc_path = os.path.join(
    std_output_dir,
    (
        "SVD_V200_OLR_T2m_and_raw_T2m_"
        "standardized_annual_STD_1961_2022.nc"
    ),
)


encoding = {
    "b_t2m_score_z_std": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(1e20),
    },
    "b_v200_score_z_std": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(1e20),
    },
    "d_t2m_score_z_std": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(1e20),
    },
    "d_olr_score_z_std": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(1e20),
    },
    "raw_t2m_grid_z_std": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(1e20),
    },
    "raw_t2m_area_mean_z_std": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(1e20),
    },
    "year": {
        "dtype": "int32",
    },
}


try:
    std_ds.to_netcdf(
        std_nc_path,
        mode="w",
        format="NETCDF4",
        engine="netcdf4",
        encoding=encoding,
    )

except (ImportError, ModuleNotFoundError):
    print(
        "netCDF4 engine is unavailable. "
        "Trying h5netcdf."
    )

    std_ds.to_netcdf(
        std_nc_path,
        mode="w",
        engine="h5netcdf",
        encoding=encoding,
    )


with xr.open_dataset(std_nc_path) as std_check:
    written_minimum_values = validate_nonnegative_std(
        std_check,
        tolerance=1e-7,
    )

    print("=" * 84)
    print("Written NetCDF validation")

    for variable_name, minimum_value in (
        written_minimum_values.items()
    ):
        print(
            f"{variable_name:<30s}: "
            f"minimum={minimum_value:.10f}"
        )

    print("\nWritten Dataset:")
    print(std_check)

    print("=" * 84)


print(
    "Annual standardized STD NetCDF successfully saved:"
)

print(std_nc_path)


# =========================
# Independent vertical colorbars
# =========================
fig.canvas.draw()


def add_aligned_vertical_colorbar(
    map_ax,
    middle_ax,
    contour,
    ticks,
):
    """
    Add a vertical colorbar between one map and its middle panel.

    The colorbar bottom, top and height are aligned with the map.
    """
    map_position = map_ax.get_position()
    middle_position = middle_ax.get_position()

    available_gap = (
        middle_position.x0
        - map_position.x1
    )

    if available_gap <= 0:
        raise ValueError(
            "No horizontal gap is available for the vertical colorbar."
        )

    colorbar_width = min(
        0.006,
        available_gap * 0.14,
    )

    colorbar_x = (
        map_position.x1
        + available_gap * 0.08
    )

    colorbar_y = map_position.y0
    colorbar_height = map_position.height

    colorbar_ax = fig.add_axes(
        [
            colorbar_x,
            colorbar_y,
            colorbar_width,
            colorbar_height,
        ]
    )

    colorbar = fig.colorbar(
        contour,
        cax=colorbar_ax,
        orientation="vertical",
        drawedges=True,
    )

    colorbar.set_ticks(ticks)

    colorbar.ax.tick_params(
        length=0,
        labelsize=9,
        pad=2,
    )

    colorbar.dividers.set_linewidth(2.0)
    colorbar.outline.set_linewidth(2.0)

    return colorbar_ax, colorbar


cax_v200, cb_v200 = add_aligned_vertical_colorbar(
    map_ax=ax200,
    middle_ax=ax200_daily,
    contour=cf_v200_mode,
    ticks=right_levels_v200,
)


cax_olr, cb_olr = add_aligned_vertical_colorbar(
    map_ax=ax850,
    middle_ax=ax850_daily,
    contour=cf_olr_mode,
    ticks=right_levels_olr,
)


# =========================
# Third column: lag correlations
# =========================
gs_right = gridspec.GridSpecFromSubplotSpec(
    3,
    1,
    subplot_spec=gs[:, 2],
    hspace=0.25,
)

max_lag = 30


lag_v200, r_v200, sig_v200 = lag_corr(
    v200_score,
    max_lag=max_lag,
    use_neff=True,
    within_same_summer=True,
)

lag_cross, r_cross, sig_cross = lag_corr(
    v200_score,
    olr_score,
    max_lag=max_lag,
    use_neff=True,
    within_same_summer=True,
)

lag_olr, r_olr, sig_olr = lag_corr(
    olr_score,
    max_lag=max_lag,
    use_neff=True,
    within_same_summer=True,
)


lag_panels = [
    (
        lag_v200,
        r_v200,
        sig_v200,
        "(e) Auto-V200",
    ),
    (
        lag_cross,
        r_cross,
        sig_cross,
        "(f) V200&OLR",
    ),
    (
        lag_olr,
        r_olr,
        sig_olr,
        "(g) Auto-OLR",
    ),
]


for i, (
    lags,
    rs,
    rsig,
    title,
) in enumerate(lag_panels):
    ax_lag = fig.add_subplot(
        gs_right[i, 0]
    )

    ax_lag.plot(
        lags,
        rs,
        color="black",
        linewidth=2.2,
    )

    ax_lag.axhline(
        0,
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )

    ax_lag.axvline(
        0,
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )

    ax_lag.plot(
        lags,
        rsig,
        color="red",
        linewidth=1.2,
        linestyle=(
            0,
            (
                8,
                5,
            ),
        ),
        alpha=0.8,
    )

    ax_lag.plot(
        lags,
        -rsig,
        color="red",
        linewidth=1.2,
        linestyle=(
            0,
            (
                8,
                5,
            ),
        ),
        alpha=0.8,
    )

    ax_lag.set_xlim(
        -max_lag,
        max_lag,
    )

    max_y = np.nanmax(
        np.abs(rs)
    )

    if not np.isfinite(max_y) or max_y == 0:
        max_y = 0.1

    ax_lag.set_ylim(
        -(max_y + 0.1),
        max_y + 0.1,
    )

    ax_lag.set_xticks(
        np.arange(
            -30,
            31,
            10,
        )
    )

    ytick_min = (
        -np.round(
            (max_y + 0.1) * 5
        )
        / 5
    )

    ytick_max = (
        np.round(
            (max_y + 0.1) * 5
        )
        / 5
    )

    ax_lag.set_yticks(
        np.arange(
            ytick_min,
            ytick_max + 1e-9,
            0.4,
        )
    )

    ax_lag.xaxis.set_minor_locator(
        MultipleLocator(2)
    )

    ax_lag.yaxis.set_minor_locator(
        MultipleLocator(0.1)
    )

    ax_lag.tick_params(
        labelsize=11,
        direction="out",
        length=4,
        width=1.2,
    )

    ax_lag.tick_params(
        which="minor",
        length=2,
        width=1,
    )

    if i < 2:
        ax_lag.set_xticklabels([])

    ax_lag.set_title(
        title,
        loc="left",
        fontsize=16,
    )

    for spine in ax_lag.spines.values():
        spine.set_linewidth(2.0)


# =========================
# T2m horizontal colorbar
# =========================
cax = inset_axes(
    ax850,
    width="100%",
    height="5%",
    loc="lower left",
    bbox_to_anchor=(
        0,
        -0.18,
        1,
        1,
    ),
    bbox_transform=ax850.transAxes,
    borderpad=0,
)

cb = fig.colorbar(
    cf_last_t2m,
    cax=cax,
    orientation="horizontal",
    drawedges=True,
)

cb.set_ticks(clevs_t2m)

cb.ax.tick_params(
    length=0,
    labelsize=10,
)

cb.dividers.set_linewidth(2.5)
cb.outline.set_linewidth(2.5)

cb.ax.set_title(
    "",
    fontsize=9,
    pad=3,
)


# =========================
# Borders
# =========================
for ax_ in fig.axes:
    for spine in ax_.spines.values():
        spine.set_linewidth(2.0)


# =========================
# Save figure
# =========================
output_dir = fr"{PYFILE}/p4/pic"

os.makedirs(
    output_dir,
    exist_ok=True,
)

plt.savefig(
    fr"{output_dir}/"
    "SVD_UVZ_all_1961_2022.png",
    bbox_inches="tight",
    dpi=600,
)

plt.savefig(
    fr"{output_dir}/"
    "SVD_UVZ_all_1961_2022.pdf",
    bbox_inches="tight",
)

plt.show()
