import os
import numpy as np
import pandas as pd
import xarray as xr
from climkit.masked import masked
from climkit.filter import *
import matplotlib.pyplot as plt

# =========================================================
# ====================== User settings =====================
# =========================================================

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

# ERA5 and CN05.1 input files
ERA5_NC_FILE = "/Volumes/TiPlus7100/p4/data/ERA5_daily_t2m_sum.nc"
CN05_NC_FILE = "/Volumes/TiPlus7100/data/CN05.1/CN05.1_Tm_1961_2023_daily.nc"

# Yangtze River basin shapefile
SHP_FILE = fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"

# Output directory
OUT_DIR = fr"{PYFILE}/p4/pic/"
os.makedirs(OUT_DIR, exist_ok=True)

# Output figure
OUT_FIG = os.path.join(OUT_DIR, "Fig3_t2m_daily_actual_ERA5_CN051_combined")

# Climatology period
CLIM_START = 1991
CLIM_END = 2020

# Composite years
COMPOSITE_YEARS = [
    1965, 1966, 1968, 1974, 1976, 1980, 1982, 1983, 1986,
    1987, 1989, 1992, 1993, 1997, 1999, 2004, 2008, 2014
]  # 0.5 std

# Target year
TARGET_YEAR = 2015

# Filter and plot settings
FILTER_MONTHS = [5, 6, 7, 8, 9]   # for filtering
PLOT_MONTHS = [6, 7, 8]            # for plotting
NWTS = 61


# =========================================================
# ====================== Helper funcs ======================
# =========================================================

def detect_main_var(ds: xr.Dataset, preferred=None) -> str:
    """Automatically detect the main temperature variable."""
    if preferred is None:
        preferred = ["t2m", "2m_temperature", "tem"]

    for v in preferred:
        if v in ds.data_vars:
            return v

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise ValueError(f"Cannot automatically detect temperature variable: {list(ds.data_vars)}")


def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Standardize longitude/latitude coordinate names to lon/lat."""
    rename_dict = {}

    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"

    if rename_dict:
        ds = ds.rename(rename_dict)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("No lon/lat or longitude/latitude coordinates found.")

    return ds


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """Basin mean. This keeps your original simple spatial mean logic."""
    return da.mean(dim=("lat", "lon"))


def parse_time_column(s: pd.Series) -> pd.Series:
    """Robustly parse time values from datetime64 or YYYYMMDD-like integers/strings."""
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s)

    s_str = s.astype(str)

    if s_str.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(s_str, format="%Y%m%d")

    return pd.to_datetime(s, errors="raise")


def build_summer_day_index(time_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Summer day index:
    Jun-01 -> 1
    ...
    Aug-31 -> 92
    """
    month = time_index.month
    day = time_index.day
    offsets = np.where(month == 6, 0, np.where(month == 7, 30, 61))
    return offsets + day


def build_warm_day_index(time_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Warm-season day index:
    May-01 -> 1
    ...
    Sep-30 -> 153
    """
    month = time_index.month
    day = time_index.day

    offsets = np.select(
        [
            month == 5,
            month == 6,
            month == 7,
            month == 8,
            month == 9,
        ],
        [
            0,
            31,
            61,
            92,
            123,
        ],
        default=np.nan,
    )

    if np.any(np.isnan(offsets)):
        raise ValueError("build_warm_day_index only supports May-September.")

    return offsets.astype(int) + day


def ensure_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert Kelvin-like temperature to Celsius."""
    vmin = float(da.min().values)
    vmax = float(da.max().values)

    if vmin > 150 and vmax < 400:
        print("Detected Kelvin-like temperature values. Converted to Celsius.")
        da = da - 273.15
        da.attrs["units"] = "degC"

    return da


def pad_filtered_result(original: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    """Pad shortened filtered sequence back to original length using NaN on both sides."""
    out = np.full(len(original), np.nan)
    m = len(filtered)

    if m > len(original):
        raise ValueError("Filtered sequence is longer than the original sequence.")

    pad_left = (len(original) - m) // 2
    pad_right = len(original) - m - pad_left

    if pad_right == 0:
        out[pad_left:] = filtered
    else:
        out[pad_left:-pad_right] = filtered

    return out


def month_day_to_summer_day(month: int, day: int) -> int:
    """Convert month/day to summer day index: Jun-01 -> 1, Aug-31 -> 92."""
    if month == 6:
        return day
    if month == 7:
        return 30 + day
    if month == 8:
        return 61 + day

    raise ValueError("Only June-August are supported.")


def validate_nonempty(df: pd.DataFrame, name: str):
    """Raise a clear error if a required DataFrame is empty."""
    if df.empty:
        raise ValueError(
            f"{name} is empty. Please check the input data time range, "
            f"COMPOSITE_YEARS, CLIM_START/CLIM_END, and TARGET_YEAR."
        )


# =========================================================
# ===================== Data processing ====================
# =========================================================

def prepare_panel_data(nc_file: str, dataset_label: str, preferred_vars=None) -> pd.DataFrame:
    """
    Read one dataset, calculate basin-mean daily climatology, composite,
    target-year values, and filtered anomalies for plotting.
    """
    print(f"\n===== Processing {dataset_label} =====")
    print("1) Reading nc data...")

    ds = xr.open_dataset(nc_file)
    ds = standardize_latlon(ds)

    var_name = detect_main_var(ds, preferred=preferred_vars)
    print(f"Detected variable name: {var_name}")

    da = ds[var_name]

    if "time" not in da.coords:
        if "valid_time" in da.coords:
            da = da.rename({"valid_time": "time"})
        else:
            raise ValueError("No time or valid_time coordinate found.")

    data_start = min([CLIM_START, TARGET_YEAR] + COMPOSITE_YEARS)
    data_end = max([CLIM_END, TARGET_YEAR] + COMPOSITE_YEARS)

    da = da.sel(time=slice(f"{data_start}-05-01", f"{data_end}-09-30"))
    da = ensure_celsius(da)

    print("2) Clipping to Yangtze River basin...")
    da_clip = masked(da, SHP_FILE)

    print("3) Calculating basin mean...")
    ts = area_weighted_mean(da_clip)

    df = ts.to_dataframe(name="t2m").reset_index()
    df["time"] = parse_time_column(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day

    df = df[df["month"].isin(FILTER_MONTHS)].copy()
    df["warm_day"] = build_warm_day_index(pd.DatetimeIndex(df["time"]))

    print("4) Calculating daily climatology...")
    clim_warm_df = (
        df[(df["year"] >= CLIM_START) & (df["year"] <= CLIM_END)]
        .groupby("warm_day", as_index=False)["t2m"]
        .mean()
        .rename(columns={"t2m": "climatology_actual"})
    )
    validate_nonempty(clim_warm_df, f"{dataset_label} climatology")

    print("5) Calculating composite daily mean and standard deviation...")
    comp_warm_stat_df = (
        df[df["year"].isin(COMPOSITE_YEARS)]
        .groupby("warm_day")["t2m"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "composite_actual",
                "std": "composite_std",
            }
        )
    )
    validate_nonempty(comp_warm_stat_df, f"{dataset_label} composite")
    comp_warm_stat_df["composite_std"] = comp_warm_stat_df["composite_std"].fillna(0.0)

    print(f"6) Calculating {TARGET_YEAR} daily values...")
    y_target_warm_df = (
        df[df["year"] == TARGET_YEAR][["warm_day", "t2m"]]
        .groupby("warm_day", as_index=False)
        .mean()
        .rename(columns={"t2m": f"{TARGET_YEAR}_actual"})
    )
    validate_nonempty(y_target_warm_df, f"{dataset_label} target year")

    print("7) Merging warm-season sequence and filtering...")
    warm_plot_df = clim_warm_df.merge(comp_warm_stat_df, on="warm_day", how="inner")
    warm_plot_df = warm_plot_df.merge(y_target_warm_df, on="warm_day", how="inner")
    validate_nonempty(warm_plot_df, f"{dataset_label} merged warm-season data")

    calendar_df = (
        df[df["year"] == TARGET_YEAR][["warm_day", "month", "day"]]
        .drop_duplicates(subset=["warm_day"])
        .sort_values("warm_day")
    )
    warm_plot_df = warm_plot_df.merge(calendar_df, on="warm_day", how="left")

    clim_y_warm = warm_plot_df["climatology_actual"].values
    comp_y_warm = warm_plot_df["composite_actual"].values
    y_target_y_warm = warm_plot_df[f"{TARGET_YEAR}_actual"].values

    y_target_anom_filt = LanczosFilter(
        y_target_y_warm - clim_y_warm,
        "bandpass",
        [10, 30],
        nwts=NWTS,
    ).filted()

    comp_anom_filt = LanczosFilter(
        comp_y_warm - clim_y_warm,
        "bandpass",
        [10, 30],
        nwts=NWTS,
    ).filted()

    warm_plot_df["target_filt_plot"] = pad_filtered_result(
        y_target_y_warm,
        y_target_anom_filt,
    ) + 24

    warm_plot_df["comp_filt_plot"] = pad_filtered_result(
        comp_y_warm,
        comp_anom_filt,
    ) + 24

    plot_df = warm_plot_df[warm_plot_df["month"].isin(PLOT_MONTHS)].copy()

    plot_df["summer_day"] = build_summer_day_index(
        pd.DatetimeIndex(
            pd.to_datetime(
                pd.DataFrame(
                    {
                        "year": np.full(len(plot_df), TARGET_YEAR),
                        "month": plot_df["month"].values,
                        "day": plot_df["day"].values,
                    }
                )
            )
        )
    )

    plot_df = plot_df.sort_values("summer_day").reset_index(drop=True)
    validate_nonempty(plot_df, f"{dataset_label} final plotting data")

    ds.close()
    return plot_df


# =========================================================
# ======================= Plot funcs =======================
# =========================================================

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase


class HandlerBiColorLine(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        xm = x0 + width / 2.0

        line1 = Line2D(
            [x0, xm],
            [y0 + height / 2, y0 + height / 2],
            color="orangered",
            linewidth=1.5,
        )
        line2 = Line2D(
            [xm, x0 + width],
            [y0 + height / 2, y0 + height / 2],
            color="green",
            linewidth=1.5,
        )

        line1.set_transform(trans)
        line2.set_transform(trans)
        return [line1, line2]


class HandlerBiColorPatch(HandlerBase):
    """Bi-color fill legend: red on the left, blue on the right."""
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        w2 = width / 2.0

        patch1 = Rectangle(
            (x0, y0),
            w2,
            height,
            facecolor="lightcoral",
            edgecolor="none",
            alpha=0.35,
            transform=trans,
        )
        patch2 = Rectangle(
            (x0 + w2, y0),
            w2,
            height,
            facecolor="deepskyblue",
            edgecolor="none",
            alpha=0.35,
            transform=trans,
        )

        return [patch1, patch2]


def plot_panel(
    ax,
    plot_df: pd.DataFrame,
    panel_title: str,
    dataset_name: str,
    show_legend: bool = True,
):
    """Draw one ERA5/CN05.1 panel on the given axis."""
    x = plot_df["summer_day"].values
    clim_y = plot_df["climatology_actual"].values
    comp_y = plot_df["composite_actual"].values
    comp_std = plot_df["composite_std"].values
    y_target_y = plot_df[f"{TARGET_YEAR}_actual"].values
    y_target_y_filt = plot_df["target_filt_plot"].values
    comp_y_filt = plot_df["comp_filt_plot"].values

    # Study period background: Jun-15 to Jul-31.
    ax.axvspan(14, 61, color="#959595", alpha=0.3, zorder=0)

    # 2015 vs climatology fill.
    ax.fill_between(
        x,
        y_target_y,
        clim_y,
        where=(y_target_y > clim_y),
        interpolate=True,
        color="lightcoral",
        alpha=0.35,
        zorder=1,
    )
    ax.fill_between(
        x,
        y_target_y,
        clim_y,
        where=(y_target_y < clim_y),
        interpolate=True,
        color="deepskyblue",
        alpha=0.35,
        zorder=1,
    )

    # 2015 below composite fill.
    ax.fill_between(
        x,
        y_target_y,
        comp_y,
        where=(y_target_y < comp_y),
        interpolate=True,
        color="#007bbb",
        alpha=0.6,
        zorder=2,
    )

    # Climatology and composite lines.
    ax.plot(x, clim_y, color="black", linestyle="-", linewidth=1, label="Clim.", zorder=4)
    ax.plot(x, comp_y, color="blue", linestyle="--", linewidth=1.5, label="Comp.", zorder=4)

    # Keep original target-year actual field invisible.
    ax.plot(x, y_target_y, color="#959595", linestyle="-", linewidth=0, zorder=5)

    # Composite daily std bars.
    y_base = 19.5
    ax.bar(
        x,
        comp_std,
        color="limegreen",
        bottom=y_base,
        alpha=0.25,
        linewidth=0,
        zorder=1.5,
    )

    # Filtered target-year anomaly line.
    ax.plot(
        x,
        np.ma.masked_where(y_target_y_filt > 24, y_target_y_filt),
        color="green",
        linestyle="-",
        linewidth=1.3,
        zorder=5,
    )
    ax.plot(
        x,
        y_target_y_filt,
        color="orangered",
        linestyle="-",
        linewidth=1.3,
        zorder=4,
    )

    # Right secondary y-axis: filtered anomaly values are shown as value + 24.
    secax = ax.secondary_yaxis("right")
    secax.set_yticks(np.arange(20, 29, 2))
    secax.set_yticklabels(np.arange(-4, 5, 2), fontsize=12)
    secax.tick_params(axis="y", labelsize=12)

    # Typhoon marks.
    typhoons = [
        {"start": (6, 30), "end": (7, 13)},
        {"start": (7, 2), "end": (7, 10)},
        {"start": (7, 30), "end": (8, 10)},
    ]

    for ty_index, tp in enumerate(typhoons):
        start_day = month_day_to_summer_day(*tp["start"])
        end_day = month_day_to_summer_day(*tp["end"])

        for iday in range(start_day, end_day + 1):
            ax.text(
                iday,
                19.6 if ty_index in [0, 2] else 19.8,
                "·",
                ha="center",
                va="center",
                color="red",
                fontsize=25,
                clip_on=False,
                zorder=10,
            )

    ax.set_xlim(1, 92)
    ax.set_ylim(y_base, 28.5)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(panel_title, loc="left", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)

    # =====================================================
    # 新增：在图内正上方中间添加 ERA5 / CN05.1 黑框标注
    # =====================================================
    ax.text(
        0.5,
        0.97,
        dataset_name,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="square,pad=0.25",
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            alpha=1.0,
        ),
        zorder=30,
    )

    if show_legend:
        plt.rcParams["legend.fontsize"] = 8

        line_clim = Line2D([0], [0], color="black", lw=1, linestyle="-", label="Clim.")
        line_comp = Line2D([0], [0], color="blue", lw=1.5, linestyle="--", label="Cool sum. comp.")
        bi_line = Line2D([0], [0], color="none", label="2015 filtered anom.")

        legend1 = ax.legend(
            handles=[bi_line, line_comp, line_clim],
            handler_map={bi_line: HandlerBiColorLine()},
            frameon=False,
            loc="upper left",
            borderaxespad=0.0,
        )
        ax.add_artist(legend1)

        bar_handle = Rectangle(
            (0, 0),
            1,
            1,
            facecolor="limegreen",
            edgecolor="none",
            alpha=0.25,
            label="Comp. std",
        )

        fill_handle = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none")
        fill_handle.set_label("2015 anom.")

        typhoon_handle = Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            linestyle="None",
            markersize=3,
            label="Typhoon",
        )

        ax.legend(
            handles=[fill_handle, bar_handle, typhoon_handle],
            handler_map={fill_handle: HandlerBiColorPatch()},
            frameon=False,
            loc="upper right",
            borderaxespad=0.0,
        )

    return secax


# =========================================================
# =========================== Main =========================
# =========================================================

def main():
    era5_df = prepare_panel_data(
        ERA5_NC_FILE,
        dataset_label="ERA5",
        preferred_vars=["t2m", "2m_temperature", "tem"],
    )

    cn05_df = prepare_panel_data(
        CN05_NC_FILE,
        dataset_label="CN05.1",
        preferred_vars=["tem", "t2m", "2m_temperature"],
    )

    print("\n===== Plotting combined figure =====")

    tick_positions = [1, 16, 31, 46, 62, 77, 92]
    tick_labels = ["Jun-01", "Jun-16", "Jul-01", "Jul-16", "Aug-01", "Aug-16", "Aug-31"]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6, 6),
        sharex=True,
    )

    secax1 = plot_panel(
        axes[0],
        era5_df,
        panel_title="(a) Index",
        dataset_name="ERA5",
        show_legend=True,
    )

    secax2 = plot_panel(
        axes[1],
        cn05_df,
        panel_title="",
        dataset_name="CN05.1",
        show_legend=True,
    )

    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels)
    axes[1].set_xlabel("")

    # Hide upper x tick labels to make the two-panel figure cleaner.
    axes[0].tick_params(labelbottom=False)

    # Make spines thicker for all main and secondary axes.
    for _ax in [axes[0], axes[1], secax1, secax2]:
        for spine in _ax.spines.values():
            spine.set_linewidth(1.5)

    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig(OUT_FIG + ".png", dpi=600, bbox_inches="tight")
    plt.savefig(OUT_FIG + ".pdf", bbox_inches="tight")
    plt.close()

    print(f"Combined figure saved to: {OUT_FIG}.png")
    print(f"Combined figure saved to: {OUT_FIG}.pdf")
import os
import numpy as np
import pandas as pd
import xarray as xr
from climkit.masked import masked
from climkit.filter import *
import matplotlib.pyplot as plt

# =========================================================
# ====================== User settings =====================
# =========================================================

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

# ERA5 and CN05.1 input files
ERA5_NC_FILE = "/Volumes/TiPlus7100/p4/data/ERA5_daily_t2m_sum.nc"
CN05_NC_FILE = "/Volumes/TiPlus7100/data/CHN_GRID_DATA/tem/Tmean_19510101-20250714_0.5.nc"

# Yangtze River basin shapefile
SHP_FILE = fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"

# Output directory
OUT_DIR = fr"{PYFILE}/p4/pic/"
os.makedirs(OUT_DIR, exist_ok=True)

# Output figure
OUT_FIG = os.path.join(OUT_DIR, "Fig3_t2m_daily_actual_ERA5_CN051_combined")

# Climatology period
CLIM_START = 1991
CLIM_END = 2020

# Composite years
COMPOSITE_YEARS = [
    1965, 1966, 1968, 1974, 1976, 1980, 1982, 1983, 1986,
    1987, 1989, 1992, 1993, 1997, 1999, 2004, 2008, 2014
]  # 0.5 std

# Target year
TARGET_YEAR = 2015

# Filter and plot settings
FILTER_MONTHS = [5, 6, 7, 8, 9]   # for filtering
PLOT_MONTHS = [6, 7, 8]            # for plotting
NWTS = 61


# =========================================================
# ====================== Helper funcs ======================
# =========================================================

def detect_main_var(ds: xr.Dataset, preferred=None) -> str:
    """Automatically detect the main temperature variable."""
    if preferred is None:
        preferred = ["t2m", "2m_temperature", "tem"]

    for v in preferred:
        if v in ds.data_vars:
            return v

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise ValueError(f"Cannot automatically detect temperature variable: {list(ds.data_vars)}")


def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Standardize longitude/latitude coordinate names to lon/lat."""
    rename_dict = {}

    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"

    if rename_dict:
        ds = ds.rename(rename_dict)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("No lon/lat or longitude/latitude coordinates found.")

    return ds


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """Basin mean. This keeps your original simple spatial mean logic."""
    return da.mean(dim=("lat", "lon"))


def parse_time_column(s: pd.Series) -> pd.Series:
    """Robustly parse time values from datetime64 or YYYYMMDD-like integers/strings."""
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s)

    s_str = s.astype(str)

    if s_str.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(s_str, format="%Y%m%d")

    return pd.to_datetime(s, errors="raise")


def build_summer_day_index(time_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Summer day index:
    Jun-01 -> 1
    ...
    Aug-31 -> 92
    """
    month = time_index.month
    day = time_index.day
    offsets = np.where(month == 6, 0, np.where(month == 7, 30, 61))
    return offsets + day


def build_warm_day_index(time_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Warm-season day index:
    May-01 -> 1
    ...
    Sep-30 -> 153
    """
    month = time_index.month
    day = time_index.day

    offsets = np.select(
        [
            month == 5,
            month == 6,
            month == 7,
            month == 8,
            month == 9,
        ],
        [
            0,
            31,
            61,
            92,
            123,
        ],
        default=np.nan,
    )

    if np.any(np.isnan(offsets)):
        raise ValueError("build_warm_day_index only supports May-September.")

    return offsets.astype(int) + day


def ensure_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert Kelvin-like temperature to Celsius."""
    vmin = float(da.min().values)
    vmax = float(da.max().values)

    if vmin > 150 and vmax < 400:
        print("Detected Kelvin-like temperature values. Converted to Celsius.")
        da = da - 273.15
        da.attrs["units"] = "degC"

    return da


def pad_filtered_result(original: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    """Pad shortened filtered sequence back to original length using NaN on both sides."""
    out = np.full(len(original), np.nan)
    m = len(filtered)

    if m > len(original):
        raise ValueError("Filtered sequence is longer than the original sequence.")

    pad_left = (len(original) - m) // 2
    pad_right = len(original) - m - pad_left

    if pad_right == 0:
        out[pad_left:] = filtered
    else:
        out[pad_left:-pad_right] = filtered

    return out


def month_day_to_summer_day(month: int, day: int) -> int:
    """Convert month/day to summer day index: Jun-01 -> 1, Aug-31 -> 92."""
    if month == 6:
        return day
    if month == 7:
        return 30 + day
    if month == 8:
        return 61 + day

    raise ValueError("Only June-August are supported.")


def validate_nonempty(df: pd.DataFrame, name: str):
    """Raise a clear error if a required DataFrame is empty."""
    if df.empty:
        raise ValueError(
            f"{name} is empty. Please check the input data time range, "
            f"COMPOSITE_YEARS, CLIM_START/CLIM_END, and TARGET_YEAR."
        )


# =========================================================
# ===================== Data processing ====================
# =========================================================

def prepare_panel_data(nc_file: str, dataset_label: str, preferred_vars=None) -> pd.DataFrame:
    """
    Read one dataset, calculate basin-mean daily climatology, composite,
    target-year values, and filtered anomalies for plotting.
    """
    print(f"\n===== Processing {dataset_label} =====")
    print("1) Reading nc data...")

    ds = xr.open_dataset(nc_file)
    ds = standardize_latlon(ds)

    var_name = detect_main_var(ds, preferred=preferred_vars)
    print(f"Detected variable name: {var_name}")

    da = ds[var_name]

    if "time" not in da.coords:
        if "valid_time" in da.coords:
            da = da.rename({"valid_time": "time"})
        else:
            raise ValueError("No time or valid_time coordinate found.")

    data_start = min([CLIM_START, TARGET_YEAR] + COMPOSITE_YEARS)
    data_end = max([CLIM_END, TARGET_YEAR] + COMPOSITE_YEARS)

    da = da.sel(time=slice(f"{data_start}-05-01", f"{data_end}-09-30"))
    da = ensure_celsius(da)

    print("2) Clipping to Yangtze River basin...")
    da_clip = masked(da, SHP_FILE)

    print("3) Calculating basin mean...")
    ts = area_weighted_mean(da_clip)

    df = ts.to_dataframe(name="t2m").reset_index()
    df["time"] = parse_time_column(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day

    df = df[df["month"].isin(FILTER_MONTHS)].copy()
    df["warm_day"] = build_warm_day_index(pd.DatetimeIndex(df["time"]))

    print("4) Calculating daily climatology...")
    clim_warm_df = (
        df[(df["year"] >= CLIM_START) & (df["year"] <= CLIM_END)]
        .groupby("warm_day", as_index=False)["t2m"]
        .mean()
        .rename(columns={"t2m": "climatology_actual"})
    )
    validate_nonempty(clim_warm_df, f"{dataset_label} climatology")

    print("5) Calculating composite daily mean and standard deviation...")
    comp_warm_stat_df = (
        df[df["year"].isin(COMPOSITE_YEARS)]
        .groupby("warm_day")["t2m"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "composite_actual",
                "std": "composite_std",
            }
        )
    )
    validate_nonempty(comp_warm_stat_df, f"{dataset_label} composite")
    comp_warm_stat_df["composite_std"] = comp_warm_stat_df["composite_std"].fillna(0.0)

    print(f"6) Calculating {TARGET_YEAR} daily values...")
    y_target_warm_df = (
        df[df["year"] == TARGET_YEAR][["warm_day", "t2m"]]
        .groupby("warm_day", as_index=False)
        .mean()
        .rename(columns={"t2m": f"{TARGET_YEAR}_actual"})
    )
    validate_nonempty(y_target_warm_df, f"{dataset_label} target year")

    print("7) Merging warm-season sequence and filtering...")
    warm_plot_df = clim_warm_df.merge(comp_warm_stat_df, on="warm_day", how="inner")
    warm_plot_df = warm_plot_df.merge(y_target_warm_df, on="warm_day", how="inner")
    validate_nonempty(warm_plot_df, f"{dataset_label} merged warm-season data")

    calendar_df = (
        df[df["year"] == TARGET_YEAR][["warm_day", "month", "day"]]
        .drop_duplicates(subset=["warm_day"])
        .sort_values("warm_day")
    )
    warm_plot_df = warm_plot_df.merge(calendar_df, on="warm_day", how="left")

    clim_y_warm = warm_plot_df["climatology_actual"].values
    comp_y_warm = warm_plot_df["composite_actual"].values
    y_target_y_warm = warm_plot_df[f"{TARGET_YEAR}_actual"].values

    y_target_anom_filt = LanczosFilter(
        y_target_y_warm - clim_y_warm,
        "bandpass",
        [10, 30],
        nwts=NWTS,
    ).filted()

    comp_anom_filt = LanczosFilter(
        comp_y_warm - clim_y_warm,
        "bandpass",
        [10, 30],
        nwts=NWTS,
    ).filted()

    warm_plot_df["target_filt_plot"] = pad_filtered_result(
        y_target_y_warm,
        y_target_anom_filt,
    ) + 24

    warm_plot_df["comp_filt_plot"] = pad_filtered_result(
        comp_y_warm,
        comp_anom_filt,
    ) + 24

    plot_df = warm_plot_df[warm_plot_df["month"].isin(PLOT_MONTHS)].copy()

    plot_df["summer_day"] = build_summer_day_index(
        pd.DatetimeIndex(
            pd.to_datetime(
                pd.DataFrame(
                    {
                        "year": np.full(len(plot_df), TARGET_YEAR),
                        "month": plot_df["month"].values,
                        "day": plot_df["day"].values,
                    }
                )
            )
        )
    )

    plot_df = plot_df.sort_values("summer_day").reset_index(drop=True)
    validate_nonempty(plot_df, f"{dataset_label} final plotting data")

    ds.close()
    return plot_df


# =========================================================
# ======================= Plot funcs =======================
# =========================================================

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase


class HandlerBiColorLine(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        xm = x0 + width / 2.0

        line1 = Line2D(
            [x0, xm],
            [y0 + height / 2, y0 + height / 2],
            color="orangered",
            linewidth=1.5,
        )
        line2 = Line2D(
            [xm, x0 + width],
            [y0 + height / 2, y0 + height / 2],
            color="green",
            linewidth=1.5,
        )

        line1.set_transform(trans)
        line2.set_transform(trans)
        return [line1, line2]


class HandlerBiColorPatch(HandlerBase):
    """Bi-color fill legend: red on the left, blue on the right."""
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        w2 = width / 2.0

        patch1 = Rectangle(
            (x0, y0),
            w2,
            height,
            facecolor="lightcoral",
            edgecolor="none",
            alpha=0.35,
            transform=trans,
        )
        patch2 = Rectangle(
            (x0 + w2, y0),
            w2,
            height,
            facecolor="deepskyblue",
            edgecolor="none",
            alpha=0.35,
            transform=trans,
        )

        return [patch1, patch2]


def plot_panel(
    ax,
    plot_df: pd.DataFrame,
    panel_title: str,
    dataset_name: str,
    show_legend: bool = True,
):
    """Draw one ERA5/CN05.1 panel on the given axis."""
    x = plot_df["summer_day"].values
    clim_y = plot_df["climatology_actual"].values
    comp_y = plot_df["composite_actual"].values
    comp_std = plot_df["composite_std"].values
    y_target_y = plot_df[f"{TARGET_YEAR}_actual"].values
    y_target_y_filt = plot_df["target_filt_plot"].values
    comp_y_filt = plot_df["comp_filt_plot"].values

    # Study period background: Jun-15 to Jul-31.
    ax.axvspan(14.5, 61.5, color="#959595", alpha=0.3, zorder=0)

    # 2015 vs climatology fill.
    ax.fill_between(
        x,
        y_target_y,
        clim_y,
        where=(y_target_y > clim_y),
        interpolate=True,
        color="lightcoral",
        alpha=0.35,
        zorder=1,
    )
    ax.fill_between(
        x,
        y_target_y,
        clim_y,
        where=(y_target_y < clim_y),
        interpolate=True,
        color="deepskyblue",
        alpha=0.35,
        zorder=1,
    )

    # 2015 below composite fill.
    ax.fill_between(
        x,
        y_target_y,
        comp_y,
        where=(y_target_y < comp_y),
        interpolate=True,
        color="#007bbb",
        alpha=0.6,
        zorder=2,
    )

    # Climatology and composite lines.
    ax.plot(x, clim_y, color="black", linestyle="-", linewidth=1, label="Clim.", zorder=4)
    ax.plot(x, comp_y, color="blue", linestyle="--", linewidth=1.5, label="Comp.", zorder=4)

    # Keep original target-year actual field invisible.
    ax.plot(x, y_target_y, color="#959595", linestyle="-", linewidth=0, zorder=5)

    # Composite daily std bars.
    y_base = 19.5
    ax.bar(
        x,
        comp_std,
        color="limegreen",
        bottom=y_base,
        alpha=0.25,
        linewidth=0,
        zorder=1.5,
    )

    # Filtered target-year anomaly line.
    ax.plot(
        x,
        np.ma.masked_where(y_target_y_filt > 24, y_target_y_filt),
        color="green",
        linestyle="-",
        linewidth=1.3,
        zorder=5,
    )
    ax.plot(
        x,
        y_target_y_filt,
        color="orangered",
        linestyle="-",
        linewidth=1.3,
        zorder=4,
    )

    # Right secondary y-axis: filtered anomaly values are shown as value + 24.
    secax = ax.secondary_yaxis("right")
    secax.set_yticks(np.arange(20, 29, 2))
    secax.set_yticklabels(np.arange(-4, 5, 2), fontsize=12)
    secax.tick_params(axis="y", labelsize=12)

    # Typhoon marks.
    # typhoons = [
    #     {"start": (6, 30), "end": (7, 13)},
    #     {"start": (7, 2), "end": (7, 10)},
    #     {"start": (7, 30), "end": (8, 10)},
    # ]
    #
    # for ty_index, tp in enumerate(typhoons):
    #     start_day = month_day_to_summer_day(*tp["start"])
    #     end_day = month_day_to_summer_day(*tp["end"])
    #
    #     for iday in range(start_day, end_day + 1):
    #         ax.text(
    #             iday,
    #             19.6 if ty_index in [0, 2] else 19.8,
    #             "·",
    #             ha="center",
    #             va="center",
    #             color="red",
    #             fontsize=25,
    #             clip_on=False,
    #             zorder=10,
    #         )

    ax.set_xlim(1, 92)
    ax.set_ylim(y_base, 28.5)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(panel_title, loc="left", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)

    # =====================================================
    # 新增：在图内正上方中间添加 ERA5 / CN05.1 黑框标注
    # =====================================================
    ax.text(
        0.5,
        0.92,
        dataset_name,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="square",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.8,
        ),
        zorder=30,
    )

    if show_legend:
        plt.rcParams["legend.fontsize"] = 8

        line_clim = Line2D([0], [0], color="black", lw=1, linestyle="-", label="Clim.")
        line_comp = Line2D([0], [0], color="blue", lw=1.5, linestyle="--", label="Cool sum. comp.")
        bi_line = Line2D([0], [0], color="none", label="2015 filtered anom.")

        legend1 = ax.legend(
            handles=[bi_line, line_comp, line_clim],
            handler_map={bi_line: HandlerBiColorLine()},
            frameon=False,
            loc="upper left",
            borderaxespad=0.0,
        )
        ax.add_artist(legend1)

        bar_handle = Rectangle(
            (0, 0),
            1,
            1,
            facecolor="limegreen",
            edgecolor="none",
            alpha=0.25,
            label="Comp. std",
        )

        fill_handle = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none")
        fill_handle.set_label("2015 anom.")

        # typhoon_handle = Line2D(
        #     [0],
        #     [0],
        #     marker="o",
        #     color="red",
        #     linestyle="None",
        #     markersize=3,
        #     label="Typhoon",
        # )

        ax.legend(
            handles=[fill_handle, bar_handle],
            handler_map={fill_handle: HandlerBiColorPatch()},
            frameon=False,
            loc="upper right",
            borderaxespad=0.0,
        )

    return secax


# =========================================================
# =========================== Main =========================
# =========================================================

def main():
    era5_df = prepare_panel_data(
        ERA5_NC_FILE,
        dataset_label="ERA5",
        preferred_vars=["t2m", "2m_temperature", "tem"],
    )

    cn05_df = prepare_panel_data(
        CN05_NC_FILE,
        dataset_label="CN05.1",
        preferred_vars=["tem", "t2m", "2m_temperature"],
    )

    print("\n===== Plotting combined figure =====")

    tick_positions = [1, 16, 31, 46, 62, 77, 92]
    tick_labels = ["Jun-01", "Jun-16", "Jul-01", "Jul-16", "Aug-01", "Aug-16", "Aug-31"]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6, 6),
        sharex=True,
    )

    secax1 = plot_panel(
        axes[0],
        era5_df,
        panel_title="Summer daily T2m",
        dataset_name="ERA5",
        show_legend=True,
    )

    secax2 = plot_panel(
        axes[1],
        cn05_df,
        panel_title="",
        dataset_name="CN05.1",
        show_legend=True,
    )

    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels)
    axes[1].set_xlabel("")

    # Hide upper x tick labels to make the two-panel figure cleaner.
    axes[0].tick_params(labelbottom=False)

    # Make spines thicker for all main and secondary axes.
    for _ax in [axes[0], axes[1], secax1, secax2]:
        for spine in _ax.spines.values():
            spine.set_linewidth(1.5)

    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig(OUT_FIG + ".png", dpi=600, bbox_inches="tight")
    plt.savefig(OUT_FIG + ".pdf", bbox_inches="tight")
    plt.close()

    print(f"Combined figure saved to: {OUT_FIG}.png")
    print(f"Combined figure saved to: {OUT_FIG}.pdf")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
