import os
import numpy as np
import pandas as pd
import xarray as xr
from climkit.masked import masked
import matplotlib.pyplot as plt

# =========================================================
# ====================== User settings =====================
# =========================================================

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

# CN05.1 input file
CN05_NC_FILE = "/Volumes/TiPlus7100/p4/data/ERA5_daily_tcc_sum.nc"

# Yangtze River basin shapefile
SHP_FILE = fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"

# Output directory
OUT_DIR = fr"{PYFILE}/p4/pic/"
os.makedirs(OUT_DIR, exist_ok=True)

# Output figure
OUT_FIG = os.path.join(OUT_DIR, "Fig3_tcc_daily_actual_CN051")

# Climatology period
CLIM_START = 1991
CLIM_END = 2020

# Target year
TARGET_YEAR = 2015

# Plot months
PLOT_MONTHS = [6, 7, 8]


# =========================================================
# ====================== Helper funcs ======================
# =========================================================

def detect_main_var(ds: xr.Dataset, preferred=None) -> str:
    """Automatically detect the main precipitation variable."""
    if preferred is None:
        preferred = ["tcc", "tcc", "tcc"]

    for v in preferred:
        if v in ds.data_vars:
            return v

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise ValueError(f"Cannot automatically detect precipitation variable: {list(ds.data_vars)}")


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


def validate_nonempty(df: pd.DataFrame, name: str):
    """Raise a clear error if a required DataFrame is empty."""
    if df.empty:
        raise ValueError(
            f"{name} is empty. Please check the input data time range, "
            f"CLIM_START/CLIM_END and TARGET_YEAR."
        )


# =========================================================
# ===================== Data processing ====================
# =========================================================

def prepare_panel_data(nc_file: str, dataset_label: str, preferred_vars=None) -> pd.DataFrame:
    """
    Read CN05.1 dataset, calculate basin-mean daily climatology,
    target-year daily precipitation, and cumulative precipitation.
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

    data_start = min(CLIM_START, TARGET_YEAR)
    data_end = max(CLIM_END, TARGET_YEAR)

    da = da.sel(time=slice(f"{data_start}-06-01", f"{data_end}-08-31"))

    print("2) Clipping to Yangtze River basin...")
    da_clip = masked(da, SHP_FILE)

    print("3) Calculating basin mean...")
    ts = area_weighted_mean(da_clip)

    df = ts.to_dataframe(name="pre").reset_index()
    df["time"] = parse_time_column(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day

    df = df[df["month"].isin(PLOT_MONTHS)].copy()
    df["summer_day"] = build_summer_day_index(pd.DatetimeIndex(df["time"]))

    print("4) Calculating daily climatology...")
    clim_df = (
        df[(df["year"] >= CLIM_START) & (df["year"] <= CLIM_END)]
        .groupby("summer_day", as_index=False)["pre"]
        .mean()
        .rename(columns={"pre": "climatology_actual"})
    )
    validate_nonempty(clim_df, f"{dataset_label} climatology")

    print(f"5) Calculating {TARGET_YEAR} daily precipitation...")
    target_df = (
        df[df["year"] == TARGET_YEAR]
        .groupby("summer_day", as_index=False)["pre"]
        .mean()
        .rename(columns={"pre": f"{TARGET_YEAR}_actual"})
    )
    validate_nonempty(target_df, f"{dataset_label} target year")

    print("6) Merging and calculating cumulative precipitation...")
    plot_df = clim_df.merge(target_df, on="summer_day", how="inner")
    validate_nonempty(plot_df, f"{dataset_label} final plotting data")

    plot_df = plot_df.sort_values("summer_day").reset_index(drop=True)

    plot_df["clim_cum"] = plot_df["climatology_actual"].cumsum()
    plot_df[f"{TARGET_YEAR}_cum"] = plot_df[f"{TARGET_YEAR}_actual"].cumsum()

    ds.close()
    return plot_df


# =========================================================
# ======================= Plot funcs =======================
# =========================================================

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def plot_panel(
    ax,
    plot_df: pd.DataFrame,
    panel_title: str,
    dataset_name: str,
    show_legend: bool = True,
):
    """Draw CN05.1 daily precipitation bars and cumulative precipitation lines."""
    x = plot_df["summer_day"].values

    clim_daily = plot_df["climatology_actual"].values
    target_daily = plot_df[f"{TARGET_YEAR}_actual"].values

    clim_cum = plot_df["clim_cum"].values
    target_cum = plot_df[f"{TARGET_YEAR}_cum"].values

    # Study period background: Jun-15 to Jul-31.
    ax.axvspan(14.5, 61.5, color="#959595", alpha=0.3, zorder=0)

    # 2015 original daily precipitation field: green filled bars.
    ax.bar(
        x,
        target_daily,
        width=0.85,
        color="limegreen",
        edgecolor="none",
        alpha=0.45,
        linewidth=0,
        label=f"{TARGET_YEAR} tcc",
        zorder=2,
    )

    # Climatological daily precipitation: black-edged, no-fill bars.
    ax.bar(
        x,
        clim_daily,
        width=0.78,
        facecolor="none",
        edgecolor="black",
        linewidth=0.3,
        label="Clim. daily tcc",
        zorder=3,
    )

    # Right axis for cumulative precipitation.
    ax2 = ax.twinx()


    ax.set_xlim(1, 92)

    daily_ymax = max(np.nanmax(target_daily), np.nanmax(clim_daily))
    ax.set_ylim(0, daily_ymax * 1.25)

    ax.set_title(panel_title, loc="left", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Dataset label box.
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

        bar_2015 = Rectangle(
            (0, 0),
            1,
            1,
            facecolor="limegreen",
            edgecolor="none",
            alpha=0.45,
            label=f"{TARGET_YEAR} daily tcc",
        )

        bar_clim = Rectangle(
            (0, 0),
            1,
            1,
            facecolor="none",
            edgecolor="black",
            linewidth=0.7,
            label="Clim. daily tcc",
        )

    return ax2


# =========================================================
# =========================== Main =========================
# =========================================================

def main():
    cn05_df = prepare_panel_data(
        CN05_NC_FILE,
        dataset_label="CN05.1",
        preferred_vars=["tcc", "tcc", "tcc"],
    )

    print("\n===== Plotting CN05.1 figure =====")

    tick_positions = [1, 16, 31, 46, 62, 77, 92]
    tick_labels = ["Jun-01", "Jun-16", "Jul-01", "Jul-16", "Aug-01", "Aug-16", "Aug-31"]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(6, 3.2),
    )

    ax2 = plot_panel(
        ax,
        cn05_df,
        panel_title="Summer daily tcc",
        dataset_name="ERA5",
        show_legend=True,
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("")

    # Make spines thicker for main and secondary axes.
    for _ax in [ax, ax2]:
        for spine in _ax.spines.values():
            spine.set_linewidth(1.5)

    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig(OUT_FIG + ".png", dpi=600, bbox_inches="tight")
    plt.savefig(OUT_FIG + ".pdf", bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {OUT_FIG}.png")
    print(f"Figure saved to: {OUT_FIG}.pdf")


if __name__ == "__main__":
    main()
