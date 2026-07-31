from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. File paths
# ============================================================
nc_file = Path(
    r"/Volumes/TiPlus7100/PyFile/p4/data/SVD_V200_OLR_T2m_and_raw_T2m_standardized_annual_STD_1961_2022.nc"
)

output_file = Path("/Volumes/TiPlus7100/PyFile/p4/pic/V200_OLR_sum_vs_raw_T2m_TCC.png")


# ============================================================
# 2. Function for reading a variable and handling fill values
# ============================================================
def read_variable(nc, variable_name):
    """
    Read a variable from an HDF5/NetCDF4 file and convert fill values to NaN.
    """
    if variable_name not in nc:
        raise KeyError(
            f"Variable '{variable_name}' was not found.\n"
            f"Available variables: {list(nc.keys())}"
        )

    values = np.asarray(nc[variable_name][...], dtype=float).squeeze()

    if "_FillValue" in nc[variable_name].attrs:
        fill_value = np.asarray(nc[variable_name].attrs["_FillValue"]).squeeze()
        values[np.isclose(values, fill_value)] = np.nan

    values[np.abs(values) > 1.0e19] = np.nan
    return values


# ============================================================
# 3. Read data
# ============================================================
with h5py.File(nc_file, mode="r") as nc:
    years = read_variable(nc, "year").astype(int)
    b_v200 = read_variable(nc, "b_v200_score_z_std")
    d_olr = read_variable(nc, "d_olr_score_z_std")
    raw_t2m = read_variable(nc, "raw_t2m_area_mean_z_std")


# ============================================================
# 4. Check array lengths
# ============================================================
series_lengths = {
    "year": len(years),
    "b_v200_score_z_std": len(b_v200),
    "d_olr_score_z_std": len(d_olr),
    "raw_t2m_area_mean_z_std": len(raw_t2m),
}

if len(set(series_lengths.values())) != 1:
    raise ValueError(
        "The variables do not have the same length:\n"
        f"{series_lengths}"
    )


# ============================================================
# 5. Compute the summed series and shifted series
# ============================================================
v200_olr_sum = b_v200 + d_olr
v200_olr_sum_shifted = v200_olr_sum - 1.0   # shift downward by 1 unit


# ============================================================
# 6. Calculate TCC
#    TCC is calculated from the ORIGINAL summed series
#    and raw_t2m_area_mean_z_std
# ============================================================
valid = np.isfinite(v200_olr_sum) & np.isfinite(raw_t2m)

if np.count_nonzero(valid) < 3:
    raise ValueError(
        "There are fewer than three valid paired observations, "
        "so TCC cannot be calculated."
    )

tcc = np.corrcoef(v200_olr_sum[valid], raw_t2m[valid])[0, 1]

print(f"Number of valid years: {np.count_nonzero(valid)}")
print(f"TCC: {tcc:.6f}")
print(f"TCC with two significant digits: {tcc:.2g}")


# ============================================================
# 7. Create figure
# ============================================================
fig, ax1 = plt.subplots(figsize=(15, 6.5))
x = np.arange(len(years))

# ---- Left y-axis: red bars ----
bar_width = 0.62
ax1.bar(
    x,
    raw_t2m,
    width=bar_width,
    color="red",
    edgecolor="black",
    linewidth=0.35,
    alpha=0.70,
    label="raw_t2m_area_mean_z_std",
)

ax1.set_xlabel("Year", fontsize=13)
ax1.set_ylabel("raw_t2m_area_mean_z_std", fontsize=13, color="red")
ax1.tick_params(axis="y", labelcolor="red")
ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

# ---- Right y-axis: blue line ----
ax2 = ax1.twinx()
ax2.plot(
    x,
    v200_olr_sum_shifted,
    color="blue",
    linewidth=2.2,
    marker="o",
    markersize=4.5,
    label="b_v200_score_z_std + d_olr_score_z_std (shifted down by 1)",
)

ax2.set_ylabel(
    "b_v200_score_z_std + d_olr_score_z_std (shifted down by 1)",
    fontsize=13,
    color="blue"
)
ax2.tick_params(axis="y", labelcolor="blue")


# ============================================================
# 8. X-axis ticks
# ============================================================
tick_indices = np.arange(0, len(years), 5)
if tick_indices[-1] != len(years) - 1:
    tick_indices = np.append(tick_indices, len(years) - 1)

ax1.set_xticks(tick_indices)
ax1.set_xticklabels(years[tick_indices], rotation=45, ha="right")
ax1.set_xlim(-0.8, len(years) - 0.2)


# ============================================================
# 9. Title
# ============================================================
ax1.set_title(
    "Shifted V200+OLR Line and Raw T2m Bar Chart",
    fontsize=15,
    pad=12,
)


# ============================================================
# 10. Add TCC annotation in the upper-right corner
# ============================================================
ax2.text(
    0.98,
    0.97,
    f"TCC = {tcc:.2g}",
    transform=ax2.transAxes,
    horizontalalignment="right",
    verticalalignment="top",
    fontsize=14,
    fontweight="bold",
    bbox={
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "black",
        "alpha": 0.85,
    },
)


# ============================================================
# 11. Combine legends from both axes
# ============================================================
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left",
    fontsize=10,
    frameon=True,
)


# ============================================================
# 12. Save and display
# ============================================================
fig.tight_layout()
fig.savefig(output_file, dpi=300, bbox_inches="tight")

print(f"Figure saved to: {output_file}")

plt.show()
