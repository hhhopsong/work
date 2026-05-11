import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# COOL_YEAR = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014, 2015] #1 std
COOL_YEAR = [1965, 1966, 1968, 1974, 1976, 1980, 1982, 1983, 1986, 1987, 1989, 1992, 1993, 1997, 1999, 2004, 2008, 2014, 2015] # 0.5 std

wavelet = xr.open_dataset(
    "/Volumes/TiPlus7100/p4/data/CN05_daily_ano_all_JJA_wavelet.nc"
)

period = wavelet["period"]
global_power = wavelet["global_power"]
global_signif = wavelet["global_signif"]

# 年份维度名按你的数据实际情况修改
year_dim = "time"

# 只保留 period <= 64
mask_period = period <= 64
period_sel = period.where(mask_period, drop=True)

period_dim = period.dims[0]

global_power = global_power.sel({period_dim: period_sel})
global_signif = global_signif.sel({period_dim: period_sel})

x = np.log2(period_sel.values)

cool_years_desc = COOL_YEAR[::-1]

# 字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

fig, ax = plt.subplots(figsize=(5, 4))

# =========================
# 1. 除 2015 年外冷年合成
# =========================
cool_years_no2015 = [y for y in COOL_YEAR if y != 2015]
warm_years = [y for y in range(1961, 2023) if (y != 2015) & (y not in cool_years_no2015)]

gp_warm = global_power.sel({year_dim: warm_years}).mean(dim=year_dim, skipna=True)
gs_warm = global_signif.sel({year_dim: warm_years}).mean(dim=year_dim, skipna=True)

gp_comp = global_power.sel({year_dim: cool_years_no2015}).mean(dim=year_dim, skipna=True)
gs_comp = global_signif.sel({year_dim: cool_years_no2015}).mean(dim=year_dim, skipna=True)

gp_2015 = global_power.sel({year_dim: 2015}).squeeze()
gs_2015 = global_signif.sel({year_dim: 2015}).squeeze()

# 转为 numpy
gp_warm_vals = gp_warm.squeeze().values
gs_warm_vals = gs_warm.squeeze().values

gp_comp_vals = gp_comp.squeeze().values
gs_comp_vals = gs_comp.squeeze().values

gp_2015_vals = gp_2015.squeeze().values
gs_2015_vals = gs_2015.squeeze().values

# =========================
# 2. 可选：统一归一化
# =========================
scale = np.nanmax([np.nanmax(gp_comp_vals), np.nanmax(gp_2015_vals)])
if scale == 0 or np.isnan(scale):
    scale = 1.0
scale = 1.0
gp_warm_plot = gp_warm_vals / scale
gs_warm_plot = gs_warm_vals / scale

gp_comp_plot = gp_comp_vals / scale
gs_comp_plot = gs_comp_vals / scale

gp_2015_plot = gp_2015_vals / scale
gs_2015_plot = gs_2015_vals / scale

# =========================
# 3. 显著性区域
# =========================
sig_warm = gp_warm_vals > gs_warm_vals
sig_comp = gp_comp_vals > gs_comp_vals
sig_2015 = gp_2015_vals > gs_2015_vals

ax.plot(
    x,
    gp_warm_plot,
    color="k",
    linewidth=1.8,
    label="Uncool-sum. Comp.",
    zorder=2
)

# 除2015外合成线
ax.plot(
    x,
    gp_comp_plot,
    color="deepskyblue",
    linewidth=1.8,
    label="Cool-sum. Comp.",
    zorder=3
)

# 2015功率谱线
ax.plot(
    x,
    gp_2015_plot,
    color="red",
    linewidth=1.8,
    label="2015",
    zorder=4
)

# 合成显著性区域
ax.fill_between(
    x,
    gs_warm_plot,
    gp_warm_plot,
    where=sig_warm,
    color="k",
    alpha=0.25,
    interpolate=True,
    zorder=1
)

ax.fill_between(
    x,
    gs_comp_plot,
    gp_comp_plot,
    where=sig_comp,
    color="deepskyblue",
    alpha=0.25,
    interpolate=True,
    zorder=1.5
)

# 2015显著性区域
ax.fill_between(
    x,
    gs_2015_plot,
    gp_2015_plot,
    where=sig_2015,
    color="red",
    alpha=0.25,
    interpolate=True,
    zorder=2.5
)

# 如果你想显示显著性检验线，可以取消下面注释
ax.plot(
    x,
    gs_warm_plot,
    color="k",
    linestyle="--",
    linewidth=1.0,
    alpha=0.7,
    label=""
)

ax.plot(
    x,
    gs_comp_plot,
    color="deepskyblue",
    linestyle="--",
    linewidth=1.0,
    alpha=0.7,
    label=""
)

ax.plot(
    x,
    gs_2015_plot,
    color="red",
    linestyle="--",
    linewidth=1.0,
    alpha=0.7,
    label=""
)

ax.set_xlabel("Period", fontsize=12)
ax.set_ylabel("", fontsize=12)

tick_periods = np.array([4, 8, 16, 32, 64])
ax.set_xticks(np.log2(tick_periods))
ax.set_xticklabels(tick_periods)

ax.set_xlim(np.log2(4), np.log2(32))
ax.set_ylim(0, 40)

ax.set_title("Global Wavelet Power", fontsize=14, loc="left")

ax.grid(axis="x", linestyle="--", alpha=0.3)

ax.legend(frameon=False, fontsize=9)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()

PYFILE = r"/volumes/TiPlus7100/PyFile"

plt.savefig(fr"{PYFILE}/p4/pic/功率谱_冷年合成_vs_2015_CN05.pdf", bbox_inches="tight")
plt.savefig(fr"{PYFILE}/p4/pic/功率谱_冷年合成_vs_2015_CN05.png", bbox_inches="tight", dpi=600)
