import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# COOL_YEAR = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014, 2015] #1 std
COOL_YEAR = [1965, 1966, 1968, 1974, 1976, 1980, 1982, 1983, 1986, 1987, 1989, 1992, 1993, 1997, 1999, 2004, 2008, 2014, 2015] # 0.5 std
# COOL_YEAR = range(1961, 2023)

wavelet = xr.open_dataset(
    "/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_ano_all_JJA_wavelet.nc"
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

fig, ax = plt.subplots(figsize=(4, 0.6*len(COOL_YEAR)))

offset = 0.0
offset_step = 1.25

for year in cool_years_desc:
    gp = global_power.sel({year_dim: year}).squeeze().values
    gs = global_signif.sel({year_dim: year}).squeeze().values

    # 每一年单独归一化，便于瀑布图展示
    scale = np.nanmax(gp)
    if scale == 0 or np.isnan(scale):
        scale = 1.0

    gp_plot = gp / scale + offset
    gs_plot = gs / scale + offset

    sig_mask = gp > gs

    # 功率谱线
    ax.plot(
        x,
        gp_plot,
        color="k",
        linewidth=1.2,
        zorder=3
    )

    # 显著性检验线
    # ax.plot(
    #     x,
    #     gs_plot,
    #     color="r",
    #     linestyle="--",
    #     linewidth=0.9,
    #     zorder=4
    # )

    # 只在显著性线以上、功率谱线以下填红色
    ax.fill_between(
        x,
        gs_plot,
        gp_plot,
        where=sig_mask,
        color="red",
        alpha=0.35,
        interpolate=True,
        zorder=2
    )

    ax.text(
        np.log2(4)-0.05,
        offset,
        str(year),
        va="center",
        ha="right",
        fontsize=12
    )

    offset += offset_step

ax.set_xlabel("Period", fontsize=12)
ax.set_ylabel("", fontsize=12)

# x轴周期显示到64
tick_periods = np.array([4, 8, 16, 32, 64])
ax.set_xticks(np.log2(tick_periods))
ax.set_xticklabels(tick_periods)

ax.set_xlim(np.log2(4), np.log2(64))

ax.set_title("Global Wavelet Power", fontsize=14, loc="left")

ax.set_yticks([])
ax.grid(axis="x", linestyle="--", alpha=0.3)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
plt.savefig(fr"{PYFILE}/p4/pic/功率谱_冷年.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p4/pic/功率谱_冷年", bbox_inches='tight', dpi=600)
