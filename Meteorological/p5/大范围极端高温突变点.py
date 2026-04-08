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
from matplotlib.ticker import MultipleLocator
from scipy import ndimage

from climkit.Cquiver import *
from climkit.corr_reg import regress
from climkit.masked import masked
from climkit.significance_test import r_test, corr_test
from climkit.lonlat_transform import *
from climkit.data_read import *
from climkit.corr_reg import *

from matplotlib import ticker
from metpy.calc import vertical_velocity
from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"


t2m = xr.open_dataset(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc")['t2m'] - 273.15
t2m = masked(t2m.sel(date=slice('1961-01-01', '2024-12-31')), fr"{PYFILE}/map/self/长江_TP/长江_tp.shp")
t2m = xr.Dataset(
    {'t2m': (['time', 'lat', 'lon'], t2m.data)},
    coords={'time': pd.to_datetime(t2m['date'], format="%Y%m%d"),
            'lat': t2m['latitude'].data,
            'lon': t2m['longitude'].data})
t2m = t2m.sel(time=slice('1961-01-01', '2022-12-31'))
t2m = t2m.sel(time=t2m['time.month'].isin([7, 8]))
t2m = t2m.mean(dim=['lat', 'lon'])
t2m78 = t2m.groupby('time.year').mean('time')

EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
# 找出EHCI>30%的每年日数
EHCI30 = EHCI.apply(lambda x: (x > 0.3).sum())
EHCI_mean = EHCI.mean('time')

corr = np.corrcoef([EHCI30['EHCI'].data, EHCI_mean['EHCI'].data, t2m78['t2m'].data])
print(corr)

EHCI30 = (EHCI30 - EHCI30.mean()) / EHCI30.std('year')
EHCI_mean = (EHCI_mean - EHCI_mean.mean()) / EHCI_mean.std('year')
t2m78 = (t2m78 - t2m78.mean()) / t2m78.std('year')


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(1, 1, 1)

ax.set_title('Index', fontsize=14)
ax.plot(EHCI30['EHCI'].data, label='Basin-wide EHT')
ax.plot(EHCI_mean['EHCI'].data, label='Year-by-year EHCI')
ax.plot(t2m78['t2m'].data, label='T2m')

ax.legend()

ax.set_xticks([0, 10 ,20, 30, 40, 50, 60])
ax.set_xticklabels(['1961', '1971', '1981', '1991', '2001', '2011', '2021'], fontsize=12)

# 设置y轴范围
ymax = 3
ax.set_ylim(-3, 4)
ax.set_xlim(-.5, 61.5)

# 仅当 KType == 1 时添加y刻度标签
ax.set_yticks(np.arange(-3, 5, 1))
ax.set_yticklabels(np.arange(-3, 5, 1), fontsize=12)

# 添加零线
ax.axhline(0, color='black', lw=1)

for ax in fig.axes:
    # 遍历每个子图中的所有艺术家对象 (artist)
    for artist in ax.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽

plt.savefig(f"{PYFILE}/p5/pic/Basin_wide_EHT_t2m.png", dpi=1000, bbox_inches='tight')
plt.savefig(f"{PYFILE}/p5/pic/Basin_wide_EHT_t2m.pdf", bbox_inches='tight')
plt.close()

corr_EHCI_i = []
day_EHCI_i = []
i_index = [0.3, 0.7, 100]
for i in np.linspace(*i_index):
    EHCI_i = EHCI.apply(lambda x: (x > i).sum())
    EHCI_i = (EHCI_i - EHCI_i.mean()) / EHCI_i.std('year')
    corr_EHCI_i.append(np.corrcoef([EHCI_i['EHCI'].data, EHCI_mean['EHCI'].data, t2m78['t2m'].data])[0, 2])
    day_EHCI_i.append(EHCI.apply(lambda x: (x > i).sum()).sum()['EHCI'].data/62/62)

corr_EHCI_i = xr.Dataset({
    'corr': (['thres'], corr_EHCI_i)}, coords={'thres': np.linspace(*i_index)})
day_EHCI_i = xr.Dataset({
    'days': (['thres'], day_EHCI_i)}, coords={'thres': np.linspace(*i_index)})


import ruptures as rpt
from scipy.stats import ttest_ind

# =========================
# 1. multiple mean change-point detection
# =========================
def detect_mean_cp(x, y, pen=0.01, min_size=5):
    x = np.asarray(x)
    y = np.asarray(y)

    signal = y.reshape(-1, 1)
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
    bkps = algo.predict(pen=pen)   # last one is always len(y)

    cp_idx = np.array(bkps[:-1], dtype=int)
    cp_x = x[cp_idx] if len(cp_idx) > 0 else np.array([])

    return cp_idx, cp_x, bkps


# =========================
# 2. significance test for each change-point
# =========================
def test_cp_significance(x, y, bkps, min_size=5):
    x = np.asarray(x)
    y = np.asarray(y)

    results = []
    start = 0

    for i in range(len(bkps) - 1):
        cp = bkps[i]
        end = bkps[i + 1]

        left = y[start:cp]
        right = y[cp:end]

        if len(left) < min_size or len(right) < min_size:
            stat, pval = np.nan, np.nan
        else:
            stat, pval = ttest_ind(left, right, equal_var=False, nan_policy='omit')

        results.append({
            'cp_idx': cp,
            'cp_x': x[cp],
            'left_n': len(left),
            'right_n': len(right),
            'left_mean': np.nanmean(left),
            'right_mean': np.nanmean(right),
            'mean_diff': np.nanmean(right) - np.nanmean(left),
            't_stat': stat,
            'p_value': pval
        })

        start = cp

    return results


# =========================
# 3. plotting function
# =========================
def plot_mean_cp(ax, x, y, bkps, cp_idx, sig_results=None,
                 ylabel='Value', title='Mean change points'):
    x = np.asarray(x)
    y = np.asarray(y)

    ax.plot(x, y, label='Corr.', lw=1.5)

    start = 0
    for end in bkps:
        seg_mean = y[start:end].mean()
        ax.hlines(seg_mean, x[start], x[end-1], colors='#454545', ls='--', lw=2)
        start = end

    for i, idx in enumerate(cp_idx):
        ax.axvspan(x[idx], 1, color="#959595", alpha=0.3, zorder=0)

    if sig_results is not None and len(sig_results) > 0:
        y_top = np.nanmax(y)
        y_bot = np.nanmin(y)
        y_span = y_top - y_bot if y_top > y_bot else 1.0

    ax.set_xlabel('EHCI Threshold')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


# =========================
# 4. prepare data
# =========================
x_corr = corr_EHCI_i['thres'].values
y_corr = corr_EHCI_i['corr'].values

# =========================
# 5. detect change-points
# =========================
cp_idx_corr, cp_x_corr, bkps_corr = detect_mean_cp(
    x_corr, y_corr, pen=0.1, min_size=5
)

print('Corr change-point indices:', cp_idx_corr)
print('Corr change-point thresholds:', cp_x_corr)

# =========================
# 6. print significance
# =========================
sig_results_corr = test_cp_significance(x_corr, y_corr, bkps_corr, min_size=5)

print('\nSignificance of mean change-points:')
for r in sig_results_corr:
    print(
        f"cp_idx={r['cp_idx']}, "
        f"thres={r['cp_x']:.4f}, "
        f"left_n={r['left_n']}, "
        f"right_n={r['right_n']}, "
        f"left_mean={r['left_mean']:.4f}, "
        f"right_mean={r['right_mean']:.4f}, "
        f"diff={r['mean_diff']:.4f}, "
        f"t={r['t_stat']:.4f}, "
        f"p={r['p_value']:.4g}"
    )

# =========================
# 7. plot
# =========================
fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharex=True)

plot_mean_cp(
    ax,
    x_corr, y_corr,
    bkps_corr, cp_idx_corr,
    sig_results=sig_results_corr,
    ylabel='Correlation',
    title='Possible Mean Change-Points of Corr.'
)

plt.tight_layout()


# axes.plot(corr_EHCI_i['thres'], corr_EHCI_i['corr'], label='Corr.')
# axes.plot(day_EHCI_i['thres'], day_EHCI_i['days'], label='Day')

plt.xlabel('EHCI Threshold', fontsize=12)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.title('EHCI Corr.', fontsize=14)

plt.legend()
plt.xlim(0.3, .7)
plt.ylim(0.4, 1)

plt.savefig(f"{PYFILE}/p5/pic/EHCI极端突变点.png", dpi=1000, bbox_inches='tight')
plt.savefig(f"{PYFILE}/p5/pic/EHCI极端突变点.pdf", bbox_inches='tight')

