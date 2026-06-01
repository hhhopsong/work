from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import filters

from climkit.significance_test import corr_test, r_test
from climkit.TN_WaveActivityFlux import TN_WAF_3D, TN_WAF
from climkit.Cquiver import *
from climkit.data_read import *
from climkit.masked import *


PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
EHCI = EHCI.apply(lambda x: (x > 0.5).sum())

EHCI = EHCI['EHCI'].data


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

t2m78 = (t2m78 - t2m78.mean()) / t2m78.std('year')

years = np.arange(1961, 2023)
# 线性趋势（基于原始距平）
coeffs = np.polyfit(years, EHCI, 1)
trend_values = np.polyval(coeffs, years)
trend = xr.DataArray(trend_values, coords={'year': years}, dims=['year'])

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

# =====================
# 左轴：EHCI days
# =====================
ax.bar(years, EHCI, color='#AAAAAA', alpha=0.8, linewidth=1.8, label='EHTBW days')

ax.plot(
    years,
    trend,
    color='k',
    ls='--',
    linewidth=1,
    label='EHTBW days trend'
)

ax.set_xlim(1960, 2023)
ax.set_xticks(np.arange(1961, 2023, 10))
ax.set_title('YRB_T2m & EHTBW_Days', loc='left', fontsize=18)

ax.set_xlabel('', fontsize=16)
ax.set_ylabel('', fontsize=16)

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.tick_params(axis='both', which='major', labelsize=11, width=2, length=6)


# =====================
# 右轴：t2m
# =====================
ax2 = ax.twinx()

ax2.plot(
    years,
    t2m78['t2m'].values,
    color='red',
    linestyle='-',
    linewidth=1.2,
    marker='o',
    markersize=2.5,
    label='YRB T2m'
)

ax2.set_ylabel('', fontsize=16, color='red')

# 右轴刻度和刻度值设为红色
ax2.tick_params(
    axis='y',
    which='major',
    labelsize=11,
    width=2,
    length=6,
    colors='red'
)

ax2.spines['right'].set_color('red')
ax2.spines['right'].set_linewidth(2)


# =====================
# 让 t2m 的 y=0 对齐 EHCI 的平均值
# =====================
ehci_mean = np.mean(EHCI)

left_ymin, left_ymax = ax.get_ylim()
frac = (ehci_mean - left_ymin) / (left_ymax - left_ymin)

right_ymin, right_ymax = ax2.get_ylim()

# 保证右轴范围包含 t2m 数据，同时使 0 位于与 EHCI mean 相同的相对高度
span_lower = (0 - right_ymin) / frac
span_upper = (right_ymax - 0) / (1 - frac)
span = max(span_lower, span_upper) * 1.05

ax2.set_ylim(
    0 - frac * span,
    0 + (1 - frac) * span
)


# =====================
# 相关系数文字
# =====================
corr = np.corrcoef(t2m78['t2m'].values, EHCI)[0, 1]

ax.text(
    0.92,
    0.89,
    f'Corr = {corr:.2f}',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=10,
    zorder=10
)

# =====================
# Legend：合并左轴和右轴图例
# =====================
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc='upper left',
    fontsize=10,
    frameon=False
)

plt.savefig(fr'{PYFILE}/p5/pic/EHCI50日数.pdf', bbox_inches='tight')
plt.savefig(fr'{PYFILE}/p5/pic/EHCI50日数.png', bbox_inches='tight', dpi=600)
plt.show()

