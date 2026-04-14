from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import filters

from climkit.significance_test import corr_test, r_test
from climkit.TN_WaveActivityFlux import TN_WAF_3D, TN_WAF
from climkit.Cquiver import *
from climkit.data_read import *


PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
EHCI = EHCI.apply(lambda x: (x > 0.6).sum())

EHCI = EHCI['EHCI'].data

years = np.arange(1961, 2023)
# 线性趋势（基于原始距平）
coeffs = np.polyfit(years, EHCI, 1)
trend_values = np.polyval(coeffs, years)
trend = xr.DataArray(trend_values, coords={'year': years}, dims=['year'])

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

ax.bar(years, EHCI, color='red', alpha=0.8, linewidth=1.8)

ax.set_xlim(1960, 2023)
ax.set_xticks(np.arange(1961, 2023, 10))
ax.set_title('EHTBW Days', loc='left',fontsize=18)

ax.plot(years, trend, color='k', ls='--', linewidth=0.8, label='Trend')

ax.set_xlabel('', fontsize=16)
ax.set_ylabel('Days', fontsize=16)

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.tick_params(axis='both', which='major', labelsize=11, width=2, length=6)

plt.savefig(fr'{PYFILE}/p5/pic/EHCI60日数.pdf', bbox_inches='tight')
plt.savefig(fr'{PYFILE}/p5/pic/EHCI60日数.png', bbox_inches='tight', dpi=600)
plt.show()
