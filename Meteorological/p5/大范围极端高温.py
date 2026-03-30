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
plt.savefig(f"{PYFILE}/p5/pic0/Basin_wide_EHT_t2m.pdf", bbox_inches='tight')
plt.show()
