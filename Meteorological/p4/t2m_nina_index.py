from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.lines import lineStyles
from matplotlib.pyplot import quiverkey
from matplotlib.ticker import MultipleLocator, FixedLocator
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmaps

from scipy.ndimage import filters
import xarray as xr
import numpy as np
import multiprocessing
import sys
import tqdm as tq
import time
import pandas as pd

from climkit.significance_test import corr_test, r_test
from climkit.TN_WaveActivityFlux import TN_WAF_3D, TN_WAF
from climkit.Cquiver import *
from climkit.data_read import *
from climkit.masked import masked
from climkit.corr_reg import corr, regress
from climkit.lonlat_transform import transform

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

# =========================
# ENSO 指数
# =========================
nino34 = xr.open_dataset(f'{DATA}/NOAA/ERSSTv5/nina34.anom.nc').sel(time=slice("1960-01-01", "2023-12-31"))['value']
nino34_12 = nino34.sel(time=nino34['time.month'].isin([12])).shift(time=1).sel(time=slice("1961-01-01", "2023-12-31"))
nino34_1 = nino34.sel(time=nino34['time.month'].isin([1])).sel(time=slice("1961-01-01", "2023-12-31"))
nino34_2 = nino34.sel(time=nino34['time.month'].isin([2])).sel(time=slice("1961-01-01", "2023-12-31"))
nino34_1212 = (nino34_12.values + nino34_1.values + nino34_2.values) / 3

EL_LA = xr.Dataset(
    {'e': (['time'], [1 if i > 0.5 else (-1 if i < -0.5 else 0) for i in nino34_1212])},
    coords={'time': nino34_1['time'].data}
)

# =========================
# 读取并处理 t2m
# =========================
t2m = xr.open_dataset(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc")['t2m']
t2m = masked(t2m.sel(date=slice('1961-01-01', '2024-12-31')), fr"{PYFILE}/map/self/长江_TP/长江_tp.shp")
t2m = xr.Dataset(
    {'t2m': (['time', 'lat', 'lon'], t2m.data)},
    coords={'time': pd.to_datetime(t2m['date'], format="%Y%m%d"),
            'lat': t2m['latitude'].data,
            'lon': t2m['longitude'].data}
)

t2m = t2m.sel(time=slice('1961-01-01', '2024-12-31'))
t2m = t2m.sel(time=t2m['time.month'].isin([6, 7, 8]))
t2m = t2m.mean(dim=['lat', 'lon'])
t2m678 = t2m.groupby('time.year').mean('time')

# 未去趋势原始距平
t2m_ano_raw = t2m678 - t2m678.mean('year')

# 线性趋势（基于原始距平）
years = t2m_ano_raw['year'].values
y_raw = t2m_ano_raw['t2m'].values
coeffs = np.polyfit(years, y_raw, 1)
trend_values = np.polyval(coeffs, years)
trend = xr.DataArray(trend_values, coords={'year': years}, dims=['year'])

# 去趋势后的距平
t2m_ano = t2m_ano_raw - trend

# 标准差（基于去趋势后的序列）
sigma = t2m_ano.std('year')

# =========================
# 整理数据
# =========================
years = t2m_ano['year'].values
vals = t2m_ano['t2m'].values              # 去趋势后的距平
vals_raw = t2m_ano_raw['t2m'].values      # 未去趋势原始距平
trend_vals = trend.values                 # 趋势线

el_la_values = EL_LA['e'].values
el_la_years = pd.to_datetime(EL_LA['time'].values).year
print(np.array_equal(years, el_la_years))  # True 为正常

# 提取 EL/LA 年份标记
el_la_series = EL_LA.sel(time=EL_LA.time.dt.year.isin(years))
el_la_years = el_la_series['time.year'].values
el_la_vals = el_la_series['e'].values
el_la_dict = {int(y): int(v) for y, v in zip(el_la_years, el_la_vals)}

# 柱子颜色：正红负蓝
bar_colors = ['red' if v > 0 else 'blue' for v in vals]

# =========================
# 绘图
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 4))

# 1. 柱状图：去趋势后的距平
bars = ax.bar(
    years, vals,
    color=bar_colors, width=0.8, alpha=0.7,
    edgecolor='none', linewidth=0.5,
    zorder=2
)

# 2. 未去趋势原始距平：紫色线 + 棕色点
raw_line, = ax.plot(
    years, vals_raw,
    color='saddlebrown', #e91e63
    linewidth=1.8,
    marker='o',
    markersize=4,
    markerfacecolor='none',
    markeredgecolor='none',
    markeredgewidth=1.0,
    label='Original',
    zorder=10
)

# 3. 趋势线：黑色虚线
trend_line, = ax.plot(
    years, trend_vals,
    color='black',
    linestyle='--',
    linewidth=0.8,
    label='Trend',
    zorder=15
)

# 0线
ax.axhline(0, color='k', linewidth=1)

# ±1 sigma
# ax.axhline(sigma['t2m'].data, color='purple', ls='--', linewidth=1)
ax.axhline(-sigma['t2m'].data, color='green', ls='--', linewidth=2)

# ENSO 标记偏移量
offset = 0.03 * (np.nanmax(vals) - np.nanmin(vals))
if offset == 0:
    offset = 0.05

# 给 EL/LA 年份加标记
for year, val in zip(years, vals):
    tag = el_la_dict.get(int(year), 0)

    if val >= 0:
        y_mark = val + offset
    else:
        y_mark = val - offset

    if tag == 1:
        ax.scatter(year, y_mark, marker='.', s=40, color='red', zorder=20)
    elif tag == -1:
        ax.scatter(year, y_mark, marker='.', s=40, color='blue', zorder=20)

# 坐标轴设置
ax.set_xlim(years.min() - 1, years.max() + 1)
ax.set_xlabel('', fontsize=14)
ax.set_ylabel('', fontsize=14)
ax.set_title('JJA_T2M_ano', loc='left', fontsize=16)

ax.set_xticks(years[::2])
ax.tick_params(axis='x', labelrotation=45, labelsize=10)
ax.tick_params(axis='y', labelsize=12, rotation=0)

# 图例
elnino_handle = ax.scatter([], [], marker='.', s=40, color='red', label='El Niño')
lanina_handle = ax.scatter([], [], marker='.', s=40, color='blue', label='La Niña')

from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
class HandlerBiColorPatch(HandlerBase):
    """双色填色图例：左红右蓝"""

    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        w2 = width / 2.0

        patch1 = Rectangle(
            (x0, y0), w2, height,
            facecolor="red", edgecolor="none", alpha=0.7,
            transform=trans
        )
        patch2 = Rectangle(
            (x0 + w2, y0), w2, height,
            facecolor="blue", edgecolor="none", alpha=0.7,
            transform=trans
        )
        return [patch1, patch2]

detrend_line = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none")
detrend_line.set_label("Detrend")
ax.legend(
    handles=[raw_line, trend_line, detrend_line, elnino_handle, lanina_handle],
    handler_map={detrend_line: HandlerBiColorPatch()},
    loc='upper center',
    ncol=2,
    frameon=False,
    fontsize=12,
    borderaxespad=0.
)

# 边框
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig(fr'{PYFILE}/p4/pic/index.pdf', bbox_inches='tight')
plt.savefig(fr'{PYFILE}/p4/pic/index.png', dpi=600, bbox_inches='tight')
plt.show()
