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

nino34 = xr.open_dataset(f'{DATA}/NOAA/ERSSTv5/nina34.anom.nc').sel(time=slice("1960-01-01", "2023-12-31"))['value']
nino34_12 = nino34.sel(time=nino34['time.month'].isin([12])).shift(time=1).sel(time=slice("1961-01-01", "2023-12-31"))  # 12月的nino34指数
nino34_1 = nino34.sel(time=nino34['time.month'].isin([1])).sel(time=slice("1961-01-01", "2023-12-31"))  # 1月的nino34指数
nino34_2 = nino34.sel(time=nino34['time.month'].isin([2])).sel(time=slice("1961-01-01", "2023-12-31"))  # 2月的nino34指数
nino34_1212 = (nino34_12.values + nino34_1.values + nino34_2.values) / 3
EL_LA = xr.Dataset(
    {'e': (['time'], [1 if i > 0.5 else (-1 if i < -0.5 else 0) for i in nino34_1212])},
    coords={'time': nino34_1['time'].data})  # 超过0.5度标记为厄尔尼诺或拉尼娜

t2m = xr.open_dataset(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc")['t2m']
t2m = masked(t2m.sel(date=slice('1961-01-01', '2024-12-31')), fr"{PYFILE}/map/self/长江_TP/长江_tp.shp")
t2m = xr.Dataset(
    {'t2m': (['time', 'lat', 'lon'], t2m.data)},
    coords={'time': pd.to_datetime(t2m['date'], format="%Y%m%d"),
            'lat': t2m['latitude'].data,
            'lon': t2m['longitude'].data})
t2m = t2m.sel(time=slice('1961-01-01', '2024-12-31'))
t2m = t2m.sel(time=t2m['time.month'].isin([6, 7, 8]))
t2m = t2m.mean(dim=['lat', 'lon'])
t2m678 = t2m.groupby('time.year').mean('time')
t2m_ano = t2m678 - t2m678.mean('year')  # 距平
# detrend
years = t2m_ano['year'].values
y = t2m_ano['t2m'].values
coeffs = np.polyfit(years, y, 1) # p=0.0015
trend_values = np.polyval(coeffs, years)
trend = xr.DataArray(trend_values, coords={'year': years}, dims=['year'])
t2m_ano = t2m_ano - trend     # 去线性趋势
# 标准差 sigma
sigma = t2m_ano.std('year')


# =========================
# 整理数据
# =========================
years = t2m_ano['year'].values
t2m_values = t2m_ano['t2m'].values
el_la_values = EL_LA['e'].values

# 保证年份一一对应
# 这里 EL_LA 的 time 对应 1961-2023，与 t2m_ano['year'] 应一致
# 若担心不一致，可加一个检查：
el_la_years = pd.to_datetime(EL_LA['time'].values).year
print(np.array_equal(years, el_la_years))  # True 为正常

# 给不同 ENSO 年份设置填充色
facecolors = []
for i in el_la_values:
    if i == 1:
        facecolors.append('red')   # 厄尔尼诺
    elif i == -1:
        facecolors.append('blue')    # 拉尼娜
    else:
        facecolors.append('none')   # 中性年：空心

# =========================
# 绘图
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 提取年份和异常值
years = t2m_ano['year'].values
vals = t2m_ano['t2m'].values

# 提取 EL/LA 年份标记
el_la_series = EL_LA.sel(time=EL_LA.time.dt.year.isin(years))
el_la_years = el_la_series['time.year'].values
el_la_vals = el_la_series['e'].values

# 做成“年份 -> 标记”字典，便于对应
el_la_dict = {int(y): int(v) for y, v in zip(el_la_years, el_la_vals)}

# 柱子颜色：正红负蓝
bar_colors = ['red' if v > 0 else 'blue' for v in vals]

fig, ax = plt.subplots(figsize=(8, 4))

# trend
#ax.plot(years, trend, color='#757575', linestyle='--', linewidth=2.0, label='Trend')

bars = ax.bar(years, vals, color=bar_colors, width=0.8, edgecolor='black', linewidth=0.5)

# 0线
ax.axhline(0, color='k', linewidth=1)

# std
ax.axhline(sigma['t2m'].data , color='purple', ls='--', linewidth=1)
ax.axhline(-sigma['t2m'].data, color='purple', ls='--', linewidth=1)

# 三角形标记位置偏移量
offset = 0.03 * (np.nanmax(vals) - np.nanmin(vals))
if offset == 0:
    offset = 0.05

# 给 EL/LA 年份加三角形
for year, val in zip(years, vals):
    tag = el_la_dict.get(int(year), 0)

    # 三角放在“柱子尽头”
    if val >= 0:
        y_mark = val + offset
        va = 'bottom'
    else:
        y_mark = val - offset
        va = 'top'

    if tag == 1:
        # EL_LA = 1 -> 小蓝色三角
        ax.scatter(year, y_mark, marker='.', s=40, color='blue', zorder=5)
    elif tag == -1:
        # EL_LA = -1 -> 小红色三角
        ax.scatter(year, y_mark, marker='.', s=40, color='red', zorder=5)

# 坐标轴设置
ax.set_xlim(years.min() - 1, years.max() + 1)
ax.set_xlabel('', fontsize=14)
ax.set_ylabel('', fontsize=14)
ax.set_title('JJA_T2M_ano', loc='left', fontsize=16)

# x轴刻度可按需要调稀疏一点
ax.set_xticks(years[::2])
ax.tick_params(axis='x', labelrotation=45, labelsize=10)
ax.tick_params(axis='y', labelsize=12, rotation=0)

# ===== legend 句柄（不会额外画到图中）=====
elnino_handle = ax.scatter([], [], marker='.', s=40, color='blue', label='El Niño')
lanina_handle = ax.scatter([], [], marker='.', s=40, color='red', label='La Niña')

# 图例放到图外右上角，避免遮挡柱线
ax.legend(handles=[elnino_handle, lanina_handle],
          loc='upper left',
          bbox_to_anchor=(0.80, 1.0),
          frameon=False,
          fontsize=12,
          borderaxespad=0.)

# 边框
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig(fr'{PYFILE}/p4/pic/index.pdf', bbox_inches='tight')
plt.savefig(fr'{PYFILE}/p4/pic/index.png', dpi=600, bbox_inches='tight')
