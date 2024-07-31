from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from cnmaps import get_adm_maps, draw_maps
from matplotlib import ticker
import cmaps
from matplotlib.ticker import MultipleLocator
from toolbar.masked import masked   # 气象工具函数
import pandas as pd
import tqdm
import seaborn as sns


# 数据读取
EHDstations_zone = xr.open_dataset(r"cache\EHDstations_zone.nc")  # 读取缓存
# 绘图
sns.set(style='ticks')
palette = sns.xkcd_palette(["windows blue"])
sns.relplot(x="day", y="__xarray_dataarray_variable__", kind="line", data=EHDstations_zone.to_dataframe(), palette=palette)
sns.despine(top=False, right=False, left=False, bottom=False)   # 加上边框
# 设置横坐标的刻度范围和标记
ax = plt.gca()
# 设置横坐标的刻度范围和标记
x = np.arange(0, 121, 1)
ax.set_xlim(0, 121)
ax.set_xticks([0, 14, 30, 44, 61, 75, 92, 106, 121])
ax.set_xticklabels(["06/01", "06/15", "07/01", "07/15", "08/01", "08/15", "09/01", "09/15", "09/30"])
# 设置纵坐标的刻度范围和标记
ax.set_ylim(0, .5)
ax.set_yticks([0, .1, .2, .3, .4, .5])
ax.set_yticklabels([f"{i}" for i in range(0, 51, 10)])
plt.xlabel('Date')
plt.ylabel('Grids(%)')
# 保存为16:9
plt.gcf().set_size_inches(16, 9)
plt.savefig(r'C:\Users\10574\desktop\图3.png', dpi=1500, bbox_inches='tight')
plt.show()
