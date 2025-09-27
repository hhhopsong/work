import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.path import Path

from scipy.stats import linregress

from toolbar.masked import masked
from toolbar.significance_test import r_test

import tqdm as tq

corr = np.load(r"D:\PyFile\p2\data\corr.npy")

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors, ticker
import matplotlib.pyplot as plt
import seaborn as sns
import cmaps

# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.set_title("BWHT Correlation across Thresholds", fontsize=14, loc='left')
levels = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
norm = BoundaryNorm(levels, cmaps.sunshine_9lev.N)
contf = ax.contourf(np.arange(0.1, 0.901, 0.01), np.arange(30, 38.01, 0.1), corr[:, :], cmap=cmaps.sunshine_9lev, levels=[0, .2, .4, .6, .7, .8, .9, .95, 1], extend='both', norm=norm)
cont = ax.contour(np.arange(0.1, 0.901, 0.01), np.arange(30, 38.01, 0.1), corr[:, :], levels=[.4, .6, .7, .8, .9, .95, 1.], colors='k', linewidths=.5)

# plt.rcParams['hatch.color'] = '#454545'
# plt.rcParams['hatch.linewidth'] = 0.5
# xianzhu_contf = ax.contourf(np.arange(0.1, 0.901, 0.01), np.arange(30, 38.01, 0.1), corr[:, :], [0, .2500349005300471, .3248184473571816, 1],
#                             hatches=[None, r'++', r'//'], colors="none", zorder=3, corner_mask=False)

plt.rcParams['hatch.color'] = '#c3c3c3'
plt.rcParams['hatch.linewidth'] = 2.2
xianzhu_contf = ax.contourf(np.arange(0.1, 0.901, 0.01), np.arange(30, 38.01, 0.1), corr[:, :], [0, .2500349005300471, .3248184473571816, 1],
                            hatches=[None, r'..', r'/'], colors="none", zorder=3, corner_mask=False)
corr_edge = np.where(np.isnan(corr), 0, corr)
# edge_cont = ax.contour(np.arange(0.1, 0.901, 0.01), np.arange(30, 38.01, 0.1), corr_edge[:, :], levels=[0], colors='red', linewidths=2)
edge_contf = ax.contourf(np.arange(0.1, 0.901, 0.01), np.arange(30, 38.01, 0.1), corr_edge[:, :], [-99, 0], colors="#454545", alpha=0.3)
# for collection in edge_contf.collections:
#     collection.set_edgecolor('#454545')#-----打点颜色设置

# 统一加粗所有四个边框
for spine in ax.spines.values():
    spine.set_linewidth(1.5)  # 设置边框线宽

# 设置x=0.3的线
ax.axvline(0.3, color='g', linestyle='-', linewidth=1)
# 设置y=31.76的线
ax.axhline(31.76, color='b', linestyle='-', linewidth=1)
# 设置(0.3, 31.76)的点
ax.scatter(0.3, 31.76, color='k', s=15, zorder=2)
# 设置x轴
ax.set_xlabel("EHCI", fontsize=14)
ax.set_xlim(0.1, 0.9)
ax.set_xticks(np.arange(0.1, 0.901, 0.1))
ax.set_xticklabels([f"{i*100:.0f}%" for i in np.arange(0.1, 0.901, 0.1)], fontsize=10)
# 设置x轴主次刻度
ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(0.1, 0.901, 0.1)))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
# 设置y轴主次刻度
ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(30, 38.01, 1)))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
# 设置y轴
ax.set_ylabel("Temperature", fontsize=14, rotation=90)
ax.set_ylim(30, 38)
ax.set_yticks(np.arange(30, 38.01, 1))
ax.set_yticklabels([f"{i:.0f}°C" for i in np.arange(30, 38.01, 1)], fontsize=10)
ax.grid(False)  # 不显示网格
# 最大刻度、最小刻度的刻度线长短，粗细设置
ax.tick_params(which='major', length=4, width=1, color='black')  # 最大刻度长度，宽度设置，
ax.tick_params(which='minor', length=2, width=.5, color='black')  # 最小刻度长度，宽度设置
ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
# 边框显示为黑色
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
# 显示颜色条
ax_colorbar = inset_axes(ax, width="3%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
cb1 = plt.colorbar(contf, cax=ax_colorbar, orientation='vertical', drawedges=True)
cb1.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
cb1.dividers.set_color('black') # 将colorbar内间隔线调为黑色
cb1.ax.tick_params(length=0, labelsize=10)  # length为刻度线的长度
cb1.locator = ticker.FixedLocator(np.array([0, .2, .4, .6, .7, .8, .9, .95, 1]))
cb1.set_ticklabels([" 0   ", " 0.2", " 0.4", " 0.6", " 0.7", " 0.8 ", " 0.9", " 0.95", " 1 "])
plt.savefig(r"D:\PyFile\p2\pic\corr.png", dpi=600, bbox_inches='tight')
plt.show()