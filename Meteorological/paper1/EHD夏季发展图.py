from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from matplotlib import gridspec
import matplotlib.colors as colors
from cnmaps import get_adm_maps, draw_maps
import cmaps
from toolbar.masked import masked   # 气象工具函数
from toolbar.significance_test import *
import pandas as pd
import tqdm
import seaborn as sns
from scipy import stats


# 数据读取
time = [1961, 2022]
info = input(r"请输入阈值(\35\37\40):")
EHDstations_zone = xr.open_dataset(fr"D:\PyFile\paper1\EHD{info}stations_happended_zone.nc").sel(year=slice(f"{time[0]}", f"{time[1]}"))  # 读取缓存
# 绘图
sns.set(style='ticks')
fig = plt.figure()
fig.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[2, 1], height_ratios=[1, 2])  # 设置子图比例

# 自定义色标 (https://mp.weixin.qq.com/s/X3Yi1NGncoB39Lm0UMTLMQ)
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
custom_colors = ["#FFFFFF", "#FDDDB1", "#FDB57E", "#F26E4c", "#CA1E14", "#7F0000"]
custom_cmap = colors.ListedColormap(custom_colors)
norm = colors.BoundaryNorm(bins, custom_cmap.N)
ax1 = sns.heatmap(EHDstations_zone["__xarray_dataarray_variable__"].to_numpy().T, cmap=custom_cmap, norm=norm, ax=fig.add_subplot(spec[1, 0]), cbar=False)  # 长江流域极端高温格点逐日占比热力图
ax1.invert_yaxis()  # 热力图y轴反向

##设置ax1坐标##
# ax1.collections[0].colorbar.remove()  # 隐藏ax1的色标colormap
ax = plt.gca()
# 设置横坐标的刻度范围和标记
x = np.arange(time[0], time[1] + 1, 1)
ax.set_xlim(0, time[1] - time[0] + 1)
ax.set_xticks([0.5, 4.5, 9.5, 14.5, 19.5, 24.5, 29.5, 34.5, 39.5, 44.5, 49.5, 54.5, 59.5])
ax.set_xticklabels(["1961", "1965", "1970", "1975", "1980", "1985", "1990", "1995", "2000", "2005", "2010", "2015", "2020"], rotation=0)

# 设置纵坐标的刻度范围和标记
y = np.arange(0, 91, 1)
ax.set_ylim(0, 92)
ax.set_yticks([0.5, 14.5, 30.5, 44.5, 61.5, 75.5, 91.5])
ax.set_yticklabels(["06/01", "06/15", "07/01", "07/15", "08/01", "08/15", "08/30"])
ax.spines['top'].set_visible(True)  # 显示上边框
ax.spines['right'].set_visible(True)  # 显示右边框
ax.spines['bottom'].set_visible(True)  # 显示下边框
ax.spines['left'].set_visible(True)  # 显示左边框
plt.xlabel('Year')
plt.ylabel('Date')
# 设置色标
ax_cbar = fig.add_axes([0.66, 0.68, 0.20, 0.015])
cbar = plt.colorbar(ax1.collections[0], cax=ax_cbar, orientation='horizontal', drawedges=True)
cbar.set_ticklabels(["0", "5", "10", "20", "30", "40", "50"])
cbar.ax.set_title('Proportion of EHT-Grids(%)', fontsize=10)
cbar.ax.tick_params(length=0)  # 设置色标刻度长度
cbar.dividers.set_linewidth(1.0)  # 设置分割线宽度
cbar.outline.set_linewidth(1.0)  # 设置色标轮廓宽度
##设置ax1坐标结束##

ax2 = sns.barplot(data=EHDstations_zone.mean('day').to_dataframe()*100, x="year", y="__xarray_dataarray_variable__", ax=fig.add_subplot(spec[0, 0]), errorbar=('ci', 0))  # 长江流域极端高温格点逐日占比
##设置ax2坐标##
ax2.xaxis.set_visible(False)  # ax2隐藏x轴标签
ax = plt.gca()
ax.set_xlim(-.5, time[1] - time[0] + 1 - .5)
ax.set_ylim(0, 0.35*100)
ax.set_yticks([5, 10, 15, 20, 25, 30, 35])
ax.set_yticklabels(["5%", "10%", "15%", "20%", "25%", "30%", "35%"])
ax.tick_params(axis='y', direction='in')  # 设置y轴刻度方向
ax.spines['top'].set_visible(False)  # 隐藏上边框
ax.spines['right'].set_visible(False)  # 隐藏右边框
ax.spines['bottom'].set_visible(True)  # 显示下边框
ax.spines['left'].set_visible(True)  # 显示左边框
plt.ylabel('Annual mean')
##设置ax2坐标结束##

ax2_reg = ax2.twinx()
ax2_reg = sns.regplot(data=EHDstations_zone.mean('day')*100, x=[i for i in range(time[1] - time[0] + 1)], y="__xarray_dataarray_variable__", ax=ax2_reg, scatter=False, color='#74C476')  # 长江流域极端高温格点逐年占比
# 计算原数据是否符合正态分布,回归的斜率、截距、相关系数、p 值等
p = normal_test(EHDstations_zone.mean('day').to_array().data[0])[1]
slope, intercept, r_value, p_value, std_err = stats.linregress([i for i in range(time[1] - time[0] + 1)], EHDstations_zone.mean('day')["__xarray_dataarray_variable__"]*100)
##设置ax2_reg坐标##
ax2_reg.yaxis.set_visible(False)  # ax2隐藏y轴标签
ax = plt.gca()
ax.set_ylim(0, 0.35*100)
ax.spines['top'].set_visible(False)  # 隐藏上边框
ax.spines['right'].set_visible(False)  # 隐藏右边框
ax.spines['bottom'].set_visible(False)  # 显示下边框
ax.spines['left'].set_visible(False)  # 显示左边框

##设置ax2_reg坐标结束##

ax3 = sns.barplot(data=EHDstations_zone.to_dataframe()*100, x='__xarray_dataarray_variable__', y='day', orient='h', ax=fig.add_subplot(spec[1, 1]), errorbar=('ci', 0), width=0, color='None', edgecolor='#756BB1')  # 长江流域极端高温格点逐日占比
ax3_6_line = plt.axvline(x=EHDstations_zone.mean('year').to_dataframe()[:30].mean().to_numpy()*100, ymax=30/92, color='#74C476', linestyle='-')  # 添加50%置信度分割线
ax3_78_line = plt.axvline(x=EHDstations_zone.mean('year').to_dataframe()[30:].mean().to_numpy()*100, ymin=30/92, color="#CA1E14", linestyle='-')
##设置ax3坐标##
ax3.yaxis.set_visible(False)  # ax3隐藏y轴标签
ax = plt.gca()
ax.set_xlim(0, 0.25*100)
ax.set_xticks([5, 10, 15, 20, 25])
ax.set_xticklabels(["5%", "10%", "15%", "20%", "25%"])
ax.tick_params(axis='x', direction='in')  # 设置x轴刻度方向
ax.spines['top'].set_visible(False)  # 隐藏上边框
ax.spines['right'].set_visible(False)  # 隐藏右边框
ax.spines['bottom'].set_visible(True)  # 显示下边框
ax.spines['left'].set_visible(True)  # 显示左边框
plt.xlabel('Daily average')
plt.gca().invert_yaxis()
##设置ax3坐标结束##



# 保存为1:1
plt.gcf().set_size_inches(10, 10)
plt.savefig(fr'D:\PyFile\pic\图3_{info}_happended.png', dpi=1000, bbox_inches='tight')
plt.show()
