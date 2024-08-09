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
fig = plt.figure()
fig.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距

# 自定义色标 (https://mp.weixin.qq.com/s/X3Yi1NGncoB39Lm0UMTLMQ)
custom_colors = ["#FFFFFF", "#FDDDB1", "#FDB57E", "#F26E4c", "#CA1E14", "#7F0000"]
custom_cmap = sns.color_palette(custom_colors)
ax1 = sns.heatmap(EHDstations_zone["__xarray_dataarray_variable__"].to_numpy().T, ax=fig.add_subplot(223), cmap=custom_cmap)  # 长江流域极端高温格点逐日占比热力图
ax1.invert_yaxis()  # 热力图y轴反向

##设置ax1坐标##
# 隐藏ax1的色标colormap
ax1.collections[0].colorbar.remove()
ax = plt.gca()
# 设置横坐标的刻度范围和标记
x = np.arange(1979, 2023, 1)
ax.set_xlim(0, 44)
ax.set_xticks([1.5, 6.5, 11.5, 16.5, 21.5, 26.5, 31.5, 36.5, 41.5])
ax.set_xticklabels(["1980", "1985", "1990", "1995", "2000", "2005", "2010", "2015", "2020"])
# 设置纵坐标的刻度范围和标记
y = np.arange(0, 121, 1)
ax.set_ylim(0, 122)
ax.set_yticks([0.5, 14.5, 30.5, 44.5, 61.5, 75.5, 92.5, 106.5, 121.5])
ax.set_yticklabels(["06/01", "06/15", "07/01", "07/15", "08/01", "08/15", "09/01", "09/15", "09/30"])
ax.spines['top'].set_visible(True)  # 显示上边框
ax.spines['right'].set_visible(True)  # 显示右边框
ax.spines['bottom'].set_visible(True)  # 显示下边框
ax.spines['left'].set_visible(True)  # 显示左边框
plt.xlabel('Year')
plt.ylabel('Date')
##设置ax1坐标结束##

ax2 = sns.barplot(data=EHDstations_zone.to_dataframe()*100, x="year", y="__xarray_dataarray_variable__", ax=fig.add_subplot(221), errorbar=('ci', 0))  # 长江流域极端高温格点逐日占比
##设置ax2坐标##
ax2.xaxis.set_visible(False)  # ax2隐藏x轴标签
ax = plt.gca()
ax.set_ylim(0, 0.6*100)
ax.tick_params(axis='y', direction='in')  # 设置y轴刻度方向
ax.spines['top'].set_visible(False)  # 隐藏上边框
ax.spines['right'].set_visible(False)  # 隐藏右边框
ax.spines['bottom'].set_visible(True)  # 显示下边框
ax.spines['left'].set_visible(True)  # 显示左边框
plt.ylabel('Grids(%)')
##设置ax2坐标结束##

ax3 = sns.barplot(data=EHDstations_zone.to_dataframe()*100, x='__xarray_dataarray_variable__', y='day', orient='h', ax=fig.add_subplot(224), errorbar=('ci', 0), width=0, color='None',edgecolor='r')  # 长江流域极端高温格点逐年占比
##设置ax3坐标##
ax3.yaxis.set_visible(False)  # ax3隐藏y轴标签
ax = plt.gca()
ax.set_xlim(0, 0.6*100)
ax.tick_params(axis='x', direction='in')  # 设置x轴刻度方向
ax.spines['top'].set_visible(False)  # 隐藏上边框
ax.spines['right'].set_visible(False)  # 隐藏右边框
ax.spines['bottom'].set_visible(True)  # 显示下边框
ax.spines['left'].set_visible(True)  # 显示左边框
plt.xlabel('Grids(%)')
##设置ax3坐标结束##



# 保存为16:9
plt.gcf().set_size_inches(16, 9)
plt.savefig(r'C:\Users\10574\desktop\图3.png', dpi=1500, bbox_inches='tight')
plt.show()
