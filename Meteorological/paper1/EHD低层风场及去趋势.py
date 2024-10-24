from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import pymannkendall as mk
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from cartopy.util import add_cyclic_point
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, FixedLocator
from matplotlib import gridspec
import matplotlib.colors as colors
from cnmaps import get_adm_maps, draw_maps
from eofs.standard import Eof
import cmaps
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
from toolbar.masked import masked  # 气象工具函数
from toolbar.significance_test import corr_test
from toolbar.sub_adjust import adjust_sub_axes
from toolbar.pre_whitening import ws2001
from toolbar.curved_quivers_master.modplot import velovect, velovect_key
import pandas as pd
import tqdm as tq
import seaborn as sns
import multiprocessing


# 数据读取
# 读取CN05.1逐日最高气温数据
time = [1961, 2022]
alpha = 0.05 # 显著性水平

EHDstations_zone = xr.open_dataset(fr"D:\PyFile\paper1\EHD35stations_happended_zone.nc").sel(year=slice(f"{time[0]}", f"{time[1]}"))  # 读取缓存
ols = np.load(r"D:\PyFile\paper1\OLS35_detrended.npy")  # 读取缓存
EHD = xr.open_dataset(r"D:\PyFile\paper1\EHD35.nc").sel(time=slice('1961-01-01', '2022-12-31'))  # 读取缓存
EHD = EHD.sel(time=EHD['time.month'].isin([7, 8])).groupby('time.year').sum('time').mean('year')
# 将零值替换为缺测值
EHD = EHD.where(EHD > 0)
EHD = masked(EHD, r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")

# uvz
u_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\u_same.nc")['u'].sel(p=850).transpose('lat', 'lon', 'year')
u_corr = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_u850_same.npy")
v_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\v_same.nc")['v'].sel(p=850).transpose('lat', 'lon', 'year')
v_corr = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_v850_same.npy")
z_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\z_same.nc")['z'].sel(p=850).transpose('lat', 'lon', 'year')
z_corr = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_z850_same.npy")
u显著性检验结果 = corr_test(ols, u_corr, alpha=alpha)
v显著性检验结果 = corr_test(ols, v_corr, alpha=alpha)
z显著性检验结果 = corr_test(ols, z_corr, alpha=alpha)

print('数据加载完成')

uv显著性检验结果 = np.where(np.where(u显著性检验结果 == 1, 1, 0) + np.where(v显著性检验结果 == 1, 1, 0) >= 1, 1, np.nan)
u_np = np.where(uv显著性检验结果 != 1, u_corr, np.nan)
v_np = np.where(uv显著性检验结果 != 1, v_corr, np.nan)
u_np = np.where(u_np ** 2 + v_np ** 2 >= 0.15 ** 2, u_np, np.nan)
v_np = np.where(u_np ** 2 + v_np ** 2 >= 0.15 ** 2, v_np, np.nan)
u_corr = np.where(uv显著性检验结果 == 1, u_corr, np.nan)
v_corr = np.where(uv显著性检验结果 == 1, v_corr, np.nan)
z_corr, z_lon = add_cyclic_point(z_corr, coord=z_diff['lon'])

print('显著性检验完成')

# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))  # 创建画布
fig.subplots_adjust(wspace=0, hspace=0)
spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], height_ratios=[1])  # 设置子图比例

extent_changjiang = [88, 124, 22, 38]  # 中国大陆经度范围，纬度范围
extent_CN = [85, 145, 12, 48]  # 中国大陆经度范围，纬度范围
proj = ccrs.PlateCarree()  # 投影方式
ax1 = fig.add_subplot(spec[0, 0], projection=proj)  # 添加子图
# 设置ax1 figsize=(9, 4)
ax1.set_extent(extent_CN, crs=proj)  # 设置地图范围
level = [0, 5, 10, 15, 20]  # 等值线间隔
custom_colors = ["#FDB57E", "#F26E4c", "#CA1E14", "#7F0000"]
custom_cmap = colors.ListedColormap(custom_colors)
a1 = ax1.contourf(EHD['lon'], EHD['lat'], EHD['tmax'], cmap=custom_cmap, levels=level, extend='max', transform=proj)
####
level2 = [-.4, -.3, -.2, -.1, 0, .1, .2, .3, .4]
z_r = gaussian_filter(z_corr, 3)
a1_h = ax1.contour(z_lon, z_diff['lat'], z_r, transform=proj, levels=level2[:4], colors='blue', linewidths=1.5, linestyles='--',alpha=1)
a1_hm = ax1.contour(z_lon, z_diff['lat'], z_r, transform=proj, levels=[0], colors='gray', linewidths=1.5, linestyles='-', alpha=1)
a1_h1 = ax1.contour(z_lon, z_diff['lat'], z_r, transform=proj, levels=level2[5:], colors='red', linewidths=1.5, linestyles='-', alpha=1)
ax1.clabel(a1_h, inline=True, fontsize=14, fmt='%.01f', colors='blue', zorder=1)
ax1.clabel(a1_hm, inline=True, fontsize=14, fmt='%.01f', colors='gray', zorder=1)
ax1.clabel(a1_h1, inline=True, fontsize=14, fmt='%.01f', colors='red', zorder=1)
ax1.text(122, 29.5, 'A', fontsize=20, fontweight='bold', color='blue', zorder=20)
####
uv_np_ = velovect(ax1, u_diff['lon'], u_diff['lat'],
                  np.array(np.where(np.isnan(u_np), 0, u_np).tolist()),
                  np.array(np.where(np.isnan(v_np), 0, v_np).tolist()),
                  arrowsize=1, scale=5, linewidth=1, regrid=20,
                  color='gray', transform=ccrs.PlateCarree(central_longitude=0))
uv_ = velovect(ax1, u_diff['lon'], u_diff['lat'],
               np.array(np.where(np.isnan(u_corr), 0, u_corr).tolist()),
               np.array(np.where(np.isnan(v_corr), 0, v_corr).tolist()),
               arrowsize=1, scale=5, linewidth=1, regrid=20,
               color='black', transform=ccrs.PlateCarree(central_longitude=0))
velovect_key(fig, ax1, uv_, U=.5, label='0.5')
ax1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray')  # 添加陆地并且陆地部分全部填充成浅灰色
#ax1.add_geometries(get_adm_maps(level='国')[0],
#                   ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=.5)
ax1.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(),
                   ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4)
ax1.add_geometries(Reader(r'D:\PyFile\map\地图线路数据\长江\长江.shp').geometries(),
                   ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=0.2)
ax1.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_长江流域\TPBoundary2500m_长江流域.shp').geometries(),
                   ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1, hatch='//')
ax1.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_del长江流域\TPBoundary2500m_del长江流域.shp').geometries(),
                   ccrs.PlateCarree(), facecolor='gray', edgecolor='gray', linewidth=.1, hatch='.', zorder=2)
# ax1.add_feature(provinces, lw=0.5, zorder=2)
# 设置坐标轴
xticks1=np.arange(extent_CN[0], extent_CN[1]+1, 10)
yticks1=np.arange(extent_CN[2], extent_CN[3]+1, 10)
ax1.set_xticks(xticks1, crs=proj)
ax1.set_yticks(yticks1, crs=proj)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
xmajorLocator = MultipleLocator(5)#先定义xmajorLocator，再进行调用
ax1.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
xminorLocator = MultipleLocator(2)
ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ymajorLocator = MultipleLocator(4)
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
yminorLocator = MultipleLocator(1)
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度


sns.set(style='ticks')
ax1_pc = fig.add_axes(ax1.get_position())
# 设置坐标轴
ax1_pc_xmajor = FixedLocator([1+i*5 for i in range(9)]) #先定义xmajorLocator，再进行调用
ax1_pc_xminor = MultipleLocator(1)
ax1_pc.xaxis.set_major_locator(ax1_pc_xmajor)  #x轴最大刻度
ax1_pc.xaxis.set_minor_locator(ax1_pc_xminor)  #x轴最小刻度
ax1_pc_ymajor = MultipleLocator(1)  #先定义xmajorLocator，再进行调用
ax1_pc_yminor = MultipleLocator(.5)
ax1_pc.yaxis.set_major_locator(ax1_pc_ymajor)  #x轴最大刻度
ax1_pc.yaxis.set_minor_locator(ax1_pc_yminor)  #x轴最小刻度
# 画条形图,正值为红色，负值为蓝色
a1_pc = sns.barplot(x=[i for i in range(1961, 2023)], y=ols, ax=ax1_pc)
for i in range(62):
    if ols[i] > 0:
        a1_pc.get_children()[i].set_color('#D85F4F')
    elif ols[i] == 0:
        a1_pc.get_children()[i].set_color('#F7F7F7')
    else:
        a1_pc.get_children()[i].set_color('#1F6AA0')
ax1_pc.set_xlim(-.5, 61.5)
ax1_pc.set_ylim(-3, 3)
#设定子图ax2大小位置
adjust_sub_axes(ax1, ax1_pc, shrink=1, lr=-.2, ud=1.0)
ax1_pc_reg = ax1_pc.twinx()
k, b = mk.sens_slope(ws2001(ols))  # Theil-Sen 斜率, 截距
#  pd.Series(ols).autocorr(2)  自相关计算=0.26
# color bar位置
position = fig.add_axes(ax1.get_position())#位置[x,y,width,height][0.296, 0.05, 0.44, 0.015]
#竖向colorbar,无尖角
cb1 = plt.colorbar(a1, cax=position, orientation='vertical', pad=0.05)#orientation为水平或垂直
cb1.ax.tick_params(color='black')#length为刻度线的长度
cb1.ax.tick_params(which='major',direction='in', labelsize=12, length=11)
cb1.ax.tick_params(which='minor',direction='in', length=11)
cb1.ax.yaxis.set_minor_locator(MultipleLocator(.1))#显示x轴副刻度
cb1.locator = ticker.FixedLocator([0, 5, 10, 15, 20])  # colorbar上的刻度值个数
adjust_sub_axes(ax1, position, shrink=1, lr=-3, ud=1.0, width=0.025)

plt.savefig(r'D:\PyFile\pic\EHD低层风高场及去趋势.png', dpi=1000, bbox_inches='tight')
plt.show()
