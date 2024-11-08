from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as path_effects
from cnmaps import get_adm_maps, draw_maps
from matplotlib import ticker
import cmaps
from matplotlib.ticker import MultipleLocator
from toolbar.significance_test import corr_test
from eofs.standard import Eof
import geopandas as gpd
import salem


# 数据读取
sst = xr.open_dataset(r"E:\data\NOAA\ERSSTv5\sst.mnmean.nc")  # NetCDF-4文件路径不可含中文
# 截取sst数据为5N-5S，40E-80W
time_data = [1961, 2022]
sst = sst.sel(lat=slice(5, -5), lon=slice(40, 360-80))['sst']
lon_sst = sst['lon']
sst_term = sst.sel(time=slice(f'{time_data[0]}-01-01', f'{time_data[1]}-12-31'))
sst_lastyear = sst.sel(time=slice(f'{time_data[0] - 1}-01-01', f'{time_data[1] - 1}-12-31'))
sst_nextyear = sst.sel(time=slice(f'{time_data[0] + 1}-01-01', f'{time_data[1] + 1}-12-31'))
# 计算sst经向平均值
sst_term_lonavg = sst_term.mean(dim='lat')
sst_lastyear_lonavg = sst_lastyear.mean(dim='lat')
sst_nextyear_lonavg = sst_nextyear.mean(dim='lat')
PC = np.load(r"D:\PyFile\paper1\IMFs.npy")  # 读取缓存
PC = PC[0] + PC[1]
# 计算sst距平
sst_term_anom = sst_term_lonavg - sst_term_lonavg.mean(dim='time')
sst_lastyear_anom = sst_lastyear_lonavg - sst_lastyear_lonavg.mean(dim='time')
sst_nextyear_anom = sst_nextyear_lonavg - sst_nextyear_lonavg.mean(dim='time')
# 计算sst距平与EOF的超前滞后相关系数，滞后范围为5
num_lead_lag_corr = 18
lead_lag_corr = np.zeros((num_lead_lag_corr, len(lon_sst)))
lead_lag_corr.fill(np.nan)
sst_leadlag = np.zeros(((time_data[1] - time_data[0] + 1) * 18, len(lon_sst)))
for i in range(time_data[1] - time_data[0] + 1):
    sst_leadlag[i*18:i*18+18, :] = np.append(sst_lastyear_anom[i*12+9:i*12+12, :], np.append(sst_term_anom[i*12:i*12+12, :], sst_nextyear_anom[i*12:i*12+3, :], axis=0), axis=0)
for i in range(num_lead_lag_corr):
    for j in range(len(lon_sst)):
        lead_lag_corr[i, j] = np.corrcoef(sst_leadlag[i::num_lead_lag_corr, j], PC)[0, 1]
lead_lag_corr2 = np.zeros((num_lead_lag_corr, len(lon_sst)))
lead_lag_corr2.fill(np.nan)
for i in range(num_lead_lag_corr):
    for j in range(len(lon_sst)):
        lead_lag_corr2[i, j] = np.corrcoef(sst_leadlag[i::num_lead_lag_corr, j], PC)[0, 1]
# 进行显著性检验
p_lead_lag_corr = corr_test(PC, lead_lag_corr, alpha=0.1)

# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.subplots_adjust(wspace=0.1, hspace=0.001)  # wspace、hspace左右、上下的间距
# plt.subplots_adjust(wspace=0.1, hspace=0.32)  # wspace、hspace左右、上下的间距
extent1 = [40, 360-80, 1, num_lead_lag_corr]  # 经度范围，纬度范围
time = np.arange(extent1[2], extent1[3] + 1, 1)
xticks1 = np.arange(extent1[0], extent1[1] + 1, 10)
yticks1 = np.arange(extent1[2], extent1[3] + 1, 1)
yticks2 = ['Last-Oct', 'Last-Nov', 'Last-Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Next-Jan', 'Next-Feb', 'Next-Mar']
proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(16, 8))
# ##ax1 Corr. PC1 & JA SST,2mT
level1 = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
ax1 = fig.add_subplot(111)
print('开始绘制图1')
ax1.set_title('(a)Lead-lag corr. N/A & SST', fontsize=22, loc='left')
a1 = ax1.contourf(lon_sst, time, lead_lag_corr, cmap=cmaps.cmp_b2r, levels=level1, extend='both')
# 通过打点显示出通过显著性检验的区域
a1_p = ax1.contourf(lon_sst, time, p_lead_lag_corr, levels=[0, 1], hatches=['//', None], colors="none", add_colorbar=False, zorder=5)
ax1.axhline(10, color='black', linewidth=2)
ax1.axhline(11, color='black', linewidth=2)

# 刻度线设置
ax1.set_xticks(xticks1)
ax1.set_yticks(yticks1)
ax1.set_yticklabels(yticks2)
lon_formatter = LongitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)

font = {'family': 'Arial', 'weight': 'bold', 'size': 22}

xmajorLocator = MultipleLocator(60)  # 先定义xmajorLocator，再进行调用
ax1.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
xminorLocator = MultipleLocator(10)
ax1.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
ymajorLocator = MultipleLocator(1)
ax1.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
# ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
# 最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=11, width=2, color='darkgray')  # 最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=8, width=1.8, color='darkgray')  # 最小刻度长度，宽度设置
ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
# 调整刻度值字体大小
ax1.tick_params(axis='both', labelsize=22, colors='black')
# 设置坐标刻度值的大小以及刻度值的字体
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
font2 = {'family': 'Arial', 'weight': 'bold', 'size': 28}

# color bar位置
position = fig.add_axes([0.296, 0.01, 0.44, 0.02])
cb1 = plt.colorbar(a1, cax=position, orientation='horizontal')#orientation为水平或垂直
cb1.ax.tick_params(length=1, labelsize=20, color='lightgray')#length为刻度线的长度
cb1.locator = ticker.FixedLocator([-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]) # colorbar上的刻度值个数


plt.savefig(r'D:\PyFile\pic\热带海温异常-imf.png', dpi=1000, bbox_inches='tight')
plt.show()
