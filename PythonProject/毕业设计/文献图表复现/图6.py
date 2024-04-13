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
from eofs.standard import Eof
import geopandas as gpd
import salem


# 数据读取
sst = xr.open_dataset(r"C:\Users\10574\Desktop\data\sst.mnmean.nc")  # NetCDF-4文件路径不可含中文
# 截取sst数据为5N-5S，40E-80W
sst = sst.sel(lat=slice(5, -5), lon=slice(40, 360-80))['sst']
lon_sst = sst['lon']
sst_8009 = sst.sel(time=slice('1979-01-01', '2014-12-31'))
# 计算sst经向平均值
sst_8009_lonavg = sst_8009.mean(dim='lat')
std_q78 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_q78.nc')
shp = fr"D:/CODES/Python/PythonProject/map/shp/south_china/中国南方.shp"
split_shp = gpd.read_file(shp)
split_shp.crs = 'wgs84'
std_q78 = std_q78.salem.roi(shape=split_shp)
# eof分解
eof_78 = Eof(std_q78['tmax'].to_numpy())  # 进行eof分解
EOF_78 = eof_78.eofs(eofscaling=2, neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
PC_78 = eof_78.pcs(pcscaling=1, npcs=2)  # 同上 npcs决定输出的时间序列个数
s_78 = eof_78.varianceFraction(neigs=2)  # 得到前neig个模态的方差贡献
# 计算sst距平
sst_8009_anom = sst_8009_lonavg - sst_8009_lonavg.mean(dim='time')
# 计算sst距平与EOF的超前滞后相关系数，滞后范围为5
lead_lag_corr = np.zeros((12, len(lon_sst)))
lead_lag_corr.fill(np.nan)
for i in range(12):
    for j in range(len(lon_sst)):
        lead_lag_corr[i, j] = np.corrcoef(sst_8009_anom[i::12, j], PC_78[:, 0])[0, 1]
lead_lag_corr2 = np.zeros((12, len(lon_sst)))
lead_lag_corr2.fill(np.nan)
for i in range(12):
    for j in range(len(lon_sst)):
        lead_lag_corr2[i, j] = np.corrcoef(sst_8009_anom[i::12, j], PC_78[:, 1])[0, 1]
# 进行显著性0.1检验
from scipy.stats import t
# 计算自由度
n = len(PC_78[:, 0])
# 计算t值
t_lead_lag_corr = lead_lag_corr * np.sqrt((n - 2) / (1 - lead_lag_corr ** 2))
t_lead_lag_corr2 = lead_lag_corr2 * np.sqrt((n - 2) / (1 - lead_lag_corr2 ** 2))
# 计算临界值
t_critical = t.ppf(0.95, n - 2)
# 进行显著性检验
p_lead_lag_corr = np.zeros((12, len(lon_sst)))
p_lead_lag_corr.fill(np.nan)
p_lead_lag_corr[np.abs(t_lead_lag_corr) > t_critical] = 1

p_lead_lag_corr2 = np.zeros((12, len(lon_sst)))
p_lead_lag_corr2.fill(np.nan)
p_lead_lag_corr2[np.abs(t_lead_lag_corr2) > t_critical] = 1



# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.subplots_adjust(wspace=0.1, hspace=0.001)  # wspace、hspace左右、上下的间距
# plt.subplots_adjust(wspace=0.1, hspace=0.32)  # wspace、hspace左右、上下的间距
extent1 = [40, 360-80, 1, 12]  # 经度范围，纬度范围
time = np.arange(extent1[2], extent1[3] + 1, 1)
xticks1 = np.arange(extent1[0], extent1[1] + 1, 10)
yticks1 = np.arange(extent1[2], extent1[3] + 1, 1)
yticks2 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(16, 8))
# ##ax1 Corr. PC1 & JA SST,2mT
level1 = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
ax1 = fig.add_subplot(121)
print('开始绘制图1')
ax1.set_title('(a)Lead-lag corr. PC1 & SST', fontsize=22, loc='left')
a1 = ax1.contourf(lon_sst, time, lead_lag_corr, cmap=cmaps.cmp_b2r, levels=level1, extend='both')
# 通过打点显示出通过显著性检验的区域
a1_p = ax1.contourf(lon_sst, time, p_lead_lag_corr, levels=[0, 1], hatches=['..', None], colors="none", add_colorbar=False, zorder=5)
ax1.axhline(7, color='black', linewidth=2)
ax1.axhline(8, color='black', linewidth=2)

ax2 = fig.add_subplot(122)
print('开始绘制图2')
ax2.set_title('(b)Lead-lag corr. PC2 & SST', fontsize=22, loc='left')
a2 = ax2.contourf(lon_sst, time, lead_lag_corr2, cmap=cmaps.cmp_b2r, levels=level1, extend='neither')
# 通过打点显示出通过显著性检验的区域
a2_p = ax2.contourf(lon_sst, time, p_lead_lag_corr2, levels=[0, 1], hatches=['..', None], colors="none", add_colorbar=False, zorder=5)
ax2.axhline(7, color='black', linewidth=2)
ax2.axhline(8, color='black', linewidth=2)

# 刻度线设置
ax1.set_xticks(xticks1)
ax1.set_yticks(yticks1)
ax1.set_yticklabels(yticks2)
lon_formatter = LongitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax2.set_xticks(xticks1)
ax2.set_yticks(yticks1)
ax2.set_yticklabels(yticks2)
ax2.xaxis.set_major_formatter(lon_formatter)

font = {'family': 'Arial', 'weight': 'bold', 'size': 22}

xmajorLocator = MultipleLocator(60)  # 先定义xmajorLocator，再进行调用
ax1.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
ax2.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
xminorLocator = MultipleLocator(10)
ax1.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
ax2.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
ymajorLocator = MultipleLocator(1)
ax1.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
ax2.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
# ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
# 最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=11, width=2, color='darkgray')  # 最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=8, width=1.8, color='darkgray')  # 最小刻度长度，宽度设置
ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
ax2.tick_params(which='major', length=11, width=2, color='darkgray')  # 最大刻度长度，宽度设置，
ax2.tick_params(which='minor', length=8, width=1.8, color='darkgray')  # 最小刻度长度，宽度设置
ax2.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
# 调整刻度值字体大小
ax1.tick_params(axis='both', labelsize=22, colors='black')
ax2.tick_params(axis='both', labelsize=22, colors='black')
# 设置坐标刻度值的大小以及刻度值的字体
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
font2 = {'family': 'Arial', 'weight': 'bold', 'size': 28}

# color bar位置
position = fig.add_axes([0.296, 0.01, 0.44, 0.02])
cb1 = plt.colorbar(a1, cax=position, orientation='horizontal')#orientation为水平或垂直
cb1.ax.tick_params(length=1, labelsize=20, color='lightgray')#length为刻度线的长度
cb1.locator = ticker.FixedLocator([-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]) # colorbar上的刻度值个数


plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\图6.png', dpi=1000, bbox_inches='tight')
plt.show()
