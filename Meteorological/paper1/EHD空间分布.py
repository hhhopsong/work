from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from cnmaps import get_adm_maps, draw_maps
from matplotlib import ticker, colors
import cmaps
from matplotlib.ticker import MultipleLocator
from toolbar.masked import masked   # 气象工具函数


# 数据读取
# 读取CN05.1逐日最高气温数据
EHD = xr.open_dataset(r"D:\PyFile\paper1\EHD35.nc").sel(time=slice('1961-01-01', '2022-12-31'))  # 读取缓存
EHD = EHD.sel(time=EHD['time.month'].isin([7, 8])).groupby('time.year').sum('time').mean('year')
# 将0值替换为缺测值
EHD = EHD.where(EHD > 0)
EHD = masked(EHD, r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")
EHD35_lat = EHD['lat'][np.argwhere(~np.isnan(EHD['tmax'].data))[:, 0]]
EHD35_lon = EHD['lon'][np.argwhere(~np.isnan(EHD['tmax'].data))[:, 1]]
EHD35_map = np.array([EHD35_lat, EHD35_lon]).T
np.save(r"D:\PyFile\paper1\EHD35_map.npy", EHD35_map)
np.save(r"D:\PyFile\paper1\EHD35_index.npy", np.argwhere(~np.isnan(EHD['tmax'].data)))
# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
extent_CN = [88, 124, 22, 38]  # 中国大陆经度范围，纬度范围
proj = ccrs.PlateCarree()   # 投影方式
fig = plt.figure(figsize=(16, 9))   # 创建画布
ax1 = fig.add_subplot(111, projection=proj)  # 添加子图
ax1.set_extent(extent_CN, crs=proj) # 设置地图范围
level = [0, 5, 10, 15, 20]  # 等值线间隔
bins = [0, 5, 10, 15, 20]
custom_colors = ["#FDB57E", "#F26E4c", "#CA1E14", "#7F0000"]
custom_cmap = colors.ListedColormap(custom_colors)
a1 = ax1.contourf(EHD['lon'], EHD['lat'], EHD['tmax'], cmap=custom_cmap, levels=level, extend='max', transform=proj)
ax1.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
ax1.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4)
ax1.add_geometries(Reader(r'D:\PyFile\map\地图线路数据\长江\长江.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=0.2)

# 刻度线设置
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
xminorLocator = MultipleLocator(1)
ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ymajorLocator = MultipleLocator(4)
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
yminorLocator = MultipleLocator(1)
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
# 调整刻度值字体大小
ax1.tick_params(axis='both', labelsize=16, colors='black')
# 最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=11, width=2, color='black')  # 最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=6, width=1.8, color='black')  # 最小刻度长度，宽度设置
ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)

# color bar位置
# position = fig.add_axes([0.296, 0.08, 0.44, 0.011])#位置[左,下,右,上]
cb1 = plt.colorbar(a1, orientation='horizontal', aspect=30, shrink=0.6)#orientation为水平或垂直
cb1.ax.tick_params(length=1, labelsize=16, color='lightgray')#length为刻度线的长度
tick_locator = ticker.FixedLocator([0, 5, 10, 15, 20])  # colorbar上的刻度值个数

plt.savefig(r'D:\PyFile\pic\图2.png', dpi=666, bbox_inches='tight')
plt.show()
pass
