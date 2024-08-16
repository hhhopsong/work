from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, FixedLocator
from matplotlib import gridspec
import matplotlib.colors as colors
from cnmaps import get_adm_maps, draw_maps
from eofs.standard import Eof
import cmaps
from toolbar.masked import masked  # 气象工具函数
from toolbar.sub_adjust import adjust_sub_axes
import pandas as pd
import tqdm
import seaborn as sns

# 数据读取
EHD = xr.open_dataset(r"cache\EHD.nc")  # 读取缓存
EHD = masked(EHD,
             r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
# 截取7月16日-8月19日数据
EHD_7 = EHD.sel(time=EHD['time.month'].isin([7]))
EHD_7 = EHD_7.sel(time=EHD_7['time.day'].isin(range(16, 32)))
EHD_8 = EHD.sel(time=EHD['time.month'].isin([8]))
EHD_8 = EHD_8.sel(time=EHD_8['time.day'].isin(range(1, 20)))
# 合并数据,并按时间排序
EHD_concat = xr.concat([EHD_7, EHD_8], dim='time').sortby('time')
EHD_concat.fillna(0)  # 数据二值化处理(1:极端高温,0:非极端高温)
EHD_concat = EHD_concat['tmax'].groupby('time.year').sum('time')  # 计算7月16日-8月15日累计极端高温日数
EHD_concat = masked(EHD_concat,
                    r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
# 计算EOF
eof = Eof(EHD_concat.to_numpy())  #进行eof分解
Modality = eof.eofs(eofscaling=2,
                    neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
PC = eof.pcs(pcscaling=1, npcs=2)  # 同上 npcs决定输出的时间序列个数
s = eof.varianceFraction(neigs=2)  # 得到前neig个模态的方差贡献
# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))  # 创建画布
spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], height_ratios=[1])  # 设置子图比例

extent_CN = [88, 124, 22, 38]  # 中国大陆经度范围，纬度范围
proj = ccrs.PlateCarree()  # 投影方式
ax1 = fig.add_subplot(spec[0, 0], projection=proj)  # 添加子图
# 设置ax1 figsize=(9, 4)
ax1.set_extent(extent_CN, crs=proj)  # 设置地图范围
a1 = ax1.contourf(EHD['lon'], EHD['lat'], Modality[0], cmap=cmaps.WhiteBlueGreenYellowRed,
                  levels=[0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
                  extend='both',transform=proj)
cbar = plt.colorbar(a1, ax=ax1, orientation='horizontal', pad=0.05, aspect=50, shrink=0.8)
ax1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray')  # 添加陆地并且陆地部分全部填充成浅灰色
ax1.add_geometries(Reader(
    r'C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(),
                   ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4)
ax1.add_geometries(Reader(r'D:\CODES\Python\Meteorological\maps\cnriver\长江\长江.shp').geometries(),
                   ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=0.2)
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
ax1_pc.set_title(f'{s[0] * 100:.2f}%', loc='right')
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
a1_pc = sns.barplot(x=[i for i in range(1979, 2023)], y=PC[:, 0], ax=ax1_pc)
for i in range(44):
    if PC[i, 0] > 0:
        a1_pc.get_children()[i].set_color('#D85F4F')
    elif PC[i, 0] == 0:
        a1_pc.get_children()[i].set_color('#F7F7F7')
    else:
        a1_pc.get_children()[i].set_color('#1F6AA0')
ax1_pc.set_xlim(-.5, 43.5)
ax1_pc.set_ylim(-3, 3)
#设定子图ax2大小位置
adjust_sub_axes(ax1, ax1_pc, shrink=1, lr=-.1, ud=1.0)
ax1_pc_reg = ax1_pc.twinx()
ax1_pc_reg = sns.regplot(x=[i for i in range(44)], y=PC[:, 0], ax=ax1_pc_reg, scatter=False, color='#74C476', ci=95)
ax1_pc_reg.set_ylim(-3, 3)
ax1_pc_reg.yaxis.set_visible(False)  # 隐藏y轴标签
ax1_pc_reg.spines['top'].set_visible(False)  # 隐藏上边框
ax1_pc_reg.spines['right'].set_visible(False)  # 隐藏右边框
ax1_pc_reg.spines['bottom'].set_visible(False)  # 显示下边框
ax1_pc_reg.spines['left'].set_visible(False)  # 显示左边框

plt.savefig(r'C:\Users\10574\desktop\EHD高发期EOF.png', dpi=1500, bbox_inches='tight')
plt.show()
