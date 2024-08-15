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
import pandas as pd
import tqdm
import seaborn as sns

# 数据读取
EHD = xr.open_dataset(r"cache\EHD.nc")  # 读取缓存
EHD = masked(EHD,
             r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
# 截取7月16日-8月15日数据
EHD_7 = EHD.sel(time=EHD['time.month'].isin([7]))
EHD_7 = EHD_7.sel(time=EHD_7['time.day'].isin(range(16, 32)))
EHD_8 = EHD.sel(time=EHD['time.month'].isin([8]))
EHD_8 = EHD_8.sel(time=EHD_8['time.day'].isin(range(1, 16)))
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
extent_CN = [88, 124, 22, 38]  # 中国大陆经度范围，纬度范围
proj = ccrs.PlateCarree()  # 投影方式
fig = plt.figure(figsize=(16, 9))  # 创建画布
ax1 = fig.add_subplot(121, projection=proj)  # 添加子图
ax1.set_extent(extent_CN, crs=proj)  # 设置地图范围
a1 = ax1.contourf(EHD['lon'], EHD['lat'], Modality[0], cmap=cmaps.WhiteBlueGreenYellowRed, levels=15, extend='both',
                  transform=proj)
cbar = plt.colorbar(a1, ax=ax1, orientation='horizontal', pad=0.05, aspect=50, shrink=0.8)
ax1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray')  # 添加陆地并且陆地部分全部填充成浅灰色
ax1.add_geometries(Reader(
    r'C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(),
                   ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4)
ax1.add_geometries(Reader(r'D:\CODES\Python\Meteorological\maps\cnriver\长江\长江.shp').geometries(),
                   ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=0.2)

sns.set(style='ticks')
ax1_pc = fig.add_subplot(122)
ax1_pc.set_title(f'{s[0] * 100:.2f}%', loc='right')
# 设置坐标轴
ax1_pc_xmajor = MultipleLocator(5)  #先定义xmajorLocator，再进行调用
ax1_pc_xminor = MultipleLocator(1)
ax1_pc.xaxis.set_major_locator(ax1_pc_xmajor)  #x轴最大刻度
ax1_pc.xaxis.set_minor_locator(ax1_pc_xminor)  #x轴最小刻度
# 画条形图,正值为红色，负值为蓝色
a1_pc = sns.barplot(x=[i for i in range(1979, 2023)], y=PC[:, 0], ax=ax1_pc)
for i in range(44):
    if PC[i, 0] > 0:
        a1_pc.get_children()[i].set_color('#D85F4F')
    elif PC[i, 0] == 0:
        a1_pc.get_children()[i].set_color('#F7F7F7')
    else:
        a1_pc.get_children()[i].set_color('#1F6AA0')

plt.savefig(r'C:\Users\10574\desktop\EHD高发期EOF.png', dpi=1500, bbox_inches='tight')
plt.show()
