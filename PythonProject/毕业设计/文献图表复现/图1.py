from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from cnmaps import get_adm_maps, draw_maps
from matplotlib import ticker
import cmaps
from matplotlib.ticker import MultipleLocator

# 数据读取
q = xr.open_dataset(r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\CN05.1\CN05.1_Tmax_1961_2021_daily_025x025.nc")
q_sort95 = q['tmax'].sel(time=slice('1979-01-01', '2014-12-31')).quantile(0.95, dim='time')
# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
#plt.subplots_adjust(wspace=0.1, hspace=0.32)  # wspace、hspace左右、上下的间距
extent1=[70, 140, 15, 55]  # 经度范围，纬度范围
xticks1=np.arange(extent1[0], extent1[1]+1, 10)
yticks1=np.arange(extent1[2], extent1[3]+1, 10)
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(16, 9))
# ##修正前
level1 = [14+i*2 for i in range(11)]
ax1 = fig.add_subplot(111, projection=proj)
#ax1.set_title(f'Fig. 1 The distribution of the 95th percentile thresholds (shading; °C)\n'
#              f'defined for extreme high temperature over China during 1980–2009', color='black', fontsize=15)
ax1.set_extent(extent1, crs=proj)
a1 = ax1.contourf(q['lon'], q['lat'], q_sort95, cmap=cmaps.WhiteBlueGreenYellowRed, levels=level1, extend='both', transform=proj)
ax1.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
draw_maps(get_adm_maps(level='国'), linewidth=0.4)
draw_maps(get_adm_maps(level='省'), linewidth=0.2)
# 刻度线设置
ax1.set_xticks(xticks1, crs=proj)
ax1.set_yticks(yticks1, crs=proj)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
xmajorLocator = MultipleLocator(10)#先定义xmajorLocator，再进行调用
ax1.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
xminorLocator = MultipleLocator(5)
ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ymajorLocator = MultipleLocator(10)
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
yminorLocator = MultipleLocator(2)
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
# ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
#最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=8,width=1.8,color='darkgray')#最小刻度长度，宽度设置
ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外
#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=20)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Arial') for label in labels]

# color bar位置
# position = fig.add_axes([0.296, 0.08, 0.44, 0.011])#位置[左,下,右,上]
cb1 = plt.colorbar(a1, orientation='horizontal', aspect=30, shrink=0.6)#orientation为水平或垂直
cb1.ax.tick_params(length=1, labelsize=16, color='lightgray')#length为刻度线的长度
tick_locator = ticker.MaxNLocator(nbins=7)  # colorbar上的刻度值个数

# 南海小地图
def adjust_sub_axes(ax_main, ax_sub, shrink, lr=1.0, ud=1.0):
    '''
    将ax_sub调整到ax_main的右下角.shrink指定缩小倍数。
    当ax_sub是GeoAxes时,需要在其设定好范围后再使用此函数
    '''
    bbox_main = ax_main.get_position()
    bbox_sub = ax_sub.get_position()
    wratio = bbox_main.width / bbox_sub.width
    hratio = bbox_main.height / bbox_sub.height
    wnew = bbox_sub.width * shrink
    hnew = bbox_sub.height * shrink
    bbox_new = mtransforms.Bbox.from_extents(
        bbox_main.x1 - lr*wnew, bbox_main.y0 + (ud-1)*hnew,
        bbox_main.x1 - (lr-1)*wnew, bbox_main.y0 + ud*hnew
    )
    ax_sub.set_position(bbox_new)


ax_sub = fig.add_axes(ax1.get_position(), projection=proj)
ax_sub.set_extent([105, 125, 0, 25], crs=proj)
a_sub = ax_sub.contourf(q['lon'], q['lat'], q_sort95, cmap=cmaps.WhiteBlueGreenYellowRed, levels=level1, extend='both', transform=proj)
ax_sub.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
draw_maps(get_adm_maps(level='国'), linewidth=0.4)
draw_maps(get_adm_maps(level='省'), linewidth=0.2)
adjust_sub_axes(ax1, ax_sub, shrink=.45)
plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\图1.png', dpi=1500, bbox_inches='tight')
plt.show()
