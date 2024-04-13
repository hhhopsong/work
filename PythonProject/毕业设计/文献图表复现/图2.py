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


# 数据读取
q = xr.open_dataset(r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\CN05.1\CN05.1_Tmax_1961_2021_daily_025x025.nc")
lon = q['lon']
lat = q['lat']
'''q_sort95 = q['tmax'].sel(time=slice('1979-01-01', '2014-12-31')).quantile(0.95, dim='time')
q_8009 = q['tmax'].sel(time=slice('1979-01-01', '2014-12-31'))
q_8009_6 = q_8009.sel(time=q_8009['time.month'] == 6)
q_8009_7 = q_8009.sel(time=q_8009['time.month'] == 7)
q_8009_8 = q_8009.sel(time=q_8009['time.month'] == 8)
q_8009_9 = q_8009.sel(time=q_8009['time.month'] == 9)
# 选取数据中1980-2009年每年6月的数据
###
q_6 = np.zeros([163, 283])
q_7 = np.zeros([163, 283])
q_8 = np.zeros([163, 283])
q_9 = np.zeros([163, 283])
for long in range(len(lon)):
    print(f"进度:{long}/282")
    for lati in range(len(lat)):
        q_6[lati, long] = np.sum(q_8009_6.sel(lon=lon[long], lat=lat[lati]) > q_sort95.sel(lon=lon[long], lat=lat[lati]))/30
        q_7[lati, long] = np.sum(q_8009_7.sel(lon=lon[long], lat=lat[lati]) > q_sort95.sel(lon=lon[long], lat=lat[lati]))/30
        q_8[lati, long] = np.sum(q_8009_8.sel(lon=lon[long], lat=lat[lati]) > q_sort95.sel(lon=lon[long], lat=lat[lati]))/30
        q_9[lati, long] = np.sum(q_8009_9.sel(lon=lon[long], lat=lat[lati]) > q_sort95.sel(lon=lon[long], lat=lat[lati]))/30
        if q_6[lati, long] == 0:
            q_6[lati, long] = np.nan
        if q_7[lati, long] == 0:
            q_7[lati, long] = np.nan
        if q_8[lati, long] == 0:
            q_8[lati, long] = np.nan
        if q_9[lati, long] == 0:
            q_9[lati, long] = np.nan
np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\q_6.npy', q_6)
np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\q_7.npy', q_7)
np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\q_8.npy', q_8)
np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\q_9.npy', q_9)'''
# 数据读取
q_6 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\q_6.npy')
q_7 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\q_7.npy')
q_8 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\q_8.npy')
q_9 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\q_9.npy')

# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.subplots_adjust(wspace=0.01, hspace=0.1)#wspace、hspace左右、上下的间距
extent1=[105, 136, 15, 55]  # 经度范围，纬度范围
xticks1=np.arange(extent1[0], extent1[1]+1, 10)
yticks1=np.arange(extent1[2], extent1[3]+1, 10)
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(15, 17))
# ##修正前
level1 = [i for i in range(18)]
ax1 = fig.add_subplot(221, projection=proj)
#ax1.set_title(f'Fig. 1 The distribution of the 95th percentile thresholds (shading; °C)\n'
#              f'defined for extreme high temperature over China during 1980–2009', color='black', fontsize=15)
ax1.set_extent(extent1, crs=proj)
# 标题水平居左
ax1.set_title('(a) EHDs in Jun.', fontsize=28, loc='left')
a1 = ax1.contourf(q['lon'], q['lat'], q_6, cmap=cmaps.WhiteBlueGreenYellowRed, levels=level1, extend='both', transform=proj)
ax1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
draw_maps(get_adm_maps(level='国'), linewidth=0.4)
draw_maps(get_adm_maps(level='省'), linewidth=0.2)
# 在ax1中画出Yangtze-River middle and lower
ax1.plot([105.2, 105.2], [33, 23.2], color='red', linewidth=2, linestyle='-',transform=proj)
ax1.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\shp\south_china\中国南方.shp').geometries(), proj, facecolor='none',edgecolor='red',linewidth=3)
ax_sub1 = fig.add_axes(ax1.get_position(), projection=proj)
ax_sub1.set_extent([104, 125, 0, 25], crs=proj)
a_sub1 = ax_sub1.contourf(q['lon'], q['lat'], q_6, cmap=cmaps.WhiteBlueGreenYellowRed, levels=level1, extend='both', transform=proj)
ax_sub1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
draw_maps(get_adm_maps(level='国'), linewidth=0.4)
draw_maps(get_adm_maps(level='省'), linewidth=0.2)
adjust_sub_axes(ax1, ax_sub1, shrink=.4)

ax2 = fig.add_subplot(222, projection=proj)
ax2.set_extent(extent1, crs=proj)
ax2.set_title('(b) EHDs in Jul.', fontsize=28, loc='left')
a2 = ax2.contourf(q['lon'], q['lat'], q_7, cmap=cmaps.WhiteBlueGreenYellowRed, levels=level1, extend='both', transform=proj)
ax2.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
draw_maps(get_adm_maps(level='国'), linewidth=0.4)
draw_maps(get_adm_maps(level='省'), linewidth=0.2)
ax2.plot([105.2, 105.2], [33, 23.2], color='red', linewidth=2, linestyle='-',transform=proj)
ax2.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\shp\south_china\中国南方.shp').geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='red',linewidth=3)
ax_sub2 = fig.add_axes(ax2.get_position(), projection=proj)
ax_sub2.set_extent([104, 125, 0, 25], crs=proj)
a_sub2 = ax_sub2.contourf(q['lon'], q['lat'], q_7, cmap=cmaps.WhiteBlueGreenYellowRed, levels=level1, extend='both', transform=proj)
ax_sub2.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
draw_maps(get_adm_maps(level='国'), linewidth=0.4)
draw_maps(get_adm_maps(level='省'), linewidth=0.2)
adjust_sub_axes(ax2, ax_sub2, shrink=.4)

ax3 = fig.add_subplot(223, projection=proj)
ax3.set_extent(extent1, crs=proj)
ax3.set_title('(c) EHDs in Aug.', fontsize=28, loc='left')
a3 = ax3.contourf(q['lon'], q['lat'], q_8, cmap=cmaps.WhiteBlueGreenYellowRed, levels=level1, extend='both', transform=proj)
ax3.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
draw_maps(get_adm_maps(level='国'), linewidth=0.4)
draw_maps(get_adm_maps(level='省'), linewidth=0.2)
ax3.plot([105.2, 105.2], [33, 23.2], color='red', linewidth=2, linestyle='-',transform=proj)
ax3.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\shp\south_china\中国南方.shp').geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='red',linewidth=3)
ax_sub3 = fig.add_axes(ax3.get_position(), projection=proj)
ax_sub3.set_extent([104, 125, 0, 25], crs=proj)
a_sub3 = ax_sub3.contourf(q['lon'], q['lat'], q_8, cmap=cmaps.WhiteBlueGreenYellowRed, levels=level1, extend='both', transform=proj)
ax_sub3.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
draw_maps(get_adm_maps(level='国'), linewidth=0.4)
draw_maps(get_adm_maps(level='省'), linewidth=0.2)
adjust_sub_axes(ax3, ax_sub3, shrink=.4)

ax4 = fig.add_subplot(224, projection=proj)
ax4.set_extent(extent1, crs=proj)
ax4.set_title('(d) EHDs in Sep.', fontsize=28, loc='left')
a4 = ax4.contourf(q['lon'], q['lat'], q_9, cmap=cmaps.WhiteBlueGreenYellowRed, levels=level1, extend='both', transform=proj)
ax4.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
draw_maps(get_adm_maps(level='国'), linewidth=0.4)
draw_maps(get_adm_maps(level='省'), linewidth=0.2)
ax4.plot([105.2, 105.2], [33, 23.2], color='red', linewidth=2, linestyle='-',transform=proj)
ax4.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\shp\south_china\中国南方.shp').geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='red',linewidth=3)
ax_sub4 = fig.add_axes(ax4.get_position(), projection=proj)
ax_sub4.set_extent([104, 125, 0, 25], crs=proj)
a_sub4 = ax_sub4.contourf(q['lon'], q['lat'], q_9, cmap=cmaps.WhiteBlueGreenYellowRed, levels=level1, extend='both', transform=proj)
ax_sub4.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
draw_maps(get_adm_maps(level='国'), linewidth=0.4)
draw_maps(get_adm_maps(level='省'), linewidth=0.2)
adjust_sub_axes(ax4, ax_sub4, shrink=.4)

# 刻度线设置
ax1.set_xticks(xticks1, crs=proj)
ax1.set_yticks(yticks1, crs=proj)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
#
ax2.set_xticks(xticks1, crs=proj)
ax2.set_yticks(yticks1, crs=proj)
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
#
ax3.set_xticks(xticks1, crs=proj)
ax3.set_yticks(yticks1, crs=proj)
ax3.xaxis.set_major_formatter(lon_formatter)
ax3.yaxis.set_major_formatter(lat_formatter)
#
ax4.set_xticks(xticks1, crs=proj)
ax4.set_yticks(yticks1, crs=proj)
ax4.xaxis.set_major_formatter(lon_formatter)
ax4.yaxis.set_major_formatter(lat_formatter)


xmajorLocator = MultipleLocator(10)#先定义xmajorLocator，再进行调用
xminorLocator = MultipleLocator(2)
ax1.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ax2.xaxis.set_major_locator(xmajorLocator)
ax2.xaxis.set_minor_locator(xminorLocator)
ax3.xaxis.set_major_locator(xmajorLocator)
ax3.xaxis.set_minor_locator(xminorLocator)
ax4.xaxis.set_major_locator(xmajorLocator)
ax4.xaxis.set_minor_locator(xminorLocator)
ymajorLocator = MultipleLocator(10)
yminorLocator = MultipleLocator(2)
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
ax2.yaxis.set_major_locator(ymajorLocator)
ax2.yaxis.set_minor_locator(yminorLocator)
ax3.yaxis.set_major_locator(ymajorLocator)
ax3.yaxis.set_minor_locator(yminorLocator)
ax4.yaxis.set_major_locator(ymajorLocator)
ax4.yaxis.set_minor_locator(yminorLocator)
#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=28)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Arial') for label in labels]

# 调整刻度值字体大小
ax1.tick_params(axis='both', labelsize=28, colors='black')
ax2.tick_params(axis='both', labelsize=28, colors='black')
ax3.tick_params(axis='both', labelsize=28, colors='black')
ax4.tick_params(axis='both', labelsize=28, colors='black')
#最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=8,width=1.8,color='darkgray')#最小刻度长度，宽度设置
ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
ax2.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax2.tick_params(which='minor', length=8,width=1.8,color='darkgray')#最小刻度长度，宽度设置
ax2.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
ax3.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax3.tick_params(which='minor', length=8,width=1.8,color='darkgray')#最小刻度长度，宽度设置
ax3.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
ax4.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax4.tick_params(which='minor', length=8,width=1.8,color='darkgray')#最小刻度长度，宽度设置
ax4.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外
# color bar位置
position = fig.add_axes([0.296, 0.05, 0.44, 0.015])#位置[左,下,右,上]
cb1 = plt.colorbar(a1, cax=position, orientation='horizontal', aspect=30, shrink=0.6)#orientation为水平或垂直
cb1.ax.tick_params(length=1, labelsize=20, color='lightgray')#length为刻度线的长度
tick_locator = ticker.MaxNLocator(nbins=12)  # colorbar上的刻度值个数

plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\图2-1.png', dpi=1500, bbox_inches='tight')
plt.show()