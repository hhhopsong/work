import pprint
import geopandas as gpd
import salem
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
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
import matplotlib.patches as patches # 导入遮盖模块
from scipy.ndimage import filters


def adjust_sub_axes(ax_main, ax_sub, shrink, size=(0, 0),lr=1.0, ud=1.0):
    '''
    将ax_sub调整到ax_main的右下角.shrink指定缩小倍数。
    当ax_sub是GeoAxes时,需要在其设定好范围后再使用此函数
    '''
    bbox_main = ax_main.get_position()
    bbox_sub = ax_sub.get_position()
    wratio = bbox_main.width / bbox_sub.width
    hratio = bbox_main.height / bbox_sub.height
    if size == (0, 0):
        wnew = bbox_sub.width * shrink
        hnew = bbox_sub.height * shrink
    else:
        wnew, hnew = bbox_sub.width * shrink, bbox_sub.height * shrink * size[1] / size[0]
    bbox_new = mtransforms.Bbox.from_extents(
        bbox_main.x1 - lr*wnew, bbox_main.y0 + (ud-1)*hnew,
        bbox_main.x1 - (lr-1)*wnew, bbox_main.y0 + ud*hnew
    )
    ax_sub.set_position(bbox_new)


# 数据读取
q = xr.open_dataset(r"C:/Users/10574/OneDrive/File/Graduation Thesis/ThesisData/CN05.1/CN05.1_Tmax_1961_2021_daily_025x025.nc")
lon = q['lon'].sel(lon=slice(105, 124))
lat = q['lat'].sel(lat=slice(15, 35))
q_sort95 = q['tmax'].sel(time=slice('1979-01-01', '2014-12-31')).quantile(0.95, dim='time')
q_8009 = q['tmax'].sel(time=slice('1979-01-01', '2014-12-31'))
q_8009_78 = q_8009.sel(time=q_8009.time.dt.month.isin([7, 8]))
# 选取数据中1979-2021年每年6月的数据
try:
    std_q78 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_q78.nc')
except:
    ###
    q_78 = np.zeros([36, 81, 77])
    # 将数组元素变为-9999
    q_78 = np.full_like(q_78, np.nan)
    for yr in range(1979, 2015):
        print(f"进度:{yr-1979}/36")
        for long in range(len(lon)):
            for lati in range(len(lat)):
                q_78[yr-1979, lati, long] = np.sum(q_8009_78.sel(time=q_8009_78['time.year'] == yr).sel(lon=lon[long], lat=lat[lati]) > q_sort95.sel(lon=lon[long], lat=lat[lati]))
    std_q78 = (q_78 - np.mean(q_78, axis=0)) / np.std(q_78, axis=0)
    std_q78 = xr.Dataset({'tmax': (('time', 'lat', 'lon'), std_q78)}, coords={'time': np.arange(1979, 2015), 'lat': lat, 'lon': lon})
    std_q78.to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_q78.nc')
# eof分解
shp = fr"D:/CODES/Python/PythonProject/map/shp/south_china/中国南方.shp"
split_shp = gpd.read_file(shp)
split_shp.crs = 'wgs84'
std_q78 = std_q78.salem.roi(shape=split_shp)
DBATP = r"D:\CODES\Python\PythonProject\map\DBATP\DBATP_Polygon.shp"
provinces = cfeature.ShapelyFeature(Reader(DBATP).geometries(), crs=ccrs.PlateCarree(), facecolor='gray', alpha=1)

eof_78 = Eof(std_q78['tmax'].to_numpy())   #进行eof分解
EOF_78 = eof_78.eofs(eofscaling=2, neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
PC_78 = eof_78.pcs(pcscaling=1, npcs=2)  # 同上 npcs决定输出的时间序列个数
s_78 = eof_78.varianceFraction(neigs=2)   # 得到前neig个模态的方差贡献

# 850hPa风场高度场
wh = xr.open_dataset(r"C:/Users/10574/OneDrive/File/Graduation Thesis/ThesisData/ERA5/ERA5_Geo_U_V_200_500_850.nc")
lon_wh = wh['longitude'].sel(longitude=slice(70, 145))
lat_wh = wh['latitude'].sel(latitude=slice(55, 15))
h_8009 = wh['z'].sel(time=slice('1979-01-01', '2014-12-31')).sel(level=850)
h_8009_78 = h_8009.sel(time=h_8009.time.dt.month.isin([7, 8])).sel(longitude=slice(70, 145)).sel(latitude=slice(55, 15))
u_8009 = wh['u'].sel(time=slice('1979-01-01', '2014-12-31')).sel(level=850)
u_8009_78 = u_8009.sel(time=u_8009.time.dt.month.isin([7, 8])).sel(longitude=slice(70, 145)).sel(latitude=slice(55, 15))
v_8009 = wh['v'].sel(time=slice('1979-01-01', '2014-12-31')).sel(level=850)
v_8009_78 = v_8009.sel(time=v_8009.time.dt.month.isin([7, 8])).sel(longitude=slice(70, 145)).sel(latitude=slice(55, 15))
#将h,u,v七八月份数据进行每年平均
h_8009_78 = h_8009_78.groupby('time.year').mean('time')
u_8009_78 = u_8009_78.groupby('time.year').mean('time')
v_8009_78 = v_8009_78.groupby('time.year').mean('time')
'''# 标准化
std_h78 = (h_8009_78 - np.mean(h_8009_78, axis=0)) / np.std(h_8009_78, axis=0)
std_u78 = (u_8009_78 - np.mean(u_8009_78, axis=0)) / np.std(u_8009_78, axis=0)
std_v78 = (v_8009_78 - np.mean(v_8009_78, axis=0)) / np.std(v_8009_78, axis=0)
std_h78 = np.array(std_h78)
std_u78 = np.array(std_u78)
std_v78 = np.array(std_v78)'''

std_h78 = np.array(h_8009_78)
std_u78 = np.array(u_8009_78)
std_v78 = np.array(v_8009_78)

# 计算h,u,v分别与主成分PC1的简单相关系数
r_h78 = np.zeros((len(lat_wh), len(lon_wh)))
r_u78 = np.zeros((len(lat_wh), len(lon_wh)))
r_v78 = np.zeros((len(lat_wh), len(lon_wh)))
for i in range(len(lat_wh)):
    for j in range(len(lon_wh)):
        r_h78[i, j] = np.corrcoef(std_h78[:, i, j], PC_78[:, 0])[0, 1]
        r_u78[i, j] = np.corrcoef(std_u78[:, i, j], PC_78[:, 0])[0, 1]
        r_v78[i, j] = np.corrcoef(std_v78[:, i, j], PC_78[:, 0])[0, 1]
#计算h,u,v分别与主成分PC2的简单相关系数
r_h78_2 = np.zeros((len(lat_wh), len(lon_wh)))
r_u78_2 = np.zeros((len(lat_wh), len(lon_wh)))
r_v78_2 = np.zeros((len(lat_wh), len(lon_wh)))
for i in range(len(lat_wh)):
    for j in range(len(lon_wh)):
        r_h78_2[i, j] = np.corrcoef(std_h78[:, i, j], PC_78[:, 1])[0, 1]
        r_u78_2[i, j] = np.corrcoef(std_u78[:, i, j], PC_78[:, 1])[0, 1]
        r_v78_2[i, j] = np.corrcoef(std_v78[:, i, j], PC_78[:, 1])[0, 1]
#筛选出相关系数绝对值大于0.35(uv其一大于即可)的格点
U1threshold = np.where(np.abs(r_u78) > 0.35, 1, 0)
V1threshold = np.where(np.abs(r_v78) > 0.35, 1, 0)
UV1threshold = np.where((U1threshold + V1threshold) >= 1, 1, 0)
r_u78 = np.where(UV1threshold == 1, r_u78, np.nan)
r_v78 = np.where(UV1threshold == 1, r_v78, np.nan)

U2threshold = np.where(np.abs(r_u78_2) > 0.35, 1, 0)
V2threshold = np.where(np.abs(r_v78_2) > 0.35, 1, 0)
UV2threshold = np.where((U2threshold + V2threshold) >= 1, 1, 0)
r_u78_2 = np.where(UV2threshold == 1, r_u78_2, np.nan)
r_v78_2 = np.where(UV2threshold == 1, r_v78_2, np.nan)

'''# eof分解
eof_h78 = Eof(std_h78)   #进行eof分解
EOF_h78 = eof_h78.eofs(eofscaling=2, neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
PC_h78 = eof_h78.pcs(pcscaling=1, npcs=2)  # 同上 npcs决定输出的时间序列个数
s_h78 = eof_h78.varianceFraction(neigs=2)   # 得到前neig个模态的方差贡献
# eof分解
eof_u78 = Eof(std_u78)   #进行eof分解
EOF_u78 = eof_u78.eofs(eofscaling=2, neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
PC_u78 = eof_u78.pcs(pcscaling=1, npcs=2)  # 同上 npcs决定输出的时间序列个数
s_u78 = eof_u78.varianceFraction(neigs=2)   # 得到前neig个模态的方差贡献
# eof分解
eof_v78 = Eof(std_v78)   #进行eof分解
EOF_v78 = eof_v78.eofs(eofscaling=2, neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
PC_v78 = eof_v78.pcs(pcscaling=1, npcs=2)  # 同上 npcs决定输出的时间序列个数
s_v78 = eof_v78.varianceFraction(neigs=2)   # 得到前neig个模态的方差贡献
'''
# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
#plt.subplots_adjust(wspace=0.01, hspace=0.1)#wspace、hspace左右、上下的间距
plt.tight_layout()
font = {'family' : 'Arial','weight' : 'bold','size' : 12}
extent1=[70, 145, 15, 55]  # 经度范围，纬度范围
xticks1=np.arange(extent1[0], extent1[1]+1, 20)
yticks1=np.arange(extent1[2], extent1[3]+1, 5)
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(15, 13))
# ##修正前
level1 = [-1.+i*0.1 for i in range(21)]
level2 = [-0.4 + i * 0.2 for i in range(5)]
ax1 = fig.add_subplot(221, projection=proj)
ax1.set_extent(extent1, crs=proj)
# 标题水平居左
ax1.set_title('(a)EOF1', fontsize=20, loc='left')
a1 = ax1.contourf(lon, lat, EOF_78[0, :, :], cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='neither', transform=proj)
r_h78 = filters.gaussian_filter(r_h78, 4)
a1_h = ax1.contour(lon_wh, lat_wh, r_h78, transform=proj, levels=level2[:2], colors='blue', linewidths=1.5, linestyles='--',alpha=1)
a1_hm = ax1.contour(lon_wh, lat_wh, r_h78, transform=proj, levels=[0], colors='gray', linewidths=1.5, linestyles='-', alpha=1)
a1_h1 = ax1.contour(lon_wh, lat_wh, r_h78, transform=proj, levels=level2[3:], colors='red', linewidths=1.5, linestyles='-', alpha=1)
#创建等值线标签，并将字体设为'Times New Roman'，大小为5
ax1.clabel(a1_h, inline=True, fontsize=14, fmt='%.01f', colors='blue')
ax1.clabel(a1_hm, inline=True, fontsize=14, fmt='%.01f', colors='gray')
ax1.clabel(a1_h1, inline=True, fontsize=14, fmt='%.01f', colors='red')
# 画风场,且箭头稀疏
n = 6
a1_uv = ax1.quiver(lon_wh[::n], lat_wh[::n], r_u78[::n, ::n], r_v78[::n, ::n], transform=proj, pivot='mid',scale=25,
                   headlength=3,headaxislength=3)
ax1.quiverkey(a1_uv,  X=0.946, Y=1.03, U=.5,angle = 0,  label='0.5m/s',
              labelpos='N', color='black',labelcolor = 'k', fontproperties = font,linewidth=0.8)#linewidth=1为箭头的大小
ax1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
ax1.text(127, 30.45, 'A', fontsize=20, fontweight='bold', color='blue', zorder=20)
ax1.add_feature(provinces, lw=0.5, zorder=2)
draw_maps(get_adm_maps(level='国'), linewidth=0.5)

ax3 = fig.add_subplot(222, projection=proj)
ax3.set_extent(extent1, crs=proj)
ax3.set_title('(c)EOF2', fontsize=20, loc='left')
a3 = ax3.contourf(lon, lat, EOF_78[1, :, :], cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='neither', transform=proj)
r_h78_2 = filters.gaussian_filter(r_h78_2, 4)
a3_h = ax3.contour(lon_wh, lat_wh, r_h78_2, transform=proj, levels=level2[:2], colors='blue', linewidths=1.5, linestyles='--', alpha=1)
a3_hm = ax3.contour(lon_wh, lat_wh, r_h78_2, transform=proj, levels=[0], colors='gray', linewidths=1.5, linestyles='-', alpha=1)
a3_h1 = ax3.contour(lon_wh, lat_wh, r_h78_2, transform=proj, levels=level2[3:], colors='red', linewidths=1.5, linestyles='-', alpha=1)
ax3.clabel(a3_h, inline=True, fontsize=14, fmt='%.01f', colors='blue')
ax3.clabel(a3_hm, inline=True, fontsize=14, fmt='%.01f', colors='gray')
ax3.clabel(a3_h1, inline=True, fontsize=14, fmt='%.01f', colors='red')
# 画风场,且箭头稀疏
a3_uv = ax3.quiver(lon_wh[::n], lat_wh[::n], r_u78_2[::n, ::n], r_v78_2[::n, ::n], transform=proj, pivot='mid',scale=25,
                   headlength=3,headaxislength=3)
ax3.quiverkey(a3_uv,  X=0.946, Y=1.03, U=.5, angle=0,  label='0.5m/s',
              labelpos='N', color='black', labelcolor='k', fontproperties=font, linewidth=0.8)#linewidth=1为箭头的大小
ax3.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
ax3.text(137.5, 51.6, 'A', fontsize=20, fontweight='bold', color='blue', zorder=20)
ax3.text(130, 35, 'C', fontsize=20, fontweight='bold', color='red', zorder=20)
ax3.text(115, 21, 'A', fontsize=20, fontweight='bold', color='blue', zorder=20)
ax3.add_feature(provinces, lw=0.5, zorder=2)
draw_maps(get_adm_maps(level='国'), linewidth=0.5)

ax2 = fig.add_axes(ax1.get_position())
ax2.set_title(f'(b)PC1', fontsize=20, loc='left')
ax2.set_title(f'{s_78[0]*100:.2f}%', fontsize=20, loc='right')
x = np.arange(1979, 2015, 1)
a2 = ax2.bar(x, PC_78[:, 0], color='crimson')
for bar,height in zip(a2, PC_78[:, 0]):
    if height < 0:
        bar.set(color='dodgerblue')
ax2.set_xlim([1978, 2015])
ax2.set_ylim([-3, 3])
ax2.set_xticks(np.arange(1979, 2015, 5))
ax2.axhline(0, color='darkgray', linewidth=1)
#设定子图ax2大小位置
adjust_sub_axes(ax1, ax2, shrink=1, size=(16, 15), lr=1.0, ud=-.5)


ax4 = fig.add_axes(ax3.get_position())
ax4.set_title(f'(d)PC2', fontsize=20, loc='left')
ax4.set_title(f'{s_78[1]*100:.2f}%', fontsize=20, loc='right')
x = np.arange(1979, 2015, 1)
a4 = ax4.bar(x, PC_78[:, 1], color='crimson')
for bar, height in zip(a4, PC_78[:, 1]):
    if height < 0:
        bar.set(color='dodgerblue')
ax4.set_xlim([1978, 2015])
ax4.set_ylim([-3.0, 3.0])
ax4.set_xticks(np.arange(1979, 2015, 5))
ax4.axhline(0, color='darkgray', linewidth=1)
adjust_sub_axes(ax3, ax4, shrink=1, size=(16, 15), lr=1.0, ud=-.5)

# 画图设置
# 刻度线设置
ax1.set_xticks(xticks1, crs=proj)
ax1.set_yticks(yticks1, crs=proj)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
#
ax3.set_xticks(xticks1, crs=proj)
ax3.set_yticks(yticks1, crs=proj)
ax3.xaxis.set_major_formatter(lon_formatter)
ax3.yaxis.set_major_formatter(lat_formatter)
font = {'family': 'Arial', 'weight': 'bold', 'size': 28}

xmajorLocator = MultipleLocator(20)#先定义xmajorLocator，再进行调用
xminorLocator = MultipleLocator(10)
ax1.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ax3.xaxis.set_major_locator(xmajorLocator)
ax3.xaxis.set_minor_locator(xminorLocator)
ymajorLocator = MultipleLocator(5)
yminorLocator = MultipleLocator(2.5)
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
ax3.yaxis.set_major_locator(ymajorLocator)
ax3.yaxis.set_minor_locator(yminorLocator)
#设置ax2 ax4坐标刻度值的大小以及刻度值的字体
xmajorLocator = MultipleLocator(5)#先定义xmajorLocator，再进行调用
xminorLocator = MultipleLocator(1)
ax2.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
ax2.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ax4.xaxis.set_major_locator(xmajorLocator)
ax4.xaxis.set_minor_locator(xminorLocator)
ymajorLocator = MultipleLocator(1)
yminorLocator = MultipleLocator(0.2)
ax2.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax2.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
ax4.yaxis.set_major_locator(ymajorLocator)
ax4.yaxis.set_minor_locator(yminorLocator)
#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=22)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
font2 = {'family': 'Arial', 'weight': 'bold','size' : 28}
# 调整刻度值字体大小
ax1.tick_params(axis='both', labelsize=20, colors='black')
ax2.tick_params(axis='both', labelsize=20, colors='black')
ax3.tick_params(axis='both', labelsize=20, colors='black')
ax4.tick_params(axis='both', labelsize=20, colors='black')
#最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=8,width=1.8,color='darkgray')#最小刻度长度，宽度设置
ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
ax2.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax2.tick_params(which='minor', length=5,width=1,color='darkgray')#最小刻度长度，宽度设置
ax2.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
ax3.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax3.tick_params(which='minor', length=8,width=1.8,color='darkgray')#最小刻度长度，宽度设置
ax3.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
ax4.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax4.tick_params(which='minor', length=5,width=1,color='darkgray')#最小刻度长度，宽度设置
ax4.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外
# color bar位置
position = fig.add_axes([0.92, 0.605, 0.01, 0.2])#位置[x,y,width,height][0.296, 0.05, 0.44, 0.015]
#竖向colorbar,无尖角
cb1 = plt.colorbar(a1, cax=position, orientation='vertical')#orientation为水平或垂直
cb1.ax.tick_params(color='black')#length为刻度线的长度
cb1.ax.tick_params(which='major',direction='in', labelsize=12, length=11)
cb1.ax.tick_params(which='minor',direction='in', length=11)
cb1.ax.yaxis.set_minor_locator(MultipleLocator(.1))#显示x轴副刻度
cb1.locator = ticker.FixedLocator([-1., -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1.])  # colorbar上的刻度值个数

plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\图3.png', dpi=1000, bbox_inches='tight')
plt.show()
