from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as path_effects
import matplotlib.path as mpath
from cnmaps import get_adm_maps, draw_maps
from matplotlib import ticker
import cmaps
from matplotlib.ticker import MultipleLocator, FixedLocator
from eofs.standard import Eof
from scipy.ndimage import filters
from tqdm import tqdm
import geopandas as gpd
import salem
from tools.TN_WaveActivityFlux import TN_WAF
import pprint


std_q78 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_q78.nc')
shp = fr"D:/CODES/Python/PythonProject/map/shp/south_china/中国南方.shp"
split_shp = gpd.read_file(shp)
split_shp.crs = 'wgs84'
std_q78 = std_q78.salem.roi(shape=split_shp)
pc_index = 0
# eof分解
eof_78 = Eof(std_q78['tmax'].to_numpy())  # 进行eof分解
EOF_78 = eof_78.eofs(eofscaling=2,
                     neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
PC_78 = eof_78.pcs(pcscaling=1, npcs=2)  # 同上 npcs决定输出的时间序列个数
s_78 = eof_78.varianceFraction(neigs=2)  # 得到前neig个模态的方差贡献
# 数据读取
sat = xr.open_dataset(r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\ERA5\ERA5_2mTemperature_MeanSlp.nc")

# 数据切片
T2m = sat['t2m'].sel(time=slice('1979-01-01', '2014-12-31'))
T2m_78 = T2m.sel(time=T2m.time.dt.month.isin([7, 8]))
# 经纬度
lon_sat = T2m['longitude']
lat_sat = T2m['latitude']

# 将七八月份数据进行每年平均
sat_78 = T2m_78.groupby('time.year').mean('time')
sat_78 = np.array(sat_78)
try:
    # 读取相关系数
    reg_sat = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_sat.nc')
except:
    # 将数据回归到PC上
    reg_sat = [[np.polyfit(PC_78[:, pc_index], sat_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_sat))] for ilat in tqdm(range(len(lat_sat)), desc='计算SAT', position=0, leave=True)]
    xr.DataArray(reg_sat, coords=[lat_sat, lon_sat], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_sat.nc')
    ###数据再读取
    reg_sat = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_sat.nc')

# 进行显著性0.05检验
from scipy.stats import t

# 计算自由度.
n = len(PC_78[:, 0])
# 使用t检验计算回归系数的的显著性
# 计算t值
Lxx = np.sum((PC_78[:, pc_index] - np.mean(PC_78[:, pc_index])) ** 2)
# SST
Sr_sat = reg_sat**2 * Lxx
St_sat = np.sum((sat_78 - np.mean(sat_78, axis=0)) ** 2, axis=0)
σ_sat = np.sqrt((St_sat - Sr_sat) / (n - 2))
t_sat = reg_sat * np.sqrt(Lxx) / σ_sat

# 计算临界值
t_critical = t.ppf(0.975, n - 2)
# 进行显著性检验
p_sat78 = np.zeros((len(lat_sat), len(lon_sat)))
p_sat78.fill(np.nan)
p_sat78[np.abs(t_sat['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1

# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.subplots_adjust(wspace=0.1, hspace=0.001)  # wspace、hspace左右、上下的间距
font = {'family' : 'Arial','weight' : 'bold','size' : 12}
# plt.subplots_adjust(wspace=0.1, hspace=0.32)  # wspace、hspace左右、上下的间距
extent1 = [0, 360, 50, 90]  # 经度范围，纬度范围
xticks1 = np.arange(extent1[0], extent1[1] + 1, 10)
yticks1 = np.arange(extent1[2], extent1[3] + 1, 10)

fig = plt.figure(figsize=(10, 10))

# ##ax1 Corr. PC1 & JA SST,2mT
level1 = [-1, -.7, -.4, -.1, -.05, .05, .1, .4, .7, 1]
ax1 = fig.add_subplot(1,1,1, projection=ccrs.NorthPolarStereo(central_longitude=90))
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
# 去除fill_value 1e+20
# 去除180白线!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!加上截距
reg_sat, a1_sat_lon = add_cyclic_point(reg_sat['__xarray_dataarray_variable__'].to_numpy(), coord=lon_sat)
print('开始绘制地图1')
ax1.set_title('(a)Reg 2mT', fontsize=20, loc='left')
a1 = ax1.contourf(a1_sat_lon, lat_sat, reg_sat, cmap=cmaps.GMT_polar[4:10]+cmaps.CBR_wet[0]+cmaps.GMT_polar[10:16], levels=level1, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
ax1.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.3)  # 添加海岸线
ax1.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black',linewidth=2)

p_sat, a1_lon_sat = add_cyclic_point(p_sat78, coord=lon_sat)
p_sat = np.where(p_sat == 1, 0, np.nan)
# 显著性打点
a1_uv = ax1.quiver(a1_lon_sat, lat_sat, p_sat, p_sat, scale=20, color='black', headlength=3,
                   regrid_shape=60, headaxislength=3, transform=ccrs.PlateCarree(central_longitude=0), width=0.005)
# 框选预测因子
ax1.plot([70, 70, 50, 50, 70], [33, 64.4, 64.4, 33, 33], color='red', linewidth=2, linestyle='-',transform=ccrs.PlateCarree(central_longitude=90))
grid_lon = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='grey',linestyle='--')
grid_lon.xlocator = FixedLocator(np.linspace(-180,180,13))
grid_lon.ylocator = FixedLocator([65, 80])
grid_lon.xlabel_style = {'size': 14}
grid_lon.ylabel_style = {'size': 0}

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax1.set_boundary(circle, transform=ax1.transAxes)

'''# 刻度线设置
ax1.set_xticks(xticks1, crs=proj)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
font = {'family': 'Arial', 'weight': 'bold', 'size': 28}

xmajorLocator = MultipleLocator(60)  # 先定义xmajorLocator，再进行调用
ax1.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
xminorLocator = MultipleLocator(10)
ax1.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度

# ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
# 最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=11, width=2, color='darkgray')  # 最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=8, width=1.8, color='darkgray')  # 最小刻度长度，宽度设置
ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
# 调整刻度值字体大小
ax1.tick_params(axis='both', labelsize=28, colors='black')
# 设置坐标刻度值的大小以及刻度值的字体
labels = ax1.get_xticklabels()
[label.set_fontname('Arial') for label in labels]
font2 = {'family': 'Arial', 'weight': 'bold', 'size': 28}'''

# color bar位置
# position = fig.add_axes([0.296, 0.08, 0.44, 0.011])#位置[左,下,右,上]
position1 = fig.add_axes([0.146, 0.01, 0.74, 0.03])
cb1 = plt.colorbar(a1, cax=position1, orientation='horizontal')  # orientation为水平或垂直
cb1.ax.tick_params(length=1, labelsize=14)  # length为刻度线的长度
cb1.locator = ticker.FixedLocator([-1, -.7, -.4, -.1, 0, .1, .4, .7, 1]) # colorbar上的刻度值个数


plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\2m温度观测场回归.png', dpi=1000, bbox_inches='tight')
plt.show()
