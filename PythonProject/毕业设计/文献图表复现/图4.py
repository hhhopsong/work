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
from cnmaps import get_adm_maps, draw_maps
from matplotlib import ticker
import cmaps
from matplotlib.ticker import MultipleLocator
from eofs.standard import Eof
from tqdm import tqdm
import geopandas as gpd
import salem

import pprint


std_q78 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_q78.nc')
shp = fr"D:/CODES/Python/PythonProject/map/shp/south_china/中国南方.shp"
split_shp = gpd.read_file(shp)
split_shp.crs = 'wgs84'
std_q78 = std_q78.salem.roi(shape=split_shp)
# eof分解
eof_78 = Eof(std_q78['tmax'].to_numpy())  # 进行eof分解
EOF_78 = eof_78.eofs(eofscaling=2,
                     neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
PC_78 = eof_78.pcs(pcscaling=1, npcs=2)  # 同上 npcs决定输出的时间序列个数
s_78 = eof_78.varianceFraction(neigs=2)  # 得到前neig个模态的方差贡献
# 数据读取
mlp_T2m = xr.open_dataset(r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\ERA5\ERA5_2mTemperature_MeanSlp.nc")
sst = xr.open_dataset(r"C:\Users\10574\Desktop\data\sst.mnmean.nc")  # NetCDF-4文件路径不可含中文
gpcp = xr.open_dataset(r"C:\Users\10574\Desktop\data\precip.mon.mean.nc")
wh = xr.open_dataset(r"C:/Users/10574/OneDrive/File/Graduation Thesis/ThesisData/ERA5/ERA5_Geo_U_V_200_500_850.nc")
# 数据切片
T2m = mlp_T2m['t2m'].sel(time=slice('1979-01-01', '2014-12-31'))
slp_2mT_78 = T2m.sel(time=T2m.time.dt.month.isin([7, 8]))
sst = sst['sst'].sel(time=slice('1979-01-01', '2014-12-31'))
sst = sst.sel(time=sst.time.dt.month.isin([7, 8]))
pre = gpcp['precip'].sel(time=slice('1979-01-01', '2014-12-31'))
pre = pre.sel(time=pre.time.dt.month.isin([7, 8]))
u850 = wh['u'].sel(level=850, time=slice('1979-01-01', '2014-12-31'))
u850 = u850.sel(time=u850.time.dt.month.isin([7, 8]))
v850 = wh['v'].sel(level=850, time=slice('1979-01-01', '2014-12-31'))
v850 = v850.sel(time=v850.time.dt.month.isin([7, 8]))
z850 = wh['z'].sel(level=850, time=slice('1979-01-01', '2014-12-31'))
z850 = z850.sel(time=z850.time.dt.month.isin([7, 8]))
u500 = wh['u'].sel(level=500, time=slice('1979-01-01', '2014-12-31'))
u500 = u500.sel(time=u500.time.dt.month.isin([7, 8]))
v500 = wh['v'].sel(level=500, time=slice('1979-01-01', '2014-12-31'))
v500 = v500.sel(time=v500.time.dt.month.isin([7, 8]))
z500 = wh['z'].sel(level=500, time=slice('1979-01-01', '2014-12-31'))
z500 = z500.sel(time=z500.time.dt.month.isin([7, 8]))
u200 = wh['u'].sel(level=200, time=slice('1979-01-01', '2014-12-31'))
u200 = u200.sel(time=u200.time.dt.month.isin([7, 8]))
v200 = wh['v'].sel(level=200, time=slice('1979-01-01', '2014-12-31'))
v200 = v200.sel(time=v200.time.dt.month.isin([7, 8]))
z200 = wh['z'].sel(level=200, time=slice('1979-01-01', '2014-12-31'))
z200 = z200.sel(time=z200.time.dt.month.isin([7, 8]))
# 经纬度
lon_T2m = T2m['longitude']
lat_T2m = T2m['latitude']
lon_sst = sst['lon']
lat_sst = sst['lat']
lon_pre = pre['lon']
lat_pre = pre['lat']
lon_uvz = u850['longitude']
lat_uvz = u850['latitude']

try:
    # 读取相关系数
    r_t2m78 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_2mT.npy')
    r_t2m78_2 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_2mT.npy')
    r_sst78 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_sst.npy')
    r_sst78_2 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_sst.npy')
    r_pre78 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_pre.npy')
    r_pre78_2 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_pre.npy')
    r_u85078 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_u850.npy')
    r_u85078_2 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_u850.npy')
    r_v85078 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_v850.npy')
    r_v85078_2 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_v850.npy')
    r_z50078 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_z500.npy')
    r_z50078_2 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_z500.npy')
except:
    #将七八月份数据进行每年平均
    slp_2mT_78 = slp_2mT_78.groupby('time.year').mean('time')
    sst_78 = sst.groupby('time.year').mean('time')
    pre_78 = pre.groupby('time.year').mean('time')
    u850_78 = u850.groupby('time.year').mean('time')
    v850_78 = v850.groupby('time.year').mean('time')
    z850_78 = z850.groupby('time.year').mean('time')
    u500_78 = u500.groupby('time.year').mean('time')
    v500_78 = v500.groupby('time.year').mean('time')
    z500_78 = z500.groupby('time.year').mean('time')
    u200_78 = u200.groupby('time.year').mean('time')
    v200_78 = v200.groupby('time.year').mean('time')
    z200_78 = z200.groupby('time.year').mean('time')
    slp_2mT_78 = np.array(slp_2mT_78)
    sst_78 = np.array(sst_78)
    pre_78 = np.array(pre_78)
    u850_78 = np.array(u850_78)
    v850_78 = np.array(v850_78)
    z850_78 = np.array(z850_78)
    u500_78 = np.array(u500_78)
    v500_78 = np.array(v500_78)
    z500_78 = np.array(z500_78)
    u200_78 = np.array(u200_78)
    v200_78 = np.array(v200_78)
    z200_78 = np.array(z200_78)
    # 将数据回归到PC上
    r_t2m78 = np.zeros((len(lat_T2m), len(lon_T2m)))
    r_sst78 = np.zeros((len(lat_sst), len(lon_sst)))
    r_pre78 = np.zeros((len(lat_pre), len(lon_pre)))
    r_u85078 = np.zeros((len(lat_uvz), len(lon_uvz)))
    r_v85078 = np.zeros((len(lat_uvz), len(lon_uvz)))
    r_z50078 = np.zeros((len(lat_uvz), len(lon_uvz)))
    for i in tqdm(range(len(lat_T2m)), desc='计算T2m和PC1的相关系数', position=0, leave=True):
        for j in range(len(lon_T2m)):
            r_t2m78[i, j] = np.corrcoef(std_slp_2mT_78[:, i, j], PC_78[:, 0])[0, 1]
    for i in tqdm(range(len(lat_sst)), desc='计算SST和PC1的相关系数', position=0, leave=True):
        for j in range(len(lon_sst)):
            r_sst78[i, j] = np.corrcoef(std_sst_78[:, i, j], PC_78[:, 0])[0, 1]
    for i in tqdm(range(len(lat_pre)), desc='计算PRE和PC1的相关系数', position=0, leave=True):
        for j in range(len(lon_pre)):
            r_pre78[i, j] = np.corrcoef(std_pre_78[:, i, j], PC_78[:, 0])[0, 1]
    for i in tqdm(range(len(lat_uvz)), desc='计算U850和PC1的相关系数', position=0, leave=True):
        for j in range(len(lon_uvz)):
            r_u85078[i, j] = np.corrcoef(std_u850_78[:, i, j], PC_78[:, 0])[0, 1]
    for i in tqdm(range(len(lat_uvz)), desc='计算V850和PC1的相关系数', position=0, leave=True):
        for j in range(len(lon_uvz)):
            r_v85078[i, j] = np.corrcoef(std_v850_78[:, i, j], PC_78[:, 0])[0, 1]
    for i in tqdm(range(len(lat_uvz)), desc='计算Z500和PC1的相关系数', position=0, leave=True):
        for j in range(len(lon_uvz)):
            r_z50078[i, j] = np.corrcoef(std_z500_78[:, i, j], PC_78[:, 0])[0, 1]
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_2mT.npy', r_t2m78)
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_sst.npy', r_sst78)
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_pre.npy', r_pre78)
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_u850.npy', r_u85078)
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_v850.npy', r_v85078)
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC1_z500.npy', r_z50078)
    #计算和PC2的相关系数
    print('计算和PC2的相关系数中......')
    r_t2m78_2 = np.zeros((len(lat_T2m), len(lon_T2m)))
    r_sst78_2 = np.zeros((len(lat_sst), len(lon_sst)))
    r_pre78_2 = np.zeros((len(lat_pre), len(lon_pre)))
    r_u85078_2 = np.zeros((len(lat_uvz), len(lon_uvz)))
    r_v85078_2 = np.zeros((len(lat_uvz), len(lon_uvz)))
    r_z50078_2 = np.zeros((len(lat_uvz), len(lon_uvz)))
    for i in tqdm(range(len(lat_T2m)), desc='计算T2m和PC2的相关系数', position=0, leave=True):
        for j in range(len(lon_T2m)):
            r_t2m78_2[i, j] = np.corrcoef(std_slp_2mT_78[:, i, j], PC_78[:, 1])[0, 1]
    for i in tqdm(range(len(lat_sst)), desc='计算SST和PC2的相关系数', position=0, leave=True):
        for j in range(len(lon_sst)):
            r_sst78_2[i, j] = np.corrcoef(std_sst_78[:, i, j], PC_78[:, 1])[0, 1]
    for i in tqdm(range(len(lat_pre)), desc='计算PRE和PC2的相关系数', position=0, leave=True):
        for j in range(len(lon_pre)):
            r_pre78_2[i, j] = np.corrcoef(std_pre_78[:, i, j], PC_78[:, 1])[0, 1]
    for i in tqdm(range(len(lat_uvz)), desc='计算U850和PC2的相关系数', position=0, leave=True):
        for j in range(len(lon_uvz)):
            r_u85078_2[i, j] = np.corrcoef(std_u850_78[:, i, j], PC_78[:, 1])[0, 1]
    for i in tqdm(range(len(lat_uvz)), desc='计算V850和PC2的相关系数', position=0, leave=True):
        for j in range(len(lon_uvz)):
            r_v85078_2[i, j] = np.corrcoef(std_v850_78[:, i, j], PC_78[:, 1])[0, 1]
    for i in tqdm(range(len(lat_uvz)), desc='计算Z500和PC2的相关系数', position=0, leave=True):
        for j in range(len(lon_uvz)):
            r_z50078_2[i, j] = np.corrcoef(std_z500_78[:, i, j], PC_78[:, 1])[0, 1]
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_2mT.npy', r_t2m78_2)
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_sst.npy', r_sst78_2)
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_pre.npy', r_pre78_2)
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_u850.npy', r_u85078_2)
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_v850.npy', r_v85078_2)
    np.save(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\PC2_z500.npy', r_z50078_2)

# 进行显著性0.05检验
from scipy.stats import t

# 计算自由度
n = len(PC_78[:, 0])
# 计算t值
t_t2m78 = r_t2m78 * np.sqrt((n - 2) / (1 - r_t2m78 ** 2))
t_t2m78_2 = r_t2m78_2 * np.sqrt((n - 2) / (1 - r_t2m78_2 ** 2))
t_sst78 = r_sst78 * np.sqrt((n - 2) / (1 - r_sst78 ** 2))
t_sst78_2 = r_sst78_2 * np.sqrt((n - 2) / (1 - r_sst78_2 ** 2))
t_pre78 = r_pre78 * np.sqrt((n - 2) / (1 - r_pre78 ** 2))
t_pre78_2 = r_pre78_2 * np.sqrt((n - 2) / (1 - r_pre78_2 ** 2))
t_u85078 = r_u85078 * np.sqrt((n - 2) / (1 - r_u85078 ** 2))
t_u85078_2 = r_u85078_2 * np.sqrt((n - 2) / (1 - r_u85078_2 ** 2))
t_v85078 = r_v85078 * np.sqrt((n - 2) / (1 - r_v85078 ** 2))
t_v85078_2 = r_v85078_2 * np.sqrt((n - 2) / (1 - r_v85078_2 ** 2))
t_z50078 = r_z50078 * np.sqrt((n - 2) / (1 - r_z50078 ** 2))
t_z50078_2 = r_z50078_2 * np.sqrt((n - 2) / (1 - r_z50078_2 ** 2))
# 计算临界值
t_critical = t.ppf(0.975, n - 2)
# 进行显著性检验
p_t2m78 = np.zeros((len(lat_T2m), len(lon_T2m)))
p_t2m78.fill(np.nan)
p_t2m78[np.abs(t_t2m78) > t_critical] = 1

p_sst78 = np.zeros((len(lat_sst), len(lon_sst)))
p_sst78.fill(np.nan)
p_sst78[np.abs(t_sst78) > t_critical] = 1

p_pre78 = np.zeros((len(lat_pre), len(lon_pre)))
p_pre78.fill(np.nan)
p_pre78[np.abs(t_pre78) > t_critical] = 1

p_z50078 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_z50078.fill(np.nan)
p_z50078[np.abs(t_z50078) > t_critical] = 1

# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.subplots_adjust(wspace=0.1, hspace=0.001)  # wspace、hspace左右、上下的间距
# plt.subplots_adjust(wspace=0.1, hspace=0.32)  # wspace、hspace左右、上下的间距
extent1 = [-180, 180, -30, 80]  # 经度范围，纬度范围
xticks1 = np.arange(extent1[0], extent1[1] + 1, 10)
yticks1 = np.arange(extent1[2], extent1[3] + 1, 10)
proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(16, 22))
# ##ax1 Corr. PC1 & JA SST,2mT
level1 = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
ax1 = fig.add_subplot(311, projection=proj)
ax1.set_extent(extent1, crs=proj)
# 去除180白线
r_t2m78, a1_lon = add_cyclic_point(r_t2m78, coord=lon_T2m)
r_sst78, a1_sst_lon = add_cyclic_point(r_sst78, coord=lon_sst)
p_t2m78, a1_lon_p = add_cyclic_point(p_t2m78, coord=lon_T2m)
p_sst78, a1_sst_lon_p = add_cyclic_point(p_sst78, coord=lon_sst)
print('开始绘制地图1')
ax1.set_title('(a)Corr. PC1 & JA SST,2mT', fontsize=28, loc='left')
a1 = ax1.contourf(a1_lon, lat_T2m, r_t2m78, cmap=cmaps.GMT_polar, levels=level1, extend='both',
                  transform=ccrs.PlateCarree(central_longitude=0))
a1_sst = ax1.contourf(a1_sst_lon, lat_sst, r_sst78, cmap=cmaps.GMT_polar, levels=level1, extend='both',
                      transform=ccrs.PlateCarree(central_longitude=0))
# 通过打点显示出通过显著性检验的区域
a1_p = ax1.contourf(a1_lon_p, lat_T2m, p_t2m78, levels=[0, 1], transform=ccrs.PlateCarree(central_longitude=0),
                    hatches=['..', None], colors="none", add_colorbar=False, zorder=5)
a1_sst_p = ax1.contourf(a1_sst_lon_p, lat_sst, p_sst78, levels=[0, 1], transform=ccrs.PlateCarree(central_longitude=0),
                        hatches=['..', None], colors="none", add_colorbar=False, zorder=5)
ax1.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1)  # 添加海岸线
ax1.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black',linewidth=2)


# ax2 Corr. PC2 & JA PRE,850UV
level2 = [-0.5, -0.4, -0.3, -0.2, 0.2, 0.3, 0.4, 0.5]
n = 20
radio = 0.3
print('开始绘制地图2')
ax2 = fig.add_subplot(312, projection=proj)
ax2.set_extent(extent1, crs=proj)
# 筛选出大于0.35的风场格点
U1threshold = np.where(np.abs(r_u85078) > 0.35, 1, 0)
V1threshold = np.where(np.abs(r_v85078) > 0.35, 1, 0)
UV1threshold = np.where((U1threshold + V1threshold) >= 1, 1, 0)
r_u85078 = np.where(UV1threshold == 1, r_u85078, np.nan)
r_v85078 = np.where(UV1threshold == 1, r_v85078, np.nan)
# 求PC1与uv两个自变量的复相关系数Rpc_uv
# 去除180白线
r_pre78, a2_lon = add_cyclic_point(r_pre78, coord=lon_pre)
r_u85078, a2_uv_lon = add_cyclic_point(r_u85078, coord=lon_uvz)
r_v85078, a2_uv_lon = add_cyclic_point(r_v85078, coord=lon_uvz)
p_pre78, a2_lon_p = add_cyclic_point(p_pre78, coord=lon_pre)
ax2.set_title('(b)Corr. PC1 & JA PRE,850UV', fontsize=28, loc='left')
a2 = ax2.contourf(a2_lon, lat_pre, r_pre78, cmap=cmaps.MPL_BrBG[32:96], levels=level2, extend='both',
                  transform=ccrs.PlateCarree(central_longitude=0))
a2_uv = ax2.quiver(a2_uv_lon[::n], lat_uvz[::n], r_u85078[::n, ::n], r_v85078[::n, ::n], scale=30, color='black', headlength=3,
                   headaxislength=3, transform = ccrs.PlateCarree(central_longitude=0))
a2_p = ax2.contourf(a2_lon_p, lat_pre, p_pre78, levels=[0, 1], transform=ccrs.PlateCarree(central_longitude=0),
                    hatches=['..', None], colors="none", add_colorbar=False, zorder=5)
ax2.text(127, 30.45, 'A', fontsize=28, fontweight='bold', color='blue', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax2.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1)  # 添加海岸线
ax2.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black',linewidth=2)


# ax3 Corr. PC1 & JA Z500
level3 = [-0.5, -0.4, -0.3, -0.2, 0.2, 0.3, 0.4, 0.5]
print('开始绘制地图3')
ax3 = fig.add_subplot(313, projection=proj)
ax3.set_extent(extent1, crs=proj)
r_z50078, a3_lon = add_cyclic_point(r_z50078, coord=lon_uvz)
p_z50078, a3_lon_p = add_cyclic_point(p_z50078, coord=lon_uvz)
ax3.set_title('(c)Corr. PC1 & JA Z500', fontsize=28, loc='left')
a3 = ax3.contourf(a3_lon, lat_uvz, r_z50078, cmap=cmaps.CBR_coldhot[1:10], levels=level3, extend='both',
                  transform=ccrs.PlateCarree(central_longitude=0))
a3_p = ax3.contourf(a3_lon_p, lat_uvz, p_z50078, levels=[0, 1], transform=ccrs.PlateCarree(central_longitude=0),
                    hatches=['..', None], colors="none", add_colorbar=False, zorder=5)
ax3.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1)  # 添加海岸线
ax3.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black',linewidth=2)


# 刻度线设置
ax1.set_xticks(xticks1, crs=proj)
ax1.set_yticks(yticks1, crs=proj)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
ax2.set_xticks(xticks1, crs=proj)
ax2.set_yticks(yticks1, crs=proj)
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
ax3.set_xticks(xticks1, crs=proj)
ax3.set_yticks(yticks1, crs=proj)
ax3.xaxis.set_major_formatter(lon_formatter)
ax3.yaxis.set_major_formatter(lat_formatter)
font = {'family': 'Arial', 'weight': 'bold', 'size': 28}

xmajorLocator = MultipleLocator(60)  # 先定义xmajorLocator，再进行调用
ax1.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
ax2.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
ax3.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
xminorLocator = MultipleLocator(10)
ax1.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
ax2.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
ax3.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
ymajorLocator = MultipleLocator(30)
ax1.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
ax2.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
ax3.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
yminorLocator = MultipleLocator(10)
ax1.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
ax2.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
ax3.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
# ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
# 最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=11, width=2, color='darkgray')  # 最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=8, width=1.8, color='darkgray')  # 最小刻度长度，宽度设置
ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
ax2.tick_params(which='major', length=11, width=2, color='darkgray')  # 最大刻度长度，宽度设置，
ax2.tick_params(which='minor', length=8, width=1.8, color='darkgray')  # 最小刻度长度，宽度设置
ax2.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
ax3.tick_params(which='major', length=11, width=2, color='darkgray')  # 最大刻度长度，宽度设置，
ax3.tick_params(which='minor', length=8, width=1.8, color='darkgray')  # 最小刻度长度，宽度设置
ax3.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
# 调整刻度值字体大小
ax1.tick_params(axis='both', labelsize=28, colors='black')
ax2.tick_params(axis='both', labelsize=28, colors='black')
ax3.tick_params(axis='both', labelsize=28, colors='black')
# 设置坐标刻度值的大小以及刻度值的字体
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
font2 = {'family': 'Arial', 'weight': 'bold', 'size': 28}

# color bar位置
# position = fig.add_axes([0.296, 0.08, 0.44, 0.011])#位置[左,下,右,上]
position1 = fig.add_axes([0.296, 0.64, 0.44, 0.011])
cb1 = plt.colorbar(a1, cax=position1, orientation='horizontal')  # orientation为水平或垂直
cb1.ax.tick_params(length=1, labelsize=24)  # length为刻度线的长度

position2 = fig.add_axes([0.296, 0.37, 0.44, 0.011])
cb2 = plt.colorbar(a2, cax=position2, orientation='horizontal')
cb2.ax.tick_params(length=1, labelsize=24)  # length为刻度线的长度

position3 = fig.add_axes([0.296, 0.10, 0.44, 0.011])
cb3 = plt.colorbar(a3, cax=position3, orientation='horizontal')
cb3.ax.tick_params(length=1, labelsize=24)  # length为刻度线的长度
tick_locator = ticker.MaxNLocator(nbins=7)  # colorbar上的刻度值个数

plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\图4.png', dpi=1000, bbox_inches='tight')
plt.show()
