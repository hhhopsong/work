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

# 将七八月份数据进行每年平均
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
z200_c = z200.mean('time')
u200_c = u200.mean('time')
v200_c = v200.mean('time')
try:
    # 读取相关系数
    reg_waf_x = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_waf_x.nc')
    reg_waf_y = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_waf_y.nc')
    reg_z200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_z200.nc')
    reg_z500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_z500.nc')
    reg_z850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_z850.nc')
    reg_u200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_u200.nc')
    reg_u500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_u500.nc')
    reg_u850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_u850.nc')
    reg_v200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_v200.nc')
    reg_v500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_v500.nc')
    reg_v850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_v850.nc')
    reg_sst = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_sst.nc')
    reg_pre = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_pre.nc')
except:
    waf200_x = np.array([TN_WAF(z200_c, u200_c, v200_c, z200_78[i], lon_uvz, lat_uvz, 200, 2)[0]
                         for i in range(len(z200_78))])
    waf200_y = np.array([TN_WAF(z200_c, u200_c, v200_c, z200_78[i], lon_uvz, lat_uvz, 200, 2)[1]
                         for i in range(len(z200_78))])
    # 将数据回归到PC上
    reg_waf_x = [[np.polyfit(PC_78[:, pc_index], waf200_x[:, ilat, ilon], 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算WAF X方向分量', position=0, leave=True)]
    reg_waf_y = [[np.polyfit(PC_78[:, pc_index], waf200_y[:, ilat, ilon], 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算WAF Y方向分量', position=0, leave=True)]
    reg_z200 = [[np.polyfit(PC_78[:, pc_index], z200_78[:, ilat, ilon]/9.8, 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算Z200', position=0, leave=True)]
    reg_u200 = [[np.polyfit(PC_78[:, pc_index], u200_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算U200', position=0, leave=True)]
    reg_v200 = [[np.polyfit(PC_78[:, pc_index], v200_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算V200', position=0, leave=True)]
    xr.DataArray(reg_waf_x, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_waf_x.nc')
    xr.DataArray(reg_waf_y, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_waf_y.nc')
    xr.DataArray(reg_z200, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_z200.nc')
    xr.DataArray(reg_u200, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_u200.nc')
    xr.DataArray(reg_v200, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_v200.nc')
    ###
    reg_z500 = [[np.polyfit(PC_78[:, pc_index], z500_78[:, ilat, ilon]/9.8, 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算Z500', position=0, leave=True)]
    reg_u500 = [[np.polyfit(PC_78[:, pc_index], u500_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算U500', position=0, leave=True)]
    reg_v500 = [[np.polyfit(PC_78[:, pc_index], v500_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算V500', position=0, leave=True)]
    reg_sst = [[np.polyfit(PC_78[:, pc_index], sst_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_sst))] for ilat in tqdm(range(len(lat_sst)), desc='计算SST', position=0, leave=True)]
    xr.DataArray(reg_z500, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_z500.nc')
    xr.DataArray(reg_u500, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_u500.nc')
    xr.DataArray(reg_v500, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_v500.nc')
    xr.DataArray(reg_sst, coords=[lat_sst, lon_sst], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_sst.nc')
    ###
    reg_z850 = [[np.polyfit(PC_78[:, pc_index], z850_78[:, ilat, ilon]/9.8, 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算Z850', position=0, leave=True)]
    reg_u850 = [[np.polyfit(PC_78[:, pc_index], u850_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算U850', position=0, leave=True)]
    reg_v850 = [[np.polyfit(PC_78[:, pc_index], v850_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算V850', position=0, leave=True)]
    reg_pre = [[np.polyfit(PC_78[:, pc_index], pre_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_pre))] for ilat in tqdm(range(len(lat_pre)), desc='计算PRE', position=0, leave=True)]
    xr.DataArray(reg_z850, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_z850.nc')
    xr.DataArray(reg_u850, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_u850.nc')
    xr.DataArray(reg_v850, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_v850.nc')
    xr.DataArray(reg_pre, coords=[lat_pre, lon_pre], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_pre.nc')
    ###数据再读取
    reg_waf_x = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_waf_x.nc')
    reg_waf_y = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_waf_y.nc')
    reg_z200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_z200.nc')
    reg_z500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_z500.nc')
    reg_z850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_z850.nc')
    reg_u200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_u200.nc')
    reg_u500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_u500.nc')
    reg_u850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_u850.nc')
    reg_v200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_v200.nc')
    reg_v500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_v500.nc')
    reg_v850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_v850.nc')
    reg_sst = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_sst.nc')
    reg_pre = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_pre.nc')

# 进行显著性0.05检验
from scipy.stats import t

# 计算自由度.
n = len(PC_78[:, 0])
# 使用t检验计算回归系数的的显著性
# 计算t值
Lxx = np.sum((PC_78[:, pc_index] - np.mean(PC_78[:, pc_index])) ** 2)
# SST
Sr_sst = reg_sst**2 * Lxx
St_sst = np.sum((sst_78 - np.mean(sst_78, axis=0)) ** 2, axis=0)
σ_sst = np.sqrt((St_sst - Sr_sst) / (n - 2))
t_sst = reg_sst * np.sqrt(Lxx) / σ_sst

Sr_pre = reg_pre**2 * Lxx
St_pre = np.sum((pre_78 - np.mean(pre_78, axis=0)) ** 2, axis=0)
σ_pre = np.sqrt((St_pre - Sr_pre) / (n - 2))
t_pre = reg_pre * np.sqrt(Lxx) / σ_pre

Sr_z200 = reg_z200**2 * Lxx
St_z200 = np.sum((z200_78/9.8 - np.mean(z200_78/9.8, axis=0)) ** 2, axis=0)
σ_z200 = np.sqrt((St_z200 - Sr_z200) / (n - 2))
t_z200 = reg_z200 * np.sqrt(Lxx) / σ_z200

Sr_u500 = reg_u500**2 * Lxx
St_u500 = np.sum((u500_78 - np.mean(u500_78, axis=0)) ** 2, axis=0)
σ_u500 = np.sqrt((St_u500 - Sr_u500) / (n - 2))
t_u500 = reg_u500 * np.sqrt(Lxx) / σ_u500

Sr_v500 = reg_v500**2 * Lxx
St_v500 = np.sum((v500_78 - np.mean(v500_78, axis=0)) ** 2, axis=0)
σ_v500 = np.sqrt((St_v500 - Sr_v500) / (n - 2))
t_v500 = reg_v500 * np.sqrt(Lxx) / σ_v500

Sr_u850 = reg_u850**2 * Lxx
St_u850 = np.sum((u850_78 - np.mean(u850_78, axis=0)) ** 2, axis=0)
σ_u850 = np.sqrt((St_u850 - Sr_u850) / (n - 2))
t_u850 = reg_u850 * np.sqrt(Lxx) / σ_u850

Sr_v850 = reg_v850**2 * Lxx
St_v850 = np.sum((v850_78 - np.mean(v850_78, axis=0)) ** 2, axis=0)
σ_v850 = np.sqrt((St_v850 - Sr_v850) / (n - 2))
t_v850 = reg_v850 * np.sqrt(Lxx) / σ_v850
# 计算临界值
t_critical = t.ppf(0.975, n - 2)
# 进行显著性检验
p_sst78 = np.zeros((len(lat_sst), len(lon_sst)))
p_sst78.fill(np.nan)
p_sst78[np.abs(t_sst['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1

p_pre78 = np.zeros((len(lat_pre), len(lon_pre)))
p_pre78.fill(np.nan)
p_pre78[np.abs(t_pre['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1

p_z20078 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_z20078.fill(np.nan)
p_z20078[np.abs(t_z200['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1

p_u50078 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_u50078.fill(0)
p_u50078[np.abs(t_u500['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_v50078 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_v50078.fill(0)
p_v50078[np.abs(t_v500['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_uv500 = np.where((p_u50078 + p_v50078) < 1, 0, 1)

p_u85078 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_u85078.fill(0)
p_u85078[np.abs(t_u850['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_v85078 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_v85078.fill(0)
p_v85078[np.abs(t_v850['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_uv850 = np.where((p_u85078 + p_v85078) < 1, 0, 1)
# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.subplots_adjust(wspace=0.1, hspace=0.001)  # wspace、hspace左右、上下的间距
font = {'family' : 'Arial','weight' : 'bold','size' : 12}
# plt.subplots_adjust(wspace=0.1, hspace=0.32)  # wspace、hspace左右、上下的间距
extent1 = [-180, 180, -30, 80]  # 经度范围，纬度范围
xticks1 = np.arange(extent1[0], extent1[1] + 1, 10)
yticks1 = np.arange(extent1[2], extent1[3] + 1, 10)
proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(16, 22))
n = 15
# ##ax1 Corr. PC1 & JA SST,2mT
level1 = [-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24]
ax1 = fig.add_subplot(311, projection=proj)
ax1.set_extent(extent1, crs=proj)
# 去除fill_value 1e+20
# 去除180白线!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!加上截距
reg_waf_x, reg_waf_y = TN_WAF(z200_c, u200_c, v200_c, reg_z200['__xarray_dataarray_variable__'].to_numpy()*9.8+z200_c,lon_uvz, lat_uvz, 200, 2)
reg_waf_x = np.where(reg_waf_x.m is np.nan, 0, reg_waf_x)
reg_waf_y = np.where(reg_waf_y.m is np.nan, 0, reg_waf_y)
reg_waf_x = filters.gaussian_filter(reg_waf_x, 3)
reg_waf_y = filters.gaussian_filter(reg_waf_y, 3)
reg_waf_x = np.where(np.abs(reg_waf_x) > 0.025, reg_waf_x, np.nan)
reg_waf_y = np.where(np.abs(reg_waf_y) > 0.025, reg_waf_y, np.nan)
reg_waf_x, a1_waf_lon = add_cyclic_point(reg_waf_x, coord=lon_uvz)
reg_waf_y, a1_waf_lon = add_cyclic_point(reg_waf_y, coord=lon_uvz)
reg_z200, a1_z_lon = add_cyclic_point(reg_z200['__xarray_dataarray_variable__'].to_numpy(), coord=lon_uvz)
print('开始绘制地图1')
ax1.set_title('(a)Reg 200WAF & 200Z', fontsize=28, loc='left')
a1 = ax1.contourf(a1_z_lon, lat_uvz[:280], reg_z200[:280,], cmap=cmaps.MPL_BrBG_r[23:105], levels=level1, extend='both',
                  transform=ccrs.PlateCarree(central_longitude=0))
a1_waf = ax1.quiver(a1_waf_lon[::n], lat_uvz[:360:n], reg_waf_x[:360:n, ::n], reg_waf_y[:360:n, ::n],
                    scale=13, color='black', headlength=2, headaxislength=2, transform = ccrs.PlateCarree(central_longitude=0))
ax1.quiverkey(a1_waf,  X=0.946, Y=1.03, U=.5,angle = 0,  label='0.5 m$^2$/s$^2$',
              labelpos='N', color='black',labelcolor = 'k', fontproperties=font,linewidth=0.8)#linewidth=1为箭头的大小
ax1.text(40, 59.5, 'A', fontsize=28, fontweight='bold', color='blue', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax1.text(68, 43, 'C', fontsize=28, fontweight='bold', color='red', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax1.text(113, 30.75, 'A', fontsize=28, fontweight='bold', color='blue', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax1.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=.3)  # 添加海岸线
ax1.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black',linewidth=2) # orientation为水平或垂直

# ax2 Corr. PC2 & JA PRE,850UV
level_sst = [-.3, -.25, -.2, -.15, -.1, -.05, .05, .1, .15, .2, .25, .3]
level_z = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20]
level2 = [-0.5, -0.4, -0.3, -0.2, 0.2, 0.3, 0.4, 0.5]
radio = 0.3
n = 25
print('开始绘制地图2')
ax2 = fig.add_subplot(312, projection=proj)
ax2.set_extent(extent1, crs=proj)
# 去除180白线
reg_sst, a2_sst_lon = add_cyclic_point(reg_sst['__xarray_dataarray_variable__'].to_numpy(), coord=lon_sst)
p_sst, a2_lon_sst = add_cyclic_point(p_sst78, coord=lon_sst)
p_sst = np.where(p_sst == 1, 0, np.nan)
reg_z500, a2_uv_lon = add_cyclic_point(reg_z500['__xarray_dataarray_variable__'].to_numpy(), coord=lon_uvz)
reg_u500 = np.where(p_uv500 == 1, reg_u500['__xarray_dataarray_variable__'].to_numpy(), np.nan)
reg_v500 = np.where(p_uv500 == 1, reg_v500['__xarray_dataarray_variable__'].to_numpy(), np.nan)
reg_u500, a2_uv_lon = add_cyclic_point(reg_u500, coord=lon_uvz)
reg_v500, a2_uv_lon = add_cyclic_point(reg_v500, coord=lon_uvz)
ax2.set_title('(b)Reg 500ZUV & SST', fontsize=28, loc='left')
a2 = ax2.contourf(a2_sst_lon, lat_sst, reg_sst, cmap=cmaps.GMT_polar[4:10]+cmaps.CBR_wet[0]+cmaps.GMT_polar[10:16], levels=level_sst, extend='both',
                  transform=ccrs.PlateCarree(central_longitude=0))
a2_uv = ax2.quiver(a2_uv_lon[::n], lat_uvz[::n], reg_u500[::n, ::n], reg_v500[::n, ::n], scale=30, color='black', headlength=3,
                   headaxislength=3, transform = ccrs.PlateCarree(central_longitude=0))
ax2.quiverkey(a2_uv,  X=0.946, Y=1.03, U=1,angle = 0,  label='1 m/s',
              labelpos='N', color='black',labelcolor = 'k', fontproperties=font,linewidth=0.8)#linewidth=1为箭头的大小

reg_z500 = filters.gaussian_filter(reg_z500, 4)
a2_low = ax2.contour(a2_uv_lon, lat_uvz, reg_z500, cmap=cmaps.BlueDarkRed18[0], levels=level_z[:6], linewidths=1.5, linestyles='--', alpha=1, transform=ccrs.PlateCarree(central_longitude=0))
a2_high = ax2.contour(a2_uv_lon, lat_uvz, reg_z500, cmap=cmaps.BlueDarkRed18[17], levels=level_z[6:], linewidths=1.5, linestyles='-', alpha=1, transform=ccrs.PlateCarree(central_longitude=0))

plt.clabel(a2_low, inline=True, fontsize=10, fmt='%d', inline_spacing=5)
plt.clabel(a2_high, inline=True, fontsize=10, fmt='%d', inline_spacing=5)

a2_p = ax2.quiver(a2_lon_sst, lat_sst, p_sst, p_sst, scale=20, color='black', headlength=2, headaxislength=2, transform = ccrs.PlateCarree(central_longitude=0))
ax2.text(52, 59.5, 'A', fontsize=28, fontweight='bold', color='blue', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax2.text(72, 43, 'C', fontsize=28, fontweight='bold', color='red', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax2.text(120, 30.75, 'A', fontsize=28, fontweight='bold', color='blue', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax2.text(145, 17, 'C', fontsize=28, fontweight='bold', color='red', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax2.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray')# 添加陆地并且陆地部分全部填充成浅灰色
ax2.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=.3)  # 添加海岸线
ax2.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black',linewidth=2)


# ax3 Corr. PC1 & JA Z500
level_pre = [-.8, -.6, -.4, -.2, .2, .4, .6, .8]
level_z = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
n = 20
print('开始绘制地图3')
ax3 = fig.add_subplot(313, projection=proj)
ax3.set_extent(extent1, crs=proj)
reg_z850, a3_uv_lon = add_cyclic_point(reg_z850['__xarray_dataarray_variable__'].to_numpy(), coord=lon_uvz)
reg_u850 = np.where(p_uv850 == 1, reg_u850['__xarray_dataarray_variable__'].to_numpy(), np.nan)
reg_v850 = np.where(p_uv850 == 1, reg_v850['__xarray_dataarray_variable__'].to_numpy(), np.nan)
reg_u850, a3_uv_lon = add_cyclic_point(reg_u850, coord=lon_uvz)
reg_v850, a3_uv_lon = add_cyclic_point(reg_v850, coord=lon_uvz)
reg_pre, a3_pre_lon = add_cyclic_point(reg_pre['__xarray_dataarray_variable__'].to_numpy(), coord=lon_pre)
p_pre, a3_lon_ppre = add_cyclic_point(p_pre78, coord=lon_pre)
p_pre = np.where(p_pre == 1, 0, np.nan)


ax3.set_title('(c)Reg 850ZUV&PRE', fontsize=28, loc='left')
a3 = ax3.contourf(a3_pre_lon, lat_pre, reg_pre, cmap=cmaps.MPL_RdYlGn[32:56]+cmaps.CBR_wet[0]+cmaps.MPL_RdYlGn[72:96], levels=level_pre, extend='both',
                  transform=ccrs.PlateCarree(central_longitude=0))
a3_uv = ax3.quiver(a3_uv_lon[::n], lat_uvz[::n], reg_u850[::n, ::n], reg_v850[::n, ::n], scale=30, color='black', headlength=3,
                   headaxislength=3, transform=ccrs.PlateCarree(central_longitude=0))
ax3.quiverkey(a3_uv,  X=0.946, Y=1.03, U=1,angle = 0,  label='1 m/s',
              labelpos='N', color='black',labelcolor = 'k', fontproperties=font,linewidth=0.8)#linewidth=1为箭头的大小

reg_z850 = filters.gaussian_filter(reg_z850, 4)
a3_low = ax3.contour(a3_uv_lon, lat_uvz, reg_z850, cmap=cmaps.BlueDarkRed18[0], levels=level_z[:5], linewidths=1.5, linestyles='--', alpha=1, transform=ccrs.PlateCarree(central_longitude=0))
a3_0 = ax3.contour(a3_uv_lon, lat_uvz, reg_z850, cmap='gray', levels=level_z[6], linewidths=1.5, linestyles='--', alpha=1, transform=ccrs.PlateCarree(central_longitude=0))

a3_high = ax3.contour(a3_uv_lon, lat_uvz, reg_z850, cmap=cmaps.BlueDarkRed18[17], levels=level_z[6:], linewidths=1.5, linestyles='-', alpha=1, transform=ccrs.PlateCarree(central_longitude=0))

plt.clabel(a3_low, inline=True, fontsize=10, fmt='%d', inline_spacing=5)
plt.clabel(a3_0, inline=True, fontsize=10, fmt='%d', inline_spacing=5)
plt.clabel(a3_high, inline=True, fontsize=10, fmt='%d', inline_spacing=5)

a3_p = ax3.quiver(a3_lon_ppre, lat_pre, p_pre, p_pre, scale=20, color='black', headlength=2, headaxislength=2, transform = ccrs.PlateCarree(central_longitude=0))
DBATP = r"D:\CODES\Python\PythonProject\map\DBATP\DBATP_Polygon.shp"
provinces = cfeature.ShapelyFeature(Reader(DBATP).geometries(), crs=ccrs.PlateCarree(), facecolor='gray', alpha=1)
ax3.add_feature(provinces, lw=0.5, zorder=2)

ax3.text(57, 59.5, 'A', fontsize=28, fontweight='bold', color='blue', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax3.text(78, 43, 'C', fontsize=28, fontweight='bold', color='red', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax3.text(122, 28, 'A', fontsize=28, fontweight='bold', color='blue', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax3.text(117, 9, 'C', fontsize=28, fontweight='bold', color='red', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))

ax3.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=.3)  # 添加海岸线
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
cb1.ax.tick_params(length=1, labelsize=20)  # length为刻度线的长度
cb1.locator = ticker.FixedLocator([-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24]) # colorbar上的刻度值个数

position2 = fig.add_axes([0.296, 0.37, 0.44, 0.011])
cb2 = plt.colorbar(a2, cax=position2, orientation='horizontal')
cb2.ax.tick_params(length=1, labelsize=20)  # length为刻度线的长度
cb2.locator = ticker.FixedLocator([-.3, -.2, -.1, .1, .2, .3]) # colorbar上的刻度值个数

position3 = fig.add_axes([0.296, 0.10, 0.44, 0.011])
cb3 = plt.colorbar(a3, cax=position3, orientation='horizontal')
cb3.ax.tick_params(length=1, labelsize=20)  # length为刻度线的长度
cb3.locator = ticker.FixedLocator([-.8, -.6, -.4, -.2, .2, .4, .6, .8]) # colorbar上的刻度值个数

plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\图4改.png', dpi=1000, bbox_inches='tight')
plt.show()
