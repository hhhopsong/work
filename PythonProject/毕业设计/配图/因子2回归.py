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


# 数据读取
wh = xr.open_dataset(r"C:/Users/10574/OneDrive/File/Graduation Thesis/ThesisData/ERA5/ERA5_Geo_U_V_200_500_850.nc")
sst = xr.open_dataset(r"C:\Users\10574\Desktop\data\sst.mnmean.nc")  # NetCDF-4文件路径不可含中文
gpcp = xr.open_dataset(r"C:\Users\10574\Desktop\data\precip.mon.mean.nc")
# 时间范围
time = ['1979', '2014']
# 地理范围
tos_lonlat = [54, 100, -8, 5]
# 框选预测因子
lon1, lon2, lat1, lat2 = tos_lonlat[0], tos_lonlat[1], tos_lonlat[2], tos_lonlat[3]
# 数据切片
sst = sst['sst'].sel(time=slice('1979-01-01', '2014-12-31'))
sst = sst.sel(time=sst.time.dt.month.isin([7, 8]))
sst = sst.sel(lon=slice(tos_lonlat[0], tos_lonlat[1]), lat=slice(tos_lonlat[3], tos_lonlat[2]))
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
lon_uvz = u850['longitude']
lat_uvz = u850['latitude']
lon_pre = pre['lon']
lat_pre = pre['lat']

# 将七八月份数据进行每年平均
sst_78 = sst.groupby('time.year').mean('time').mean(['lat', 'lon'])
sst_78 = np.array(sst_78)
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
    reg_lbm_t2m_z200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_z200.nc')
    reg_lbm_t2m_u200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_u200.nc')
    reg_lbm_t2m_v200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_v200.nc')
    reg_lbm_t2m_z500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_z500.nc')
    reg_lbm_t2m_u500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_u500.nc')
    reg_lbm_t2m_v500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_v500.nc')
    reg_lbm_t2m_z850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_z850.nc')
    reg_lbm_t2m_u850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_u850.nc')
    reg_lbm_t2m_v850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_v850.nc')
    reg_lbm_t2m_pre = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_pre.nc')
except:
    # 将数据回归到PC上
    reg_z200 = [[np.polyfit(sst_78[:], z200_78[:, ilat, ilon]/9.8,1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM sst Z200', position=0, leave=True)]
    xr.DataArray(reg_z200, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_z200.nc')
    reg_u200 = [[np.polyfit(sst_78[:], u200_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM sst U200', position=0, leave=True)]
    xr.DataArray(reg_u200, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_u200.nc')
    reg_v200 = [[np.polyfit(sst_78[:], v200_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM sst V200', position=0, leave=True)]
    xr.DataArray(reg_v200, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_v200.nc')
    reg_z500 = [[np.polyfit(sst_78[:], z500_78[:, ilat, ilon]/9.8,1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM sst Z500', position=0, leave=True)]
    xr.DataArray(reg_z500, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_z500.nc')
    reg_u500 = [[np.polyfit(sst_78[:], u500_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM sst U500', position=0, leave=True)]
    xr.DataArray(reg_u500, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_u500.nc')
    reg_v500 = [[np.polyfit(sst_78[:], v500_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM sst V500', position=0, leave=True)]
    xr.DataArray(reg_v500, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_v500.nc')
    reg_z850 = [[np.polyfit(sst_78[:], z850_78[:, ilat, ilon]/9.8,1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM sst Z850', position=0, leave=True)]
    xr.DataArray(reg_z850, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_z850.nc')
    reg_u850 = [[np.polyfit(sst_78[:], u850_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM sst U850', position=0, leave=True)]
    xr.DataArray(reg_u850, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_u850.nc')
    reg_v850 = [[np.polyfit(sst_78[:], v850_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM sst V850', position=0, leave=True)]
    xr.DataArray(reg_v850, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_v850.nc')
    reg_pre = [[np.polyfit(sst_78[:], pre_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_pre))] for ilat in tqdm(range(len(lat_pre)), desc='计算LBM sst pre', position=0, leave=True)]
    xr.DataArray(reg_pre, coords=[lat_pre, lon_pre], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_pre.nc')
    ###数据再读取
    reg_lbm_t2m_z200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_z200.nc')
    reg_lbm_t2m_u200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_u200.nc')
    reg_lbm_t2m_v200 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_v200.nc')
    reg_lbm_t2m_z500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_z500.nc')
    reg_lbm_t2m_u500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_u500.nc')
    reg_lbm_t2m_v500 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_v500.nc')
    reg_lbm_t2m_z850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_z850.nc')
    reg_lbm_t2m_u850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_u850.nc')
    reg_lbm_t2m_v850 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_v850.nc')
    reg_lbm_t2m_pre = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_lbm_sst_pre.nc')
# 进行显著性0.05检验
from scipy.stats import t

# 计算自由度.
n = len(sst_78[:])
# 使用t检验计算回归系数的的显著性
# 计算t值
Lxx = np.sum((sst_78[:] - np.mean(sst_78[:])) ** 2)
# lbm_t2m_z200
Sr_lbm_t2m_z200 = reg_lbm_t2m_z200**2 * Lxx
St_lbm_t2m_z200 = np.sum((z200_78/9.8 - np.mean(z200_78/9.8, axis=0)) ** 2, axis=0)
σ_lbm_t2m_z200 = np.sqrt((St_lbm_t2m_z200 - Sr_lbm_t2m_z200) / (n - 2))
t_lbm_t2m_z200 = reg_lbm_t2m_z200 * np.sqrt(Lxx) / σ_lbm_t2m_z200
# lbm_t2m_u200
Sr_lbm_t2m_u200 = reg_lbm_t2m_u200**2 * Lxx
St_lbm_t2m_u200 = np.sum((u200_78 - np.mean(u200_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_u200 = np.sqrt((St_lbm_t2m_u200 - Sr_lbm_t2m_u200) / (n - 2))
t_lbm_t2m_u200 = reg_lbm_t2m_u200 * np.sqrt(Lxx) / σ_lbm_t2m_u200
# lbm_t2m_v200
Sr_lbm_t2m_v200 = reg_lbm_t2m_v200**2 * Lxx
St_lbm_t2m_v200 = np.sum((v200_78 - np.mean(v200_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_v200 = np.sqrt((St_lbm_t2m_v200 - Sr_lbm_t2m_v200) / (n - 2))
t_lbm_t2m_v200 = reg_lbm_t2m_v200 * np.sqrt(Lxx) / σ_lbm_t2m_v200
# lbm_t2m_z500
Sr_lbm_t2m_z500 = reg_lbm_t2m_z500**2 * Lxx
St_lbm_t2m_z500 = np.sum((z500_78/9.8 - np.mean(z500_78/9.8, axis=0)) ** 2, axis=0)
σ_lbm_t2m_z500 = np.sqrt((St_lbm_t2m_z500 - Sr_lbm_t2m_z500) / (n - 2))
t_lbm_t2m_z500 = reg_lbm_t2m_z500 * np.sqrt(Lxx) / σ_lbm_t2m_z500
# lbm_t2m_u500
Sr_lbm_t2m_u500 = reg_lbm_t2m_u500**2 * Lxx
St_lbm_t2m_u500 = np.sum((u500_78 - np.mean(u500_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_u500 = np.sqrt((St_lbm_t2m_u500 - Sr_lbm_t2m_u500) / (n - 2))
t_lbm_t2m_u500 = reg_lbm_t2m_u500 * np.sqrt(Lxx) / σ_lbm_t2m_u500
# lbm_t2m_v500
Sr_lbm_t2m_v500 = reg_lbm_t2m_v500**2 * Lxx
St_lbm_t2m_v500 = np.sum((v500_78 - np.mean(v500_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_v500 = np.sqrt((St_lbm_t2m_v500 - Sr_lbm_t2m_v500) / (n - 2))
t_lbm_t2m_v500 = reg_lbm_t2m_v500 * np.sqrt(Lxx) / σ_lbm_t2m_v500
# lbm_t2m_z850
Sr_lbm_t2m_z850 = reg_lbm_t2m_z850**2 * Lxx
St_lbm_t2m_z850 = np.sum((z850_78/9.8 - np.mean(z850_78/9.8, axis=0)) ** 2, axis=0)
σ_lbm_t2m_z850 = np.sqrt((St_lbm_t2m_z850 - Sr_lbm_t2m_z850) / (n - 2))
t_lbm_t2m_z850 = reg_lbm_t2m_z850 * np.sqrt(Lxx) / σ_lbm_t2m_z850
# lbm_t2m_u850
Sr_lbm_t2m_u850 = reg_lbm_t2m_u850**2 * Lxx
St_lbm_t2m_u850 = np.sum((u850_78 - np.mean(u850_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_u850 = np.sqrt((St_lbm_t2m_u850 - Sr_lbm_t2m_u850) / (n - 2))
t_lbm_t2m_u850 = reg_lbm_t2m_u850 * np.sqrt(Lxx) / σ_lbm_t2m_u850
# lbm_t2m_v850
Sr_lbm_t2m_v850 = reg_lbm_t2m_v850**2 * Lxx
St_lbm_t2m_v850 = np.sum((v850_78 - np.mean(v850_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_v850 = np.sqrt((St_lbm_t2m_v850 - Sr_lbm_t2m_v850) / (n - 2))
t_lbm_t2m_v850 = reg_lbm_t2m_v850 * np.sqrt(Lxx) / σ_lbm_t2m_v850
# lbm_t2m_pre
Sr_lbm_t2m_pre = reg_lbm_t2m_pre**2 * Lxx
St_lbm_t2m_pre = np.sum((pre_78 - np.mean(pre_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_pre = np.sqrt((St_lbm_t2m_pre - Sr_lbm_t2m_pre) / (n - 2))
t_lbm_t2m_pre = reg_lbm_t2m_pre * np.sqrt(Lxx) / σ_lbm_t2m_pre
# 计算临界值
t_critical = t.ppf(0.95, n - 2)
t_critical_95 = t.ppf(0.95, n - 2)
# 进行显著性检验
p_lbm_t2m_z200 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_z200.fill(np.nan)
p_lbm_t2m_z200[np.abs(t_lbm_t2m_z200['__xarray_dataarray_variable__'].to_numpy()) > t_critical_95] = 1

p_lbm_t2m_u200 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_u200.fill(0)
p_lbm_t2m_u200[np.abs(t_lbm_t2m_u200['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_lbm_t2m_v200 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_v200.fill(0)
p_lbm_t2m_v200[np.abs(t_lbm_t2m_v200['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_uv200 = np.where((p_lbm_t2m_u200 + p_lbm_t2m_v200) < 1, 0, 1)

p_lbm_t2m_z500 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_z500.fill(np.nan)
p_lbm_t2m_z500[np.abs(t_lbm_t2m_z500['__xarray_dataarray_variable__'].to_numpy()) > t_critical_95] = 1

p_lbm_t2m_u500 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_u500.fill(0)
p_lbm_t2m_u500[np.abs(t_lbm_t2m_u500['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_lbm_t2m_v500 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_v500.fill(0)
p_lbm_t2m_v500[np.abs(t_lbm_t2m_v500['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_uv500 = np.where((p_lbm_t2m_u500 + p_lbm_t2m_v500) < 1, 0, 1)

p_lbm_t2m_z850 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_z850.fill(np.nan)
p_lbm_t2m_z850[np.abs(t_lbm_t2m_z850['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1

p_lbm_t2m_u850 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_u850.fill(0)
p_lbm_t2m_u850[np.abs(t_lbm_t2m_u850['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_lbm_t2m_v850 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_v850.fill(0)
p_lbm_t2m_v850[np.abs(t_lbm_t2m_v850['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_uv850 = np.where((p_lbm_t2m_u850 + p_lbm_t2m_v850) < 1, 0, 1)

p_lbm_t2m_pre = np.zeros((len(lat_pre), len(lon_pre)))
p_lbm_t2m_pre.fill(np.nan)
p_lbm_t2m_pre[np.abs(t_lbm_t2m_pre['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
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
fig = plt.figure(figsize=(9, 13))

#
shp = fr"D:/CODES/Python/PythonProject/map/shp/south_china/中国南方.shp"
split_shp = gpd.read_file(shp)
split_shp.crs = 'wgs84'
# ##ax1 Corr. PC1 & JA SST,2mT
level1 = [-1, -.7, -.4, -.1, -.05, .05, .1, .4, .7, 1]
level1_z = [-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24]
ax1 = fig.add_subplot(311, projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
# 风场显著性筛选
u200 = np.where(p_uv200 == 1, reg_lbm_t2m_u200['__xarray_dataarray_variable__'].to_numpy(), np.nan)
v200 = np.where(p_uv200 == 1, reg_lbm_t2m_v200['__xarray_dataarray_variable__'].to_numpy(), np.nan)
u200_np = np.where(p_uv200 == 0, reg_lbm_t2m_u200['__xarray_dataarray_variable__'].to_numpy(), np.nan)
v200_np = np.where(p_uv200 == 0, reg_lbm_t2m_v200['__xarray_dataarray_variable__'].to_numpy(), np.nan)
# 去除180白线
u200, a1_uv200_lon = add_cyclic_point(u200, coord=lon_uvz)
v200, a1_uv200_lon = add_cyclic_point(v200, coord=lon_uvz)
u200_np, a1_uv200_lon = add_cyclic_point(u200_np, coord=lon_uvz)
v200_np, a1_uv200_lon = add_cyclic_point(v200_np, coord=lon_uvz)
z200, a1_z_lon = add_cyclic_point(reg_lbm_t2m_z200['__xarray_dataarray_variable__'].to_numpy(), coord=lon_uvz)
print('开始绘制地图1')
ax1.set_title('(a)Reg. 200ZUV onto IST', fontsize=20, loc='left')
a1 = ax1.contourf(a1_z_lon, lat_uvz, z200, cmap=cmaps.MPL_BrBG_r[23:105], levels=level1_z, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
a1_waf = ax1.quiver(a1_uv200_lon, lat_uvz, u200, v200, regrid_shape=32,
                    scale=80, color='black', headlength=2, headaxislength=2, transform=ccrs.PlateCarree(central_longitude=0))
a1_uv_p = ax1.quiver(a1_uv200_lon, lat_uvz, u200_np, v200_np, regrid_shape=32,
                    scale=80, color='gray', headlength=2, headaxislength=2, transform=ccrs.PlateCarree(central_longitude=0))
ax1.quiverkey(a1_waf,  X=0.946, Y=1.03, U=2, angle=0,  label='2 m/s',
              labelpos='N', color='black', labelcolor='k', fontproperties=font,linewidth=0.8)#linewidth=1为箭头的大小
ax1.text(125, 28, 'A', fontsize=16, fontweight='bold', color='blue', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax1.text(135, 42, 'C', fontsize=16, fontweight='bold', color='red', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))

# 显著性打点
p_lbm_t2m_z200, a1_lon_p = add_cyclic_point(p_lbm_t2m_z200, coord=lon_uvz)
p_lbm_t2m_z200 = np.where(p_lbm_t2m_z200 == 1, 0, np.nan)
p_uv = ax1.quiver(a1_lon_p, lat_uvz, p_lbm_t2m_z200, p_lbm_t2m_z200, scale=20, color='black', headlength=3,
                   regrid_shape=60, headaxislength=3, transform=ccrs.PlateCarree(central_longitude=0), width=0.002)


smoonth = 100
lon_ = [i for i in np.linspace(lon1, lon2, smoonth)] + [i for i in np.linspace(lon1, lon2, smoonth)][::-1] + [lon1]
lat_ = [lat1 for i in range(smoonth)] + [lat2 for i in range(smoonth)] + [lat1]
ax1.plot(lon_, lat_, color='blue', linewidth=1, linestyle='--', transform=ccrs.PlateCarree(central_longitude=0))
ax1.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black',linewidth=1) # orientation为水平或垂直
ax1.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.3)  # 添加海岸线

# ax2 Reg 500ZUV onto AST
level_z = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
print('开始绘制地图2')
ax2 = fig.add_subplot(312, projection=ccrs.PlateCarree(central_longitude=180))
ax2.set_extent(extent1, crs=ccrs.PlateCarree())
# 去除180白线
z500, a2_z500_lon = add_cyclic_point(reg_lbm_t2m_z500['__xarray_dataarray_variable__'].to_numpy(), coord=lon_uvz)
u500 = np.where(p_uv500 == 1, reg_lbm_t2m_u500['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 显著风场
v500 = np.where(p_uv500 == 1, reg_lbm_t2m_v500['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 显著风场
u500_np = np.where(p_uv500 == 0, reg_lbm_t2m_u500['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 不显著风场
v500_np = np.where(p_uv500 == 0, reg_lbm_t2m_v500['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 不显著风场
u500, a2_uv500_lon = add_cyclic_point(u500, coord=lon_uvz)
v500, a2_uv500_lon = add_cyclic_point(v500, coord=lon_uvz)
u500_np, a2_uv500_lon = add_cyclic_point(u500_np, coord=lon_uvz)
v500_np, a2_uv500_lon = add_cyclic_point(v500_np, coord=lon_uvz)
ax2.set_title('(b)Reg. 500ZUV onto IST', fontsize=20, loc='left')
#reg_z500 = filters.gaussian_filter(reg_z500, 3)
a2 = ax2.contourf(a2_z500_lon, lat_uvz, z500, cmap=cmaps.MPL_BrBG_r[23:105], levels=level_z, extend='both', transform=ccrs.PlateCarree())

a2_uv = ax2.quiver(a2_uv500_lon, lat_uvz, u500, v500, scale=40, color='black', headlength=3, regrid_shape=25,
                   headaxislength=3, transform=ccrs.PlateCarree())
a2_uv_p = ax2.quiver(a2_uv500_lon, lat_uvz, u500_np, v500_np, scale=40, color='gray', headlength=3, regrid_shape=25,
                     headaxislength=3, transform=ccrs.PlateCarree())
ax2.quiverkey(a2_uv,  X=0.946, Y=1.03, U=2, angle=0,  label='2 m/s',
              labelpos='N', color='black', labelcolor='k', fontproperties=font,linewidth=0.8)#linewidth=1为箭头的大小

# 显著性打点
p_z500, a2_p_z500 = add_cyclic_point(reg_lbm_t2m_z500['__xarray_dataarray_variable__'].to_numpy(), coord=lon_uvz)
p_z500 = np.where(p_z500 == 1, 0, np.nan)
a2_p = ax2.quiver(a2_p_z500, lat_uvz, p_z500, p_z500, scale=30, color='black', headlength=3,
                   regrid_shape=60, headaxislength=3, transform=ccrs.PlateCarree(), width=0.002)
ax2.text(124, 22, 'A', fontsize=16, fontweight='bold', color='blue', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax2.text(140, 40, 'C', fontsize=16, fontweight='bold', color='red', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax2.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.3)  # 添加海岸线
ax2.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black', linewidth=1)
ax2.plot(lon_, lat_, color='blue', linewidth=1, linestyle='--', transform=ccrs.PlateCarree(central_longitude=0))

# ax3 Reg 850ZUV onto AST
level_z = [-7, -5, -3, -1, 0, 1, 3, 5, 7]
level_pre = [-.6, -.4, -.2, -.1, .1, .2, .4, .6]
print('开始绘制地图3')
ax3 = fig.add_subplot(313, projection=ccrs.PlateCarree(central_longitude=180))
ax3.set_extent(extent1, crs=ccrs.PlateCarree())
# 去除180白线
pre, a3_pre_lon = add_cyclic_point(reg_lbm_t2m_pre['__xarray_dataarray_variable__'].to_numpy(), coord=lon_pre)
z850, a3_z850_lon = add_cyclic_point(reg_lbm_t2m_z850['__xarray_dataarray_variable__'].to_numpy(), coord=lon_uvz)
u850 = np.where(p_uv850 == 1, reg_lbm_t2m_u850['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 显著风场
v850 = np.where(p_uv850 == 1, reg_lbm_t2m_v850['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 显著风场
u850_np = np.where(p_uv850 == 0, reg_lbm_t2m_u850['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 不显著风场
v850_np = np.where(p_uv850 == 0, reg_lbm_t2m_v850['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 不显著风场
u850, a3_uv850_lon = add_cyclic_point(u850, coord=lon_uvz)
v850, a3_uv850_lon = add_cyclic_point(v850, coord=lon_uvz)
u850_np, a3_uv850_lon = add_cyclic_point(u850_np, coord=lon_uvz)
v850_np, a3_uv850_lon = add_cyclic_point(v850_np, coord=lon_uvz)
ax3.set_title('(c)Reg. 850ZUV&PRE onto IST', fontsize=20, loc='left')
#reg_z500 = filters.gaussian_filter(reg_z500, 3)
a3 = ax3.contourf(a3_pre_lon, lat_pre, pre, cmap=cmaps.MPL_RdYlGn[32:56]+cmaps.CBR_wet[0]+cmaps.MPL_RdYlGn[72:96], levels=level_pre, extend='both', transform=ccrs.PlateCarree())

a3_uv = ax3.quiver(a3_uv850_lon, lat_uvz, u850, v850, scale=32, color='black', headlength=3, regrid_shape=30,
                   headaxislength=3, transform=ccrs.PlateCarree())
a3_uv_p = ax3.quiver(a3_uv850_lon, lat_uvz, u850_np, v850_np, scale=32, color='gray', headlength=3, regrid_shape=30,
                        headaxislength=3, transform=ccrs.PlateCarree())
ax3.quiverkey(a3_uv,  X=0.946, Y=1.03, U=1, angle=0,  label='1 m/s',
              labelpos='N', color='black', labelcolor='k', fontproperties=font, linewidth=0.8)#linewidth=1为箭头的大小
# 高度场
z850 = filters.gaussian_filter(z850, 4)
a3_low = ax3.contour(a3_z850_lon, lat_uvz, z850, cmap=cmaps.BlueDarkRed18[0], levels=level_z[:5], linewidths=1, linestyles='--', alpha=1, transform=ccrs.PlateCarree())
a3_0 = ax3.contour(a3_z850_lon, lat_uvz, z850, cmap='gray', levels=level_z[6], linewidths=1, linestyles='--', alpha=1, transform=ccrs.PlateCarree())
a3_high = ax3.contour(a3_z850_lon, lat_uvz, z850, cmap=cmaps.BlueDarkRed18[17], levels=level_z[6:], linewidths=1, linestyles='-', alpha=1, transform=ccrs.PlateCarree())

plt.clabel(a3_low, inline=True, fontsize=10, fmt='%d', inline_spacing=5)
plt.clabel(a3_0, inline=True, fontsize=10, fmt='%d', inline_spacing=5)
plt.clabel(a3_high, inline=True, fontsize=10, fmt='%d', inline_spacing=5)


# 显著性打点
p_pre, a3_p_pre = add_cyclic_point(p_lbm_t2m_pre, coord=lon_pre)
p_pre = np.where(p_pre == 1, 0, np.nan)
a3_p = ax3.quiver(a3_p_pre, lat_pre, p_pre, p_pre, scale=30, color='black', headlength=3,
                   regrid_shape=60, headaxislength=3, transform=ccrs.PlateCarree(), width=0.002)
ax3.text(125, 23, 'A', fontsize=16, fontweight='bold', color='blue', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax3.text(140, 38, 'C', fontsize=16, fontweight='bold', color='red', zorder=20, transform=ccrs.PlateCarree(central_longitude=0))
ax3.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.3)  # 添加海岸线
ax3.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black', linewidth=1)
DBATP = r"D:\CODES\Python\PythonProject\map\DBATP\TP_2500m\TPBoundary_2500m.shp"
provinces = cfeature.ShapelyFeature(Reader(DBATP).geometries(), crs=ccrs.PlateCarree(), facecolor='gray', alpha=1)
ax3.add_feature(provinces, lw=0.5, zorder=2)
ax3.plot(lon_, lat_, color='blue', linewidth=1, linestyle='--', transform=ccrs.PlateCarree(central_longitude=0))
# 刻度线设置
# ax1
ax1.set_xticks(xticks1, crs=ccrs.PlateCarree())
ax1.set_yticks(yticks1, crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
# ax2
ax2.set_xticks(xticks1, crs=ccrs.PlateCarree())
ax2.set_yticks(yticks1, crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
# ax3
ax3.set_xticks(xticks1, crs=ccrs.PlateCarree())
ax3.set_yticks(yticks1, crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax3.xaxis.set_major_formatter(lon_formatter)
ax3.yaxis.set_major_formatter(lat_formatter)

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
ax1.tick_params(axis='both', labelsize=16, colors='black')
ax2.tick_params(axis='both', labelsize=16, colors='black')
ax3.tick_params(axis='both', labelsize=16, colors='black')
# 设置坐标刻度值的大小以及刻度值的字体
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Arial') for label in labels]


# color bar位置
# position = fig.add_axes([0.296, 0.08, 0.44, 0.011])#位置[左,下,右,上]
position1 = fig.add_axes([0.296, 0.64, 0.44, 0.011])
cb1 = plt.colorbar(a1, cax=position1, orientation='horizontal')  # orientation为水平或垂直
cb1.ax.tick_params(length=1, labelsize=14)  # length为刻度线的长度
cb1.locator = ticker.FixedLocator([-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24]) # colorbar上的刻度值个数

position2 = fig.add_axes([0.296, 0.37, 0.44, 0.011])
cb2 = plt.colorbar(a2, cax=position2, orientation='horizontal')
cb2.ax.tick_params(length=1, labelsize=14)  # length为刻度线的长度
cb2.locator = ticker.FixedLocator([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]) # colorbar上的刻度值个数

position3 = fig.add_axes([0.296, 0.10, 0.44, 0.011])
cb3 = plt.colorbar(a3, cax=position3, orientation='horizontal')
cb3.ax.tick_params(length=1, labelsize=14)  # length为刻度线的长度
cb3.locator = ticker.FixedLocator([-.6, -.4, -.2, -.1, .1, .2, .4, .6]) # colorbar上的刻度值个数

plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\LBM平替2.png', dpi=1000, bbox_inches='tight')
plt.show()
