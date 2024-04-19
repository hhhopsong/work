import os
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
import matplotlib.patches as mpatches
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


# 时间范围
time = ['1979', '2014']
# 地理范围
tas_lonlat = [36, 62, 54, 70]
tos_lonlat = [55, 100, -8, 5]
# 数据路径
mlp_T2m = xr.open_dataset(r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\ERA5\ERA5_2mTemperature_MeanSlp.nc")
sst = xr.open_dataset(r"C:\Users\10574\Desktop\data\sst.mnmean.nc")  # NetCDF-4文件路径不可含中文
tas_dataurl = r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\CMIP6\historical\CMIP6_historical_tas\amon"#数据路径
tos_dataurl = r"D:/CODES/Python/PythonProject/cache/CMIP6_tos/interp"#数据路径
Model_Name = os.listdir(tas_dataurl)
Model_Name_tos = os.listdir(tos_dataurl)

T2m = mlp_T2m['t2m'].sel(time=slice('1979-01-01', '2014-12-31'))
slp_2mT_78 = T2m.sel(time=T2m.time.dt.month.isin([7, 8]))
slp_2mT_78 = slp_2mT_78.sel(longitude=slice(tas_lonlat[0], tas_lonlat[1]), latitude=slice(tas_lonlat[3], tas_lonlat[2]))
sst = sst['sst'].sel(time=slice('1979-01-01', '2014-12-31'))
sst = sst.sel(time=sst.time.dt.month.isin([7, 8]))
sst = sst.sel(lon=slice(tas_lonlat[0], tas_lonlat[1]), lat=slice(tas_lonlat[3], tas_lonlat[2]))
slp_2mT_78 = slp_2mT_78.groupby('time.year').mean('time')
sst_78 = sst.groupby('time.year').mean('time')
slp_2mT_78 = np.array(slp_2mT_78)
sst_78 = np.array(sst_78)

# 进行显著性0.05检验
from scipy.stats import t
# 计算自由度
N = len(PC_78[:, 0])
# 计算临界值
t_critical = t.ppf(0.975, N - 2)

# 总存
corr_sat = []
corr_sot = []
for iModle in range(len(Model_Name)):
    ModelName = Model_Name[iModle]
    url = os.listdir(tas_dataurl + '/' + ModelName)
    q = xr.open_dataset(tas_dataurl + '/' + ModelName + '/' + url[0])
    for iurl in tqdm(url[1:], desc=f'\t读取{iModle + 1} {ModelName}模式 数据', unit='文件', position=0, colour='green'):
        q = xr.concat([q, xr.open_dataset(tas_dataurl + '/' + ModelName + '/' + iurl)], dim='time')
    q_term = q['tas'].sel(time=slice(time[0] + '-01-01', time[1] + '-12-31')) - 273.15
    q_term = q_term.sel(lon=slice(tas_lonlat[0], tas_lonlat[1]), lat=slice(tas_lonlat[2], tas_lonlat[3]))
    q_term_78 = q_term.sel(time=q_term.time.dt.month.isin([7, 8]))  # 7-8月2m气温
    q_term_78 = q_term_78.groupby('time.year').mean('time')
    r = np.corrcoef(PC_78[:, pc_index], q_term_78.mean(['lat', 'lon']))[0, 1]
    pas = 0
    t_ = r * np.sqrt((N - 2) / (1 - r ** 2))
    if t_ > t_critical:
        pas = 1
    corr_sat.append([ModelName, r, pas])
for i in range(len(corr_sat)):
    print(f'{corr_sat[i][0]}\t{corr_sat[i][1]:.2f}\t{corr_sat[i][2]}')
q_obs = slp_2mT_78 - 273.15
r = np.corrcoef(PC_78[:, pc_index], q_obs.mean(axis=(1, 2)))[0, 1]
pas = 0
t_ = r * np.sqrt((N - 2) / (1 - r ** 2))
if t_ > t_critical:
    pas = 1
print(f'Obs\t{np.corrcoef(PC_78[:, pc_index], q_obs.mean(axis=(1, 2)))[0, 1]:.2f}\t{pas}')


for iModle in range(len(Model_Name_tos)):
    ModelName_tos = Model_Name_tos[iModle]
    q = xr.open_dataset(tos_dataurl + '/' + ModelName_tos)
    q_term = q['__xarray_dataarray_variable__'].sel(time=slice(time[0] + '-01-01', time[1] + '-12-31'))
    q_term = q_term.sel(lon=slice(tos_lonlat[0], tos_lonlat[1]), lat=slice(tos_lonlat[2], tos_lonlat[3]))
    q_term_78 = np.zeros((len(q_term.time)//2, len(q_term.lat), len(q_term.lon)))
    q_term_78.fill(np.nan)
    for i in range(len(q_term)//2):
        q_term_78[i, :, :] = q_term[i*2:i*2+2].to_numpy().mean(axis=0)
    corr = np.corrcoef(PC_78[:, pc_index], np.array([i[~np.isnan(i)].mean() for i in q_term_78]))[0, 1]
    pas = 0
    t_ = corr * np.sqrt((N - 2) / (1 - corr ** 2))
    if t_ > t_critical:
        pas = 1
    corr_sot.append([ModelName_tos, corr, pas])
for i in range(len(corr_sot)):
    print(f'{corr_sot[i][0]}\t{corr_sot[i][1]:.2f}\t{corr_sot[i][2]}')
q_obs = sst_78
r = np.corrcoef(PC_78[:, pc_index], np.array([i[~np.isnan(i)].mean() for i in q_obs]))[0, 1]
pas = 0
t_ = r * np.sqrt((N - 2) / (1 - r ** 2))
if t_ > t_critical:
    pas = 1
print(f'Obs\t{np.corrcoef(PC_78[:, pc_index], np.array([i[~np.isnan(i)].mean() for i in q_obs]))[0, 1]:.2f}\t{pas}')
