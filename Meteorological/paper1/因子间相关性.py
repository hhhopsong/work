import pandas as pd
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import filters
from tqdm import tqdm
import geopandas as gpd
import salem
from toolbar.TN_WaveActivityFlux import TN_WAF_3D, TN_WAF
from toolbar.curved_quivers.modplot import Curlyquiver
import pprint


# 数据读取
year = [1961, 2022]
try:
    # 读取相关系数
    sat = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\sat.nc')
    sst = xr.open_dataset(r"E:\data\NOAA\ERSSTv5\sst.mnmean.nc")
    wh = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\wh.nc')
    gpcp = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\gpcp.nc')
    olr = xr.open_dataset(r"D:\PyFile\paper1\cache\olr\olr_same.nc")
except:
    # 读取数据
    sat = xr.open_dataset(r"E:\data\ERA5\ERA5_singleLev\ERA5_sgLEv.nc")
    sst = xr.open_dataset(r"E:\data\NOAA\ERSSTv5\sst.mnmean.nc")
    wh = xr.open_dataset(r"E:\data\ERA5\ERA5_pressLev\era5_pressLev.nc")
    gpcp = xr.open_dataset(r"E:\data\NOAA\PREC\precip.mon.anom.nc")
    olr = xr.open_dataset(r"D:\PyFile\paper1\cache\olr\olr_same.nc")
    # 数据描述信息修改
    change = sat.sel(   date=slice(str(year[0] - 1) + '-01-01', str(year[1] + 1) + '-12-31'),
                        latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])['t2m']
    sat = xr.DataArray(change.data, coords=[('time', pd.to_datetime(change['date'], format="%Y%m%d")),
                                            ('lat', change['latitude'].data),
                                            ('lon', change['longitude'].data)]).to_dataset(name='t2m')
    sst = sst.sel(  time=slice(str(year[0] - 1) + '-01-01', str(year[1] + 1) + '-12-31'),
                    latitude=[90 - i * 2 for i in range(89)], longitude=[i * 2 for i in range(180)])['sst']
    change_u = wh.sel(   date=slice(str(year[0] - 1) + '-01-01', str(year[1] + 1) + '-12-31'),
                         pressure_level=[200, 300, 400, 500, 600, 700, 850],
                        latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])['u']
    change_v = wh.sel(   date=slice(str(year[0] - 1) + '-01-01', str(year[1] + 1) + '-12-31'),
                         pressure_level=[200, 300, 400, 500, 600, 700, 850],
                        latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])['v']
    change_z = wh.sel(   date=slice(str(year[0] - 1) + '-01-01', str(year[1] + 1) + '-12-31'),
                         pressure_level=[200, 300, 400, 500, 600, 700, 850],
                        latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])['z']
    wh = xr.Dataset({  'u': (['time', 'level', 'lat', 'lon'], change_u.data),
                                'v': (['time', 'level', 'lat', 'lon'], change_v.data),
                                'z': (['time', 'level', 'lat', 'lon'], change_z.data)},
                      coords={  'time': pd.to_datetime(change_u['date'], format="%Y%m%d"),
                                'level': change_u['pressure_level'].data,
                                'lat': change_u['latitude'].data,
                                'lon': change_u['longitude'].data})
    sat.to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\sat.nc')
    wh.to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\wh.nc')
    gpcp.to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\gpcp.nc')
# 数据切片
olr = olr['olr']
pre = gpcp['precip'].sel(time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
pre = pre.sel(time=pre.time.dt.month.isin([7, 8]))
sst = sst['sst'].sel(time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
sst = sst.sel(time=sst.time.dt.month.isin([7, 8]))
T2m = sat['t2m'].sel(time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
T2m_78 = T2m.sel(time=T2m.time.dt.month.isin([7, 8]))
u850 = wh['u'].sel(level=850, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
u850 = u850.sel(time=u850.time.dt.month.isin([7, 8]))
v850 = wh['v'].sel(level=850, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
v850 = v850.sel(time=v850.time.dt.month.isin([7, 8]))
z850 = wh['z'].sel(level=850, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
z850 = z850.sel(time=z850.time.dt.month.isin([7, 8]))
u700 = wh['u'].sel(level=700, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
u700 = u700.sel(time=u700.time.dt.month.isin([7, 8]))
v700 = wh['v'].sel(level=700, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
v700 = v700.sel(time=v700.time.dt.month.isin([7, 8]))
z700 = wh['z'].sel(level=700, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
z700 = z700.sel(time=z700.time.dt.month.isin([7, 8]))
u500 = wh['u'].sel(level=500, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
u500 = u500.sel(time=u500.time.dt.month.isin([7, 8]))
v500 = wh['v'].sel(level=500, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
v500 = v500.sel(time=v500.time.dt.month.isin([7, 8]))
z500 = wh['z'].sel(level=500, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
z500 = z500.sel(time=z500.time.dt.month.isin([7, 8]))
u200 = wh['u'].sel(level=200, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
u200 = u200.sel(time=u200.time.dt.month.isin([7, 8]))
v200 = wh['v'].sel(level=200, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
v200 = v200.sel(time=v200.time.dt.month.isin([7, 8]))
z200 = wh['z'].sel(level=200, time=slice(f'{year[0]}-01-01', f'{year[1]}-12-31'))
z200 = z200.sel(time=z200.time.dt.month.isin([7, 8]))
# 经纬度
lon_uvz = u850['lon']
lat_uvz = u850['lat']
lon_pre = pre['lon']
lat_pre = pre['lat']
lon_sst = sst['lon']
lat_sst = sst['lat']
lon_t2m = T2m['lon']
lat_t2m = T2m['lat']
lon_olr = olr['lon']
lat_olr = olr['lat']

# 北极附近暖异常
lon1, lon2, lat1, lat2 = 40, 82.5, 50, 72
sat_78 = T2m_78.groupby('time.year').mean('time')
sat_78 = sat_78.sel(lat=slice(lat2, lat1), lon=slice(lon1, lon2))
sat_78 = sat_78.mean(['lat', 'lon'])
sat_78 = np.array(sat_78)
sat_detrend = np.polyfit(np.arange(len(sat_78)), sat_78[:],1)
sat_detrend = np.polyval(sat_detrend, np.arange(len(sat_78)))
sat_78 = sat_78 - sat_detrend

# 热带对流活动
lon1, lon2, lat1, lat2 = [122.5, 155], [-180, -85], [5, -5], [5, -5]
sst_78 = sst.groupby('time.year').mean('time')
sst_78_1 = sst_78.sel(lat=slice(lat1[0], lat1[1]), lon=slice(lon1[0], lon1[1])).mean(['lat', 'lon'])  # 赤道西太平洋
sst_78_2 = sst_78.sel(lat=slice(lat2[0], lat2[1]), lon=slice(lon2[0]+360, lon2[1]+360)).mean(['lat', 'lon'])  # 赤道东太平洋
sst_78 = sst_78_1 - sst_78_2
sst_78 = np.array(sst_78)
sst_detrend = np.polyfit(np.arange(len(sst_78)), sst_78[:],1)
sst_detrend = np.polyval(sst_detrend, np.arange(len(sst_78)))
sst_78 = sst_78 - sst_detrend

# 印度洋降水异常
lon1, lat1= [60, 80], [5, -15]
pre_78 = pre.groupby('time.year').mean('time')
pre_78 = pre_78.sel(lat=slice(lat1[0], lat1[1]), lon=slice(lon1[0], lon1[1])).mean(['lat', 'lon'])  # 赤道西太平  # 赤道东太平洋
pre_78 = np.array(pre_78)
pre_detrend = np.polyfit(np.arange(len(pre_78)), pre_78[:],1)
pre_detrend = np.polyval(pre_detrend, np.arange(len(pre_78)))
pre_78 = pre_78 - pre_detrend

# obs
ols = np.load(r"D:\PyFile\paper1\OLS35_detrended.npy")  # 读取缓存

corr_t2m_sst = np.corrcoef([sat_78, sst_78])[0, 1]
corr_pre_sst = np.corrcoef([pre_78, sst_78])[0, 1]
corr_t2m_pre = np.corrcoef([sat_78, pre_78])[0, 1]
corr_all = np.corrcoef([ols, sat_78, sst_78, pre_78])
plt.axhline(0, color='gray', linestyle='-')
plt.plot(ols, label='Obs', color='k', alpha=0.5, lw=1)
plt.plot(sat_78, label='2mT', color='r', alpha=0.7, lw=1)
plt.plot(sst_78, label='sst', color='b', alpha=0.7, lw=1)
plt.plot(pre_78, label='pre', color='g', alpha=0.7, lw=1)
# 网格线
plt.grid(axis='x', linestyle='--')
plt.grid(axis='y', linestyle='--')
plt.xticks(np.arange(0, 61, 10), np.arange(1961, 2022, 10))# x轴起始为1961年
plt.legend()
plt.savefig(r'D:\PyFile\pic\corr_is.png', dpi=600, bbox_inches='tight')
plt.show()
print(f'Corr. 2mT&sst :{corr_t2m_sst}')
print(f'Corr. pre&sst :{corr_pre_sst}')
print(f'Corr. 2mT&pre :{corr_t2m_pre}')
print(f'Corr. all :{corr_all}')