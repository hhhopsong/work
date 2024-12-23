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

lon1, lat1= [50, 82.5], [40, 22]
sat_78 = pre.groupby('time.year').mean('time')
sat_78 = sat_78.sel(lat=slice(lat1[0], lat1[1]), lon=slice(lon1[0], lon1[1])).mean(['lat', 'lon'])  # 赤道西太平  # 赤道东太平洋
sat_78 = np.array(sat_78)
sat_detrend = np.polyfit(np.arange(len(sat_78)), sat_78[:],1)
sat_detrend = np.polyval(sat_detrend, np.arange(len(sat_78)))
sat_78 = sat_78 - sat_detrend

# 将七八月份数据进行每年平均
pre_78 = pre.groupby('time.year').mean('time')
sst_78 = sst.groupby('time.year').mean('time')
t2m_78 = T2m.groupby('time.year').mean('time')
u850_78 = u850.groupby('time.year').mean('time')
v850_78 = v850.groupby('time.year').mean('time')
z850_78 = z850.groupby('time.year').mean('time')
u700_78 = u700.groupby('time.year').mean('time')
v700_78 = v700.groupby('time.year').mean('time')
z700_78 = z700.groupby('time.year').mean('time')
u500_78 = u500.groupby('time.year').mean('time')
v500_78 = v500.groupby('time.year').mean('time')
z500_78 = z500.groupby('time.year').mean('time')
u200_78 = u200.groupby('time.year').mean('time')
v200_78 = v200.groupby('time.year').mean('time')
z200_78 = z200.groupby('time.year').mean('time')
pre_78 = np.array(pre_78)
sst_78 = np.array(sst_78)
t2m_78 = np.array(t2m_78)
olr_78 = np.array(olr)
u850_78 = np.array(u850_78)
v850_78 = np.array(v850_78)
z850_78 = np.array(z850_78)
u700_78 = np.array(u700_78)
v700_78 = np.array(v700_78)
z700_78 = np.array(z700_78)
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
    reg_lbm_t2m_z200 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z200.nc')
    reg_lbm_t2m_u200 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u200.nc')
    reg_lbm_t2m_v200 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v200.nc')
    reg_lbm_t2m_z500 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z500.nc')
    reg_lbm_t2m_u500 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u500.nc')
    reg_lbm_t2m_v500 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v500.nc')
    reg_lbm_t2m_z700 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z700.nc')
    reg_lbm_t2m_u700 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u700.nc')
    reg_lbm_t2m_v700 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v700.nc')
    reg_lbm_t2m_z850 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z850.nc')
    reg_lbm_t2m_u850 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u850.nc')
    reg_lbm_t2m_v850 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v850.nc')
    reg_lbm_t2m_pre = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_pre.nc')
    reg_lbm_t2m_sst = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_sst.nc')
    reg_lbm_t2m_t2m = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_t2m.nc')
    reg_lbm_t2m_olr = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_olr.nc')
except:
    # 将数据回归到PC上
    reg_z200 = [[np.polyfit(sat_78[:], z200_78[:, ilat, ilon]/9.8,1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT Z200', position=0, leave=True)]
    xr.DataArray(reg_z200, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z200.nc')
    reg_u200 = [[np.polyfit(sat_78[:], u200_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT U200', position=0, leave=True)]
    xr.DataArray(reg_u200, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u200.nc')
    reg_v200 = [[np.polyfit(sat_78[:], v200_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT V200', position=0, leave=True)]
    xr.DataArray(reg_v200, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v200.nc')
    reg_z500 = [[np.polyfit(sat_78[:], z500_78[:, ilat, ilon]/9.8,1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT Z500', position=0, leave=True)]
    xr.DataArray(reg_z500, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z500.nc')
    reg_u500 = [[np.polyfit(sat_78[:], u500_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT U500', position=0, leave=True)]
    xr.DataArray(reg_u500, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u500.nc')
    reg_v500 = [[np.polyfit(sat_78[:], v500_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT V500', position=0, leave=True)]
    xr.DataArray(reg_v500, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v500.nc')
    reg_z700 = [[np.polyfit(sat_78[:], z700_78[:, ilat, ilon]/9.8,1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT Z700', position=0, leave=True)]
    xr.DataArray(reg_z700, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z700.nc')
    reg_u700 = [[np.polyfit(sat_78[:], u700_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT U700', position=0, leave=True)]
    xr.DataArray(reg_u700, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u700.nc')
    reg_v700 = [[np.polyfit(sat_78[:], v700_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT V700', position=0, leave=True)]
    xr.DataArray(reg_v700, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v700.nc')
    reg_z850 = [[np.polyfit(sat_78[:], z850_78[:, ilat, ilon]/9.8,1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT Z850', position=0, leave=True)]
    xr.DataArray(reg_z850, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z850.nc')
    reg_u850 = [[np.polyfit(sat_78[:], u850_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT U850', position=0, leave=True)]
    xr.DataArray(reg_u850, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u850.nc')
    reg_v850 = [[np.polyfit(sat_78[:], v850_78[:, ilat, ilon],1)[0] for ilon in range(len(lon_uvz))] for ilat in tqdm(range(len(lat_uvz)), desc='计算LBM 2mT V850', position=0, leave=True)]
    xr.DataArray(reg_v850, coords=[lat_uvz, lon_uvz], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v850.nc')
    reg_pre = [[np.polyfit(sat_78[:], pre_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_pre))] for ilat in tqdm(range(len(lat_pre)), desc='计算LBM 2mT pre', position=0, leave=True)]
    xr.DataArray(reg_pre, coords=[lat_pre, lon_pre], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_pre.nc')
    reg_sst = [[np.polyfit(sat_78[:], sst_78[:, ilat, ilon], 1)[0] if not np.isnan(sst_78[:, ilat, ilon]).any() else np.nan for ilon in range(len(lon_sst))] for ilat in tqdm(range(len(lat_sst)), desc='计算LBM 2mT sst', position=0, leave=True)]
    xr.DataArray(reg_sst, coords=[lat_sst, lon_sst], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_sst.nc')
    reg_t2m = [[np.polyfit(sat_78[:], t2m_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_t2m))] for ilat in tqdm(range(len(lat_t2m)), desc='计算LBM 2mT t2m', position=0, leave=True)]
    xr.DataArray(reg_t2m, coords=[lat_t2m, lon_t2m], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_t2m.nc')
    reg_olr = [[np.polyfit(sat_78[:], olr[:, ilat, ilon], 1)[0] for ilon in range(len(lon_olr))] for ilat in tqdm(range(len(lat_olr)), desc='计算LBM 2mT olr', position=0, leave=True)]
    xr.DataArray(reg_olr, coords=[lat_olr, lon_olr], dims=['lat', 'lon']).to_netcdf(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_olr.nc')
    ###数据再读取
    reg_lbm_t2m_z200 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z200.nc')
    reg_lbm_t2m_u200 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u200.nc')
    reg_lbm_t2m_v200 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v200.nc')
    reg_lbm_t2m_z500 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z500.nc')
    reg_lbm_t2m_u500 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u500.nc')
    reg_lbm_t2m_v500 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v500.nc')
    reg_lbm_t2m_z700 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z700.nc')
    reg_lbm_t2m_u700 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u700.nc')
    reg_lbm_t2m_v700 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v700.nc')
    reg_lbm_t2m_z850 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_z850.nc')
    reg_lbm_t2m_u850 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_u850.nc')
    reg_lbm_t2m_v850 = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_v850.nc')
    reg_lbm_t2m_pre = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_pre.nc')
    reg_lbm_t2m_sst = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_sst.nc')
    reg_lbm_t2m_t2m = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_t2m.nc')
    reg_lbm_t2m_olr = xr.open_dataset(r'D:\PyFile\paper1\cache\reg_lbm\pre2\reg_lbm_t2m_olr.nc')
# 进行显著性0.05检验
from scipy.stats import t

# 计算自由度.
n = len(sat_78[:])
# 使用t检验计算回归系数的的显著性
# 计算t值
Lxx = np.sum((sat_78[:] - np.mean(sat_78[:])) ** 2)
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
# lbm_t2m_z700
Sr_lbm_t2m_z700 = reg_lbm_t2m_z700**2 * Lxx
St_lbm_t2m_z700 = np.sum((z700_78/9.8 - np.mean(z700_78/9.8, axis=0)) ** 2, axis=0)
σ_lbm_t2m_z700 = np.sqrt((St_lbm_t2m_z700 - Sr_lbm_t2m_z700) / (n - 2))
t_lbm_t2m_z700 = reg_lbm_t2m_z700 * np.sqrt(Lxx) / σ_lbm_t2m_z700
# lbm_t2m_u700
Sr_lbm_t2m_u700 = reg_lbm_t2m_u700**2 * Lxx
St_lbm_t2m_u700 = np.sum((u700_78 - np.mean(u700_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_u700 = np.sqrt((St_lbm_t2m_u700 - Sr_lbm_t2m_u700) / (n - 2))
t_lbm_t2m_u700 = reg_lbm_t2m_u700 * np.sqrt(Lxx) / σ_lbm_t2m_u700
# lbm_t2m_v700
Sr_lbm_t2m_v700 = reg_lbm_t2m_v700**2 * Lxx
St_lbm_t2m_v700 = np.sum((v700_78 - np.mean(v700_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_v700 = np.sqrt((St_lbm_t2m_v700 - Sr_lbm_t2m_v700) / (n - 2))
t_lbm_t2m_v700 = reg_lbm_t2m_v700 * np.sqrt(Lxx) / σ_lbm_t2m_v700
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
# lbm_t2m_sst
Sr_lbm_t2m_sst = reg_lbm_t2m_sst**2 * Lxx
St_lbm_t2m_sst = np.sum((sst_78 - np.mean(sst_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_sst = np.sqrt((St_lbm_t2m_sst - Sr_lbm_t2m_sst) / (n - 2))
t_lbm_t2m_sst = reg_lbm_t2m_sst * np.sqrt(Lxx) / σ_lbm_t2m_sst
# lbm_t2m_t2m
Sr_lbm_t2m_t2m = reg_lbm_t2m_t2m**2 * Lxx
St_lbm_t2m_t2m = np.sum((t2m_78 - np.mean(t2m_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_t2m = np.sqrt((St_lbm_t2m_t2m - Sr_lbm_t2m_t2m) / (n - 2))
t_lbm_t2m_t2m = reg_lbm_t2m_t2m * np.sqrt(Lxx) / σ_lbm_t2m_t2m
# lbm_t2m_olr
Sr_lbm_t2m_olr = reg_lbm_t2m_olr**2 * Lxx
St_lbm_t2m_olr = np.sum((olr_78 - np.mean(olr_78, axis=0)) ** 2, axis=0)
σ_lbm_t2m_olr = np.sqrt((St_lbm_t2m_olr - Sr_lbm_t2m_olr) / (n - 2))
t_lbm_t2m_olr = reg_lbm_t2m_olr * np.sqrt(Lxx) / σ_lbm_t2m_olr
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

p_lbm_t2m_z700 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_z700.fill(np.nan)
p_lbm_t2m_z700[np.abs(t_lbm_t2m_z700['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_lbm_t2m_u700 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_u700.fill(0)
p_lbm_t2m_u700[np.abs(t_lbm_t2m_u700['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_lbm_t2m_v700 = np.zeros((len(lat_uvz), len(lon_uvz)))
p_lbm_t2m_v700.fill(0)
p_lbm_t2m_v700[np.abs(t_lbm_t2m_v700['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
p_uv700 = np.where((p_lbm_t2m_u700 + p_lbm_t2m_v700) < 1, 0, 1)

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

p_lbm_t2m_sst = np.zeros((len(lat_sst), len(lon_sst)))
p_lbm_t2m_sst.fill(np.nan)
p_lbm_t2m_sst[np.abs(t_lbm_t2m_sst['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1

p_lbm_t2m_t2m = np.zeros((len(lat_t2m), len(lon_t2m)))
p_lbm_t2m_t2m.fill(np.nan)
p_lbm_t2m_t2m[np.abs(t_lbm_t2m_t2m['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1

p_lbm_t2m_olr = np.zeros((len(lat_olr), len(lon_olr)))
p_lbm_t2m_olr.fill(np.nan)
p_lbm_t2m_olr[np.abs(t_lbm_t2m_olr['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1
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
shp = fr"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp"
split_shp = gpd.read_file(shp)
split_shp.crs = 'wgs84'
# ##ax1 Corr. PC1 & JA SST,2mT
level1 = np.array([-4, -3, -2, -1, -.5, .5, 1, 2, 3, 4]) * 10
level1_z = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])*10
ax1 = fig.add_subplot(411, projection=ccrs.PlateCarree(central_longitude=180-67.5))
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
# WAF
reg_z200 = reg_lbm_t2m_z200['__xarray_dataarray_variable__']*9.8
Geoc = xr.DataArray(z200_c.data[np.newaxis, :, :],
                    coords=[('level', [200]),
                            ('lat', z200_c['lat'].data),
                            ('lon', z200_c['lon'].data)])
Uc = xr.DataArray(u200_c.data[np.newaxis, :, :],
                  coords=[('level', [200]),
                          ('lat', u200_c['lat'].data),
                          ('lon', u200_c['lon'].data)])
Vc = xr.DataArray(v200_c.data[np.newaxis, :, :],
                  coords=[('level', [200]),
                          ('lat', v200_c['lat'].data),
                          ('lon', v200_c['lon'].data)])
GEOa = xr.DataArray(reg_z200.data[np.newaxis, :, :],
                    coords=[('level', [200]),
                            ('lat', reg_z200['lat'].data),
                            ('lon', reg_z200['lon'].data)])
reg_waf_x, reg_waf_y = TN_WAF_3D(Geoc, Uc, Vc, GEOa, u_threshold=0, filt=3)
olr, a1_olr_lon = add_cyclic_point(reg_lbm_t2m_olr['__xarray_dataarray_variable__'].to_numpy(), coord=lon_olr)  # 去除180白线
print('开始绘制地图1')
ax1.set_title('(a)Reg. 200UV&WAF&OLR', fontsize=20, loc='left')
a1 = ax1.contourf(a1_olr_lon, lat_olr, olr,
                  cmap=cmaps.MPL_PuOr_r[11+15:56]+ cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.MPL_PuOr_r[64:106-15],
                  levels=level1, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
u200 = reg_lbm_t2m_u200['__xarray_dataarray_variable__'].to_numpy()
v200 = reg_lbm_t2m_v200['__xarray_dataarray_variable__'].to_numpy()
a1_uv = Curlyquiver(ax1, lon_uvz, lat_uvz, u200, v200, regrid=15, lon_trunc=-67.5, arrowsize=.6, scale=30, linewidth=0.8,
                                  color='k', transform=ccrs.PlateCarree(central_longitude=0))
a1_uv.key(fig, U=10, label='10 m/s', lr=2., width_shrink=0.5)
a1_waf = Curlyquiver(ax1, lon_uvz, lat_uvz[:180], reg_waf_x[:180, :], reg_waf_y[:180, :], regrid=10, lon_trunc=-67.5, arrowsize=.7, scale=7, linewidth=1.1,
                                  color='blue', transform=ccrs.PlateCarree(central_longitude=0), arrowstyle='fancy')
a1_waf.key(fig, U=.5, label='0.5 m$^2$/s$^2$', width_shrink =.5)

# 显著性打点
p_lbm_t2m_olr, a1_lon_p = add_cyclic_point(p_lbm_t2m_olr, coord=lon_olr)
p_lbm_t2m_olr = np.where(p_lbm_t2m_olr == 1, 0, np.nan)
p_uv = ax1.quiver(a1_lon_p, lat_uvz, p_lbm_t2m_olr, p_lbm_t2m_olr, scale=20, color='white', headlength=3,
                   regrid_shape=60, headaxislength=3, transform=ccrs.PlateCarree(central_longitude=0), width=0.002)

# 框选预测因子
smoonth = 100
lon_1 = [i for i in np.linspace(lon1[0], lon1[1], smoonth)] + [i for i in np.linspace(lon1[0], lon1[1], smoonth)][::-1] + [lon1[0]]
lat_1 = [lat1[1] for i in range(smoonth)] + [lat1[0] for i in range(smoonth)] + [lat1[1]]
ax1.plot(lon_1, lat_1, color='red', linewidth=1, linestyle='--', transform=ccrs.PlateCarree(central_longitude=0))

ax1.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black',linewidth=1) # orientation为水平或垂直
ax1.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.3)  # 添加海岸线

# ax2 Reg 500ZUV onto AST
level_z500 = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])*10
size_uv = 40
reshape_uv = 20
uv_min = 0
print('开始绘制地图2')
ax2 = fig.add_subplot(412, projection=ccrs.PlateCarree(central_longitude=180-67.5))
ax2.set_extent(extent1, crs=ccrs.PlateCarree())
# 去除180白线
z500, a2_z500_lon = add_cyclic_point(reg_lbm_t2m_z500['__xarray_dataarray_variable__'].to_numpy(), coord=lon_uvz)
u500 = np.where(p_uv500 == 1, reg_lbm_t2m_u500['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 显著风场
v500 = np.where(p_uv500 == 1, reg_lbm_t2m_v500['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 显著风场
u500_np = np.where(p_uv500 == 0, reg_lbm_t2m_u500['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 非显著风场
v500_np = np.where(p_uv500 == 0, reg_lbm_t2m_v500['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 非显著风场
u500 = np.where(np.abs(u500) > uv_min, u500, np.nan)
v500 = np.where(np.abs(v500) > uv_min, v500, np.nan)
u500_np = np.where(np.abs(u500_np) > uv_min, u500_np, np.nan)
v500_np = np.where(np.abs(v500_np) > uv_min, v500_np, np.nan)
ax2.set_title('(b)Reg. 500UVZ', fontsize=20, loc='left')
#reg_z500 = filters.gaussian_filter(reg_z500, 3)
a2 = ax2.contourf(a2_z500_lon, lat_uvz, z500, cmap=cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
                  levels=level_z500, extend='both', transform=ccrs.PlateCarree())
a2_uv = Curlyquiver(ax2, lon_uvz, lat_uvz, u500, v500, regrid=15, lon_trunc=-67.5, arrowsize=.6, scale=20, linewidth=0.8,
                                  color='k', transform=ccrs.PlateCarree(central_longitude=0))
a2_uv_np = Curlyquiver(ax2, lon_uvz, lat_uvz, u500_np, v500_np, regrid=15, lon_trunc=-67.5, arrowsize=.6, scale=20, linewidth=0.8,
                                  color='gray', transform=ccrs.PlateCarree(central_longitude=0), nanmax=a2_uv.nanmax)
a2_uv.key(fig, U=5, label='5 m/s', width_shrink=0.5)
# 显著性打点
p_z500, a2_p_z500 = add_cyclic_point(p_lbm_t2m_z500, coord=lon_uvz)
p_z500 = np.where(p_z500 == 1, 0, np.nan)
a2_p = ax2.quiver(a2_p_z500, lat_uvz, p_z500, p_z500, scale=30, color='white', headlength=3,
                   regrid_shape=60, headaxislength=3, transform=ccrs.PlateCarree(), width=0.002)
ax2.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.3)  # 添加海岸线
ax2.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black', linewidth=1)
# 框选预测因子
ax2.plot(lon_1, lat_1, color='red', linewidth=1, linestyle='--', transform=ccrs.PlateCarree(central_longitude=0))
# ax3 Reg 850ZUV onto AST
level_sst = np.array([-.3, -.2, -.1, -.05, .05, .1, .2, .3])*10
size_uv = 30
reshape_uv = 20
uv_min = 0
print('开始绘制地图3')
ax3 = fig.add_subplot(413, projection=ccrs.PlateCarree(central_longitude=180-67.5))
ax3.set_extent(extent1, crs=ccrs.PlateCarree())
# 去除180白线
sst, a3_sst_lon = add_cyclic_point(reg_lbm_t2m_sst['__xarray_dataarray_variable__'].to_numpy(), coord=lon_sst)
t2m, a3_t2m_lon = add_cyclic_point(reg_lbm_t2m_t2m['__xarray_dataarray_variable__'].to_numpy(), coord=lon_t2m)
z850, a3_z850_lon = add_cyclic_point(reg_lbm_t2m_z700['__xarray_dataarray_variable__'].to_numpy(), coord=lon_uvz)
u850 = np.where(p_uv850 == 1, reg_lbm_t2m_u700['__xarray_dataarray_variable__'].to_numpy(), np.nan)
v850 = np.where(p_uv850 == 1, reg_lbm_t2m_v700['__xarray_dataarray_variable__'].to_numpy(), np.nan)
u850_np = np.where(p_uv850 == 0, reg_lbm_t2m_u700['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 非显著风场
v850_np = np.where(p_uv850 == 0, reg_lbm_t2m_v700['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 非显著风场
u850 = np.where(np.abs(u850) > uv_min, u850, np.nan)
v850 = np.where(np.abs(v850) > uv_min, v850, np.nan)
u850_np = np.where(np.abs(u850_np) > uv_min, u850_np, np.nan)
v850_np = np.where(np.abs(v850_np) > uv_min, v850_np, np.nan)
ax3.set_title('(c)Reg. 700UVZ&SST&2mT', fontsize=20, loc='left')
#reg_z500 = filters.gaussian_filter(reg_z500, 3)
a3_t2m = ax3.contourf(a3_t2m_lon, lat_t2m, t2m, cmap=cmaps.GreenMagenta16[8-5:8] + cmaps.GMT_red2green_r[11:11+4],
                  levels=level_sst, extend='both', transform=ccrs.PlateCarree())
a3 = ax3.contourf(a3_sst_lon, lat_sst, sst, cmap=cmaps.BlueWhiteOrangeRed[40:-40], levels=level_sst, extend='both', transform=ccrs.PlateCarree())
a3_uv = Curlyquiver(ax3, lon_uvz, lat_uvz, u500, v500, regrid=15, lon_trunc=-67.5, arrowsize=.6, scale=20, linewidth=0.8,
                                  color='k', transform=ccrs.PlateCarree(central_longitude=0))
a3_uv_np = Curlyquiver(ax3, lon_uvz, lat_uvz, u500_np, v500_np, regrid=15, lon_trunc=-67.5, arrowsize=.6, scale=20, linewidth=0.8,
                                  color='gray', transform=ccrs.PlateCarree(central_longitude=0), nanmax=a3_uv.nanmax)
a3_uv.key(fig, U=5, label='5 m/s', width_shrink=0.5)
# 高度场
z850 = filters.gaussian_filter(z850, 4)
a3_low = ax3.contour(a3_z850_lon, lat_uvz, z850, cmap=cmaps.BlueDarkRed18[0], levels=[-8, -4], linewidths=1, linestyles='--', alpha=1, transform=ccrs.PlateCarree())
a3_high = ax3.contour(a3_z850_lon, lat_uvz, z850, cmap=cmaps.BlueDarkRed18[17], levels=[4, 8], linewidths=1, linestyles='-', alpha=1, transform=ccrs.PlateCarree())

plt.clabel(a3_low, inline=True, fontsize=10, fmt='%d', inline_spacing=5)
plt.clabel(a3_high, inline=True, fontsize=10, fmt='%d', inline_spacing=5)

# 显著性打点
p_t2m, a3_p_t2m = add_cyclic_point(p_lbm_t2m_t2m, coord=lon_t2m)
p_t2m = np.where(p_t2m == 1, 0, np.nan)
a3_p = ax3.quiver(a3_p_t2m, lat_t2m, p_t2m, p_t2m, scale=30, color='white', headlength=3,
                   regrid_shape=60, headaxislength=3, transform=ccrs.PlateCarree(), width=0.002)
p_sst, a3_p_sst = add_cyclic_point(p_lbm_t2m_sst, coord=lon_sst)
p_sst = np.where(p_sst == 1, 0, np.nan)
a3_p = ax3.quiver(a3_p_sst, lat_sst, p_sst, p_sst, scale=30, color='white', headlength=3,
                   regrid_shape=60, headaxislength=3, transform=ccrs.PlateCarree(), width=0.002)
ax3.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.3)  # 添加海岸线
ax3.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black', linewidth=1)
DBATP = r"D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary_2500m\TPBoundary_2500m.shp"
provinces = cfeature.ShapelyFeature(Reader(DBATP).geometries(), crs=ccrs.PlateCarree(), facecolor='gray', alpha=1)
ax3.add_feature(provinces, lw=0.5, zorder=2)
# 框选预测因子
ax3.plot(lon_1, lat_1, color='red', linewidth=1, linestyle='--', transform=ccrs.PlateCarree(central_longitude=0))
# ax4 Reg 850ZUV onto AST
level_pre = np.array([-.6, -.4, -.2, -.1, .1, .2, .4, .6])*10
size_uv = 30
reshape_uv = 20
uv_min = 0
print('开始绘制地图4')
ax4 = fig.add_subplot(414, projection=ccrs.PlateCarree(central_longitude=180-67.5))
ax4.set_extent(extent1, crs=ccrs.PlateCarree())
# 去除180白线
pre, a4_pre_lon = add_cyclic_point(reg_lbm_t2m_pre['__xarray_dataarray_variable__'].to_numpy(), coord=lon_pre)
z850, a4_z850_lon = add_cyclic_point(reg_lbm_t2m_z850['__xarray_dataarray_variable__'].to_numpy(), coord=lon_uvz)
u850 = np.where(p_uv850 == 1, reg_lbm_t2m_u850['__xarray_dataarray_variable__'].to_numpy(), np.nan)
v850 = np.where(p_uv850 == 1, reg_lbm_t2m_v850['__xarray_dataarray_variable__'].to_numpy(), np.nan)
u850_np = np.where(p_uv850 == 0, reg_lbm_t2m_u850['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 非显著风场
v850_np = np.where(p_uv850 == 0, reg_lbm_t2m_v850['__xarray_dataarray_variable__'].to_numpy(), np.nan) # 非显著风场
u850 = np.where(np.abs(u850) > uv_min, u850, np.nan)
v850 = np.where(np.abs(v850) > uv_min, v850, np.nan)
u850_np = np.where(np.abs(u850_np) > uv_min, u850_np, np.nan)
v850_np = np.where(np.abs(v850_np) > uv_min, v850_np, np.nan)
ax4.set_title('(d)Reg. 850UV&PRE', fontsize=20, loc='left')
#reg_z500 = filters.gaussian_filter(reg_z500, 3)
a4 = ax4.contourf(a4_pre_lon, lat_pre, pre, cmap=cmaps.MPL_RdYlGn[32+10:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:96-10], levels=level_pre, extend='both', transform=ccrs.PlateCarree())
a4_uv = Curlyquiver(ax4, lon_uvz, lat_uvz, u500, v500, regrid=15, lon_trunc=-67.5, arrowsize=.6, scale=20, linewidth=0.8,
                                  color='k', transform=ccrs.PlateCarree(central_longitude=0))
a4_uv_np = Curlyquiver(ax4, lon_uvz, lat_uvz, u500_np, v500_np, regrid=15, lon_trunc=-67.5, arrowsize=.6, scale=20, linewidth=0.8,
                                  color='gray', transform=ccrs.PlateCarree(central_longitude=0), nanmax=a4_uv.nanmax)
a4_uv.key(fig, U=5, label='5 m/s', width_shrink=0.5)

# 显著性打点
p_pre, a4_p_pre = add_cyclic_point(p_lbm_t2m_pre, coord=lon_pre)
p_pre = np.where(p_pre == 1, 0, np.nan)
a4_p = ax4.quiver(a4_p_pre, lat_pre, p_pre, p_pre, scale=30, color='white', headlength=3,
                   regrid_shape=60, headaxislength=3, transform=ccrs.PlateCarree(), width=0.002)
ax4.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.3)  # 添加海岸线
ax4.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black', linewidth=1)
DBATP = r"D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary_2500m\TPBoundary_2500m.shp"
provinces = cfeature.ShapelyFeature(Reader(DBATP).geometries(), crs=ccrs.PlateCarree(), facecolor='gray', alpha=1)
ax4.add_feature(provinces, lw=0.5, zorder=2)
# 框选预测因子
ax4.plot(lon_1, lat_1, color='red', linewidth=1, linestyle='--', transform=ccrs.PlateCarree(central_longitude=0))
# 刻度线设置
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
# ax1
ax1.set_yticks(yticks1, crs=ccrs.PlateCarree())
ax1.yaxis.set_major_formatter(lat_formatter)
# ax2
ax2.set_yticks(yticks1, crs=ccrs.PlateCarree())
ax2.yaxis.set_major_formatter(lat_formatter)
# ax3
ax3.set_yticks(yticks1, crs=ccrs.PlateCarree())
ax3.yaxis.set_major_formatter(lat_formatter)
# ax4
ax4.set_xticks(xticks1, crs=ccrs.PlateCarree())
ax4.set_yticks(yticks1, crs=ccrs.PlateCarree())
ax4.xaxis.set_major_formatter(lon_formatter)
ax4.yaxis.set_major_formatter(lat_formatter)

xmajorLocator = MultipleLocator(60)  # 先定义xmajorLocator，再进行调用
xminorLocator = MultipleLocator(10)
ymajorLocator = MultipleLocator(30)
yminorLocator = MultipleLocator(10)
ax4.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
ax4.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度

ax1.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
ax2.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
ax3.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
ax4.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度

ax1.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
ax2.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
ax3.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
ax4.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
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
ax4.tick_params(which='major', length=11, width=2, color='darkgray')  # 最大刻度长度，宽度设置，
ax4.tick_params(which='minor', length=8, width=1.8, color='darkgray')  # 最小刻度长度，宽度设置
ax4.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
# 调整刻度值字体大小
ax1.tick_params(axis='both', labelsize=16, colors='black')
ax2.tick_params(axis='both', labelsize=16, colors='black')
ax3.tick_params(axis='both', labelsize=16, colors='black')
ax4.tick_params(axis='both', labelsize=16, colors='black')
# 设置坐标刻度值的大小以及刻度值的字体
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Arial') for label in labels]


# color bar位置
ax_ins1 = inset_axes(
    ax1,
    width="1.25%",  # width: 5% of parent_bbox width
    height="100%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(1.11, 0., 1, 1),
    bbox_transform=ax1.transAxes,
    borderpad=0,
)
# position = fig.add_axes([0.296, 0.08, 0.44, 0.011])#位置[左,下,右,上]
cb1 = plt.colorbar(a1, orientation='vertical', drawedges=True, cax=ax_ins1)  # orientation为水平或垂直
cb1.ax.tick_params(length=1, labelsize=14)  # length为刻度线的长度
cb1.locator = ticker.FixedLocator(level1) # colorbar上的刻度值个数

ax_ins2 = inset_axes(
    ax2,
    width="1.25%",  # width: 5% of parent_bbox width
    height="100%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(1.11, 0., 1, 1),
    bbox_transform=ax2.transAxes,
    borderpad=0,
)
cb2 = plt.colorbar(a2, orientation='vertical', drawedges=True, cax=ax_ins2)
cb2.ax.tick_params(length=1, labelsize=14)  # length为刻度线的长度
cb2.locator = ticker.FixedLocator(level_z500) # colorbar上的刻度值个数

ax_ins3_1 = inset_axes(
    ax3,
    width="1.25%",  # width: 5% of parent_bbox width
    height="100%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(1.01, 0., 1, 1),
    bbox_transform=ax3.transAxes,
    borderpad=0,
)
ax_ins3_2 = inset_axes(
    ax3,
    width="1.25%",  # width: 5% of parent_bbox width
    height="100%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(1.11, 0., 1, 1),
    bbox_transform=ax3.transAxes,
    borderpad=0,
)
cb3_1 = plt.colorbar(a3_t2m, orientation='vertical', drawedges=True, cax=ax_ins3_1)
cb3_1.ax.tick_params(length=1, labelsize=14)  # length为刻度线的长度
cb3_1.locator = ticker.FixedLocator(level_sst) # colorbar上的刻度值个数
cb3_2 = plt.colorbar(a3, orientation='vertical', drawedges=True, cax=ax_ins3_2)
cb3_2.ax.tick_params(length=1, labelsize=14)  # length为刻度线的长度
cb3_2.locator = ticker.FixedLocator(level_sst) # colorbar上的刻度值个数

ax_ins4 = inset_axes(
    ax4,
    width="1.25%",  # width: 5% of parent_bbox width
    height="100%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(1.11, 0., 1, 1),
    bbox_transform=ax4.transAxes,
    borderpad=0,
)
cb4 = plt.colorbar(a4, orientation='vertical', drawedges=True, cax=ax_ins4)
cb4.ax.tick_params(length=1, labelsize=14)  # length为刻度线的长度

plt.savefig(r'D:\PyFile\pic\北印度洋降水回归.png', dpi=600, bbox_inches='tight')
plt.show()
