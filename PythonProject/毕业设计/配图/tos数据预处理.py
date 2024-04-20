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
from scipy.interpolate import interpolate, Rbf, interp2d, bisplrep, LinearNDInterpolator, griddata
from scipy.ndimage import filters
from tqdm import tqdm
import geopandas as gpd
import salem
from tools.TN_WaveActivityFlux import TN_WAF



zone = 'Indian_sea'
shp = fr"D:\CODES\Python\PythonProject\map\sea\India_sea.shp"
split_shp = gpd.read_file(shp)
split_shp.crs = 'wgs84'

# 时间范围
time = ['1979', '2014']
month = [7, 8]
date = []
for i in range(eval(time[0]), eval(time[1])+1):
    for j in month:
        if j == 7:
            date.append(f"{i}-07-15")
        else:
            date.append(f"{i}-08-15")
# 地理范围
tos_lonlat_0 = [0, 360, -89.5, 90]
tos_lonlat1_0 = [-180, 180, -89.5, 90]
det_grid = 1
# 数据路径
tos_dataurl = r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\CMIP6\historical\CMIP6_historical_tos\Omon"#数据路径
Model_Name_tos = os.listdir(tos_dataurl)


for iModle in range(len(Model_Name_tos)):
    ModelName_tos = Model_Name_tos[iModle]
    url = os.listdir(tos_dataurl + '/' + ModelName_tos)
    start_index = 0
    tos_lonlat = tos_lonlat_0
    tos_lonlat1 = tos_lonlat1_0
    try:
        q_grid = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + url[0])['lon']
        criterion = np.where(q_grid > 180, 1, 0)
        if criterion.sum() != 0:
            grids_lat, grids_lon = np.meshgrid(np.arange(tos_lonlat[2], tos_lonlat[3], 0.5), np.arange(tos_lonlat[0], tos_lonlat[1], 0.5))
            tos_lonlat = tos_lonlat_0
        else:
            grids_lat, grids_lon = np.meshgrid(np.arange(tos_lonlat1[2], tos_lonlat1[3], 0.5), np.arange(tos_lonlat1[0], tos_lonlat1[1], 0.5))
            tos_lonlat = tos_lonlat1_0
    except:
        try:
            q_grid = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + url[0])['longitude']
            criterion = np.where(q_grid > 180, 1, 0)
            if criterion.sum() != 0:
                grids_lat, grids_lon = np.meshgrid(np.arange(tos_lonlat[2], tos_lonlat[3], 0.5),
                                                   np.arange(tos_lonlat[0], tos_lonlat[1], 0.5))
                tos_lonlat = tos_lonlat_0
            else:
                grids_lat, grids_lon = np.meshgrid(np.arange(tos_lonlat1[2], tos_lonlat1[3], 0.5),
                                                   np.arange(tos_lonlat1[0], tos_lonlat1[1], 0.5))
                tos_lonlat = tos_lonlat1_0
        except:
            q_grid = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + url[0])['nav_lon']
            criterion = np.where(q_grid > 180, 1, 0)
            if criterion.sum() != 0:
                grids_lat, grids_lon = np.meshgrid(np.arange(tos_lonlat[2], tos_lonlat[3], 0.5), np.arange(tos_lonlat[0], tos_lonlat[1], 0.5))
                tos_lonlat = tos_lonlat_0
            else:
                grids_lat, grids_lon = np.meshgrid(np.arange(tos_lonlat1[2], tos_lonlat1[3], 0.5), np.arange(tos_lonlat1[0], tos_lonlat1[1], 0.5))
                tos_lonlat = tos_lonlat1_0
    try:
        xr.open_dataset(rf'D:\CODES\Python\PythonProject\cache\CMIP6_tos\interp_global\{ModelName_tos}.nc')
    except:
        n = 1
        try:
            q = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + url[0])
            q = q['tos'].sel(time=q.time.dt.month.isin([7, 8])).sel(time=slice(time[0] + '-01-01', time[1] + '-12-31'))
            try:
                q_t = q.sel(lon=slice(tos_lonlat[0], tos_lonlat[1]), lat=slice(tos_lonlat[2], tos_lonlat[3]))
                for iurl in url:
                    tqdm.write(f'{iurl}插值中')
                    q = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + iurl)
                    q = q['tos'].sel(time=q.time.dt.month.isin([7, 8])).sel(time=slice(time[0] + '-01-01', time[1] + '-12-31'))
                    if start_index == 0:
                        q_new = xr.DataArray(q.to_numpy(), coords=[('time', q['time'].to_numpy()), ('lat', q['lat'].to_numpy()), ('lon', q['lon'].to_numpy())])
                        start_index = 1
                    else:
                        q_new = xr.concat([q_new, xr.DataArray(q.to_numpy(), coords=[('time', q['time'].to_numpy()), ('lat', q['lat'].to_numpy()), ('lon', q['lon'].to_numpy())])], dim='time')
                q_new.to_netcdf(rf'D:\CODES\Python\PythonProject\cache\CMIP6_tos\interp_global\{ModelName_tos}.nc')
                q_new.close()
                tqdm.write(f'{ModelName_tos}插值完成')
                continue
            except:
                try:
                    q_t = q.sel(longitude=slice(tos_lonlat[0], tos_lonlat[1]), latitude=slice(tos_lonlat[2], tos_lonlat[3]))
                    for iurl in url:
                        tqdm.write(f'{iurl}插值中')
                        q = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + iurl)
                        q = q['tos'].sel(time=q.time.dt.month.isin([7, 8])).sel(time=slice(time[0] + '-01-01', time[1] + '-12-31'))
                    if start_index == 0:
                        q_new = xr.DataArray(q.to_numpy(), coords=[('time', q['time'].to_numpy()), ('lat', q['latitude'].to_numpy()), ('lon', q['longitude'].to_numpy())])
                        start_index = 1
                    else:
                        q_new = xr.concat([q_new, xr.DataArray(q.to_numpy(), coords=[('time', q['time'].to_numpy()), ('lat', q['latitude'].to_numpy()), ('lon', q['longitude'].to_numpy())])], dim='time')
                    q_new.to_netcdf(rf'D:\CODES\Python\PythonProject\cache\CMIP6_tos\interp_global\{ModelName_tos}.nc')
                    q_new.close()
                    tqdm.write(f'{ModelName_tos}插值完成')
                    continue
                except:
                    n = 1
            lon = np.array(q['lon'].to_numpy().flatten()[::n]).reshape(-1, 1)
            lat = np.array(q['lat'].to_numpy().flatten()[::n]).reshape(-1, 1)
            points = np.concatenate([lon, lat], axis=1)
            for iurl in url:
                tqdm.write(f'{iurl}插值中')
                q = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + iurl)
                q = q['tos'].sel(time=q.time.dt.month.isin([7, 8])).sel(time=slice(time[0] + '-01-01', time[1] + '-12-31'))
                for i in tqdm(q, desc=f'\t插值 {ModelName_tos}模式', unit='Year', position=0, colour='green'):
                    q = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + url[0])
                    q = q['tos'].sel(time=q.time.dt.month.isin([7, 8])).sel(time=slice(time[0] + '-01-01', time[1] + '-12-31'))
                    data = np.array(i.to_numpy().flatten()[::n]).reshape(-1, 1)
                    lon_grid, lat_grid = np.meshgrid(np.arange(tos_lonlat[0], tos_lonlat[1] + det_grid, det_grid),
                                                         np.arange(tos_lonlat[2], tos_lonlat[3] + det_grid, det_grid))
                    interp = griddata(points, data, (lon_grid, lat_grid), method='linear')
                    grid_data = interp[:, :, 0]
                    # 保证纬度从上到下是递减的
                    if lat_grid[0, 0] < lat_grid[1, 0]:
                        lat_grid = lat_grid[-1::-1]
                        grid_data = grid_data[-1::-1]
                    if start_index == 0:
                        q_new = xr.DataArray(grid_data, coords=[('lat', np.arange(tos_lonlat[2], tos_lonlat[3]+det_grid, det_grid)), ('lon', np.arange(tos_lonlat[0], tos_lonlat[1]+det_grid, det_grid))])
                        start_index = 1
                    else:
                        q_new = xr.concat([q_new, xr.DataArray(grid_data, coords=[('lat', np.arange(tos_lonlat[2], tos_lonlat[3]+det_grid, det_grid)), ('lon', np.arange(tos_lonlat[0], tos_lonlat[1]+det_grid, det_grid))])], dim='time')
        except:
            try:
                q = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + url[0])
                q = q['tos'].sel(time=q.time.dt.month.isin([7, 8])).sel(time=slice(time[0] + '-01-01', time[1] + '-12-31'))
                lon = np.array(q['longitude'].to_numpy().flatten()[::n]).reshape(-1, 1)
                lat = np.array(q['latitude'].to_numpy().flatten()[::n]).reshape(-1, 1)
                points = np.concatenate([lon, lat], axis=1)
                for iurl in url:
                    tqdm.write(f'{iurl}插值中')
                    q = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + iurl)
                    q = q['tos'].sel(time=q.time.dt.month.isin([7, 8])).sel(time=slice(time[0] + '-01-01', time[1] + '-12-31'))
                    for i in tqdm(q, desc=f'\t插值 {ModelName_tos}模式', unit='Year', position=0, colour='green'):
                        data = np.array(i.to_numpy().flatten()[::n]).reshape(-1, 1)
                        lon_grid, lat_grid = np.meshgrid(np.arange(tos_lonlat[0], tos_lonlat[1] + det_grid, det_grid),
                                                             np.arange(tos_lonlat[2], tos_lonlat[3] + det_grid, det_grid))
                        interp = griddata(points, data, (lon_grid, lat_grid), method='linear')
                        grid_data = interp[:, :, 0]
                        # 保证纬度从上到下是递减的
                        if lat_grid[0, 0] < lat_grid[1, 0]:
                            lat_grid = lat_grid[-1::-1]
                            grid_data = grid_data[-1::-1]
                        if start_index == 0:
                            q_new = xr.DataArray(grid_data, coords=[('lat', np.arange(tos_lonlat[2], tos_lonlat[3]+det_grid, det_grid)), ('lon', np.arange(tos_lonlat[0], tos_lonlat[1]+det_grid, det_grid))])
                            start_index = 1
                        else:
                            q_new = xr.concat([q_new, xr.DataArray(grid_data, coords=[('lat', np.arange(tos_lonlat[2], tos_lonlat[3]+det_grid, det_grid)), ('lon', np.arange(tos_lonlat[0], tos_lonlat[1]+det_grid, det_grid))])], dim='time')
            except:
                q = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + url[0])
                q = q['tos'].sel(time=q.time.dt.month.isin([7, 8])).sel(time=slice(time[0] + '-01-01', time[1] + '-12-31'))
                lon = np.array(q['nav_lon'].to_numpy().flatten()[::n]).reshape(-1, 1)
                lat = np.array(q['nav_lat'].to_numpy().flatten()[::n]).reshape(-1, 1)
                points = np.concatenate([lon, lat], axis=1)
                for iurl in url:
                    tqdm.write(f'{iurl}插值中')
                    q = xr.open_dataset(tos_dataurl + '/' + ModelName_tos + '/' + iurl)
                    q = q['tos'].sel(time=q.time.dt.month.isin([7, 8])).sel(time=slice(time[0] + '-01-01', time[1] + '-12-31'))
                    for i in tqdm(q, desc=f'\t插值 {ModelName_tos}模式', unit='Year', position=0, colour='green'):
                        data = np.array(i.to_numpy().flatten()[::n]).reshape(-1, 1)
                        lon_grid, lat_grid = np.meshgrid(np.arange(tos_lonlat[0], tos_lonlat[1] + det_grid, det_grid),
                                                             np.arange(tos_lonlat[2], tos_lonlat[3] + det_grid, det_grid))
                        interp = griddata(points, data, (lon_grid, lat_grid), method='linear')
                        grid_data = interp[:, :, 0]
                        # 保证纬度从上到下是递减的
                        if lat_grid[0, 0] < lat_grid[1, 0]:
                            lat_grid = lat_grid[-1::-1]
                            grid_data = grid_data[-1::-1]
                        if start_index == 0:
                            q_new = xr.DataArray(grid_data, coords=[('lat', np.arange(tos_lonlat[2], tos_lonlat[3]+det_grid, det_grid)), ('lon', np.arange(tos_lonlat[0], tos_lonlat[1]+det_grid, det_grid))])
                            start_index = 1
                        else:
                            q_new = xr.concat([q_new, xr.DataArray(grid_data, coords=[('lat', np.arange(tos_lonlat[2], tos_lonlat[3]+det_grid, det_grid)), ('lon', np.arange(tos_lonlat[0], tos_lonlat[1]+det_grid, det_grid))])], dim='time')
        q_new = xr.DataArray(q_new, coords=[('time', np.array(date)), ('lat', np.arange(tos_lonlat[2], tos_lonlat[3]+det_grid, det_grid)), ('lon', np.arange(tos_lonlat[0], tos_lonlat[1]+det_grid, det_grid))])
        q_new.to_netcdf(rf'D:\CODES\Python\PythonProject\cache\CMIP6_tos\interp_global\{ModelName_tos}.nc')
        q_new.close()
    tqdm.write(f'{ModelName_tos}插值完成')
