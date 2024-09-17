import cartopy.crs as ccrs
import cmaps

from matplotlib import pyplot as plt

from LBM.force_file import horizontal_profile as hp
from LBM.force_file import vertical_profile as vp
from LBM.force_file import mk_grads, mk_wave, interp3d_lbm

from metpy.units import units
import metpy.calc as mpcalc
from metpy.xarray import grid_deltas_from_dataarray

import xarray as xr
import numpy as np

from toolbar.significance_test import corr_test

info = xr.open_dataset(r"D:\CODES\Python\Meteorological\paper1\cache\uvz\z\diff\z_24_4_4.nc")
z200 = np.load(r"D:\CODES\Python\Meteorological\paper1\cache\uvz\z\corr1\corr_200_24_4_4.npy")
z500 = np.load(r"D:\CODES\Python\Meteorological\paper1\cache\uvz\z\corr1\corr_500_24_4_4.npy")
z600 = np.load(r"D:\CODES\Python\Meteorological\paper1\cache\uvz\z\corr1\corr_600_24_4_4.npy")
z700 = np.load(r"D:\CODES\Python\Meteorological\paper1\cache\uvz\z\corr1\corr_700_24_4_4.npy")
z850 = np.load(r"D:\CODES\Python\Meteorological\paper1\cache\uvz\z\corr1\corr_850_24_4_4.npy")
frc = xr.Dataset({'z':(['lev', 'lat', 'lon'], np.array([z850, z700, z600, z500, z200]))},
                 coords={'lev': [850, 700, 600, 500, 200], 'lat': info['lat'], 'lon': info['lon']})
# 读取强迫场
# 选择45-90N，35W-35E的区域
ols = np.load(r"cache\OLS_detrended.npy")  # 读取缓存
lon, lat = np.meshgrid(frc['lon'], frc['lat'])
mask = ((np.where(lon<= 35, 1, 0) + np.where(lon>= 325, 1, 0))
        * (np.where(lat>= 45, 1, 0) * np.where(lat<= 80, 1, 0))
        * np.where(frc['z'] <= 0, 1, 0)
        * corr_test(ols, frc['z'], alpha=0.05, other=0))
frc_mask = frc.where(mask != 0, 0)

heights = frc_mask['z'] * units('m^2/s^2')
dx, dy = mpcalc.lat_lon_grid_deltas(np.meshgrid(heights.lon, heights.lat)[0], np.meshgrid(heights.lon, heights.lat)[1])

vor = np.zeros_like(heights)
# 计算相对涡度
p=0
for i in [850, 700, 600, 500, 200]:
    ug, vg = mpcalc.geostrophic_wind(heights.sel(lev=i)*(np.std(info['z'].sel(p=i), axis=0)/np.std(ols)), dx=dx, dy=dy) # 计算从高度或重力势给出的地转风
    vor[p] = mpcalc.vorticity(ug, vg, dx=dx, dy=dy)  # 计算水平风的垂直涡度
    p += 1

vor = xr.Dataset({'v':(['lev', 'lat', 'lon'], vor)},
                    coords={'lev': [850, 700, 600, 500, 200], 'lat': info['lat'], 'lon': info['lon']})
frc_nc_sigma = interp3d_lbm(vor)
frc_nc_p = interp3d_lbm(vor, 'p')
# 绘图
# 图1
var = 'v' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
lev = 200
n = 10
extent1 = [-180, 180, -80, 80]
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
ax1.coastlines(linewidths=0.3)
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
lev_range = np.linspace(-np.max(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.max(np.abs(frc_nc_p[var].sel(lev=lev).data)), 10)
t200 = ax1.contourf(frc_nc_p[var]['lon'], frc_nc_p[var]['lat'], frc_nc_p[var].sel(lev=lev),
                    levels=lev_range, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree(central_longitude=0), extend='both')
plt.show()

if input("是否导出?(y/n)") == 'y':
    frc_nc_sigma.to_netcdf(r'D:\CODES\Python\Meteorological\frc_nc\frc.t42l20.Tingyang.nc')

