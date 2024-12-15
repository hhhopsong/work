import cartopy.crs as ccrs
import cmaps

from matplotlib import pyplot as plt

from metpy.units import units
import metpy.calc as mpcalc
from metpy.xarray import grid_deltas_from_dataarray
from toolbar.curved_quivers.modplot import Curlyquiver
from scipy.ndimage import filters

import xarray as xr
import xgrads as xg
import numpy as np

from toolbar.LBM.force_file import horizontal_profile as hp
from toolbar.LBM.force_file import vertical_profile as vp
from toolbar.LBM.force_file import mk_grads, mk_wave, interp3d_lbm
from toolbar.significance_test import corr_test

info_z = xr.open_dataset(r"D:\PyFile\paper1\cache\uvz\z_same.nc")
info_u = xr.open_dataset(r"D:\PyFile\paper1\cache\uvz\u_same.nc")
info_v = xr.open_dataset(r"D:\PyFile\paper1\cache\uvz\v_same.nc")
z200 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_z200_same.npy")
u200 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_u200_same.npy") * units('m/s')
v200 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_v200_same.npy") * units('m/s')

z500 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_z500_same.npy")
u500 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_u500_same.npy") * units('m/s')
v500 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_v500_same.npy") * units('m/s')

z600 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_z600_same.npy")
u600 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_u600_same.npy") * units('m/s')
v600 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_v600_same.npy") * units('m/s')

z700 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_z700_same.npy")
u700 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_u700_same.npy") * units('m/s')
v700 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_v700_same.npy") * units('m/s')

z850 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_z850_same.npy")
u850 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_u850_same.npy") * units('m/s')
v850 = np.load(r"D:\PyFile\paper1\cache\uvz\corr_v850_same.npy") * units('m/s')
frc = xr.Dataset({'z':(['lev', 'lat', 'lon'], np.array([z850, z700, z600, z500, z200]))},
                 coords={'lev': [850, 700, 600, 500, 200], 'lat': info_z['lat'], 'lon': info_z['lon']})
uv = xr.Dataset({'u':(['lev', 'lat', 'lon'], np.array([u850, u700, u600, u500, u200])),
                    'v':(['lev', 'lat', 'lon'], np.array([v850, v700, v600, v500, v200]))},
                    coords={'lev': [850, 700, 600, 500, 200], 'lat': info_z['lat'], 'lon': info_z['lon']})
# 读取强迫场
# 选择45-90N，35W-35E的区域
ols = np.load(r"D:\PyFile\paper1\OLS35_detrended.npy")  # 读取缓存

heights = frc['z'] * units('m^2/s^2')
wind = uv['u'] * units('m/s'), uv['v'] * units('m/s')
dx, dy = mpcalc.lat_lon_grid_deltas(np.meshgrid(heights.lon, heights.lat)[0], np.meshgrid(heights.lon, heights.lat)[1])


vor = np.zeros(heights.shape)
p=0
for i in [850, 700, 600, 500, 200]:
    ug, vg = wind[0].sel(lev=i)*(np.std(info_u['u'].sel(p=i), axis=0)/np.std(ols)), wind[1].sel(lev=i)*(np.std(info_v['v'].sel(p=i), axis=0)/np.std(ols))
    vor[p] = filters.gaussian_filter(mpcalc.vorticity(ug, vg, dx=dx, dy=dy), 3) # 计算水平风的垂直涡度
    p += 1

vor = xr.Dataset({'v':(['lev', 'lat', 'lon'], np.where(np.isnan(vor), 0, vor))},
                    coords={'lev': [850, 700, 600, 500, 200], 'lat': info_z['lat'], 'lon': info_z['lon']})

lon, lat = np.meshgrid(frc['lon'], frc['lat'])
mask = ((np.where(lon<= 87.5, 1, 0) * np.where(lon>= 35.00, 1, 0))
        * (np.where(lat>= 49.00, 1, 0) * np.where(lat<= 78.00, 1, 0))
        * np.where(vor['v'] <= 0, 1, 0) * corr_test(ols, frc['z'], alpha=0.05, other=0))

vor_mask = vor.where(mask != 0, 0)

frc_nc_sigma = interp3d_lbm(vor_mask)
frc_nc_p = interp3d_lbm(vor_mask, 'p')
# 绘图
# 图1
var = 'v' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
lev = 500
n = 10
extent1 = [-180, 180, -80, 80]
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
ax1.coastlines(linewidths=0.3)
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
lev_range = np.linspace(-np.max(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.max(np.abs(frc_nc_p[var].sel(lev=lev).data)), 10)
var200 = ax1.contourf(frc_nc_p[var]['lon'], frc_nc_p[var]['lat'], frc_nc_p[var].sel(lev=lev, time=0),
                    levels=lev_range, cmap=cmaps.GMT_polar_r, transform=ccrs.PlateCarree(central_longitude=0), extend='both')
quiver = Curlyquiver(ax1, uv['u']['lon'], uv['u']['lat'], uv['u'].sel(lev=lev), uv['v'].sel(lev=lev),
                  arrowsize=.5, scale=30, linewidth=0.4, regrid=15,
                  transform=ccrs.PlateCarree(central_longitude=0))
plt.show()

if input("是否导出?(1/0)") == '1':
    r'''template = xr.open_dataset(r"D:\CODES\Python\Meteorological\frc_nc\Template.nc")
    template['v'] = frc_nc_sigma['v'].sel(lev2=0.995)
    template['d'] = frc_nc_sigma['d'].sel(lev2=0.995)
    template['t'] = frc_nc_sigma['t'].sel(lev2=0.995)
    template['p'] = frc_nc_sigma['v'].sel(lev2=0.995)'''
    frc_nc_sigma.to_netcdf(r'D:\lbm\main\data\Forcing\frc.t42l20.nc', format='NETCDF3_CLASSIC')
    frc_nc_p.to_netcdf(r'D:\lbm\main\data\Forcing\frc_p.t42l20.nc', format='NETCDF3_CLASSIC')

