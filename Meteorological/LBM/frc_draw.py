import xarray as xr
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from toolbar.curved_quivers_master.modplot import velovect

lbm = xr.open_dataset(r'\\wsl.localhost\Ubuntu-20.04\home\hopsong\lbm\data\out\Output_frc.t42l20.Tingyang.nc')
u = lbm['u'][19:25].mean('time')
v = lbm['v'][19:25].mean('time')
t = lbm['t'][19:25].mean('time')
z = lbm['z'][19:25].mean('time')
lon = lbm['lon']
lat = lbm['lat']
# 绘图
# 图1
lev = 200
n = 10
lev_Z = np.array([-1, -0.8, -0.6, -0.4, -0.2, -0.05, 0.05, 0.2, 0.4, 0.6, 0.8, 1]) * 4
lev_T = np.array([-.1, -.09, -.08, -.07, -.06, -.05, -.04, -.03, -.02, -.01,-.002, .002, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1]) * 4
extent1 = [-180, 180, -30, 80]
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(311, projection=ccrs.PlateCarree(central_longitude=180))
ax1.coastlines()
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
T, lon_T = add_cyclic_point(t.sel(lev=lev), coord=lon)
Z, lon_Z = add_cyclic_point(z.sel(lev=lev), coord=lon)
U, lon_UV = add_cyclic_point(u.sel(lev=lev), coord=lon)
V, lon_UV = add_cyclic_point(v.sel(lev=lev), coord=lon)
t200 = ax1.contourf(lon_Z, lat, Z, levels=lev_Z, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree(central_longitude=0))
wind200 = ax1.quiver(lon_UV, lat, U, V, transform=ccrs.PlateCarree(), scale=25, color='black', regrid_shape=20)
wind200 = velovect(ax1, lon_UV, lat, U, V, arrowstyle='fancy', arrowsize=.3, scale=1.75, grains=20, linewidth=0.75,
                    color='black', transform=ccrs.PlateCarree(central_longitude=0))

# 图2
lev = 500
lev_Z = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
extent1 = [-180, 180, -30, 80]
ax2 = fig.add_subplot(312, projection=ccrs.PlateCarree(central_longitude=180))
ax2.coastlines()
ax2.set_extent(extent1, crs=ccrs.PlateCarree())
T, lon_T = add_cyclic_point(t.sel(lev=lev), coord=lon)
Z, lon_Z = add_cyclic_point(z.sel(lev=lev), coord=lon)
U, lon_UV = add_cyclic_point(u.sel(lev=lev), coord=lon)
V, lon_UV = add_cyclic_point(v.sel(lev=lev), coord=lon)
t500 = ax2.contourf(lon_Z, lat, T, levels=lev_T, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree(central_longitude=0))
wind500 = ax2.quiver(lon_UV, lat, U, V, transform=ccrs.PlateCarree(), scale=25, color='black', regrid_shape=20)

# 图1
lev = 850
lev_Z = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
extent1 = [-180, 180, -30, 80]
ax3 = fig.add_subplot(313, projection=ccrs.PlateCarree(central_longitude=180))
ax3.coastlines()
ax3.set_extent(extent1, crs=ccrs.PlateCarree())
T, lon_T = add_cyclic_point(t.sel(lev=lev), coord=lon)
Z, lon_Z = add_cyclic_point(z.sel(lev=lev), coord=lon)
U, lon_UV = add_cyclic_point(u.sel(lev=lev), coord=lon)
V, lon_UV = add_cyclic_point(v.sel(lev=lev), coord=lon)
t850 = ax3.contourf(lon_Z, lat, T, levels=lev_T, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree(central_longitude=0))
wind850 = ax3.quiver(lon_UV, lat, U, V, transform=ccrs.PlateCarree(), scale=15, color='black', regrid_shape=20)
plt.savefig(r'C:\Users\10574\Desktop\LBM_test.png', dpi=1000, bbox_inches='tight')
plt.show()
