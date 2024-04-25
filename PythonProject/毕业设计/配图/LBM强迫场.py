import xarray as xr
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import xgrads as xg


frc = xg.open_CtlDataset(r'//wsl.localhost/Ubuntu-20.04/home/hopsong/lbm/data/frc/draw.ctl')
t = frc['t'][19:30].mean('time')
lon = frc['lon']
lat = frc['lat']
# 绘图
# 图1
lev = 200
n = 10
lev_Z = [-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1]
lev_T = [-.1, -.09, -.08, -.07, -.06, -.05, -.04, -.03, -.02, -.01, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1]
extent1 = [-180, 180, -30, 80]
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(311, projection=ccrs.PlateCarree(central_longitude=180))
ax1.coastlines()
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
T, lon_T = add_cyclic_point(t.sel(lev=lev), coord=lon)
Z, lon_Z = add_cyclic_point(z.sel(lev=lev), coord=lon)
U, lon_UV = add_cyclic_point(u.sel(lev=lev), coord=lon)
V, lon_UV = add_cyclic_point(v.sel(lev=lev), coord=lon)
t200 = ax1.contourf(lon_Z, lat, Z, levels=lev_Z, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree())
wind200 = ax1.quiver(lon_UV, lat, U, V, transform=ccrs.PlateCarree(), scale=25, color='black', regrid_shape=20)
wind200 = ax1.streamplot(lon_T[::n], lat[::n], U[::n, ::n], V[::n, ::n], density=3, linewidth=0.3, arrowsize=0.5, transform=ccrs.PlateCarree())

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
t500 = ax2.contourf(lon_Z, lat, T, levels=lev_T, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree())
wind500 = ax2.quiver(lon_UV, lat, U, V, transform=ccrs.PlateCarree(), scale=25, color='black', regrid_shape=20)
wind500 = ax2.streamplot(lon_T[::n], lat[::n], U[::n, ::n], V[::n, ::n], density=4, linewidth=0.3, arrowsize=0.5, transform=ccrs.PlateCarree())

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
t850 = ax3.contourf(lon_Z, lat, T, levels=lev_T, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree())
wind850 = ax3.quiver(lon_UV, lat, U, V, transform=ccrs.PlateCarree(), scale=15, color='black', regrid_shape=20)
wind850 = ax3.streamplot(lon_T[::n], lat[::n], U[::n, ::n], V[::n, ::n], density=5, linewidth=0.3, arrowsize=0.5, transform=ccrs.PlateCarree())
plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\LBM-AST.png', dpi=1000, bbox_inches='tight')
plt.show()
pass