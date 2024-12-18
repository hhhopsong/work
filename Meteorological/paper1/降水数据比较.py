from cartopy import crs as ccrs
import cartopy.feature as cfeature
import multiprocessing
import sys
import cartopy.feature as cfeature
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr
from cartopy import crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
from cartopy.util import add_cyclic_point
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.pyplot import quiverkey
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import filters
from toolbar.significance_test import corr_test
from toolbar.TN_WaveActivityFlux import TN_WAF_3D
from toolbar.curved_quivers.modplot import *


ols = np.load(r"D:\PyFile\paper1\OLS35_detrended.npy")[19:]
gpcp_0 = xr.open_dataset(r"E:\data\NOAA\GPCP\precip.mon.mean.nc")['precip']
gpcp = gpcp_0.sel(time=slice('1979-01-01', '2022-12-31'))
gpcp = gpcp.sel(time=gpcp['time.month'].isin([7, 8])).groupby('time.year').mean('time').transpose('lat', 'lon', 'year')
shape = gpcp.shape
gpcp = gpcp.data if isinstance(gpcp, xr.DataArray) else gpcp
gpcp = gpcp.reshape(shape[0] * shape[1], shape[2])
corr_1 = np.array([np.corrcoef(d, ols)[0, 1] for d in tqdm.tqdm(gpcp)]).reshape(shape[0], shape[1])
pre_diff_0 = xr.open_dataset(fr"D:\PyFile\paper1\cache\pre\pre_same.nc")['precip'].transpose('lat', 'lon', 'year')[:, :, 19:]
shape = pre_diff_0.shape
pre_diff = pre_diff_0.data if isinstance(pre_diff_0, xr.DataArray) else pre_diff_0
pre_diff = pre_diff.reshape(shape[0] * shape[1], shape[2])
corr_2 = np.array([np.corrcoef(d, ols)[0, 1] for d in tqdm.tqdm(pre_diff)]).reshape(shape[0], shape[1])
fig = plt.figure(figsize=(12, 6))
lev = [-.4, -.35, -.3, -.25, -.2, -.15, -.1, .1, .15, .2, .25, .3, .35, .4]
ax = fig.add_subplot(121, projection=ccrs.PlateCarree(central_longitude=180-67.5))
ax.set_title('a) GPCP', loc='left')
ax.coastlines()
ax.set_extent([-180, 180, -30, 90], crs=ccrs.PlateCarree())
ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree(central_longitude=180-67.5))
ax2.set_title('b) PREC', loc='left')
ax2.coastlines()
ax2.set_extent([-180, 180, -30, 90], crs=ccrs.PlateCarree())
gpcp_corr, gpcp_lon = add_cyclic_point(corr_1, coord=gpcp_0['lon'])
ax.contourf(gpcp_lon, gpcp_0.lat, gpcp_corr, levels=lev, extend='both', transform=ccrs.PlateCarree(), cmap=cmaps.MPL_RdYlGn[32+10:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:96-10])
lon, lat = np.meshgrid(gpcp_0.lon, gpcp_0.lat)
mask = np.abs(corr_1) > 0.32
ax.scatter(lon[mask], lat[mask], marker='.', s=.5, color='k', transform=ccrs.PlateCarree())
pre_corr, pre_lon = add_cyclic_point(corr_2, coord=pre_diff_0['lon'])
ax2.contourf(pre_lon, pre_diff_0.lat, pre_corr, levels=lev, extend='both', transform=ccrs.PlateCarree(), cmap=cmaps.MPL_RdYlGn[32+10:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:96-10])
lon2, lat2 = np.meshgrid(pre_diff_0.lon, pre_diff_0.lat)
mask2 = np.abs(corr_2) > 0.32
ax2.scatter(lon2[mask2], lat2[mask2], marker='.', s=.5, color='k', transform=ccrs.PlateCarree())
plt.savefig(r"D:\PyFile\pic\pre_gpcp_corr.png", dpi=600, bbox_inches='tight')