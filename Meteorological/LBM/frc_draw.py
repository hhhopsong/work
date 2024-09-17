import xarray as xr
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import scipy.ndimage as ndimage
from toolbar.curved_quivers_master.modplot import velovect


def draw_frc():
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
    extent1 = [-180, 180, -80, 80]
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(411, projection=ccrs.PlateCarree(central_longitude=180))
    ax1.set_title('200hPa UVZ', fontsize=4, loc='left')
    ax1.coastlines(linewidths=0.3)
    ax1.set_extent(extent1, crs=ccrs.PlateCarree())
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    W = np.where(np.sqrt(U**2 + V**2) > np.sqrt(U**2 + V**2).quantile(0.70), 1, 0)
    U, V = U * W, V * W
    t200 = ax1.contourf(lon_Z, lat, Z, levels=lev_Z, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    wind200 = velovect(ax1, lon_UV, lat, U, V, arrowstyle='fancy', arrowsize=.3, scale=5, grains=30, linewidth=0.75,
                        color='black', transform=ccrs.PlateCarree(central_longitude=0))

    # 图2
    lev = 500
    lev_Z = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    extent1 = extent1
    ax2 = fig.add_subplot(412, projection=ccrs.PlateCarree(central_longitude=180))
    ax2.set_title('500hPa UVZ', fontsize=4, loc='left')
    ax2.coastlines(linewidths=0.3)
    ax2.set_extent(extent1, crs=ccrs.PlateCarree())
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    W = np.where(np.sqrt(U**2 + V**2) > np.sqrt(U**2 + V**2).quantile(0.70), 1, 0)
    U, V = U * W, V * W
    t500 = ax2.contourf(lon_Z, lat, Z, levels=lev_Z, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    wind500 = velovect(ax2, lon_UV, lat, U, V, arrowstyle='fancy', arrowsize=.3, scale=5, grains=30, linewidth=0.75,
                        color='black', transform=ccrs.PlateCarree(central_longitude=0))

    # 图1
    lev = 700
    lev_Z = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    extent1 = extent1
    ax3 = fig.add_subplot(413, projection=ccrs.PlateCarree(central_longitude=180))
    ax3.set_title('700hPa UVZ&T', fontsize=4, loc='left')
    ax3.coastlines(linewidths=0.3)
    ax3.set_extent(extent1, crs=ccrs.PlateCarree())
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    W = np.where(np.sqrt(U**2 + V**2) > np.sqrt(U**2 + V**2).quantile(0.70), 1, 0)
    U, V = U * W, V * W
    t850 = ax3.contourf(lon_Z, lat, T, levels=lev_T, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    z850 = ax3.contour(lon_Z, lat, Z, levels=4, colors='black', transform=ccrs.PlateCarree(central_longitude=0), linewidths=0.4)
    wind850 = velovect(ax3, lon_UV, lat, U, V, arrowstyle='fancy', arrowsize=.3, scale=5, grains=30, linewidth=0.75,
                        color='black', transform=ccrs.PlateCarree(central_longitude=0))

    # 图1
    lev = 850
    lev_Z = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    extent1 = extent1
    ax4 = fig.add_subplot(414, projection=ccrs.PlateCarree(central_longitude=180))
    ax4.set_title('850hPa UVZ&T', fontsize=4, loc='left')
    ax4.coastlines(linewidths=0.3)
    ax4.set_extent(extent1, crs=ccrs.PlateCarree())
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    W = np.where(np.sqrt(U**2 + V**2) > np.sqrt(U**2 + V**2).quantile(0.70), 1, 0)
    U, V = U * W, V * W
    t850 = ax4.contourf(lon_Z, lat, T, levels=lev_T, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    z850 = ax4.contour(lon_Z, lat, Z, levels=4, colors='black', transform=ccrs.PlateCarree(central_longitude=0), linewidths=0.4)
    wind850 = velovect(ax4, lon_UV, lat, U, V, arrowstyle='fancy', arrowsize=.3, scale=5, grains=30, linewidth=0.75,
                        color='black', transform=ccrs.PlateCarree(central_longitude=0))
    plt.savefig(r'D:\CODES\Python\Meteorological\frc_nc\Output.png', dpi=1000, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_frc()
