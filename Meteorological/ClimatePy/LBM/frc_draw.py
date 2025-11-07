import xarray as xr
import xgrads
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import scipy.ndimage as ndimage
from ..curved_quivers.modplot import velovect


def draw_frc():
    lbm = xr.open_dataset(r'D:\lbm\main\data\Output\Output_frc.t42l20.Tingyang.nc')
    frc_p = xr.open_dataset(r'D:\lbm\main\data\Forcing\frc_p.t42l20.nc')
    u = lbm['u'][19:25].mean('time')
    v = lbm['v'][19:25].mean('time')
    t = lbm['t'][19:25].mean('time')
    z = lbm['z'][19:25].mean('time')
    lon = lbm['lon']
    lat = lbm['lat']
    frc_lon = frc_p['lon']
    frc_lat = frc_p['lat']
    # 绘图
    # 图1
    lev = 200
    n = 10
    extent1 = [-180, 180, -80, 80]
    fig = plt.figure()
    ax1 = fig.add_subplot(411, projection=ccrs.PlateCarree(central_longitude=180))
    ax1.set_title('200hPa UVZ', fontsize=4, loc='left')
    ax1.coastlines(linewidths=0.3)
    ax1.set_extent(extent1, crs=ccrs.PlateCarree())
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    Frc, lon_Frc = frc_p['v'].sel(lev=lev, time=0), frc_lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    level_z = np.linspace(-np.nanmax(np.abs(Z)), np.nanmax(np.abs(Z)), 10)
    level_frc = np.linspace(-np.nanmax(np.abs(frc_p['v'].sel(lev=lev, time=0).data)), np.nanmax(np.abs(frc_p['v'].sel(lev=lev, time=0).data)), 8)
    var200 = ax1.contourf(lon_Z, lat, Z, levels=level_z, cmap=cmaps.GMT_polar[:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    frc200 = ax1.contour(lon_Frc, frc_lat, Frc, levels=level_frc, colors='blue', transform=ccrs.PlateCarree(central_longitude=0), linewidths=0.4, linestyles='-')
    wind200 = velovect(ax1, lon_UV, lat, U, V, arrowsize=.3, scale=20, linewidth=0.3, regrid=15,
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
    Frc, lon_Frc = frc_p['v'].sel(lev=lev, time=0), frc_lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    level_z = np.linspace(-np.nanmax(np.abs(Z)), np.nanmax(np.abs(Z)), 10)
    level_frc = np.linspace(-np.nanmax(np.abs(frc_p['v'].sel(lev=lev, time=0).data)), np.nanmax(np.abs(frc_p['v'].sel(lev=lev, time=0).data)), 8)
    var500 = ax2.contourf(lon_Z, lat, Z, levels=level_z, cmap=cmaps.GMT_polar[:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    frc200 = ax2.contour(lon_Frc, frc_lat, Frc, levels=level_frc, colors='blue', transform=ccrs.PlateCarree(central_longitude=0), linewidths=0.4, linestyles='-')
    wind500 = velovect(ax2, lon_UV, lat, U, V, arrowsize=.3, scale=20, linewidth=0.3, regrid=15,
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
    Frc, lon_Frc = frc_p['v'].sel(lev=lev, time=0), frc_lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    level_z = np.linspace(-np.nanmax(np.abs(Z)), np.nanmax(np.abs(Z)), 10)
    level_frc = np.linspace(-np.nanmax(np.abs(frc_p['v'].sel(lev=lev, time=0).data)), np.nanmax(np.abs(frc_p['v'].sel(lev=lev, time=0).data)), 8)
    var850 = ax3.contourf(lon_Z, lat, Z, levels=level_z, cmap=cmaps.GMT_polar[:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    frc200 = ax3.contour(lon_Frc, frc_lat, Frc, levels=level_frc, colors='blue', transform=ccrs.PlateCarree(central_longitude=0), linewidths=0.4, linestyles='-')
    wind850 = velovect(ax3, lon_UV, lat, U, V, arrowsize=.3, scale=20, linewidth=0.3, regrid=15,
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
    Frc, lon_Frc = frc_p['v'].sel(lev=lev, time=0), frc_lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    level_z = np.linspace(-np.nanmax(np.abs(Z)), np.nanmax(np.abs(Z)), 10)
    level_frc = np.linspace(-np.nanmax(np.abs(frc_p['v'].sel(lev=lev, time=0).data)), np.nanmax(np.abs(frc_p['v'].sel(lev=lev, time=0).data)), 8)
    var850 = ax4.contourf(lon_Z, lat, Z, levels=level_z, cmap=cmaps.GMT_polar[:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    frc200 = ax4.contour(lon_Frc, frc_lat, Frc, levels=level_frc, colors='blue', transform=ccrs.PlateCarree(central_longitude=0), linewidths=0.4, linestyles='-')
    wind850 = velovect(ax4, lon_UV, lat, U, V, arrowsize=.3, scale=20, linewidth=0.3, regrid=15,
                        color='black', transform=ccrs.PlateCarree(central_longitude=0))
    plt.savefig(r'D:\PyFile\pic\Output_frc.png', dpi=1000, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_frc()
