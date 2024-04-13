import cmaps
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
from cnmaps import get_adm_maps, draw_maps
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import filters


# 读取数据
def read_data(file_path, var, time, level):
    data = xr.open_dataset(file_path)
    slat = slice(45, 10)
    slon = slice(90, 140)
    t = data[var].loc[time, level, slat, slon]
    lat = t['latitude']
    lon = t['longitude']
    return t.data, lat.data, lon.data


# 读取数据
a = 6.371e6
rh = 'specific_humidity.nc'
t = 't.nc'
u = 'uv_wind.nc'
v = 'uv_wind.nc'
#######
aT = ['2000-04-13T00:00:00', '2000-04-13T12:00:00', '2000-04-14T00:00:00', '2000-04-14T12:00:00', '2000-04-15T00:00:00', '2000-04-15T12:00:00', '2000-04-16T00:00:00', '2000-04-16T12:00:00']
alev = [500, 700, 850, 1000]
#######
extent1=[90, 140, 15, 45]
proj = ccrs.PlateCarree()
level1 = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
xticks1=np.arange(extent1[0], extent1[1]+1, 5)
yticks1=np.arange(extent1[2], extent1[3]+1, 5)
for lev in alev:
    for T in aT:
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        fig1 = plt.figure(figsize=(15, 13))
        u_lev, u_lat, u_lon = read_data(u, 'u', T, lev)
        v_lev, v_lat, v_lon = read_data(v, 'v', T, lev)
        dx, dy = mpcalc.lat_lon_grid_deltas(u_lon, u_lat)
        u_lev = filters.gaussian_filter(u_lev, 3)
        v_lev = filters.gaussian_filter(v_lev, 3)
        r_v_a = mpcalc.advection(mpcalc.vorticity(u_lev * units("m/s"), v_lev * units("m/s"), dx=dx, dy=dy), u_lev, v_lev, dx=dx, dy=dy, x_dim=-1, y_dim=-2)
        t_lev, t_lat, t_lon = read_data(t, 't', T, lev)
        t_lev = filters.gaussian_filter(t_lev, 3)
        t_lev_a = mpcalc.advection(t_lev * units("K"), u_lev, v_lev, dx=dx, dy=dy, x_dim=-1, y_dim=-2)
        rh_lev, rh_lat, rh_lon = read_data(rh, 'q', T, lev)
        D_q = mpcalc.divergence(rh_lev * u_lev, rh_lev * v_lev, dx=dx, dy=dy) / 9.8
        ##########
        ax1 = fig1.add_subplot(111, projection=proj)
        ax1.set_extent(extent1, crs=proj)
        a1 = ax1.contourf(u_lon, u_lat, D_q*10**9, cmap=cmaps.MPL_BrBG_r, levels=level1, extend='both', transform=proj)
        n = 4
        #a11 = ax1.quiver(u_lon[::n], u_lat[::n], D_q[::n, ::n]*10**3, D_q[::n, ::n]*10**3, transform=proj)
        cb1 = plt.colorbar(a1, orientation='vertical', aspect=30, shrink=0.6)#orientation为水平或垂直
        # 刻度线设置
        ax1.set_xticks(xticks1, crs=proj)
        ax1.set_yticks(yticks1, crs=proj)
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax1.xaxis.set_major_formatter(lon_formatter)
        ax1.yaxis.set_major_formatter(lat_formatter)
        draw_maps(get_adm_maps(level='国'), linewidth=0.4)
        draw_maps(get_adm_maps(level='省'), linewidth=0.2)
        ax1.add_feature(cfeature.COASTLINE)
        ax1.tick_params(axis='both', labelsize=18, colors='black')
        plt.savefig(f"C:/Users/10574/Desktop/picture/{T[5:7]}{T[8:10]}{eval(T[11:13])+8:>02}-{lev:.0f}hPa-水汽通量散度(x10^-2).png", dpi=500, bbox_inches='tight')
        plt.close()
