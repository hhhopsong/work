import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import tqdm as tq

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MultipleLocator

from toolbar.curved_quivers.modplot import *

def pic(fig, pic_loc, lat, lon, corr_u, corr_v, corr_z, corr_t2m):
    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    ax.set_title(f'{str(picloc)[2]}) Type{str(picloc)[2]}', loc='left', fontsize=10)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    contf = ax.contourf(t2m['lon'], t2m['lat'], corr_t2m, cmap=cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
                        levels=[-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5], extend='both', transform=ccrs.PlateCarree(central_longitude=0))
    cont = ax.contour(lon, lat, corr_z, colors='red', levels=[.1, .2, .3, .4, .5], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(lon, lat, corr_z, colors='blue', levels=[-.5, -.4, -.3, -.2, -.1], linestyles='--', linewidths=0.4,
                       transform=ccrs.PlateCarree(central_longitude=0))
    cont.clabel(inline=1, fontsize=4)
    cont_.clabel(inline=1, fontsize=4)
    #cont_clim = ax.contour(lon, lat, uvz_clim['z'], colors='k', levels=20, linewidths=0.6, transform=ccrs.PlateCarree(central_longitude=0))
    #Cq = Curlyquiver(ax, lon, lat, corr_u, corr_v,
    #                              lon_trunc=-70, arrowsize=.8, scale=20, linewidth=0.6, regrid=20, color='black',
    #                              transform=ccrs.PlateCarree(central_longitude=0))
    #Cq.key(fig, U=1, label='1')
    q = ax.quiver(lon, lat, corr_u, corr_v,
                   transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=20,
                   scale=10, width=.004, headwidth=3, headlength=4, headaxislength=3.5,
                   color='black')
    ax.quiverkey(q, 0.85, 1.05, .5, r'0.5', labelpos='E',
                   fontproperties={'size': 8})
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.2)
    ax.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_长江流域\TPBoundary2500m_长江流域.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='gray', edgecolor='black', linewidth=.5)
    # 刻度线设置
    xticks1 = np.arange(60, 160, 20)
    yticks1 = np.arange(0, 60, 15)
    ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks1, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)

    ymajorLocator = MultipleLocator(15)  # 先定义xmajorLocator，再进行调用
    ax.yaxis.set_major_locator(ymajorLocator)  # x轴最大刻度
    yminorLocator = MultipleLocator(5)
    ax.yaxis.set_minor_locator(yminorLocator)  # x轴最小刻度
    xmajorLocator = MultipleLocator(30)  # 先定义xmajorLocator，再进行调用
    xminorLocator = MultipleLocator(10)
    ax.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
    ax.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
    # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
    ax.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax.tick_params(axis='both', labelsize=6, colors='black')

def corr(K, uvz):
    K_ = (K - K.mean()) / K.std()  # 标准化
    shape = uvz.shape
    uvz = uvz.reshape(shape[0] * shape[1], shape[2])
    corr_ = np.array([np.corrcoef(d, K_)[0, 1] for d in tq.tqdm(uvz)]).reshape(shape[0], shape[1])
    return corr_


K_type = xr.open_dataset(r"D:\PyFile\p2\data\Time_type_95%_0.45_4.nc")

try:
    uvz = xr.open_dataset(r"D:\PyFile\p2\data\uvz_678.nc")
except:
    uvz = xr.open_dataset(r"E:\data\ERA5\ERA5_pressLev\era5_pressLev.nc").sel(
        date=slice('1961-01-01', '2023-12-31'),
        pressure_level=[200, 300, 400, 500, 600, 700, 850],
        latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])
    uvz = xr.Dataset(
            {'u': (['time', 'p', 'lat', 'lon'],uvz['u'].data),
                      'v': (['time', 'p', 'lat', 'lon'],uvz['v'].data),
                      'z': (['time', 'p', 'lat', 'lon'],uvz['z'].data)},
                     coords={'time': pd.to_datetime(uvz['date'], format="%Y%m%d"),
                             'p': uvz['pressure_level'].data,
                             'lat': uvz['latitude'].data,
                             'lon': uvz['longitude'].data})
    uvz = uvz.sel(time=slice('1961-01-01', '2022-12-31'))
    uvz = uvz.sel(time=uvz['time.month'].isin([6, 7, 8])).groupby('time.year').mean('time')
    uvz.to_netcdf(r"D:\PyFile\p2\data\uvz_678.nc")
uvz = uvz.sel(p=500).transpose('lat', 'lon', 'year')  # 500hPa
uvz_clim = uvz.mean('year')

try:
    t2m = xr.open_dataset(r"D:\PyFile\p2\data\t2m_678.nc")
except:
    t2m = xr.open_dataset(r"E:\data\ERA5\ERA5_singleLev\ERA5_sgLEv.nc")['t2m']
    t2m = t2m.sel(date=slice('1961-01-01', '2023-12-31'))
    t2m = xr.Dataset(
            {'t2m': (['time', 'lat', 'lon'],t2m.data)},
                     coords={'time': pd.to_datetime(t2m['date'], format="%Y%m%d"),
                             'lat': t2m['latitude'].data,
                             'lon': t2m['longitude'].data})
    t2m = t2m.sel(time=slice('1961-01-01', '2022-12-31'))
    t2m = t2m.sel(time=t2m['time.month'].isin([6, 7, 8])).groupby('time.year').mean('time')
    t2m.to_netcdf(r"D:\PyFile\p2\data\t2m_678.nc")
t2m = t2m.transpose('lat', 'lon', 'year')
t2m_clim = t2m.mean('year')

fig = plt.figure(figsize=(10, 10))
for i in K_type['type']:
    picloc = int(100 + len(K_type['type'])*10 + i)
    time_ser = K_type.sel(type=i)['K'].data
    time_ser = time_ser - np.polyval(np.polyfit(range(len(time_ser)), time_ser, 1), range(len(time_ser)))
    corr_K_u = corr(time_ser, uvz['u'].data)
    corr_K_v = corr(time_ser, uvz['v'].data)
    corr_K_z = corr(time_ser, uvz['z'].data)
    corr_K_t2m = corr(time_ser, t2m['t2m'].data)
    pic(fig, picloc, uvz['lat'], uvz['lon'], corr_K_u, corr_K_v, corr_K_z, corr_K_t2m)
plt.savefig(r"D:\PyFile\p2\pic\K_uvz_t2m.png", dpi=600, bbox_inches='tight')
plt.show()