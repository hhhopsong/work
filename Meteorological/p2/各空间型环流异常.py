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
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator

from toolbar.curved_quivers.modplot import *
from toolbar.significance_test import r_test
from toolbar.lonlat_transform import *

from matplotlib import ticker
from metpy.calc import vertical_velocity
from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants


# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'

nanmax = None
type_name = ['', '500UVZ&T2M of MLB Type', '500UVZ&T2M of ALL Type', '500UVZ&T2M of UB Type']
def pic(fig, pic_loc, lat, lon, corr_u, corr_v, corr_z, corr_t2m):
    global lev_t, nanmax
    pic_ind = ['', 'd', 'e', 'f']
    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    ax.set_title(f'{pic_ind[eval(str(picloc)[2])]}) {type_name[eval(str(picloc)[2])]}', loc='left', fontsize=12)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    contf = ax.contourf(t2m['lon'], t2m['lat'], corr_t2m[0], cmap=cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
                        levels=lev_t, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
    # 显著性打点
    p_test = np.where(np.abs(corr_t2m[1]) >= r_test(62), 0, np.nan)
    p = ax.quiver(t2m['lon'], t2m['lat'], p_test, p_test, transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=60, color='k', scale=20, headlength=2, headaxislength=2)
    cont = ax.contour(lon, lat, corr_z[0], colors='red', levels=[20, 40, 60], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(lon, lat, corr_z[0], colors='blue', levels=[-60, -40, -20], linestyles='--', linewidths=0.4,
                       transform=ccrs.PlateCarree(central_longitude=0))
    cont.clabel(inline=1, fontsize=4)
    cont_.clabel(inline=1, fontsize=4)
    #cont_clim = ax.contour(lon, lat, uvz_clim['z'], colors='k', levels=20, linewidths=0.6, transform=ccrs.PlateCarree(central_longitude=0))
    if nanmax:
        Cq = Curlyquiver(ax, lon, lat, corr_u[0], corr_v[0], center_lon=110, scale=20, linewidth=0.2, arrowsize=.5,
                         regrid=15, color='k', nanmax=nanmax)
    else:
        Cq = Curlyquiver(ax, lon, lat, corr_u[0], corr_v[0], center_lon=110, scale=20, linewidth=0.2, arrowsize=.5,
                         regrid=15, color='k')
    Cq.key(fig, U=1, label='1 m/s', color='k')
    nanmax = Cq.nanmax
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.2)
    ax.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_长江流域\TPBoundary2500m_长江流域.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='gray', edgecolor='black', linewidth=.5)

    # 刻度线设置
    xticks1 = np.arange(60, 160, 20)
    yticks1 = np.arange(0, 60, 15)
    if pic_loc == 331 or pic_loc== 334 or pic_loc== 337: ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
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
    ax.tick_params(axis='both', labelsize=10, colors='black')

    return contf

def pic2(fig, pic_loc, lat, lon, lat_f, lon_f, contour_1, contourf_1, lev, lev_f, r_N, color, clabel_tf, cmap, title):
    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    ax.set_title(f'{title}', loc='left', fontsize=12)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    contf = ax.contourf(lon_f, lat_f, contourf_1[0], cmap=cmap,
                        levels=lev_f, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
    # 显著性打点
    p_test = np.where(np.abs(contourf_1[1]) >= r_test(r_N), 0, np.nan)
    p = ax.quiver(lon_f, lat_f, p_test, p_test, transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=60, color='k', scale=20, headlength=2, headaxislength=2)
    # cont = ax.contour(lon, lat, contour_1[0], colors=color[1], levels=lev[1], linewidths=0.4, transform=ccrs.PlateCarree(central_longitude=0))
    # cont_ = ax.contour(lon, lat, contour_1[0], colors=color[0], levels=lev[0], linestyles='--', linewidths=0.4,
    #                    transform=ccrs.PlateCarree(central_longitude=0))
    # if clabel_tf:
    #     cont.clabel(inline=1, fontsize=4)
    #     cont_.clabel(inline=1, fontsize=4)

    low_patch = ax.contourf(lon, lat, contour_1[0],
                            levels=[-9e9, lev[0][0], lev[0][1]],
                            colors='none', transform=ccrs.PlateCarree(central_longitude=0),
                            hatches=['xxxxxxxxxxxxxxxxxxxxxxxx', '////////////////////////'], alpha=0)
    high_patch = ax.contourf(lon, lat, contour_1[0],
                             levels=[lev[1][0], lev[1][1], 9e9],
                             colors='none', transform=ccrs.PlateCarree(central_longitude=0),
                             hatches=[r'\\\\\\\\\\\\\\\\\\\\\\\\', 'xxxxxxxxxxxxxxxxxxxxxxxx'], alpha=0)
    plt.rcParams['hatch.linewidth'] = 0.1
    for collection in low_patch.collections:
        collection.set_edgecolor(color[0])  # -----打点颜色设置
    for collection in high_patch.collections:
        collection.set_edgecolor(color[1])  # -----打点颜色设置

    #cont_clim = ax.contour(lon, lat, uvz_clim['z'], colors='k', levels=20, linewidths=0.6, transform=ccrs.PlateCarree(central_longitude=0))

    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.2)
    ax.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_长江流域\TPBoundary2500m_长江流域.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='gray', edgecolor='black', linewidth=.5)

    # 刻度线设置
    xticks1 = np.arange(60, 160, 20)
    yticks1 = np.arange(0, 60, 15)
    if pic_loc == 331 or pic_loc== 334 or pic_loc== 337: ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
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
    ax.tick_params(axis='both', labelsize=10, colors='black')

    return contf

def regress(time_series, data):
    # 将 data 重塑为二维：时间轴为第一个维度
    reshaped_data = data.reshape(len(time_series), -1)

    # 减去均值以中心化（标准化自变量和因变量）
    time_series_mean = time_series - np.mean(time_series)
    data_mean = reshaped_data - np.mean(reshaped_data, axis=0)

    # 计算分子（协方差的分子）
    numerator = np.sum(data_mean * time_series_mean[:, np.newaxis], axis=0)

    # 计算分母（自变量的平方和）
    denominator = np.sum(time_series_mean ** 2)

    # 计算回归系数
    regression_coef = numerator / denominator
    correlation = numerator / (np.sqrt(np.sum(data_mean ** 2, axis=0)) * np.sqrt(np.sum(time_series_mean ** 2)))
    # 重塑为 (lat, lon)
    regression_map = regression_coef.reshape(data.shape[1:])
    correlation_map = correlation.reshape(data.shape[1:])
    return regression_map, correlation_map


K_type = xr.open_dataset(r"D:\PyFile\p2\data\Time_type_AverFiltAll0.9%_0.3%_3.nc")

try:
    uvz = xr.open_dataset(r"D:\PyFile\p2\data\uvz_78.nc")
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
    uvz = uvz.sel(time=uvz['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    uvz.to_netcdf(r"D:\PyFile\p2\data\uvz_78.nc")
uvz = uvz.sel(p=500).transpose('year', 'lat', 'lon')  # 500hPa
uvz_clim = uvz.mean('year')

try:
    t2m = xr.open_dataset(r"D:\PyFile\p2\data\t2m_78.nc")
except:
    t2m = xr.open_dataset(r"E:\data\ERA5\ERA5_singleLev\ERA5_sgLEv.nc")['t2m']
    t2m = t2m.sel(date=slice('1961-01-01', '2023-12-31'))
    t2m = xr.Dataset(
            {'t2m': (['time', 'lat', 'lon'],t2m.data)},
                     coords={'time': pd.to_datetime(t2m['date'], format="%Y%m%d"),
                             'lat': t2m['latitude'].data,
                             'lon': t2m['longitude'].data})
    t2m = t2m.sel(time=slice('1961-01-01', '2022-12-31'))
    t2m = t2m.sel(time=t2m['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    t2m.to_netcdf(r"D:\PyFile\p2\data\t2m_78.nc")
t2m = t2m.transpose('year', 'lat', 'lon')
t2m_clim = t2m.mean('year')


w = xr.open_dataset(r"D:/PyFile/p2/data/W.nc")
w = w.sel(level=900).transpose('year', 'lat', 'lon')  # 500hPa
w_clim = w.mean('year')

try:
    qdiv = xr.open_dataset(r"D:\PyFile\p2\data\qdiv_all_78.nc")
except:
    U = xr.open_dataset(r"D:/PyFile/p2/data/U.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000])
    V = xr.open_dataset(r"D:/PyFile/p2/data/V.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000])
    Q = xr.open_dataset(r"D:/PyFile/p2/data/Q.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000])
    U = transform(U['u'], lon_name='lon', type='180->360')
    V = transform(V['v'], lon_name='lon', type='180->360')
    Q = transform(Q['q'], lon_name='lon', type='180->360')

    Qu = Q * units('kg/kg') * U * units('m/s') / constants.g * 1000
    Qv = Q * units('kg/kg') * V * units('m/s') / constants.g * 1000

    # 垂直积分
    integrated_qx = -Qu.integrate(coord='level')
    integrated_qy = -Qv.integrate(coord='level')

    dx, dy = mpcalc.lat_lon_grid_deltas(Qu.lon.data, Qu.lat.data)

    # 计算整层水汽通量散度
    # mpcalc.divergence(integrated_qx, integrated_qy, dx=dx, dy=dy)
    Q_div = np.array([mpcalc.divergence(integrated_qx[iYear,  :, :], integrated_qy[iYear,  :, :], dx=dx, dy=dy) for iYear in range(len(Qu['year']))])

    qdiv = xr.Dataset(
        {'qdiv': (['year', 'lat', 'lon'], Q_div)},
        coords={'year': Qu['year'],
                'lat': Qu['lat'],
                'lon': Qu['lon']})
    qdiv.to_netcdf(r"D:\PyFile\p2\data\qdiv_all_78.nc")
qdiv = qdiv.transpose('year', 'lat', 'lon')  # 500hPa

tcc = xr.open_dataset(r"D:\PyFile\p2\data\TCC.nc")
tcc = tcc.transpose('year', 'lat', 'lon')  # 500hPa
tcc_clim = tcc.mean('year')

q = qdiv['qdiv'].data

fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(wspace=0.1)
lev_t = np.array([-.5, -.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4, .5])
for i in K_type['type']:
    picloc = int(330 + i)
    time_ser = K_type.sel(type=i)['K'].data
    # if i == 2:
    #     time_ser = time_ser - np.polyval(np.polyfit(range(len(time_ser)), time_ser, 1), range(len(time_ser)))
    if i!=2:
        time_ser = (time_ser - time_ser.mean()) / time_ser.std()
        reg_K_u = regress(time_ser, uvz['u'].data)
        reg_K_v = regress(time_ser, uvz['v'].data)
        reg_K_z = regress(time_ser, uvz['z'].data)
        reg_K_t2m = regress(time_ser, t2m['t2m'].data)
        reg_K_w = regress(time_ser, w['w'].data)
        reg_K_qdiv = regress(time_ser, qdiv['qdiv'].data)
        reg_K_tcc = regress(time_ser, tcc['tcc'].data)
    else:
        time_ser = time_ser[:-1]
        time_ser = (time_ser - time_ser.mean()) / time_ser.std()
        reg_K_u = regress(time_ser, uvz['u'].sel(year=slice(1961, 2021)).data)
        reg_K_v = regress(time_ser, uvz['v'].sel(year=slice(1961, 2021)).data)
        reg_K_z = regress(time_ser, uvz['z'].sel(year=slice(1961, 2021)).data)
        reg_K_t2m = regress(time_ser, t2m['t2m'].sel(year=slice(1961, 2021)).data)
        reg_K_w = regress(time_ser, w['w'].sel(year=slice(1961, 2021)).data)
        reg_K_qdiv = regress(time_ser, qdiv['qdiv'].sel(year=slice(1961, 2021)).data)
        reg_K_tcc = regress(time_ser, tcc['tcc'].sel(year=slice(1961, 2021)).data)
    contourfs = pic(fig, picloc, uvz['lat'], uvz['lon'], reg_K_u, reg_K_v, reg_K_z, reg_K_t2m)
    if i == 1:
        contourfs2 = pic2(fig, picloc+3, tcc['lat'], tcc['lon'], w['lat'], w['lon'], reg_K_tcc, reg_K_w,
                          np.array([[-.04, -.02], [.02, .04]]),
                          np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5])*.025,
                          62, ['blue', 'red'], True, cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
                          f'h) 900W&TCC of MLB Type')
    elif i == 2:
        contourfs2 = pic2(fig, picloc+3, tcc['lat'], tcc['lon'], w['lat'], w['lon'], reg_K_tcc, reg_K_w,
                          np.array([[-.04, -.02], [.02, .04]]),
                          np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5])*.025,
                          61, ['blue', 'red'], True, cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
                          f'i) 900W&TCC of ALL Type')
    elif i == 3:
        contourfs2 = pic2(fig, picloc+3, tcc['lat'], tcc['lon'], w['lat'], w['lon'], reg_K_tcc, reg_K_w,
                          np.array([[-.04, -.02], [.02, .04]]),
                          np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5])*.025,
                          62, ['blue', 'red'], True, cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
                          f'j) 900W&TCC of UB Type')

# 添加全局colorbar  # 为colorbar腾出空间
cbar_ax = fig.add_axes([0.25, 0.38, 0.5, 0.01]) # [left, bottom, width, height]
cbar = fig.colorbar(contourfs, cax=cbar_ax, orientation='horizontal', drawedges=True)
cbar.locator = ticker.FixedLocator(lev_t)
cbar.set_ticklabels([str(i) for i in lev_t])
cbar.ax.tick_params(labelsize=8, length=0)

fig.subplots_adjust(hspace=0.03)
plt.savefig(r"D:\PyFile\p2\pic\图4.png", dpi=600, bbox_inches='tight')
plt.show()