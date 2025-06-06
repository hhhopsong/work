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
from scipy import ndimage

from toolbar.curved_quivers.modplot import *
from toolbar.masked import masked
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
    ax.set_aspect('auto')
    ax.set_title(f'{pic_ind[eval(str(picloc)[2])-3]}) {type_name[eval(str(picloc)[2])-3]}', loc='left', fontsize=12)
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
    xmajorLocator = MultipleLocator(20)  # 先定义xmajorLocator，再进行调用
    xminorLocator = MultipleLocator(5)
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

def pic2(fig, pic_loc, lat, lon, lat_f, lon_f, lat_pat, lon_pat, contour_1, contourf_1, contpatch, lev, lev_f, lev_pat, r_N, color, clabel_tf, cmap, color_pat,  title):
    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    ax.set_aspect('auto')
    ax.set_title(f'{title}', loc='left', fontsize=12)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    contf = ax.contourf(lon_f, lat_f, contourf_1[0], cmap=cmap,
                        levels=lev_f, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
    # 显著性打点
    p_test = np.where(np.abs(contourf_1[1]) >= r_test(r_N), 0, np.nan)
    p = ax.quiver(lon_f, lat_f, p_test, p_test, transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=60, color='k', scale=20, headlength=2, headaxislength=2)
    cont = ax.contour(lon, lat, contour_1[0], colors=color[1], levels=lev[1], linestyles='--', linewidths=0.4,
                      transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(lon, lat, contour_1[0], colors=color[0], levels=lev[0], linestyles='solid', linewidths=0.4,
                       transform=ccrs.PlateCarree(central_longitude=0))
    if clabel_tf:
        cont.clabel(inline=1, fontsize=4)
        cont_.clabel(inline=1, fontsize=4)

    plt.rcParams['hatch.color'] = color_pat[0]
    plt.rcParams['hatch.linewidth'] = 0.2
    low_patch = ax.contourf(lon_pat, lat_pat, contpatch[0],
                            levels=[-9e9, lev_pat[0][0], lev_pat[0][1]], add_colorbar=False,
                            colors='none', edgecolor='none', transform=ccrs.PlateCarree(central_longitude=0),
                            hatches=['xxxxxxxxxxxxxxxxxxxxxxxx', '////////////////////////'])

    plt.rcParams['hatch.color'] = color_pat[1]
    high_patch = ax.contourf(lon_pat, lat_pat, contpatch[0],
                             levels=[lev_pat[1][0], lev_pat[1][1], 9e9], add_colorbar=False,
                             colors='none', edgecolor='none', transform=ccrs.PlateCarree(central_longitude=0),
                             hatches=[r'\\\\\\\\\\\\\\\\\\\\\\\\', 'xxxxxxxxxxxxxxxxxxxxxxxx'])


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
    xmajorLocator = MultipleLocator(20)  # 先定义xmajorLocator，再进行调用
    xminorLocator = MultipleLocator(5)
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

if __name__ == '__main__':
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
    w = w.sel(level=500).transpose('year', 'lat', 'lon')  # 500hPa
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

    time = [1961, 2022]
    t_budget = xr.open_dataset(r'E:\data\ERA5\ERA5_pressLev\single_var\t_budget_1961_2022.nc').sel(
        time=slice(str(time[0]) + '-01', str(time[1]) + '-12'))
    dTdt_78 = t_budget['dTdt'].sel(time=t_budget['time.month'].isin([6, 7])).groupby('time.year').mean('time')
    adv_T_78 = t_budget['adv_T'].sel(time=t_budget['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    ver_78 = t_budget['ver'].sel(time=t_budget['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    Q_78 = dTdt_78 - adv_T_78 - ver_78

    reg_map, corr_map = np.zeros(
        (3, 4, len(t_budget['Q'].level), len(t_budget['Q'].lat), len(t_budget['Q'].lon))), np.zeros(
        (3, 4, len(t_budget['Q'].level), len(t_budget['Q'].lat), len(t_budget['Q'].lon)))

    p_lev = 900  # 950

    for i in tq.trange(len(K_type['type'])):
        K_series = K_type.sel(type=i + 1)['K'].data
        if i != 1:
            K_series = (K_series - np.mean(K_series)) / np.std(K_series)
            reg_map[i, 0], corr_map[i, 0] = regress(K_series, dTdt_78.data)
            reg_map[i, 1], corr_map[i, 1] = regress(K_series, adv_T_78.data)
            reg_map[i, 2], corr_map[i, 2] = regress(K_series, ver_78.data)
            reg_map[i, 3], corr_map[i, 3] = regress(K_series, Q_78.data)
        else:
            K_series = K_series[:-1]
            K_series = (K_series - np.mean(K_series)) / np.std(K_series)
            reg_map[i, 0], corr_map[i, 0] = regress(K_series, dTdt_78.sel(year=slice(1961, 2021)).data)
            reg_map[i, 1], corr_map[i, 1] = regress(K_series, adv_T_78.sel(year=slice(1961, 2021)).data)
            reg_map[i, 2], corr_map[i, 2] = regress(K_series, ver_78.sel(year=slice(1961, 2021)).data)
            reg_map[i, 3], corr_map[i, 3] = regress(K_series, Q_78.sel(year=slice(1961, 2021)).data)

    reg_map = xr.Dataset({'dTdt': (['type', 'level', 'lat', 'lon'], reg_map[:, 0]),
                          'adv_T': (['type', 'level', 'lat', 'lon'], reg_map[:, 1]),
                          'ver': (['type', 'level', 'lat', 'lon'], reg_map[:, 2]),
                          'Q': (['type', 'level', 'lat', 'lon'], reg_map[:, 3])},
                         coords={'type': K_type['type'], 'level': t_budget['Q'].level, 'lat': t_budget['Q'].lat,
                                 'lon': t_budget['Q'].lon})
    corr_map = xr.Dataset({'dTdt': (['type', 'level', 'lat', 'lon'], corr_map[:, 0]),
                           'adv_T': (['type', 'level', 'lat', 'lon'], corr_map[:, 1]),
                           'ver': (['type', 'level', 'lat', 'lon'], corr_map[:, 2]),
                           'Q': (['type', 'level', 'lat', 'lon'], corr_map[:, 3])},
                          coords={'type': K_type['type'], 'level': t_budget['Q'].level, 'lat': t_budget['Q'].lat,
                                  'lon': t_budget['Q'].lon})

    fig = plt.figure(figsize=(12.44, 7))
    plt.subplots_adjust(wspace=0.05, hspace=0.4)
    lev_t = np.array([-.5, -.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4, .5])

    # 柱状图
    for KType in range(1, 4):
        if KType == 3:
            reg_map_ = masked(reg_map, r"D:\Code\work\Meteorological\p2\map\WYTR\长江_tp.shp")
        elif KType == 1:
            reg_map_ = masked(reg_map, r"D:\Code\work\Meteorological\p2\map\EYTR\长江_tp.shp")
        elif KType == 2:
            reg_map_ = masked(reg_map, r'D:\PyFile\map\self\长江_TP\长江_tp.shp')
        reg_ = reg_map_.sel(type=KType, level=p_lev)

        adv_X_dTdt = np.nanmean(reg_['adv_T']) * 86400 * 31
        ver_X_dTdt = np.nanmean(reg_['ver']) * 86400 * 31
        Q_X_dTdt = np.nanmean(reg_['Q']) * 86400 * 31

        # 准备绘图数据
        variables = ['Adv', 'Ver', 'Q']
        values = [adv_X_dTdt, ver_X_dTdt, Q_X_dTdt]
        colors = ['#ff7373' if val > 0 else '#7373ff' for val in values]

        # 添加标题和网格
        ax = fig.add_subplot(3, 3, KType)
        ax.set_aspect('auto')
        title = ['MLB Type', 'ALL Type', 'UB Type'][KType - 1]
        ax.set_title(f'{chr(ord("a") + KType - 1)}) Temp. pert budget of {title}', fontsize=12, loc='left')
        ax.grid(True, linestyle='--', zorder=0, axis='y')

        bars = ax.bar(range(3), values, width=0.3, color=colors, edgecolor='black', zorder=2)

        # 设置坐标轴标签,字体为 Times New Roman
        ax.set_xticks(range(3))
        ax.set_xticklabels([r'$-(\mathbf{V} \cdot \nabla T)^{\prime}$',
                            r'$(\omega \sigma)^{\prime}$',
                            r'${\dot{Q}}^{\prime}$'], fontsize=12)

        # 设置y轴范围
        ymax = 3
        ax.set_ylim(-ymax, ymax)
        ax.set_xlim(-.5, 2.5)

        # 仅当 KType == 1 时添加y刻度标签
        if KType == 1:
            ax.set_yticks(np.arange(-3, 4, 1))
            ax.set_yticklabels(np.arange(-3, 4, 1), fontsize=12)
        else:
            ax.set_yticks(np.arange(-3, 4, 1))
            ax.set_yticklabels([])
            ax.tick_params(axis='y', left=False)

        # 添加零线
        ax.axhline(0, color='black', lw=1)
        # ax.axvline(2.5, color='black', lw=1)

        # 边框显示为黑色
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

    # 空间图
    for i in K_type['type']:
        picloc = int(333 + i)
        time_ser = K_type.sel(type=i)['K'].data
        # if i == 2:
        #     time_ser = time_ser - np.polyval(np.polyfit(range(len(time_ser)), time_ser, 1), range(len(time_ser)))
        # if i!=2:
        #     time_ser = (time_ser - time_ser.mean()) / time_ser.std()
        #     reg_K_u = regress(time_ser, uvz['u'].data)
        #     reg_K_v = regress(time_ser, uvz['v'].data)
        #     reg_K_z = regress(time_ser, uvz['z'].data)
        #     reg_K_t2m = regress(time_ser, t2m['t2m'].data)
        #     reg_K_w = regress(time_ser, w['w'].data)
        #     reg_K_qdiv = regress(time_ser, qdiv['qdiv'].data)
        #     reg_K_tcc = regress(time_ser, tcc['tcc'].data)
        # else:
        #     time_ser = time_ser[:-1]
        #     time_ser = (time_ser - time_ser.mean()) / time_ser.std()
        #     reg_K_u = regress(time_ser, uvz['u'].sel(year=slice(1961, 2021)).data)
        #     reg_K_v = regress(time_ser, uvz['v'].sel(year=slice(1961, 2021)).data)
        #     reg_K_z = regress(time_ser, uvz['z'].sel(year=slice(1961, 2021)).data)
        #     reg_K_t2m = regress(time_ser, t2m['t2m'].sel(year=slice(1961, 2021)).data)
        #     reg_K_w = regress(time_ser, w['w'].sel(year=slice(1961, 2021)).data)
        #     reg_K_qdiv = regress(time_ser, qdiv['qdiv'].sel(year=slice(1961, 2021)).data)
        #     reg_K_tcc = regress(time_ser, tcc['tcc'].sel(year=slice(1961, 2021)).data)
        from concurrent.futures import ProcessPoolExecutor
        if i != 2:
            time_ser = (time_ser - time_ser.mean()) / time_ser.std()
            with ProcessPoolExecutor() as executor:
                futures = {
                    'u': executor.submit(regress, time_ser, uvz['u'].data),
                    'v': executor.submit(regress, time_ser, uvz['v'].data),
                    'z': executor.submit(regress, time_ser, uvz['z'].data),
                    't2m': executor.submit(regress, time_ser, t2m['t2m'].data),
                    'w': executor.submit(regress, time_ser, w['w'].data),
                    'qdiv': executor.submit(regress, time_ser, qdiv['qdiv'].data),
                    'tcc': executor.submit(regress, time_ser, tcc['tcc'].data)
                }
                reg_K_u = futures['u'].result()
                reg_K_v = futures['v'].result()
                reg_K_z = futures['z'].result()
                reg_K_t2m = futures['t2m'].result()
                reg_K_w = futures['w'].result()
                reg_K_qdiv = futures['qdiv'].result()
                reg_K_tcc = futures['tcc'].result()
        else:
            time_ser = time_ser[:-1]
            time_ser = (time_ser - time_ser.mean()) / time_ser.std()
            with ProcessPoolExecutor() as executor:
                futures = {
                    'u': executor.submit(regress, time_ser, uvz['u'].sel(year=slice(1961, 2021)).data),
                    'v': executor.submit(regress, time_ser, uvz['v'].sel(year=slice(1961, 2021)).data),
                    'z': executor.submit(regress, time_ser, uvz['z'].sel(year=slice(1961, 2021)).data),
                    't2m': executor.submit(regress, time_ser, t2m['t2m'].sel(year=slice(1961, 2021)).data),
                    'w': executor.submit(regress, time_ser, w['w'].sel(year=slice(1961, 2021)).data),
                    'qdiv': executor.submit(regress, time_ser, qdiv['qdiv'].sel(year=slice(1961, 2021)).data),
                    'tcc': executor.submit(regress, time_ser, tcc['tcc'].sel(year=slice(1961, 2021)).data)
                }
                reg_K_u = futures['u'].result()
                reg_K_v = futures['v'].result()
                reg_K_z = futures['z'].result()
                reg_K_t2m = futures['t2m'].result()
                reg_K_w = futures['w'].result()
                reg_K_qdiv = futures['qdiv'].result()
                reg_K_tcc = futures['tcc'].result()

        contourfs = pic(fig, picloc, uvz['lat'], uvz['lon'], reg_K_u, reg_K_v, reg_K_z, reg_K_t2m)
        if i == 1:
            contourfs2 = pic2(fig, picloc+3, tcc['lat'], tcc['lon'], w['lat'], w['lon'], qdiv['lat'], qdiv['lon'], reg_K_tcc, reg_K_w, reg_K_qdiv,
                              np.array([[-.04, -.02], [.02, .04]]),
                              np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5])*.025,
                              np.array([[-.0003, -.0001], [.0001, .0003]]),
                              62, ['red', 'blue'], True, cmaps.MPL_PuOr_r[11+15:56]+ cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.MPL_PuOr_r[64:106-15],
                              ['#a35a49', '#4c7952'], f'h) 500W&TCC&QDIV of MLB Type')
        elif i == 2:
            contourfs2 = pic2(fig, picloc+3, tcc['lat'], tcc['lon'], w['lat'], w['lon'], qdiv['lat'], qdiv['lon'], reg_K_tcc, reg_K_w, reg_K_qdiv,
                              np.array([[-.04, -.02], [.02, .04]]),
                              np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5])*.025,
                              np.array([[-.0003, -.0001], [.0001, .0003]]),
                              61, ['red', 'blue'], True, cmaps.MPL_PuOr_r[11+15:56]+ cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.MPL_PuOr_r[64:106-15],
                               ['#a35a49', '#4c7952'], f'i) 500W&TCC&QDIV of ALL Type')
        elif i == 3:
            contourfs2 = pic2(fig, picloc+3, tcc['lat'], tcc['lon'], w['lat'], w['lon'], qdiv['lat'], qdiv['lon'], reg_K_tcc, reg_K_w, reg_K_qdiv,
                              np.array([[-.04, -.02], [.02, .04]]),
                              np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5])*.025,
                              np.array([[-.0003, -.0001], [.0001, .0003]]),
                              62, ['red', 'blue'], True, cmaps.MPL_PuOr_r[11+15:56]+ cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.MPL_PuOr_r[64:106-15],
                              ['#a35a49', '#4c7952'], f'j) 500W&TCC&QDIV of UB Type')

    # 添加全局colorbar  # 为colorbar腾出空间
    cbar_ax = fig.add_axes([0.915, 0.39, 0.01, 0.21]) # [left, bottom, width, height]
    cbar = fig.colorbar(contourfs, cax=cbar_ax, orientation='vertical', drawedges=True)
    cbar.locator = ticker.FixedLocator(lev_t)
    cbar.set_ticklabels([str(i) for i in lev_t])
    cbar.ax.tick_params(labelsize=10, length=0)

    lev_w = np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5])*.025
    cbar_ax1 = fig.add_axes([0.915, 0.105, 0.01, 0.21])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(contourfs2, cax=cbar_ax1, orientation='vertical', drawedges=True)
    cbar1.locator = ticker.FixedLocator(lev_w)
    cbar1.set_ticklabels(['-1.25', '-1', '-0.75', '-0.5', '-0.25', '0.25', '0.5', '0.75', '1', '1.25'])
    cbar1.set_label('×10$^{-2}$', fontsize=10, loc='bottom')
    cbar1.ax.tick_params(labelsize=10, length=0)


    plt.savefig(r"D:\PyFile\p2\pic\图4.png", dpi=600, bbox_inches='tight')
    plt.show()