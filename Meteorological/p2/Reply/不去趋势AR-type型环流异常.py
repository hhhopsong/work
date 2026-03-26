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
from matplotlib import ticker, gridspec
from matplotlib.ticker import MultipleLocator
from scipy import ndimage

from climkit.Cquiver import *
from climkit.TN_WaveActivityFlux import TN_WAF_3D
from climkit.masked import masked
from climkit.significance_test import r_test, corr_test
from climkit.lonlat_transform import *
from climkit.data_read import *
from climkit.corr_reg import *

from matplotlib import ticker
from metpy.calc import vertical_velocity
from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants


# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

nanmax = None
type_name = ['', 'MLR-type 500UVZ&T2M', 'AR-type 500UVZ&T2M', 'UR-type 500UVZ&T2M']
PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

def latlon_fmt(ax, xticks1, yticks1, xmajorLocator, xminorLocator, ymajorLocator, yminorLocator):
    ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks1, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.tick_params(which='major', length=4, width=.5, color='black')
    ax.tick_params(which='minor', length=2, width=.2, color='black')
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'
    ax.tick_params(axis='both', labelsize=10, colors='black')

def plot_text(ax, x, y, title, size, color):
    import matplotlib.patheffects as path_effects
    txt = ax.text(x, y, title,
         transform=ccrs.PlateCarree(),
         ha='center',
         va='center',
         fontsize=size,
         fontweight='bold',
         color=color,
         fontname='Times New Roman',
         zorder=1000)
    # 添加白色边缘线效果
    txt.set_path_effects([
        path_effects.Stroke(linewidth=0.5, foreground='white'),
        path_effects.Normal()
    ])
    return 0

def pic(fig, pic_loc, lat, lon, corr_u, corr_v, corr_z, corr_t2m):
    global lev_t, nanmax
    pic_ind = ['', 'b', 'b', 'b']
    ax = fig.add_subplot(gs[pic_loc], projection=ccrs.PlateCarree(central_longitude=180-70))
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')
    ax.set_title(f'(b) AR-type 500UVZ&T2M', loc='left', fontsize=12)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

    da_contour = xr.DataArray(
        corr_t2m[0],
        coords={'lat': t2m['lat'].data, 'lon': t2m['lon'].data},
        dims=('lat', 'lon')
    )
    roi_shape = ((60, 0), (160, 60))
    contf = ax.contourf(t2m['lon'], t2m['lat'], da_contour.salem.roi(corners=roi_shape), cmap=cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
                        levels=lev_t, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
    # 显著性打点
    p_test = np.where(np.abs(corr_t2m[1]) >= r_test(62), 0, np.nan)
    p = ax.quiver(t2m['lon'], t2m['lat'], p_test, p_test, transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=40, color='k', scale=10, headlength=5, headaxislength=5, width=0.005)
    cont = ax.contour(lon, lat, corr_z[0], colors='red', levels=[20, 40, 60], linewidths=0.8, transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(lon, lat, corr_z[0], colors='blue', levels=[-60, -40, -20], linestyles='--', linewidths=0.8,
                       transform=ccrs.PlateCarree(central_longitude=0))
    cont.clabel(inline=1, fontsize=4)
    cont_.clabel(inline=1, fontsize=4)
    #cont_clim = ax.contour(lon, lat, uvz_clim['z'], colors='k', levels=20, linewidths=0.6, transform=ccrs.PlateCarree(central_longitude=0))
    if nanmax:
        Cq = ax.Curlyquiver(lon, lat, corr_u[0], corr_v[0], scale=5, linewidth=0.7, arrowsize=.8, MinDistance=[0.1, 0.3], thinning=['30%', 'min'],
                         regrid=17, color='#454545', nanmax=nanmax, transform=ccrs.PlateCarree(central_longitude=0))
    else:
        Cq = ax.Curlyquiver(lon, lat, corr_u[0], corr_v[0], scale=5, linewidth=0.7, arrowsize=.8, MinDistance=[0.1, 0.3], thinning=['30%', 'min'],
                         regrid=17, color='#454545', transform=ccrs.PlateCarree(central_longitude=0))
    Cq.key(U=1, label='1 m/s', color='k', fontproperties={'size': 8})
    nanmax = Cq.nanmax
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4)
    ax.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
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
    xmajorLocator = MultipleLocator(20)  # 先定义xmajorLocator，再进行调用
    xminorLocator = MultipleLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
    ax.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
    ax.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax.tick_params(axis='both', labelsize=10, colors='black')

    return contf, ax

def pic2(fig, pic_loc, lat, lon, lat_f, lon_f, lat_pat, lon_pat, contour_1, contourf_1, contpatch, lev, lev_f, lev_pat, r_N, color, clabel_tf, cmap, color_pat,  title):
    ax = fig.add_subplot(gs[pic_loc], projection=ccrs.PlateCarree(central_longitude=180-70))
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')
    ax.set_title(f'{title}', loc='left', fontsize=12)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

    da_contour = xr.DataArray(
        contourf_1[0],
        coords={'lat': lat_f, 'lon': lon_f},
        dims=('lat', 'lon')
    )
    roi_shape = ((60, 0), (160, 60))
    contf = ax.contourf(lon_f, lat_f, da_contour.salem.roi(corners=roi_shape), cmap=cmap,
                        levels=lev_f, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
    # 显著性打点
    p_test = np.where(np.abs(contourf_1[1]) >= r_test(r_N), 0, np.nan)
    p = ax.quiver(lon_f, lat_f, p_test, p_test, transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=40, color='k', scale=10, headlength=5, headaxislength=5, width=0.005)
    cont = ax.contour(lon, lat, contour_1[0], colors=color[1], levels=lev[1], linestyles='--', linewidths=0.8,
                      transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(lon, lat, contour_1[0], colors=color[0], levels=lev[0], linestyles='solid', linewidths=0.8,
                       transform=ccrs.PlateCarree(central_longitude=0))
    if clabel_tf:
        cont.clabel(inline=1, fontsize=8)
        cont_.clabel(inline=1, fontsize=8)


    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4)
    ax.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
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
    xmajorLocator = MultipleLocator(20)  # 先定义xmajorLocator，再进行调用
    xminorLocator = MultipleLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
    ax.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
    ax.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax.tick_params(axis='both', labelsize=10, colors='black')

    return contf, ax

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
    K_type = xr.open_dataset(fr"{PYFILE}/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc")

    try:
        uvz = xr.open_dataset(fr"{PYFILE}/p2/data/uvz_78.nc")
    except:
        uvz = xr.open_dataset(fr"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc").sel(
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
        uvz.to_netcdf(fr"{PYFILE}/p2/data/uvz_78.nc")
    uvz = uvz.sel(p=500).transpose('year', 'lat', 'lon')  # 500hPa
    uvz_clim = uvz.mean('year')

    try:
        t2m = xr.open_dataset(fr"{PYFILE}/p2/data/t2m_78.nc")
    except:
        t2m = xr.open_dataset(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc")['t2m']
        t2m = t2m.sel(date=slice('1961-01-01', '2023-12-31'))
        t2m = xr.Dataset(
                {'t2m': (['time', 'lat', 'lon'],t2m.data)},
                         coords={'time': pd.to_datetime(t2m['date'], format="%Y%m%d"),
                                 'lat': t2m['latitude'].data,
                                 'lon': t2m['longitude'].data})
        t2m = t2m.sel(time=slice('1961-01-01', '2022-12-31'))
        t2m = t2m.sel(time=t2m['time.month'].isin([7, 8])).groupby('time.year').mean('time')
        t2m.to_netcdf(fr"{PYFILE}/p2/data/t2m_78.nc")
    t2m = t2m.transpose('year', 'lat', 'lon')
    t2m_clim = t2m.mean('year')

    w = xr.open_dataset(fr"{PYFILE}/p2/data/W.nc")
    w = w.sel(level=500).transpose('year', 'lat', 'lon')  # 500hPa
    w_clim = w.mean('year')

    try:
        qdiv = xr.open_dataset(fr"{PYFILE}\p2\data\qdiv_all_78.nc")
    except:
        U = xr.open_dataset(fr"{PYFILE}/p2/data/U.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000])
        V = xr.open_dataset(fr"{PYFILE}/p2/data/V.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000])
        Q = xr.open_dataset(fr"{PYFILE}/p2/data/Q.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000])
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
        qdiv.to_netcdf(fr"{PYFILE}/p2/data/qdiv_all_78.nc")
    qdiv = qdiv.transpose('year', 'lat', 'lon')  # 500hPa

    tcc = xr.open_dataset(fr"{PYFILE}/p2/data/TCC.nc")
    tcc = tcc.transpose('year', 'lat', 'lon')  # 500hPa
    tcc_clim = tcc.mean('year')

    q = qdiv['qdiv'].data

    time = [1961, 2022]
    t_budget = xr.open_dataset(fr'{DATA}/ERA5/ERA5_pressLev/single_var/t_budget_1961_2022.nc').sel(
        time=slice(str(time[0]) + '-01', str(time[1]) + '-12'))
    dTdt_78 = t_budget['dTdt'].sel(time=t_budget['time.month'].isin([6, 7])).groupby('time.year').mean('time')
    adv_T_78 = t_budget['adv_T'].sel(time=t_budget['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    ver_78 = t_budget['ver'].sel(time=t_budget['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    Q_78 = dTdt_78 - adv_T_78 - ver_78

    reg_map, corr_map = np.zeros(
        (3, 4, len(t_budget['Q'].level), len(t_budget['Q'].lat), len(t_budget['Q'].lon))), np.zeros(
        (3, 4, len(t_budget['Q'].level), len(t_budget['Q'].lat), len(t_budget['Q'].lon)))

    p_lev = 900  # 950

    K_series = K_type.sel(type=2)['K'].data
    K_series = (K_series - np.mean(K_series)) / np.std(K_series)
    reg_map[1, 0], corr_map[1, 0] = regress(K_series, dTdt_78.data)
    reg_map[1, 1], corr_map[1, 1] = regress(K_series, adv_T_78.data)
    reg_map[1, 2], corr_map[1, 2] = regress(K_series, ver_78.data)
    reg_map[1, 3], corr_map[1, 3] = regress(K_series, Q_78.data)

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

    fig = plt.figure(figsize=(11.5, 7))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1.5], height_ratios=[1, 1, 1], wspace=0.27, hspace=0.4)
    lev_t = np.array([-.5, -.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4, .5])

    # 柱状图
    reg_map_ = masked(reg_map, fr'{PYFILE}/map/self/长江_TP/长江_tp.shp')
    reg_ = reg_map_.sel(type=2, level=p_lev)

    adv_X_dTdt = np.nanmean(reg_['adv_T']) * 86400 * 31
    ver_X_dTdt = np.nanmean(reg_['ver']) * 86400 * 31
    Q_X_dTdt = np.nanmean(reg_['Q']) * 86400 * 31

    # 准备绘图数据
    variables = ['Adv', 'Ver', 'Q']
    values = [adv_X_dTdt, ver_X_dTdt, Q_X_dTdt]
    colors = ['#ff7373' if val > 0 else '#7373ff' for val in values]

    # 添加标题和网格
    ax = fig.add_subplot(gs[0])
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')
    title = ['(a) MLR-type', '(a) AR-type', '(h) UR-type'][1]
    ax.set_title(f'{title} temp_budget', fontsize=12, loc='left')
    ax.grid(True, linestyle='--', zorder=0, axis='y')

    bars = ax.bar(range(3), values, width=0.3, color=colors, edgecolor='black', zorder=2)

    # 设置坐标轴标签,字体为 Times New Roman
    ax.set_xticks(range(3))
    ax.set_xticklabels([r'$-(\mathbf{V} \cdot \nabla T)^{\prime}$',
                        r'$(\omega \sigma)^{\prime}$',
                        r'${Q}^{\prime}$'], fontsize=10)

    # 设置y轴范围
    ymax = 3
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(-.5, 2.5)

    # 仅当 KType == 1 时添加y刻度标签
    ax.set_yticks(np.arange(-3, 4, 1))
    ax.set_yticklabels(np.arange(-3, 4, 1), fontsize=10)

    # 添加零线
    ax.axhline(0, color='black', lw=1)
    # ax.axvline(2.5, color='black', lw=1)

    # 边框显示为黑色
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    # 空间图
    time_ser = K_type.sel(type=2)['K'].data
    from concurrent.futures import ProcessPoolExecutor
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

    contourfs, ax1 = pic(fig, 2, uvz['lat'], uvz['lon'], reg_K_u, reg_K_v, reg_K_z, reg_K_t2m)
    plot_text(ax1, 117, 35, 'A', 12, 'blue')
    contourfs2, ax2 = pic2(fig, 4, tcc['lat'], tcc['lon'], w['lat'], w['lon'], qdiv['lat'], qdiv['lon'], reg_K_tcc*np.array([100, 1])[:,np.newaxis,np.newaxis], np.array(reg_K_w)*np.array([10e2, 1])[:,np.newaxis,np.newaxis], reg_K_qdiv,
                      np.array([[-4, -2], [2, 4]]),
                      np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5])*.025*10e2,
                      np.array([[-.0003, -.0001], [.0001, .0003]]),
                      61, ['red', 'blue'], True, cmaps.MPL_PuOr_r[11+15:56]+ cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.MPL_PuOr_r[64:106-15],
                       ['#a35a49', '#4c7952'], f'(c) AR-type 500$\omega$&TCC')

    # 添加全局colorbar  # 为colorbar腾出空间
    cbar_ax = inset_axes(ax1, width="4%", height="100%", loc='lower left', bbox_to_anchor=(1.025, 0., 1, 1),
                              bbox_transform=ax1.transAxes, borderpad=0)
    cbar = fig.colorbar(contourfs, cax=cbar_ax, orientation='vertical', drawedges=True)
    cbar.locator = ticker.FixedLocator(lev_t)
    cbar.set_ticklabels(['-0.50', '-0.40', '-0.30', '-0.20', '-0.10', '-0.05', ' 0.05', ' 0.10', ' 0.20', ' 0.30', ' 0.40', ' 0.50'])
    cbar.ax.tick_params(labelsize=10, length=0)

    lev_w = np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5])*.025*10e2
    cbar_ax1 = inset_axes(ax2, width="4%", height="100%", loc='lower left', bbox_to_anchor=(1.025, 0., 1, 1),
                              bbox_transform=ax2.transAxes, borderpad=0)
    cbar1 = fig.colorbar(contourfs2, cax=cbar_ax1, orientation='vertical', drawedges=True)
    cbar1.locator = ticker.FixedLocator(lev_w)
    cbar1.set_ticklabels(['-1.25', '-1.00', '-0.75', '-0.50', '-0.25', ' 0.25', ' 0.50', ' 0.75', ' 1.00', ' 1.25'])
    # cbar1.set_label('×10$^{-2}$', fontsize=10, loc='bottom')
    cbar1.ax.tick_params(labelsize=10, length=0)
#---------------------SST PRE
    Pre = xr.open_dataset(fr"{PYFILE}/p2/data/pre.nc")
    Sst = xr.open_dataset(fr"{PYFILE}/p2/data/sst.nc")
    Z = xr.open_dataset(fr"{PYFILE}/p2/data/Z.nc").sel(level=[200, 500, 850])
    U = xr.open_dataset(fr"{PYFILE}/p2/data/U.nc").sel(level=[200, 500, 850])
    V = xr.open_dataset(fr"{PYFILE}/p2/data/V.nc").sel(level=[200, 500, 850])
    from cartopy.util import add_cyclic_point
    corr_pre = np.zeros((2, len(Pre['lat']), len(Pre['lon'])))
    corr_sst = np.zeros((2, len(Sst['lat']), len(Sst['lon'])))
    time_series = K_type.sel(type=2)['K'].data
    time_series = (time_series - np.mean(time_series)) / np.std(time_series)
    corr_pre[0], corr_pre[1] = regress(time_series, Pre['pre'].data)
    corr_sst[0], corr_sst[1] = regress(time_series, Sst['sst'].data)

    corr_z = np.zeros((2, len(Z['level']), len(Z['lat']), len(Z['lon'])))
    reg_z = np.zeros((len(Z['level']), len(Z['lat']), len(Z['lon'])))
    corr_u = np.zeros((2, len(U['level']), len(U['lat']), len(U['lon'])))
    corr_v = np.zeros((2, len(V['level']), len(V['lat']), len(V['lon'])))
    for j in tq.trange(len(Z['level'])):
        lev = Z['level'][j].data
        corr_z[0, j], corr_z[1, j] = regress(time_series, Z['z'].sel(level=lev).data)
        corr_u[0, j], corr_u[1, j] = regress(time_series, U['u'].sel(level=lev).data)
        corr_v[0, j], corr_v[1, j] = regress(time_series, V['v'].sel(level=lev).data)
        reg_z[j] = np.array([np.polyfit(time_series, f, 1)[0] for f in Z['z'].sel(level=lev).transpose('lat', 'lon', 'year').data.reshape(-1,len(time_series))]).reshape(Z['z'].sel(level=lev).data.shape[1], Z['z'].sel(level=lev).data.shape[2])

    corr_pre = xr.Dataset({'corr': (['lat', 'lon'], corr_pre[1]),
                           'reg': (['lat', 'lon'], corr_pre[0])},
                          coords={
                                  'lat': Pre['lat'].data,
                                  'lon': Pre['lon'].data}).interp(lon=np.arange(0, 360, 0.5),
                                                                  lat=np.arange(-90, 90.1, 0.5),
                                                                  kwargs={"fill_value": "extrapolate"})
    corr_sst = xr.Dataset({'corr': (['lat', 'lon'], corr_sst[1]),
                           'reg': (['lat', 'lon'], corr_sst[0])},
                          coords={
                                  'lat': Sst['lat'].data,
                                  'lon': Sst['lon'].data})
    corr_z = xr.Dataset({'corr': (['level', 'lat', 'lon'], corr_z[1]),
                         'reg': (['level', 'lat', 'lon'], corr_z[0])},
                        coords={
                                'level': Z['level'].data,
                                'lat': Z['lat'].data,
                                'lon': Z['lon'].data}).interp(lon=np.arange(0, 360, 0.5), lat=np.arange(-90, 90.1, 0.5))
    corr_u = xr.Dataset({'corr': (['level', 'lat', 'lon'], corr_u[1]),
                         'reg': (['level', 'lat', 'lon'], corr_u[0])},
                        coords={
                                'level': U['level'].data,
                                'lat': U['lat'].data,
                                'lon': U['lon'].data}).interp(lon=np.arange(0, 360, 0.5), lat=np.arange(-90, 90.1, 0.5))
    corr_v = xr.Dataset({'corr': (['level', 'lat', 'lon'], corr_v[1]),
                         'reg': (['level', 'lat', 'lon'], corr_v[0])},
                        coords={
                                'level': V['level'].data,
                                'lat': V['lat'].data,
                                'lon': V['lon'].data}).interp(lon=np.arange(0, 360, 0.5), lat=np.arange(-90, 90.1, 0.5))

    Uc = xr.DataArray(U['u'].sel(level=[200]).mean('year').data,
                        coords=[('level', [200]),
                                ('lat', U['lat'].data),
                                ('lon', U['lon'].data)])
    Vc = xr.DataArray(V['v'].sel(level=[200]).mean('year').data,
                        coords=[('level', [200]),
                                ('lat', V['lat'].data),
                                ('lon', V['lon'].data)])
    GEOa = xr.DataArray(corr_z['reg'].sel(level=[200]),
                        coords=[('level', [200]),
                                ('lat', U['lat'].data),
                                ('lon', U['lon'].data)])
    waf_x, waf_y = TN_WAF_3D(Uc, Vc, GEOa)

    p_th = r_test(62, 0.1)
    xticks1 = np.arange(-180, 180, 60)
    yticks1 = np.arange(-0, 91, 30)

    c_lon = 180 - 70 - 10

    ax1 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree(central_longitude=c_lon))
    ax1.set_title(f"(d) AR-type 200UVZ&WAF", fontsize=12, loc='left')
    ax1.set_aspect('auto')
    plt.rcParams['hatch.linewidth'] = 0.2
    plt.rcParams['hatch.color'] = '#FFFFFF'
    ax1.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.75, color="#a4a4a4")
    ax1.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                       facecolor='none', edgecolor='black', linewidth=.5)
    ax1.set_extent([-180, 180, -20, 90], crs=ccrs.PlateCarree(central_longitude=0))
    latlon_fmt(ax1, xticks1, yticks1, MultipleLocator(60), MultipleLocator(10), MultipleLocator(30),
               MultipleLocator(10))

    z_high = ax1.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(level=200) / 9.8, colors='red', levels=[8, 10, 14], linewidths=0.65,
                         transform=ccrs.PlateCarree(central_longitude=0))
    z_low = ax1.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(level=200) / 9.8, colors='blue', levels=[-3, -1, 0], linewidths=0.65,
                        transform=ccrs.PlateCarree(central_longitude=0), linestyles='--')
    clabel1 = z_high.clabel(inline=1, fontsize=4.5)
    clabel2 = z_low.clabel(inline=1, fontsize=4.5)
    # 循环遍历每个标签，并为它设置一个带白色背景的边界框
    clabels = clabel1 + clabel2
    for label in clabels:
        label.set_bbox(dict(facecolor='white',  # 背景色为白色
                            edgecolor='none',  # 无边框
                            pad=0,  # 标签与背景的间距
                            alpha=1,  # 背景的透明度 (0.8表示80%不透明)
                            zorder=20
                            ))

    wind = ax1.Curlyquiver(corr_u['lon'], corr_u['lat'], corr_u['reg'].sel(level=200),
                           corr_v['reg'].sel(level=200), transform=ccrs.PlateCarree(central_longitude=0),
                           arrowsize=1., scale=5, linewidth=0.8, regrid=20, zorder=30,
                           color='#454545', thinning=['30%', 'min'], MinDistance=[0.25, 0.5], nanmax=2)
    wind.key(U=2, label='2 m/s', edgecolor='none', arrowsize=4., color='k', linewidth=0.5, fontproperties={'size': 8},
             bbox_to_anchor=(0, 0.185, 1, 1))

    waf_x_ = waf_x.sel(lat=np.arange(30, 80), lon=np.r_[0:360])
    waf_y_ = waf_y.sel(lat=np.arange(30, 80), lon=np.r_[0:360])
    waf_lat = waf_x.sel(lat=np.arange(30, 80), lon=np.r_[0:360])['lat']
    waf_lon = waf_x.sel(lat=np.arange(30, 80), lon=np.r_[0:360])['lon']
    WAF_Q = ax1.Curlyquiver(waf_lon, waf_lat, waf_x_.data, waf_y_.data, regrid=30, scale=1.6, color='#0066ff', linewidth=1.6,
                            arrowsize=2.5, MinDistance=[0.5, 0.2], nanmax=0.006, transform=ccrs.PlateCarree(central_longitude=0),
                            arrowstyle='tri', thinning=[['50%', '80%'], 'range'], alpha=1, zorder=40,
                            integration_direction='stick_both')
    WAF_Q.key(U=0.02, label='0.02 m$^2$/s$^2$', loc='upper right', bbox_to_anchor=(-0.15, 0.185, 1, 1), fontproperties={'size': 8},
              arrowsize=0.05, edgecolor='none')



    ax2 = fig.add_subplot(gs[3], projection=ccrs.PlateCarree(central_longitude=c_lon))
    ax2.set_aspect('auto')
    plt.rcParams['hatch.linewidth'] = 0.2
    plt.rcParams['hatch.color'] = '#FFFFFF'
    # 统一加粗所有四个边框
    for spine in ax2.spines.values():
        spine.set_linewidth(1)  # 设置边框线宽

    ax2.set_title(f"(e) AR-type 500UVZ&SST", fontsize=12, loc='left')
    ax2.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.75, color="#a4a4a4")
    ax2.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                       facecolor='none', edgecolor='black', linewidth=.5)
    ax2.set_extent([-180, 180, -20, 90], crs=ccrs.PlateCarree(central_longitude=0))
    latlon_fmt(ax2, xticks1, yticks1, MultipleLocator(60), MultipleLocator(10), MultipleLocator(30),
               MultipleLocator(10))
    # z
    positive_values = corr_z['reg'].sel(level=500).data[
        corr_z['reg'].sel(level=500).data > 0]
    q_positive = np.round(np.percentile(positive_values, 70) / 9.8) if positive_values.size > 0 else 0
    positive_values = corr_z['reg'].sel(level=500).data[
        corr_z['reg'].sel(level=500).data < 0]
    q_positive_ = np.round(np.percentile(positive_values, 30) / 9.8) if positive_values.size > 0 else 0

    z_high = ax2.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(level=500) / 9.8,
                         colors='red', levels=[4, 6, 8], linewidths=0.65,
                         transform=ccrs.PlateCarree(central_longitude=0))
    z_low = ax2.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(level=500) / 9.8,
                        colors='blue', levels=[-4, -2, 0], linewidths=0.65,
                        transform=ccrs.PlateCarree(central_longitude=0), linestyles='--')


    clabel1 = z_high.clabel(inline=1, fontsize=4.5)
    clabel2 = z_low.clabel(inline=1, fontsize=4.5)
    # 循环遍历每个标签，并为它设置一个带白色背景的边界框
    clabels = clabel1 + clabel2
    for label in clabels:
        label.set_bbox(dict(facecolor='white',  # 背景色为白色
                            edgecolor='none',  # 无边框
                            pad=0,  # 标签与背景的间距
                            alpha=1,  # 背景的透明度 (0.8表示80%不透明)
                            zorder=20
                            ))
    reg_sst_, lon = add_cyclic_point(corr_sst['reg'], coord=corr_sst['lon'])
    corr_sst_, lon = add_cyclic_point(corr_sst['corr'], coord=corr_sst['lon'])
    # sst
    lev_sst = np.array([-.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4])
    sst = ax2.contourf(lon, corr_sst['lat'], reg_sst_,
                       cmap=cmaps.GMT_polar[2:10 - 2] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10 + 2:-2],
                       levels=lev_sst, extend='both', transform=ccrs.PlateCarree(central_longitude=0), alpha=0.65)
    p_test = np.where(np.abs(corr_sst_) > p_th, 0, np.nan)

    # 显著性
    p_hatches = ax2.contourf(lon, corr_sst['lat'], p_test, levels=[0, 1], hatches=['////////////', None],
                             colors="none", add_colorbar=False, transform=ccrs.PlateCarree(central_longitude=0),
                             edgecolor='none', linewidths=0)
    plt.rcParams['hatch.linewidth'] = 0.2
    plt.rcParams['hatch.color'] = '#FFFFFF'

    # wind
    wind = ax2.Curlyquiver(corr_u['lon'], corr_u['lat'], corr_u['reg'].sel(level=500),
                           corr_v['reg'].sel(level=500), transform=ccrs.PlateCarree(central_longitude=0),
                           arrowsize=1., scale=5, linewidth=0.8, regrid=20, zorder=30,
                           color='#454545', thinning=['30%', 'min'], MinDistance=[0.25, 0.5], nanmax=1)
    wind.key(U=1, label='1 m/s', edgecolor='none', arrowsize=2., color='k', linewidth=0.5, fontproperties={'size': 8},
             bbox_to_anchor=(0, 0.185, 1, 1))

    # 边框显示为黑色
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')

    # 色条
    ax2_colorbar = inset_axes(ax2, width="2.5%", height="100%", loc='lower left', bbox_to_anchor=(1.025, 0., 1, 1),
                              bbox_transform=ax2.transAxes, borderpad=0)
    cb2 = plt.colorbar(sst, cax=ax2_colorbar, orientation='vertical', drawedges=True)
    cb2.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
    cb2.dividers.set_color('black')  # 将colorbar内间隔线调为黑色
    cb2.locator = ticker.FixedLocator(lev_sst)
    cb2.set_ticklabels([str(f'{lev:.2f}') for lev in lev_sst])
    cb2.ax.tick_params(length=0, labelsize=10)  # length为刻度线的长度

    ax3 = fig.add_subplot(gs[5], projection=ccrs.PlateCarree(central_longitude=c_lon))
    ax3.set_aspect('auto')
    # 统一加粗所有四个边框
    for spine in ax3.spines.values():
        spine.set_linewidth(1)  # 设置边框线宽
    ax3.set_title(f"(f) AR-type 850UVZ&PRE", fontsize=12, loc='left')
    ax3.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.75, color="#a4a4a4")
    ax3.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(), ccrs.PlateCarree(),
                       facecolor='none', edgecolor='black', linewidth=.5)
    ax3.add_geometries(Reader(
        fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp').geometries(),
                       ccrs.PlateCarree(), facecolor='#909090', edgecolor='#909090', linewidth=.1, hatch='.',
                       zorder=10)
    ax3.set_extent([-180, 180, -20, 90], crs=ccrs.PlateCarree(central_longitude=0))
    # ax3.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    latlon_fmt(ax3, xticks1, yticks1, MultipleLocator(60), MultipleLocator(10), MultipleLocator(30),
               MultipleLocator(10))
    reg_pre_, lon = add_cyclic_point(corr_pre['reg'], coord=corr_pre['lon'])
    corr_pre_, lon = add_cyclic_point(corr_pre['corr'], coord=corr_pre['lon'])
    lev_pre = np.array([-.5, -.4, -.3, -.2, -.15, .15, .2, .3, .4, .5])
    # reg_pre_ = np.where((np.abs(reg_pre_) <= 0.15), np.nan, reg_pre_) if ipic!= 2 else np.where((np.abs(reg_pre_) <= 0.05), np.nan, reg_pre_)
    # pre
    pre = ax3.contourf(lon, corr_pre['lat'], reg_pre_,
                       cmap=cmaps.MPL_RdYlGn[22 + 0:56 - 10] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72 + 10:106 - 0],
                       levels=lev_pre, extend='both', transform=ccrs.PlateCarree(central_longitude=0), alpha=0.65)
    pre_edge = ax3.contour(lon, corr_pre['lat'], reg_pre_, colors='white', levels=lev_pre, linestyles='solid',
                           linewidths=0.2, transform=ccrs.PlateCarree(central_longitude=0))
    p_test = np.where(np.abs(corr_pre_) > p_th, 0, np.nan)
    # ax3.quiver(lon, corr_pre['lat'], p_test, p_test, transform=ccrs.PlateCarree(central_longitude=0), color='w', scale=20, headlength=2, headaxislength=2, regrid_shape=60)
    # z
    positive_values = corr_z['reg'].sel(level=850).data[
        corr_z['reg'].sel(level=850).data > 0]
    q_positive = np.round(np.percentile(positive_values, 80) / 9.8) if positive_values.size > 0 else 0
    positive_values = corr_z['reg'].sel(level=850).data[
        corr_z['reg'].sel(level=850).data < 0]
    q_positive_ = np.round(np.percentile(positive_values, 20) / 9.8) if positive_values.size > 0 else 0

    z_high = ax3.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(level=850) / 9.8,
                         colors='red', levels=[2, 3, 4], linewidths=0.65,
                         transform=ccrs.PlateCarree(central_longitude=0))
    z_low = ax3.contour(corr_z['lon'], corr_z['lat'], corr_z['reg'].sel(level=850) / 9.8,
                        colors='blue', levels=[-2, -1], linewidths=0.65,
                        transform=ccrs.PlateCarree(central_longitude=0), linestyles='--')


    clabel1 = z_high.clabel(inline=1, fontsize=4.5)
    clabel2 = z_low.clabel(inline=1, fontsize=4.5)
    # 循环遍历每个标签，并为它设置一个带白色背景的边界框
    clabels = clabel1 + clabel2
    for label in clabels:
        label.set_bbox(dict(facecolor='white',  # 背景色为白色
                            edgecolor='none',  # 无边框
                            pad=0,  # 标签与背景的间距
                            alpha=1,  # 背景的透明度 (0.8表示80%不透明)
                            zorder=20
                            ))

    # 显著性
    p_hatches = ax3.contourf(lon, corr_pre['lat'], p_test, levels=[0, 1], hatches=['////////////', None],
                             colors="none", add_colorbar=False, transform=ccrs.PlateCarree(central_longitude=0),
                             edgecolor='none', linewidths=0)
    plt.rcParams['hatch.linewidth'] = 0.2
    plt.rcParams['hatch.color'] = '#FFFFFF'

    # wind
    wind = ax3.Curlyquiver(corr_u['lon'], corr_u['lat'], corr_u['reg'].sel(level=850),
                           corr_v['reg'].sel(level=850), transform=ccrs.PlateCarree(central_longitude=0),
                           arrowsize=1.0, scale=5, linewidth=0.8, regrid=20, zorder=30,
                           color='#454545', nanmax=0.5, thinning=['30%', 'min'],
                           MinDistance=[0.25, 0.5])
    wind.key(U=.5, label='0.5 m/s', edgecolor='none', arrowsize=1., color='k', linewidth=0.5, fontproperties={'size': 8},
             bbox_to_anchor=(0, 0.185, 1, 1))

    # 框选因子
    lw = 1.0

    plot_text(ax1, -61, 41, 'A', 10, 'blue')
    plot_text(ax1, -12.27, 60.81, 'C', 10, 'red')
    plot_text(ax1, 40, 58, 'A', 10, 'blue')
    plot_text(ax1, 74, 47, 'C', 10, 'red')
    plot_text(ax1, 109, 39, 'A', 10, 'blue')

    plot_text(ax2, -59, 41, 'A', 10, 'blue')
    plot_text(ax2, -9.93, 62.8, 'C', 10, 'red')
    plot_text(ax2, 48, 60, 'A', 10, 'blue')
    plot_text(ax2, 74, 46, 'C', 10, 'red')
    plot_text(ax2, 118, 36, 'A', 10, 'blue')

    plot_text(ax3, -64, 34, 'A', 10, 'blue')
    plot_text(ax3, -8.83, 65.6, 'C', 10, 'red')
    plot_text(ax3, 53, 60, 'A', 10, 'blue')
    # plot_text(ax3, 105, 43, 'C', 8, 'red')
    plot_text(ax3, 124, 34, 'A', 10, 'blue')


    # 边框显示为黑色
    ax3.grid(False)
    for spine in ax3.spines.values():
        spine.set_edgecolor('black')
    # 色条
    ax3_colorbar = inset_axes(ax3, width="2.5%", height="100%", loc='lower left', bbox_to_anchor=(1.025, 0., 1, 1),
                              bbox_transform=ax3.transAxes, borderpad=0)
    cb3 = plt.colorbar(pre, cax=ax3_colorbar, orientation='vertical', drawedges=True)
    cb3.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
    cb3.dividers.set_color('black')  # 将colorbar内间隔线调为黑色
    cb3.locator = ticker.FixedLocator(lev_pre)
    cb3.set_ticklabels([str(f'{lev:.2f}') for lev in lev_pre])
    cb3.ax.tick_params(length=0, labelsize=10)  # length为刻度线的长度


    for ax in fig.axes:
        # 遍历每个子图中的所有艺术家对象 (artist)
        for artist in ax.get_children():
            # 强制开启裁剪
            artist.set_clip_on(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # 设置边框线宽

    plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r3.pdf", bbox_inches='tight')
    plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r3.png", bbox_inches='tight', dpi=600)
    plt.show()