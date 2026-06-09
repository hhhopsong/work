import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import tqdm as tq
import salem

from concurrent.futures import ProcessPoolExecutor
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator

from climkit.Cquiver import *
from climkit.masked import masked
from climkit.significance_test import r_test
from climkit.lonlat_transform import *

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

# =========================================================
# 区域平均统一范围：
# 先用全流域 shp 裁剪；
# KType=1：114E 以东，中下游流域
# KType=2：全流域
# KType=3：108E 以西，上游地区
# =========================================================
BASIN_SHP = fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"


def mask_yangtze_region(data, KType, lon_name='lon'):
    """
    区域平均统一使用全流域 shp 裁剪后再按经度划分。

    KType == 1: 114E 以东，中下游流域
    KType == 2: 全流域
    KType == 3: 108E 以西，上游地区
    """
    data_region = masked(data, BASIN_SHP)

    lon = data_region[lon_name]

    if KType == 1:
        data_region = data_region.where(lon >= 114)
    elif KType == 2:
        data_region = data_region
    elif KType == 3:
        data_region = data_region.where(lon <= 112)
    else:
        raise ValueError("KType must be 1, 2 or 3.")

    return data_region


def plot_text(ax, x, y, title, size, color):
    ax.text(
        x, y, title,
        transform=ccrs.PlateCarree(),
        ha='center',
        va='center',
        fontsize=size,
        fontweight='bold',
        color=color,
        fontname='Times New Roman',
        zorder=1000000
    )
    return 0


def regress(time_series, data):
    """
    返回：
    regression_map, correlation_map
    """
    reshaped_data = data.reshape(len(time_series), -1)

    time_series_mean = time_series - np.mean(time_series)
    data_mean = reshaped_data - np.mean(reshaped_data, axis=0)

    numerator = np.sum(data_mean * time_series_mean[:, np.newaxis], axis=0)
    denominator = np.sum(time_series_mean ** 2)

    regression_coef = numerator / denominator
    correlation = numerator / (
        np.sqrt(np.sum(data_mean ** 2, axis=0)) *
        np.sqrt(np.sum(time_series_mean ** 2))
    )

    regression_map = regression_coef.reshape(data.shape[1:])
    correlation_map = correlation.reshape(data.shape[1:])

    return regression_map, correlation_map


def scale_regcorr(regcorr, scale):
    """
    regcorr 是 regress() 返回的二元组：
    regcorr[0] = 回归系数
    regcorr[1] = 相关系数

    只缩放回归系数，不缩放相关系数。
    """
    return np.stack(
        [
            regcorr[0] * scale,
            regcorr[1]
        ],
        axis=0
    )


def calc_surface_energy_budget(K_type, radio, KType):
    """
    计算地表能量收支：
    Rn = SSR + STR
    Fs = Rn + SSHF + SLHF

    区域平均统一使用：
        {PYFILE}/map/self/长江_TP/长江_tp.shp

    然后：
        KType=1: 114E 以东，中下游流域
        KType=2: 全流域
        KType=3: 108E 以西，上游地区

    返回：
    [Fs, Rn, SSR, STR, SSHF, SLHF]
    """
    K_series = K_type.sel(type=KType)['K'].data

    if KType == 1:
        radio_ = radio.sel(year=slice(1961, 2022))

    elif KType == 2:
        # AR-type 少一年，和主程序其他变量保持一致
        K_series = K_series[:-1]
        radio_ = radio.sel(year=slice(1961, 2021))

    elif KType == 3:
        radio_ = radio.sel(year=slice(1961, 2022))

    else:
        raise ValueError("KType must be 1, 2 or 3.")

    K_series = (K_series - np.mean(K_series)) / np.std(K_series)

    ssr_reg = regress(K_series, radio_['ssr'].data / 86400)[0]
    str_reg = regress(K_series, radio_['str'].data / 86400)[0]
    sshf_reg = regress(K_series, radio_['sshf'].data / 86400)[0]
    slhf_reg = regress(K_series, radio_['slhf'].data / 86400)[0]

    reg_map = xr.Dataset(
        {
            'ssr':  (['lat', 'lon'], ssr_reg),
            'str':  (['lat', 'lon'], str_reg),
            'sshf': (['lat', 'lon'], sshf_reg),
            'slhf': (['lat', 'lon'], slhf_reg),
        },
        coords={
            'lat': radio_['lat'].data,
            'lon': radio_['lon'].data,
        }
    )

    reg_map = mask_yangtze_region(reg_map, KType, lon_name='lon')

    ssr = np.nanmean(reg_map['ssr'].data)
    str_ = np.nanmean(reg_map['str'].data)
    sshf = np.nanmean(reg_map['sshf'].data)
    slhf = np.nanmean(reg_map['slhf'].data)

    rn = ssr + str_
    fs = rn + sshf + slhf

    return [fs, rn, ssr, str_, sshf, slhf]


def add_surface_energy_inset(parent_ax, values, ylim=None):
    """
    将原 f、j 图作为 inset 放置到原 e、i 图右下角。

    注意：
    原 surface energy values = [Fs, Rn, SSR, STR, SSHF, SLHF]
    inset 中不再保留 Fs 和 Rn，只绘制：
        [SSR, STR, SSHF, SLHF]
    """
    inset_values = values[2:]
    inset_labels = [r'$SSR$', r'$STR$', r'$SSHF$', r'$SLHF$']

    axins = parent_ax.inset_axes([0.65, 0.08, 0.32, 0.36])

    for spine in axins.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')

    colors = ['#ff7373' if val > 0 else '#7373ff' for val in inset_values]

    axins.bar(
        range(4),
        inset_values,
        width=0.45,
        color=colors,
        edgecolor='black',
        linewidth=0.5,
        zorder=2
    )

    axins.axhline(0, color='black', lw=0.8)
    axins.grid(True, linestyle='--', linewidth=0.4, axis='y', zorder=0)

    axins.set_xticks(range(4))
    axins.set_xticklabels(
        inset_labels,
        fontsize=7,
        rotation=0,
        ha='center',
        va='center'
    )

    if ylim is None:
        ymin = min(-3, np.floor(np.nanmin(inset_values)) - 1)
        ymax = max(4, np.ceil(np.nanmax(inset_values)) + 1)
        axins.set_ylim(ymin, ymax)
    else:
        axins.set_ylim(ylim)

    axins.yaxis.set_major_locator(MultipleLocator(2))
    axins.tick_params(which='major', length=2.5, width=.4, color='black')
    axins.tick_params(which='minor', length=1.5, width=.2, color='black')
    axins.tick_params(axis='y', labelsize=7)
    axins.tick_params(axis='x', labelsize=7, length=0.8)

    return axins


def pic(fig, pic_loc, lat, lon, corr_u, corr_v, corr_z, corr_t2m):
    global lev_t, nanmax

    # 新版 3 行 3 列：
    # 第 2 行空间图编号：4, 5, 6
    # 对应新图序 b, e, h
    pic_ind = {
        4: 'b',
        5: 'e',
        6: 'h'
    }

    ax = fig.add_subplot(
        3,
        3,
        pic_loc,
        projection=ccrs.PlateCarree(central_longitude=180 - 70)
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_aspect('auto')

    ind = pic_loc - 3

    ax.set_title(
        f'({pic_ind[pic_loc]}) {type_name[ind]}',
        loc='left',
        fontsize=12
    )

    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

    da_contour = xr.DataArray(
        corr_t2m[0],
        coords={
            'lat': t2m['lat'].data,
            'lon': t2m['lon'].data
        },
        dims=('lat', 'lon')
    )

    roi_shape = ((60, 0), (160, 60))

    contf = ax.contourf(
        t2m['lon'],
        t2m['lat'],
        da_contour.salem.roi(corners=roi_shape),
        cmap=cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
        levels=lev_t,
        extend='both',
        transform=ccrs.PlateCarree(central_longitude=0)
    )

    # 显著性打点
    p_test = np.where(np.abs(corr_t2m[1]) >= r_test(62), 0, np.nan)

    ax.quiver(
        t2m['lon'],
        t2m['lat'],
        p_test,
        p_test,
        transform=ccrs.PlateCarree(central_longitude=0),
        regrid_shape=40,
        color='k',
        scale=10,
        headlength=5,
        headaxislength=5,
        width=0.005
    )

    cont = ax.contour(
        lon,
        lat,
        corr_z[0],
        colors='#ff0090',
        levels=[20, 40, 60],
        linewidths=1.2,
        transform=ccrs.PlateCarree(central_longitude=0),
        zorder=1000
    )

    cont_ = ax.contour(
        lon,
        lat,
        corr_z[0],
        colors='#005eff',
        levels=[-60, -40, -20],
        linestyles='--',
        linewidths=1.2,
        transform=ccrs.PlateCarree(central_longitude=0),
        zorder=1000
    )

    cont.clabel(inline=1, fontsize=4)
    cont_.clabel(inline=1, fontsize=4)

    Cq = ax.Curlyquiver(
        lon,
        lat,
        corr_u[0],
        corr_v[0],
        center_lon=110,
        scale=10,
        linewidth=0.9,
        arrowsize=.9,
        regrid=30,
        color='#727272',
        nanmax=2,
        transform=ccrs.PlateCarree(central_longitude=0),
        thinning=['30%', 'min'],
        MinDistance=[0.6, 0.5]
    )

    Cq.key(
        U=1,
        label='1 m/s',
        color='#727272',
        fontproperties={'size': 8},
        facecolor='white',
        loc="lower right"
    )

    nanmax = Cq.nanmax

    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4)

    ax.add_geometries(
        Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black',
        linewidth=.5,
        zorder=9
    )

    ax.add_geometries(
        Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='gray',
        edgecolor='black',
        linewidth=.5,
        zorder=10
    )

    xticks1 = np.arange(60, 160, 20)
    yticks1 = np.arange(0, 60, 15)

    if pic_loc in [4, 7]:
        ax.set_yticks(yticks1, crs=ccrs.PlateCarree())

    ax.set_xticks(xticks1, crs=ccrs.PlateCarree())

    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()

    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)

    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    ax.tick_params(which='major', length=4, width=.5, color='black')
    ax.tick_params(which='minor', length=2, width=.2, color='black')
    ax.tick_params(
        which='both',
        bottom=True,
        top=False,
        left=True,
        labelbottom=True,
        labeltop=False
    )

    plt.rcParams['ytick.direction'] = 'out'
    ax.tick_params(axis='both', labelsize=10, colors='black')

    return contf, ax


def pic2(
    fig,
    pic_loc,
    lat,
    lon,
    lat_f,
    lon_f,
    lat_pat,
    lon_pat,
    contour_1,
    contourf_1,
    contpatch,
    lev,
    lev_f,
    lev_pat,
    r_N,
    color,
    clabel_tf,
    cmap,
    color_pat,
    title
):
    ax = fig.add_subplot(
        3,
        3,
        pic_loc,
        projection=ccrs.PlateCarree(central_longitude=180 - 70)
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_aspect('auto')
    ax.set_title(f'{title}', loc='left', fontsize=12)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

    da_contour = xr.DataArray(
        contourf_1[0],
        coords={
            'lat': lat_f,
            'lon': lon_f
        },
        dims=('lat', 'lon')
    )

    roi_shape = ((60, 0), (160, 60))

    contf = ax.contourf(
        lon_f,
        lat_f,
        da_contour.salem.roi(corners=roi_shape),
        cmap=cmap,
        levels=lev_f,
        extend='both',
        transform=ccrs.PlateCarree(central_longitude=0)
    )

    # 显著性打点
    p_test = np.where(np.abs(contourf_1[1]) >= r_test(r_N), 0, np.nan)

    ax.quiver(
        lon_f,
        lat_f,
        p_test,
        p_test,
        transform=ccrs.PlateCarree(central_longitude=0),
        regrid_shape=40,
        color='k',
        scale=10,
        headlength=5,
        headaxislength=5,
        width=0.005
    )

    cont = ax.contour(
        lon,
        lat,
        contour_1[0],
        colors=color[1],
        levels=lev[1],
        linestyles='--',
        linewidths=1.2,
        transform=ccrs.PlateCarree(central_longitude=0)
    )

    cont_ = ax.contour(
        lon,
        lat,
        contour_1[0],
        colors=color[0],
        levels=lev[0],
        linestyles='solid',
        linewidths=1.2,
        transform=ccrs.PlateCarree(central_longitude=0)
    )

    if clabel_tf:
        cont.clabel(inline=1, fontsize=8)
        cont_.clabel(inline=1, fontsize=8)

    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4)

    ax.add_geometries(
        Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black',
        linewidth=.5,
        zorder=9
    )

    ax.add_geometries(
        Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='gray',
        edgecolor='black',
        linewidth=.5,
        zorder=10
    )

    xticks1 = np.arange(60, 160, 20)
    yticks1 = np.arange(0, 60, 15)

    if pic_loc in [4, 7]:
        ax.set_yticks(yticks1, crs=ccrs.PlateCarree())

    ax.set_xticks(xticks1, crs=ccrs.PlateCarree())

    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()

    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)

    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    ax.tick_params(which='major', length=4, width=.5, color='black')
    ax.tick_params(which='minor', length=2, width=.2, color='black')
    ax.tick_params(
        which='both',
        bottom=True,
        top=False,
        left=True,
        labelbottom=True,
        labeltop=False
    )

    plt.rcParams['ytick.direction'] = 'out'
    ax.tick_params(axis='both', labelsize=10, colors='black')

    return contf, ax


if __name__ == '__main__':

    K_type = xr.open_dataset(
        fr"{PYFILE}/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc"
    )

    try:
        uvz = xr.open_dataset(fr"{PYFILE}/p2/data/uvz_78.nc")

    except:
        uvz = xr.open_dataset(
            fr"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc"
        ).sel(
            date=slice('1961-01-01', '2023-12-31'),
            pressure_level=[200, 300, 400, 500, 600, 700, 850],
            latitude=[90 - i * 0.5 for i in range(361)],
            longitude=[i * 0.5 for i in range(720)]
        )

        uvz = xr.Dataset(
            {
                'u': (['time', 'p', 'lat', 'lon'], uvz['u'].data),
                'v': (['time', 'p', 'lat', 'lon'], uvz['v'].data),
                'z': (['time', 'p', 'lat', 'lon'], uvz['z'].data)
            },
            coords={
                'time': pd.to_datetime(uvz['date'], format="%Y%m%d"),
                'p': uvz['pressure_level'].data,
                'lat': uvz['latitude'].data,
                'lon': uvz['longitude'].data
            }
        )

        uvz = uvz.sel(time=slice('1961-01-01', '2022-12-31'))
        uvz = uvz.sel(
            time=uvz['time.month'].isin([7, 8])
        ).groupby('time.year').mean('time')

        uvz.to_netcdf(fr"{PYFILE}/p2/data/uvz_78.nc")

    uvz = uvz.sel(p=500).transpose('year', 'lat', 'lon')
    uvz_clim = uvz.mean('year')

    try:
        t2m = xr.open_dataset(fr"{PYFILE}/p2/data/t2m_78_global.nc")

    except:
        t2m = xr.open_dataset(
            fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc"
        )['t2m']

        t2m = t2m.sel(date=slice('1961-01-01', '2023-12-31'))

        t2m = xr.Dataset(
            {
                't2m': (['time', 'lat', 'lon'], t2m.data)
            },
            coords={
                'time': pd.to_datetime(t2m['date'], format="%Y%m%d"),
                'lat': t2m['latitude'].data,
                'lon': t2m['longitude'].data
            }
        )

        t2m = t2m.sel(time=slice('1961-01-01', '2022-12-31'))
        t2m = t2m.sel(
            time=t2m['time.month'].isin([7, 8])
        ).groupby('time.year').mean('time')

        t2m.to_netcdf(fr"{PYFILE}/p2/data/t2m_78.nc")

    t2m = t2m.transpose('year', 'lat', 'lon')
    t2m_clim = t2m.mean('year')

    w = xr.open_dataset(fr"{PYFILE}/p2/data/W.nc")
    w = w.sel(level=500).transpose('year', 'lat', 'lon')
    w_clim = w.mean('year')

    try:
        qdiv = xr.open_dataset(fr"{PYFILE}/p2/data/qdiv_all_78.nc")

    except:
        U = xr.open_dataset(fr"{PYFILE}/p2/data/U.nc").sel(
            level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000]
        )
        V = xr.open_dataset(fr"{PYFILE}/p2/data/V.nc").sel(
            level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000]
        )
        Q = xr.open_dataset(fr"{PYFILE}/p2/data/Q.nc").sel(
            level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000]
        )

        U = transform(U['u'], lon_name='lon', type='180->360')
        V = transform(V['v'], lon_name='lon', type='180->360')
        Q = transform(Q['q'], lon_name='lon', type='180->360')

        Qu = Q * units('kg/kg') * U * units('m/s') / constants.g * 1000
        Qv = Q * units('kg/kg') * V * units('m/s') / constants.g * 1000

        integrated_qx = -Qu.integrate(coord='level')
        integrated_qy = -Qv.integrate(coord='level')

        dx, dy = mpcalc.lat_lon_grid_deltas(Qu.lon.data, Qu.lat.data)

        Q_div = np.array(
            [
                mpcalc.divergence(
                    integrated_qx[iYear, :, :],
                    integrated_qy[iYear, :, :],
                    dx=dx,
                    dy=dy
                )
                for iYear in range(len(Qu['year']))
            ]
        )

        qdiv = xr.Dataset(
            {
                'qdiv': (['year', 'lat', 'lon'], Q_div)
            },
            coords={
                'year': Qu['year'],
                'lat': Qu['lat'],
                'lon': Qu['lon']
            }
        )

        qdiv.to_netcdf(fr"{PYFILE}/p2/data/qdiv_all_78.nc")

    qdiv = qdiv.transpose('year', 'lat', 'lon')

    tcc = xr.open_dataset(fr"{PYFILE}/p2/data/TCC.nc")
    tcc = tcc.transpose('year', 'lat', 'lon')
    tcc_clim = tcc.mean('year')

    # 地表能量收支数据
    radio = xr.open_dataset(fr"{PYFILE}/p2/data/Surface_Radio.nc")

    time = [1961, 2022]

    t_budget = xr.open_dataset(
        fr'{DATA}/ERA5/ERA5_pressLev/single_var/t_budget_1961_2022.nc'
    ).sel(
        time=slice(str(time[0]) + '-01', str(time[1]) + '-12')
    )

    dTdt_78 = t_budget['dTdt'].sel(
        time=t_budget['time.month'].isin([6, 7])
    ).groupby('time.year').mean('time')

    adv_T_78 = t_budget['adv_T'].sel(
        time=t_budget['time.month'].isin([7, 8])
    ).groupby('time.year').mean('time')

    ver_78 = t_budget['ver'].sel(
        time=t_budget['time.month'].isin([7, 8])
    ).groupby('time.year').mean('time')

    Q_78 = dTdt_78 - adv_T_78 - ver_78

    reg_map = np.zeros(
        (
            3,
            4,
            len(t_budget['Q'].level),
            len(t_budget['Q'].lat),
            len(t_budget['Q'].lon)
        )
    )

    corr_map = np.zeros(
        (
            3,
            4,
            len(t_budget['Q'].level),
            len(t_budget['Q'].lat),
            len(t_budget['Q'].lon)
        )
    )

    p_lev = 900

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

            reg_map[i, 0], corr_map[i, 0] = regress(
                K_series,
                dTdt_78.sel(year=slice(1961, 2021)).data
            )
            reg_map[i, 1], corr_map[i, 1] = regress(
                K_series,
                adv_T_78.sel(year=slice(1961, 2021)).data
            )
            reg_map[i, 2], corr_map[i, 2] = regress(
                K_series,
                ver_78.sel(year=slice(1961, 2021)).data
            )
            reg_map[i, 3], corr_map[i, 3] = regress(
                K_series,
                Q_78.sel(year=slice(1961, 2021)).data
            )

    reg_map = xr.Dataset(
        {
            'dTdt': (['type', 'level', 'lat', 'lon'], reg_map[:, 0]),
            'adv_T': (['type', 'level', 'lat', 'lon'], reg_map[:, 1]),
            'ver':   (['type', 'level', 'lat', 'lon'], reg_map[:, 2]),
            'Q':     (['type', 'level', 'lat', 'lon'], reg_map[:, 3])
        },
        coords={
            'type': K_type['type'],
            'level': t_budget['Q'].level,
            'lat': t_budget['Q'].lat,
            'lon': t_budget['Q'].lon
        }
    )

    corr_map = xr.Dataset(
        {
            'dTdt': (['type', 'level', 'lat', 'lon'], corr_map[:, 0]),
            'adv_T': (['type', 'level', 'lat', 'lon'], corr_map[:, 1]),
            'ver':   (['type', 'level', 'lat', 'lon'], corr_map[:, 2]),
            'Q':     (['type', 'level', 'lat', 'lon'], corr_map[:, 3])
        },
        coords={
            'type': K_type['type'],
            'level': t_budget['Q'].level,
            'lat': t_budget['Q'].lat,
            'lon': t_budget['Q'].lon
        }
    )

    # =========================================================
    # 新版布局：
    # 3 行 3 列
    #
    # 第 1 行：温度收支柱状图
    #          原 b、f、j 独立地表能量收支图删除
    #          原 f、j 分别作为 inset 放入原 e、i 右下角
    #          inset 不再保留 Fs、Rn，只画 SSR、STR、SSHF、SLHF
    #
    # 第 2 行：原第三行 500UVZ & T2M 空间图上移
    # 第 3 行：原第四行 500ω & TCC 空间图上移
    #
    # 新图序按列排列：
    # MLR: a, b, c
    # AR : d, e, f
    # UR : g, h, i
    # =========================================================
    fig = plt.figure(figsize=(11.5, 7.4))
    plt.subplots_adjust(wspace=0.05, hspace=0.42)

    lev_t = np.array(
        [-.5, -.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4, .5]
    )

    # ==============================
    # 地表能量收支：仅用于 inset
    # ==============================
    surface_values = {
        1: calc_surface_energy_budget(K_type, radio, KType=1),
        2: calc_surface_energy_budget(K_type, radio, KType=2),
        3: calc_surface_energy_budget(K_type, radio, KType=3)
    }

    # inset 只使用 SSR、STR、SSHF、SLHF，即 values[2:]
    inset_surface_values = np.concatenate(
        [
            np.array(surface_values[2][2:]),
            np.array(surface_values[3][2:])
        ]
    )

    inset_surface_ylim = (
        min(-3, np.floor(np.nanmin(inset_surface_values)) - 1),
        max(4, np.ceil(np.nanmax(inset_surface_values)) + 1)
    )

    # ==============================
    # 第一行：温度收支柱状图
    # ==============================
    bar_axes = {}

    temp_subplot_idx = {
        1: 1,
        2: 2,
        3: 3
    }

    # 新图序：MLR=a, AR=d, UR=g
    temp_titles = {
        1: '(a) MLR-type temp_budget',
        2: '(d) AR-type temp_budget&surf_energy',
        3: '(g) UR-type temp_budget&surf_energy'
    }

    for KType in range(1, 4):

        reg_map_ = mask_yangtze_region(reg_map, KType, lon_name='lon')
        reg_ = reg_map_.sel(type=KType, level=p_lev)

        adv_X_dTdt = np.nanmean(reg_['adv_T'].data) * 86400 * 31
        ver_X_dTdt = np.nanmean(reg_['ver'].data) * 86400 * 31
        Q_X_dTdt = np.nanmean(reg_['Q'].data) * 86400 * 31

        values = [adv_X_dTdt, ver_X_dTdt, Q_X_dTdt]
        colors = ['#ff7373' if val > 0 else '#7373ff' for val in values]

        ax = fig.add_subplot(3, 3, temp_subplot_idx[KType])
        bar_axes[KType] = ax

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.set_aspect('auto')

        ax.set_title(temp_titles[KType], fontsize=12, loc='left')

        ax.grid(True, linestyle='--', zorder=0, axis='y')

        ax.bar(
            range(3),
            values,
            width=0.3,
            color=colors,
            edgecolor='black',
            zorder=2
        )

        ax.set_xticks(range(3))
        ax.set_xticklabels(
            [
                r'$-(\mathbf{V} \cdot \nabla T)^{\prime}$',
                r'$(\omega \sigma)^{\prime}$',
                r'${Q}^{\prime}$'
            ],
            fontsize=12
        )

        ymax = 3
        ax.set_ylim(-ymax, 3)
        ax.set_xlim(-.5, 2.5)

        if KType == 1:
            ax.set_yticks(np.arange(-ymax, 3 + 1, 1))
            ax.set_yticklabels(np.arange(-ymax, 3 + 1, 1), fontsize=12)
            ax.set_ylabel(r'', fontsize=11)
        else:
            ax.set_yticks(np.arange(-ymax, 3 + 1, 1))
            ax.set_yticklabels([])
            ax.tick_params(axis='y', left=False)

        ax.axhline(0, color='black', lw=1)

        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

        # 原 f、j 分别作为 inset 放入原 e、i 图右下角
        # KType=2: 原 f -> 原 e
        # KType=3: 原 j -> 原 i
        # 不保留 Fs、Rn，只保留 SSR、STR、SSHF、SLHF
        if KType in [2, 3]:
            add_surface_energy_inset(
                parent_ax=ax,
                values=surface_values[KType],
                ylim=inset_surface_ylim
            )

    # ==============================
    # 第二、三行：空间图
    # ==============================
    for i in K_type['type'].data.astype(int):

        picloc = 3 + i   # 第二行：4, 5, 6
        time_ser = K_type.sel(type=i)['K'].data

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
                    'u': executor.submit(
                        regress,
                        time_ser,
                        uvz['u'].sel(year=slice(1961, 2021)).data
                    ),
                    'v': executor.submit(
                        regress,
                        time_ser,
                        uvz['v'].sel(year=slice(1961, 2021)).data
                    ),
                    'z': executor.submit(
                        regress,
                        time_ser,
                        uvz['z'].sel(year=slice(1961, 2021)).data
                    ),
                    't2m': executor.submit(
                        regress,
                        time_ser,
                        t2m['t2m'].sel(year=slice(1961, 2021)).data
                    ),
                    'w': executor.submit(
                        regress,
                        time_ser,
                        w['w'].sel(year=slice(1961, 2021)).data
                    ),
                    'qdiv': executor.submit(
                        regress,
                        time_ser,
                        qdiv['qdiv'].sel(year=slice(1961, 2021)).data
                    ),
                    'tcc': executor.submit(
                        regress,
                        time_ser,
                        tcc['tcc'].sel(year=slice(1961, 2021)).data
                    )
                }

                reg_K_u = futures['u'].result()
                reg_K_v = futures['v'].result()
                reg_K_z = futures['z'].result()
                reg_K_t2m = futures['t2m'].result()
                reg_K_w = futures['w'].result()
                reg_K_qdiv = futures['qdiv'].result()
                reg_K_tcc = futures['tcc'].result()

        contourfs, ax1 = pic(
            fig,
            picloc,
            uvz['lat'],
            uvz['lon'],
            reg_K_u,
            reg_K_v,
            reg_K_z,
            reg_K_t2m
        )

        tcc_for_plot = scale_regcorr(reg_K_tcc, 100)
        w_for_plot = scale_regcorr(reg_K_w, 10e2)

        if i == 1:
            plot_text(ax1, 132, 40, 'A', 12, 'blue')

            contourfs2, ax2 = pic2(
                fig,
                picloc + 3,   # 第三行：7
                tcc['lat'],
                tcc['lon'],
                w['lat'],
                w['lon'],
                qdiv['lat'],
                qdiv['lon'],
                tcc_for_plot,
                w_for_plot,
                reg_K_qdiv,
                np.array([[-4, -2], [2, 4]]),
                np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]) * .025 * 10e2,
                np.array([[-.0003, -.0001], [.0001, .0003]]),
                62,
                ['#ff0090', '#005eff'],
                True,
                cmaps.MPL_PuOr_r[11 + 15:56]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.MPL_PuOr_r[64:106 - 15],
                ['#a35a49', '#4c7952'],
                f'(c) MLR-type 500$\\omega$&TCC'
            )

        elif i == 2:
            plot_text(ax1, 120, 35, 'A', 12, 'blue')

            contourfs2, ax2 = pic2(
                fig,
                picloc + 3,   # 第三行：8
                tcc['lat'],
                tcc['lon'],
                w['lat'],
                w['lon'],
                qdiv['lat'],
                qdiv['lon'],
                tcc_for_plot,
                w_for_plot,
                reg_K_qdiv,
                np.array([[-4, -2], [2, 4]]),
                np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]) * .025 * 10e2,
                np.array([[-.0003, -.0001], [.0001, .0003]]),
                61,
                ['#ff0090', '#005eff'],
                True,
                cmaps.MPL_PuOr_r[11 + 15:56]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.MPL_PuOr_r[64:106 - 15],
                ['#a35a49', '#4c7952'],
                f'(f) AR-type 500$\\omega$&TCC'
            )

        elif i == 3:
            plot_text(ax1, 125, 38, 'A', 12, 'blue')
            plot_text(ax1, 117.5, 24, 'C', 12, 'red')

            contourfs2, ax2 = pic2(
                fig,
                picloc + 3,   # 第三行：9
                tcc['lat'],
                tcc['lon'],
                w['lat'],
                w['lon'],
                qdiv['lat'],
                qdiv['lon'],
                tcc_for_plot,
                w_for_plot,
                reg_K_qdiv,
                np.array([[-4, -2], [2, 4]]),
                np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]) * .025 * 10e2,
                np.array([[-.0003, -.0001], [.0001, .0003]]),
                62,
                ['#ff0090', '#005eff'],
                True,
                cmaps.MPL_PuOr_r[11 + 15:56]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.CBR_wet[0]
                + cmaps.MPL_PuOr_r[64:106 - 15],
                ['#a35a49', '#4c7952'],
                f'(i) UR-type 500$\\omega$&TCC'
            )

    # ==============================
    # Colorbar 1：第二行空间图
    # ==============================
    cbar_ax = fig.add_axes([0.915, 0.38, 0.01, 0.18])

    cbar = fig.colorbar(
        contourfs,
        cax=cbar_ax,
        orientation='vertical',
        drawedges=True
    )

    cbar.locator = ticker.FixedLocator(lev_t)
    cbar.set_ticklabels(
        [
            '-0.50', '-0.40', '-0.30', '-0.20', '-0.10', '-0.05',
            ' 0.05', ' 0.10', ' 0.20', ' 0.30', ' 0.40', ' 0.50'
        ]
    )
    cbar.ax.tick_params(labelsize=10, length=0)

    # ==============================
    # Colorbar 2：第三行空间图
    # ==============================
    lev_w = np.array(
        [-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]
    ) * .025 * 10e2

    cbar_ax1 = fig.add_axes([0.915, 0.12, 0.01, 0.18])

    cbar1 = fig.colorbar(
        contourfs2,
        cax=cbar_ax1,
        orientation='vertical',
        drawedges=True
    )

    cbar1.locator = ticker.FixedLocator(lev_w)
    cbar1.set_ticklabels(
        [
            '-1.25', '-1.00', '-0.75', '-0.50', '-0.25',
            ' 0.25', ' 0.50', ' 0.75', ' 1.00', ' 1.25'
        ]
    )
    cbar1.ax.tick_params(labelsize=10, length=0)

    for ax in fig.axes:
        for artist in ax.get_children():
            artist.set_clip_on(True)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.savefig(fr"{PYFILE}/p2/pic/图4.pdf", bbox_inches='tight')
    plt.savefig(fr"{PYFILE}/p2/pic/图4.png", bbox_inches='tight', dpi=600)

    plt.show()
