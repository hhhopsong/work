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
from scipy.stats import ttest_ind

from climkit.Cquiver import *
from climkit.masked import masked
from climkit.significance_test import r_test
from climkit.lonlat_transform import *

from matplotlib import ticker
from metpy.calc import vertical_velocity
from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants


# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

def plot_text(ax, x, y, title, size, color):
    ax.text(x, y, title,
         transform=ccrs.PlateCarree(),
         ha='center',
         va='center',
         fontsize=size,
         fontweight='bold',
         color=color,
         fontname='Times New Roman',
         zorder=1000)
    return 0

def pic(fig, pic_loc, lat, lon, lev, lev_t, corr_u, corr_v, corr_z, corr_t2m, title):

    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')

    idx = int(str(pic_loc)[2]) - 3
    ax.set_title(title, loc='left', fontsize=12)

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
    p_test = np.where(corr_t2m[1] < 0.05, 0, np.nan)
    p = ax.quiver(t2m['lon'], t2m['lat'], p_test, p_test, transform=ccrs.PlateCarree(central_longitude=0), regrid_shape=40, color='k', scale=10, headlength=5, headaxislength=5, width=0.005)
    cont = ax.contour(lon, lat, corr_z[0], colors='red', levels=lev[1], linewidths=0.8, transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(lon, lat, corr_z[0], colors='blue', levels=lev[0], linestyles='--', linewidths=0.8,
                       transform=ccrs.PlateCarree(central_longitude=0))
    cont.clabel(inline=1, fontsize=4)
    cont_.clabel(inline=1, fontsize=4)
    #cont_clim = ax.contour(lon, lat, uvz_clim['z'], colors='k', levels=20, linewidths=0.6, transform=ccrs.PlateCarree(central_longitude=0))

    Cq = ax.Curlyquiver(lon, lat, corr_u[0], corr_v[0], center_lon=110, scale=5, linewidth=0.5, arrowsize=1., transform=ccrs.PlateCarree(central_longitude=0), MinDistance=[0.2, 0.5],
                     regrid=12, color='#454545', nanmax=5)

    Cq.key(U=2, label='2 m/s', color='k', fontproperties={'size': 8}, linewidth=.7, arrowsize=3.)
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
    # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
    ax.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax.tick_params(axis='both', labelsize=10, colors='black')

    return contf, ax

def pic2(fig, pic_loc, lat, lon, lat_f, lon_f, contour_1, contourf_1, lev, lev_f, color, clabel_tf, cmap, title):
    ax = fig.add_subplot(pic_loc, projection=ccrs.PlateCarree(central_longitude=180-70))
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
    p_test = np.where(contourf_1[1] < 0.05, 0, np.nan)
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
    # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
    ax.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax.tick_params(axis='both', labelsize=10, colors='black')

    return contf, ax

import numpy as np
import xarray as xr
from scipy.stats import ttest_ind

def regress(time_series, data, var_name=None):
    """
    对 time_series 与 data 的第一个维度做线性回归和相关分析，并进行显著性检验。

    Parameters
    ----------
    time_series : array-like or xr.DataArray
        一维时间序列，长度应等于 data 的时间维长度
    data : xr.Dataset / xr.DataArray / np.ndarray
        待回归数据，要求第一个维度是时间维
    var_name : str, optional
        当 data 是 xr.Dataset 时，需要指定变量名；
        若不指定且 Dataset 只有一个变量，则自动取该变量
    alpha : float, default 0.05
        显著性水平

    Returns
    -------
    regression_map : xr.DataArray or np.ndarray
        回归系数场
    correlation_map : xr.DataArray or np.ndarray
        相关系数场
    pvalue_map : xr.DataArray or np.ndarray
        相关系数对应的 p 值
    significance_mask : xr.DataArray or np.ndarray
        是否通过显著性检验（True/False）
    """
    from scipy import stats
    # -----------------------------
    # 1. 统一 data 类型
    # -----------------------------
    original_is_xarray = False
    coords = None
    dims = None
    name = None

    if isinstance(data, xr.Dataset):
        if var_name is not None:
            da = data[var_name]
        else:
            data_vars = list(data.data_vars)
            if len(data_vars) == 1:
                da = data[data_vars[0]]
            else:
                raise ValueError(
                    "data 是 xr.Dataset，包含多个变量，请通过 var_name 指定变量名。"
                )
        original_is_xarray = True
        coords = {k: v for k, v in da.coords.items() if k in da.dims[1:]}
        dims = da.dims[1:]
        name = da.name if da.name is not None else "data"

    if isinstance(data, xr.DataArray):
        da = data
        original_is_xarray = True
        coords = {k: v for k, v in da.coords.items() if k in da.dims[1:]}
        dims = da.dims[1:]
        name = da.name if da.name is not None else "data"

    elif isinstance(data, np.ndarray):
        da = None
    else:
        raise TypeError("data 必须是 xr.Dataset、xr.DataArray 或 np.ndarray")

    # -----------------------------
    # 2. time_series 转 np.array
    # -----------------------------
    if isinstance(time_series, xr.DataArray):
        ts = time_series.values
    else:
        ts = np.asarray(time_series)

    ts = np.squeeze(ts)
    if ts.ndim != 1:
        raise ValueError("time_series 必须是一维数组")

    # -----------------------------
    # 3. data 转 np.array
    # -----------------------------
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        data_np = np.asarray(da.values)
    else:
        data_np = np.asarray(data)

    if data_np.shape[0] != len(ts):
        raise ValueError(
            f"time_series 长度 ({len(ts)}) 必须等于 data 第一个维度长度 ({data_np.shape[0]})"
        )

    # -----------------------------
    # 4. reshape 为 (time, space)
    # -----------------------------
    reshaped_data = data_np.reshape(len(ts), -1)

    # -----------------------------
    # 5. 去均值
    # -----------------------------
    ts_anom = ts - np.mean(ts)
    data_anom = reshaped_data - np.mean(reshaped_data, axis=0)

    # -----------------------------
    # 6. 回归系数和相关系数
    # -----------------------------
    numerator = np.sum(data_anom * ts_anom[:, np.newaxis], axis=0)
    denominator = np.sum(ts_anom ** 2)

    regression_coef = numerator / denominator

    data_std_term = np.sqrt(np.sum(data_anom ** 2, axis=0))
    ts_std_term = np.sqrt(np.sum(ts_anom ** 2))

    correlation = numerator / (data_std_term * ts_std_term)

    # 避免浮点误差导致 |r| > 1
    correlation = np.clip(correlation, -1.0, 1.0)

    # -----------------------------
    # 7. 显著性检验（对相关系数做 t 检验）
    #    t = r * sqrt((n-2)/(1-r^2))
    # -----------------------------
    n = len(ts)
    dof = n - 2

    with np.errstate(divide="ignore", invalid="ignore"):
        t_value = correlation * np.sqrt(dof / (1.0 - correlation ** 2))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), df=dof))


    # -----------------------------
    # 8. reshape 回原空间维度
    # -----------------------------
    spatial_shape = data_np.shape[1:]
    regression_map = regression_coef.reshape(spatial_shape)
    pvalue_map = p_value.reshape(spatial_shape)

    # -----------------------------
    # 9. 若原数据是 xarray，则还原成 DataArray
    # -----------------------------
    if original_is_xarray:
        regression_map = xr.DataArray(
            regression_map,
            coords=coords,
            dims=dims,
            name=f"{name}_regression"
        )

        pvalue_map = xr.DataArray(
            pvalue_map,
            coords=coords,
            dims=dims,
            name=f"{name}_pvalue"
        )

    return regression_map, pvalue_map


try:
    uvz = xr.open_dataset(fr"{PYFILE}/p2/data/uvz_78.nc")
except:
    uvz = xr.open_dataset(fr"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc").sel(
        date=slice('1961-01-01', '2023-12-31'),
        pressure_level=[200, 500, 850],
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

tcc = xr.open_dataset(fr"{PYFILE}/p2/data/TCC.nc")
tcc = tcc.transpose('year', 'lat', 'lon')  # 500hPa
tcc_clim = tcc.mean('year')

time = [1961, 2022]
t_budget = xr.open_dataset(fr'{DATA}/ERA5/ERA5_pressLev/single_var/t_budget_1961_2022.nc').sel(
    time=slice(str(time[0]) + '-01', str(time[1]) + '-12'))
dTdt_78 = t_budget['dTdt'].sel(time=t_budget['time.month'].isin([6, 7]), level=900).groupby('time.year').mean('time')
adv_T_78 = t_budget['adv_T'].sel(time=t_budget['time.month'].isin([7, 8]), level=900).groupby('time.year').mean('time')
ver_78 = t_budget['ver'].sel(time=t_budget['time.month'].isin([7, 8]), level=900).groupby('time.year').mean('time')
Q_78 = dTdt_78 - adv_T_78 - ver_78

surface_radio = xr.open_dataset(fr"{PYFILE}/p2/data/Surface_Radio.nc") # 为地面供能为正，放能为负

#%%
# 去趋势处理
def detrend(obj, dim='year', deg=1):
    if isinstance(obj, xr.DataArray):
        coef = obj.polyfit(dim=dim, deg=deg, skipna=True)
        trend = xr.polyval(obj[dim], coef.polyfit_coefficients)
        return obj - trend

    elif isinstance(obj, xr.Dataset):
        out = xr.Dataset(coords=obj.coords, attrs=obj.attrs)
        for name, da in obj.data_vars.items():
            coef = da.polyfit(dim=dim, deg=deg, skipna=True)
            trend = xr.polyval(da[dim], coef.polyfit_coefficients)
            out[name] = da - trend
        return out

    else:
        raise TypeError("obj 必须是 xarray.DataArray 或 xarray.Dataset")

# adv_T_78 = detrend(adv_T_78, dim='year')
# ver_78 = detrend(ver_78, dim='year')
# Q_78 = detrend(Q_78, dim='year')
# uvz = detrend(uvz, dim='year')
# t2m = detrend(t2m, dim='year')
# w = detrend(w, dim='year')
# tcc = detrend(tcc, dim='year')
# surface_radio = detrend(surface_radio, dim='time')

#%%

YEAR = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014]
# YEAR = [2015]

EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
EHCI30 = EHCI.apply(lambda x: (x > 0.3).sum())
EHCI30 = (EHCI30 - EHCI30.mean()) / EHCI30.std('year')
EHCI30 = EHCI30['EHCI'].data

comp_adv, p_adv = regress(EHCI30, adv_T_78)
print('adv done')
comp_ver, p_ver = regress(EHCI30, ver_78)
print('ver done')
comp_Q, p_Q = regress(EHCI30, Q_78)
print('Q done')

comp_u = regress(EHCI30, uvz['u'])
comp_v = regress(EHCI30, uvz['v'])
comp_z = regress(EHCI30, uvz['z'])
comp_t2m = regress(EHCI30, t2m['t2m'])
comp_w = regress(EHCI30, w['w'])
comp_tcc = regress(EHCI30, tcc['tcc'])
comp_ssr = regress(EHCI30, surface_radio['ssr'])
comp_str = regress(EHCI30, surface_radio['str'])
comp_sshf = regress(EHCI30, surface_radio['sshf'])
comp_slhf = regress(EHCI30, surface_radio['slhf'])



comp_map = xr.Dataset(
    {
        'adv_T': (['lat', 'lon'], comp_adv.data),
        'ver': (['lat', 'lon'], comp_ver.data),
        'Q': (['lat', 'lon'], comp_Q.data),
        'ssr': (['lat', 'lon'], comp_ssr[0].data),
        'str': (['lat', 'lon'], comp_str[0].data),
        'sshf': (['lat', 'lon'], comp_sshf[0].data),
        'slhf': (['lat', 'lon'], comp_slhf[0].data)
    },
    coords={
        'lat': dTdt_78['lat'].data,
        'lon': dTdt_78['lon'].data
    }
)

p_map = xr.Dataset(
    {
        'adv_T': (['lat', 'lon'], p_adv.data),
        'ver': (['lat', 'lon'], p_ver.data),
        'Q': (['lat', 'lon'], p_Q.data),
        'ssr': (['lat', 'lon'], comp_ssr[1].data),
        'str': (['lat', 'lon'], comp_str[1].data),
        'sshf': (['lat', 'lon'], comp_sshf[1].data),
        'slhf': (['lat', 'lon'], comp_slhf[1].data)
    },
    coords={
        'lat': dTdt_78['lat'].data,
        'lon': dTdt_78['lon'].data
    }
)

#%%
fig = plt.figure(figsize=(11.5/3, 7))
plt.subplots_adjust(wspace=0.05, hspace=0.4)
title_head = 'Reg'
# 柱状图

comp_map_ = masked(comp_map, fr'{PYFILE}/map/self/长江_TP/长江_tp.shp')
comp_ = comp_map_


ax = fig.add_subplot(3, 1, 1)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.set_aspect('auto')
ax.set_title(f'(a) {title_head} temp_budget&surf_energy', fontsize=12, loc='left')
ax.grid(True, linestyle='--', zorder=0, axis='y')

adv_X_dTdt = np.nanmean(comp_['adv_T']) * 86400 * 31
ver_X_dTdt = np.nanmean(comp_['ver']) * 86400 * 31
Q_X_dTdt = np.nanmean(comp_['Q']) * 86400 * 31

scale = 3
Rn = np.nanmean(comp_map_['ssr']) + np.nanmean(comp_map_['str'])
Rn /= 86400 * scale
SSHF = np.nanmean(comp_map_['sshf']) / 86400 / scale
SLHF = np.nanmean(comp_map_['slhf']) / 86400 / scale


variables = ['Adv', 'Ver', 'Q', 'Fs', 'Rn', 'SSHF', 'SLHF']
values = [adv_X_dTdt, ver_X_dTdt, Q_X_dTdt, Rn+SSHF+SLHF, Rn, SSHF, SLHF]
colors = ['#ff7373' if val > 0 else '#7373ff' for val in values]

bars = ax.bar(range(7), values, width=0.3, color=colors, edgecolor='black', zorder=2)

ax.set_xticks(range(7))
ax.set_xticklabels([
    r'$-(\mathbf{V} \cdot \nabla T)^{\prime}$',
    r'$(\omega \sigma)^{\prime}$',
    r'${Q}^{\prime}$',
    r'$F_s$',
    r'$R_n$',
    r'$SSHF$',
    r'$SLHF$'
], fontsize=9.5)

ax.axvspan(-.5, 2.5, color="lightcoral", alpha=0.15, zorder=0)
ax.set_ylim(-3, 4)
ax.set_xlim(-.5, 6.5)

ax.set_yticks(np.arange(-4, 5, 1))
ax.set_yticklabels(np.arange(-4, 5, 1), fontsize=12, color='#d86868')
ax.tick_params(axis='y', labelsize=12, color='#d86868')

# ========= 右侧单独显示“地表能量对应刻度” =========
ax.axvspan(2.5, 6.5, color="deepskyblue", alpha=0.15, zorder=0)
secax = ax.secondary_yaxis('right')
secax.set_yticks(np.arange(-4, 5, 1))
secax.set_yticklabels(np.arange(-12, 15, 3), fontsize=12, color='#007bbb')
secax.tick_params(axis='y', labelsize=12, color='#007bbb')

ax.axhline(0, color='black', lw=1)

ax.spines['top'].set_color('black')
ax.spines['right'].set_color('#007bbb')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('#d86868')

# 空间图
contourfs, ax1 = pic(
    fig, 312,
    uvz['lat'], uvz['lon'],
    np.array([[-40, -20, -10], [40, 60, 80]])*4,
    np.array([-.5, -.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4, .5])*2,
    comp_u, comp_v, comp_z, comp_t2m,
f'(b) {title_head} 500UVZ&T2M'
)

# plot_text(ax1, 118, 33.6, 'C', 12, 'red')
plot_text(ax1, 124.3, 35.3, 'A', 12, 'blue')

comp_tcc_plot = (comp_tcc[0] * 100,  # 场
    comp_tcc[1]  # p值
)

comp_w_plot = (comp_w[0] * 1e3,  # 场
    comp_w[1]  # p值
)

contourfs2, ax2 = pic2(
    fig, 313,
    tcc['lat'], tcc['lon'],
    w['lat'], w['lon'],
    comp_tcc_plot,
    comp_w_plot,
    np.array([[-2], [2]]),
         np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]) * .2 * 1e2,
    ['red', 'blue'],
    True,
         cmaps.MPL_PuOr_r[11 + 15:56]
         + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0]
         + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0]
         + cmaps.MPL_PuOr_r[64:106 - 15],
    f'(c) {title_head} 500$\\omega$ & TCC'
)

# 添加全局colorbar  # 为colorbar腾出空间
cbar_ax = fig.add_axes([0.915, 0.39, 0.01, 0.21]) # [left, bottom, width, height]
cbar = fig.colorbar(contourfs, cax=cbar_ax, orientation='vertical', drawedges=True)
cbar.locator = ticker.FixedLocator(np.array([-.5, -.4, -.3, -.2, -.1, -.05, .05, .1, .2, .3, .4, .5])*2)
cbar.set_ticklabels(['-1.00', '-0.80', '-0.60', '-0.40', '-0.20', '-0.10', ' 0.10', ' 0.20', ' 0.40', ' 0.60', ' 0.80', ' 1.00'])
cbar.ax.tick_params(labelsize=10, length=0)

lev_w = np.array([-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]) * .2 * 1e2
cbar_ax1 = fig.add_axes([0.915, 0.105, 0.01, 0.21])  # [left, bottom, width, height]
cbar1 = fig.colorbar(contourfs2, cax=cbar_ax1, orientation='vertical', drawedges=True)
cbar1.locator = ticker.FixedLocator(lev_w)
cbar1.set_ticklabels(['-0.10', '-0.08', '-0.06', '-0.04', '-0.02', ' 0.02', ' 0.04', ' 0.06', ' 0.08', ' 0.10'])
# cbar1.set_label('×10$^{-2}$', fontsize=10, loc='bottom')
cbar1.ax.tick_params(labelsize=10, length=0)

for ax in fig.axes:
    # 遍历每个子图中的所有艺术家对象 (artist)
    for artist in ax.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)

plt.savefig(fr"{PYFILE}/p5/pic/局地诊断.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p5/pic/局地诊断.png", bbox_inches='tight', dpi=600)
plt.show()