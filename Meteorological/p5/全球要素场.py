from cartopy import crs as ccrs
import cartopy.feature as cfeature
import multiprocessing
import sys
import cartopy.feature as cfeature
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tq
import xarray as xr
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
from cartopy.util import add_cyclic_point
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.lines import lineStyles
from matplotlib.pyplot import quiverkey
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import filters

from climkit.significance_test import corr_test, r_test
from climkit.TN_WaveActivityFlux import TN_WAF_3D, TN_WAF
from climkit.Cquiver import *
from climkit.data_read import *

def corr(time_series, data):
    # 计算相关系数
    # 将 data 重塑为二维：时间轴为第一个维度
    reshaped_data = data.reshape(len(time_series), -1)

    # 减去均值以标准化
    time_series_mean = time_series - np.mean(time_series)
    data_mean = reshaped_data - np.mean(reshaped_data, axis=0)

    # 计算分子（协方差）
    numerator = np.sum(data_mean * time_series_mean[:, np.newaxis], axis=0)

    # 计算分母（标准差乘积）
    denominator = np.sqrt(np.sum(data_mean ** 2, axis=0)) * np.sqrt(np.sum(time_series_mean ** 2))

    # 相关系数
    correlation = numerator / denominator

    # 重塑为 (lat, lon)
    correlation_map = correlation.reshape(data.shape[1:])
    return correlation_map

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

import matplotlib.patheffects as path_effects
def plot_text(ax, x, y, title, size, color):
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
    ax.tick_params(axis='both', labelsize=6, colors='black')

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

def pic2(fig, pic_loc, lat, lon, lat_f, lon_f, contour_1, contourf_1, contpatch, lev, lev_f, lev_pat, r_N, color, clabel_tf, cmap, color_pat,  title):
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

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
EHCI30 = EHCI.apply(lambda x: (x > 0.6).sum())
EHCI30 = (EHCI30 - EHCI30.mean()) / EHCI30.std('year')
EHCI30 = detrend(EHCI30, dim='year')
EHCI30 = EHCI30['EHCI'].data

#%%
try:
    uvz = xr.open_dataset(fr"{PYFILE}/p2/data/uvz_78.nc")
except:
    uvz = xr.open_dataset(fr"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc").sel(
        date=slice('1961-01-01', '2023-12-31'),
        pressure_level=[200, 300, 400, 500, 600, 700, 850],
        latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])
    uvz = xr.Dataset(
        {'u': (['time', 'p', 'lat', 'lon'], uvz['u'].data),
         'v': (['time', 'p', 'lat', 'lon'], uvz['v'].data),
         'z': (['time', 'p', 'lat', 'lon'], uvz['z'].data)},
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
        {'t2m': (['time', 'lat', 'lon'], t2m.data)},
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


# 空间图
time_ser = EHCI30
from concurrent.futures import ThreadPoolExecutor
time_ser = (time_ser - time_ser.mean()) / time_ser.std()
with ThreadPoolExecutor() as executor:
    futures = {
        'u': executor.submit(regress, time_ser, uvz['u'].data),
        'v': executor.submit(regress, time_ser, uvz['v'].data),
        'z': executor.submit(regress, time_ser, uvz['z'].data),
        't2m': executor.submit(regress, time_ser, t2m['t2m'].data),
        'w': executor.submit(regress, time_ser, w['w'].data),
        'tcc': executor.submit(regress, time_ser, tcc['tcc'].data)
    }
    reg_K_u = futures['u'].result()
    reg_K_v = futures['v'].result()
    reg_K_z = futures['z'].result()
    reg_K_t2m = futures['t2m'].result()
    reg_K_w = futures['w'].result()
    reg_K_tcc = futures['tcc'].result()
#%%
fig = plt.figure(figsize=(5.75, 7))
gs = gridspec.GridSpec(3, 1,  height_ratios=[1, 1, 1], wspace=0.27, hspace=0.4)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
#---------------------SST PRE
Pre = xr.open_dataset(fr"{PYFILE}/p2/data/pre.nc")
Sst = xr.open_dataset(fr"{PYFILE}/p2/data/sst.nc")
Z = xr.open_dataset(fr"{PYFILE}/p2/data/Z.nc").sel(level=[200, 500, 850])
U = xr.open_dataset(fr"{PYFILE}/p2/data/U.nc").sel(level=[200, 500, 850])
V = xr.open_dataset(fr"{PYFILE}/p2/data/V.nc").sel(level=[200, 500, 850])
from cartopy.util import add_cyclic_point
corr_pre = np.zeros((2, len(Pre['lat']), len(Pre['lon'])))
corr_sst = np.zeros((2, len(Sst['lat']), len(Sst['lon'])))
time_series = EHCI30
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

ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=c_lon))
ax1.set_title(f"(a) Reg 200UVZ&WAF", fontsize=12, loc='left')
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
                       color='#454545', thinning=['20%', 'min'], MinDistance=[0.25, 0.5], nanmax=2)
wind.key(U=2, label='2 m/s', edgecolor='none', arrowsize=4., color='k', linewidth=0.5, fontproperties={'size': 8},
         bbox_to_anchor=(0, 0.185, 1, 1))

waf_x_ = waf_x.sel(lat=np.arange(30, 80), lon=np.r_[0:360])
waf_y_ = waf_y.sel(lat=np.arange(30, 80), lon=np.r_[0:360])
waf_lat = waf_x.sel(lat=np.arange(30, 80), lon=np.r_[0:360])['lat']
waf_lon = waf_x.sel(lat=np.arange(30, 80), lon=np.r_[0:360])['lon']
WAF_Q = ax1.Curlyquiver(waf_lon, waf_lat, waf_x_.data, waf_y_.data, regrid=30, scale=1.6, color='#0066ff', linewidth=1.6,
                        arrowsize=2.5, MinDistance=[0.5, 0.2], nanmax=0.006, transform=ccrs.PlateCarree(central_longitude=0),
                        arrowstyle='tri', thinning=[['30%', '85%'], 'range'], alpha=1, zorder=40,
                        integration_direction='stick_both')
WAF_Q.key(U=0.02, label='0.02 m$^2$/s$^2$', loc='upper right', bbox_to_anchor=(-0.15, 0.185, 1, 1), fontproperties={'size': 8},
          arrowsize=0.05, edgecolor='none')



ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree(central_longitude=c_lon))
ax2.set_aspect('auto')
plt.rcParams['hatch.linewidth'] = 0.2
plt.rcParams['hatch.color'] = '#FFFFFF'
# 统一加粗所有四个边框
for spine in ax2.spines.values():
    spine.set_linewidth(1)  # 设置边框线宽

ax2.set_title(f"(b) Reg 500UVZ&SST", fontsize=12, loc='left')
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

ax3 = fig.add_subplot(gs[2], projection=ccrs.PlateCarree(central_longitude=c_lon))
ax3.set_aspect('auto')
# 统一加粗所有四个边框
for spine in ax3.spines.values():
    spine.set_linewidth(1)  # 设置边框线宽
ax3.set_title(f"(c) Reg 850UVZ&PRE", fontsize=12, loc='left')
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

plot_text(ax1, -53.6, 44.2, 'A', 10, 'blue')
plot_text(ax1, -12.27, 60.81, 'C', 10, 'red')
plot_text(ax1, 51.3, 61, 'A', 10, 'blue')
plot_text(ax1, 96.6, 54.7, 'C', 10, 'red')
plot_text(ax1, 115, 39.2, 'A', 10, 'blue')

plot_text(ax2, -54.8, 43.9, 'A', 10, 'blue')
plot_text(ax2, -9.93, 62.8, 'C', 10, 'red')
plot_text(ax2, 55.0, 62.5, 'A', 10, 'blue')
plot_text(ax2, 120.0, 55.16, 'C', 10, 'red')
plot_text(ax2, 126.9, 36, 'A', 10, 'blue')

plot_text(ax3, -55.25, 42.04, 'A', 10, 'blue')
# plot_text(ax3, -8.83, 65.6, 'C', 10, 'red')
plot_text(ax3, 61.8, 63.4, 'A', 10, 'blue')
plot_text(ax3, 124.9, 53.67, 'C', 10, 'red')
plot_text(ax3, 131.3, 32.34, 'A', 10, 'blue')


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

plt.savefig(fr'{PYFILE}/p5/pic/全球异常场.pdf', bbox_inches='tight')
plt.savefig(fr'{PYFILE}/p5/pic/全球异常场.png', bbox_inches='tight', dpi=600)
plt.show()