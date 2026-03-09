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
from climkit.corr_reg import regress
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

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"


# read data
lamda = 2.45e6
t2m_78 = xr.open_dataset(fr"{PYFILE}/p2/data/t2m_78.nc") - 273.15
try:
    H = xr.open_dataset(fr"{PYFILE}/p2/data/H_78.nc")
    Hp = xr.open_dataset(fr"{PYFILE}/p2/data/Hp_78.nc")
except:
    E_78 = xr.open_dataset(fr"{DATA}/ERA5/ERA5_singleLev/evaporation.nc")['e']
    E_78 = xr.Dataset(
        {'e': (['time', 'lat', 'lon'], E_78.data)},
        coords={'time': pd.to_datetime(E_78['valid_time'], format="%Y%m%d"),
                'lat': E_78['latitude'].data,
                'lon': E_78['longitude'].data})
    E_78 = E_78.sel(time=slice('1961-01-01', '2022-12-31'))
    E_78 = E_78.sel(time=E_78['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    Ep_78 = xr.open_dataset(fr"{DATA}/ERA5/ERA5_singleLev/evaporation.nc")['pev']
    Ep_78 = xr.Dataset(
        {'pev': (['time', 'lat', 'lon'], Ep_78.data)},
        coords={'time': pd.to_datetime(Ep_78['valid_time'], format="%Y%m%d"),
                'lat': Ep_78['latitude'].data,
                'lon': Ep_78['longitude'].data})
    Ep_78 = Ep_78.sel(time=slice('1961-01-01', '2022-12-31'))
    Ep_78 = Ep_78.sel(time=Ep_78['time.month'].isin([7, 8])).groupby('time.year').mean('time')
    date = xr.open_dataset(fr"{DATA}/ERA5/ERA5_singleLev/radiation_1.nc")
    ssr_78 = era5_AfterOpen(date, 1961, 2022, 'ssr')['ssr'] # 地表净短波辐射能 向下为正
    ssr_78 = ssr_78.sel(time=ssr_78['time.month'].isin([7, 8])).groupby('time.year').mean('time').interp(lon=t2m_78.lon, lat=t2m_78.lat)
    str_78 = era5_AfterOpen(date, 1961, 2022, 'str')['str'] # 地表净长波辐射能 向下为正
    str_78 = str_78.sel(time=str_78['time.month'].isin([7, 8])).groupby('time.year').mean('time').interp(lon=t2m_78.lon, lat=t2m_78.lat)
    Rn = xr.Dataset(
        {'rn': (['year', 'lat', 'lon'], ssr_78.data + str_78.data)},
        coords={'year': ssr_78['year'].data,
                'lat': ssr_78['lat'].data,
                'lon': ssr_78['lon'].data}) # 地表净辐射能 向下为正
    Rn = Rn.sel(year=slice('1961', '2022'))
    del date, str_78, ssr_78
    # H', Hp'
    H = Rn['rn'].data + lamda * E_78['e'].data * 1000
    Hp = Rn['rn'].data + lamda * Ep_78['pev'].data * 1000
    H = xr.Dataset(
        {'h': (['year', 'lat', 'lon'], H)},
        coords={'year': Rn['year'].data,
                'lat': Rn['lat'].data,
                'lon': Rn['lon'].data}) # 实际潜热
    Hp = xr.Dataset(
        {'hp': (['year', 'lat', 'lon'], Hp)},
        coords={'year': Rn['year'].data,
                'lat': Rn['lat'].data,
                'lon': Rn['lon'].data}) # 潜在潜热
    H.to_netcdf(fr"{PYFILE}/p2/data/H_78.nc")
    Hp.to_netcdf(fr"{PYFILE}/p2/data/Hp_78.nc")

K_type = xr.open_dataset(fr"{PYFILE}/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc")
K_series = K_type.sel(type=3)['K'].data
K_series = (K_series - np.mean(K_series)) / np.std(K_series)

uvz = xr.open_dataset(fr"{PYFILE}/p2/data/uvz_78.nc")
uvz = uvz.sel(p=500).transpose('year', 'lat', 'lon')  # 500hPa


u_78 = regress(K_series, uvz['u'].data)
v_78 = regress(K_series, uvz['v'].data)
z_78 = regress(K_series, uvz['z'].data)

# H = (H - H.mean(dim='year')) / H.std(dim='year')
# Hp = (Hp - Hp.mean(dim='year')) / Hp.std(dim='year')
H = regress(K_series, H['h'].data) / H.std(dim='year')
Hp = regress(K_series, Hp['hp'].data) / Hp.std(dim='year')
t2m_78 = regress(K_series, t2m_78['t2m'].data) / t2m_78.std(dim='year')
corr_pi = np.zeros_like(t2m_78.to_dataarray())

e = H['h'].data - Hp['hp'].data
e = xr.Dataset(
    {'e': (['lat', 'lon'], e)},
    coords={'lat': H['lat'].data,
            'lon': H['lon'].data})['e']

pi = e * t2m_78['t2m'].data

# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
lev_t = np.array([0., .08, .16, .24, .32, .4, .48, .56])

def pic(fig, lat, lon, corr_u, corr_v, corr_z, corr_t2m, p_test_):
    global lev_t, nanmax
    pic_ind = ['', 'b', 'b', 'b']
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180-70))
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.set_aspect('auto')
    ax.set_title(r'UR-type 500UVZ&$\pi$', loc='left', fontsize=12)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

    da_contour = xr.DataArray(
        corr_t2m,
        coords={'lat': pi['lat'].data, 'lon': pi['lon'].data},
        dims=('lat', 'lon')
    )
    roi_shape = ((60, 0), (160, 60))
    contf = ax.contourf(pi['lon'], pi['lat'], da_contour.salem.roi(corners=roi_shape), cmap=cmaps.CBR_wet[0] + cmaps.GMT_polar[11:-4],
                        levels=lev_t, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
    # 显著性打点
    p_test = np.where(np.abs(p_test_) >= r_test(62), 0, np.nan)

    cont = ax.contour(uvz['lon'], uvz['lat'], corr_z, colors='red', levels=[20, 40, 60], linewidths=0.8, transform=ccrs.PlateCarree(central_longitude=0))
    cont_ = ax.contour(uvz['lon'], uvz['lat'], corr_z, colors='blue', levels=[-60, -40, -20], linestyles='--', linewidths=0.8,
                       transform=ccrs.PlateCarree(central_longitude=0))
    cont.clabel(inline=1, fontsize=4)
    cont_.clabel(inline=1, fontsize=4)
    #cont_clim = ax.contour(lon, lat, uvz_clim['z'], colors='k', levels=20, linewidths=0.6, transform=ccrs.PlateCarree(central_longitude=0))
    Cq = ax.Curlyquiver(uvz['lon'], uvz['lat'], corr_u, corr_v, scale=5, linewidth=0.7, arrowsize=.8, MinDistance=[0.1, 0.3], thinning=['30%', 'min'],
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

fig = plt.figure(figsize=(np.array([8, 4.5])*.6))

contourfs, ax1 = pic(fig, pi['lat'], pi['lon'], u_78, v_78, z_78, pi.data, corr_pi)

# 添加全局colorbar  # 为colorbar腾出空间
cbar_ax = inset_axes(ax1, width="4%", height="100%", loc='lower left', bbox_to_anchor=(1.025, 0., 1, 1),
                          bbox_transform=ax1.transAxes, borderpad=0)
cbar = fig.colorbar(contourfs, cax=cbar_ax, orientation='vertical', drawedges=True)
cbar.locator = ticker.FixedLocator(lev_t)
cbar.set_ticklabels(['  0 ', '0.08', '0.16', '0.24', '0.32', '0.40', '0.48', '0.56'])
cbar.ax.tick_params(labelsize=7, length=0)

for ax in fig.axes:
    # 遍历每个子图中的所有艺术家对象 (artist)
    for artist in ax.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽

plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r4.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r4.png", bbox_inches='tight', dpi=600)
