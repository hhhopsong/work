import cartopy.crs as ccrs
import cmaps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, ScalarFormatter

from metpy.units import units
import metpy.calc as mpcalc
from metpy.xarray import grid_deltas_from_dataarray
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from toolbar.curved_quivers.modplot import Curlyquiver
from scipy.ndimage import filters

import xarray as xr
import xgrads as xg
import numpy as np

from toolbar.LBM.force_file import horizontal_profile as hp
from toolbar.LBM.force_file import vertical_profile as vp
from toolbar.LBM.force_file import mk_grads, mk_wave, interp3d_lbm
from toolbar.significance_test import corr_test

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

    # 重塑为 (lat, lon)
    regression_map = regression_coef.reshape(data.shape[1:])
    return regression_map

K_type = xr.open_dataset(r"D:/PyFile/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc")
info_z = xr.open_dataset(r"D:/PyFile/p2/data/Z.nc")['z']
info_u = xr.open_dataset(r"D:/PyFile/p2/data/U.nc")['u']
info_v = xr.open_dataset(r"D:/PyFile/p2/data/V.nc")['v']
info_sst = xr.open_dataset(r"D:/PyFile/p2/data/sst.nc").interp(lat=info_z['lat'], lon=info_z['lon'])['sst']

'''## 东部型
K_series = K_type.sel(type=1)['K'].data
K_series = K_series - np.polyval(np.polyfit(range(len(K_series)), K_series, 1), range(len(K_series)))
K_series = (K_series - np.mean(K_series))/np.std(K_series)
#### 北大西洋暖
zone_corr = [360-65, 360-10, 65, 30]
corr_NPW = corr(K_series, info_sst.sel(lon=slice(zone_corr[0], zone_corr[1]), lat=slice(zone_corr[2], zone_corr[3])).data)
time_series = ((info_sst.sel(lon=slice(zone_corr[0], zone_corr[1]), lat=slice(zone_corr[2], zone_corr[3]))
                -info_sst.sel(lon=slice(zone_corr[0], zone_corr[1]), lat=slice(zone_corr[2], zone_corr[3])).mean(['year']))
               *corr_NPW).mean(['lat', 'lon']).to_numpy()
time_series = time_series - np.polyval(np.polyfit(range(len(time_series)), time_series, 1), range(len(time_series)))  # 去除线性趋势
time_series = (time_series - np.mean(time_series))/np.std(time_series)
zone = [360 - 60, 360 - 0, 70, 35] #北大西洋强迫'''

## 全局一致型
K_series = K_type.sel(type=2)['K'].data
K_series = K_series - np.polyval(np.polyfit(range(len(K_series)), K_series, 1), range(len(K_series)))
K_series = (K_series - np.mean(K_series))/np.std(K_series)
#### NAO
zone_corr = [360-80, 360-15, 50, 0]
corr_NPW = corr(K_series, info_sst.sel(lon=slice(zone_corr[0], zone_corr[1]), lat=slice(zone_corr[2], zone_corr[3])).data)
time_series = ((info_sst.sel(lon=slice(zone_corr[0], zone_corr[1]), lat=slice(zone_corr[2], zone_corr[3]))
                -info_sst.sel(lon=slice(zone_corr[0], zone_corr[1]), lat=slice(zone_corr[2], zone_corr[3])).mean(['year']))
               *corr_NPW).mean(['lat', 'lon']).to_numpy()
time_series = time_series - np.polyval(np.polyfit(range(len(time_series)), time_series, 1), range(len(time_series)))  # 去除线性趋势
time_series = (time_series - np.mean(time_series))/np.std(time_series)
zone = [360-80, 360-15, 60, 0] #NAO

'''## 西部型
K_series = K_type.sel(type=3)['K'].data
K_series = K_series - np.polyval(np.polyfit(range(len(K_series)), K_series, 1), range(len(K_series)))
K_series = (K_series - np.mean(K_series))/np.std(K_series)
#### 北大西洋经向异常
zone_corr = [360-80, 360-15, 55, 10]
corr_NPW = corr(K_series, info_sst.sel(lon=slice(zone_corr[0], zone_corr[1]), lat=slice(zone_corr[2], zone_corr[3])).data)
time_series = ((info_sst.sel(lon=slice(zone_corr[0], zone_corr[1]), lat=slice(zone_corr[2], zone_corr[3]))
                -info_sst.sel(lon=slice(zone_corr[0], zone_corr[1]), lat=slice(zone_corr[2], zone_corr[3])).mean(['year']))
               *corr_NPW).mean(['lat', 'lon']).to_numpy()
time_series = time_series - np.polyval(np.polyfit(range(len(time_series)), time_series, 1), range(len(time_series)))  # 去除线性趋势
time_series = (time_series - np.mean(time_series))/np.std(time_series)
zone = [360-70, 360-20, 55, 10] #'''

#############
K_series = time_series
z1000 = np.nan_to_num(regress(K_series, info_z.sel(level=1000).data), nan=0)
z850 = np.nan_to_num(regress(K_series, info_z.sel(level=850).data), nan=0)
z500 = np.nan_to_num(regress(K_series, info_z.sel(level=500).data), nan=0)
z200 = np.nan_to_num(regress(K_series, info_z.sel(level=200).data), nan=0)
z150 = np.nan_to_num(regress(K_series, info_z.sel(level=150).data), nan=0)
z100 = np.nan_to_num(regress(K_series, info_z.sel(level=100).data), nan=0)

u1000 = np.nan_to_num(regress(K_series, info_u.sel(level=1000).data), nan=0) * units('m/s')
u850 = np.nan_to_num(regress(K_series, info_u.sel(level=850).data), nan=0) * units('m/s')
u500 = np.nan_to_num(regress(K_series, info_u.sel(level=500).data), nan=0) * units('m/s')
u200 = np.nan_to_num(regress(K_series, info_u.sel(level=200).data), nan=0) * units('m/s')
u150 = np.nan_to_num(regress(K_series, info_u.sel(level=150).data), nan=0) * units('m/s')
u100 = np.nan_to_num(regress(K_series, info_u.sel(level=100).data), nan=0) * units('m/s')

v1000 = np.nan_to_num(regress(K_series, info_v.sel(level=1000).data), nan=0) * units('m/s')
v850 = np.nan_to_num(regress(K_series, info_v.sel(level=850).data), nan=0) * units('m/s')
v500 = np.nan_to_num(regress(K_series, info_v.sel(level=500).data), nan=0) * units('m/s')
v200 = np.nan_to_num(regress(K_series, info_v.sel(level=200).data), nan=0) * units('m/s')
v150 = np.nan_to_num(regress(K_series, info_v.sel(level=150).data), nan=0) * units('m/s')
v100 = np.nan_to_num(regress(K_series, info_v.sel(level=100).data), nan=0) * units('m/s')

frc = xr.Dataset({'z':(['lev', 'lat', 'lon'], np.array([z1000, z850, z500, z200, z150, z100]))},
                 coords={'lev': [1000, 850, 500, 200, 150, 100], 'lat': info_z['lat'], 'lon': info_z['lon']})
uv = xr.Dataset({'u':(['lev', 'lat', 'lon'], np.array([u1000, u850, u500, u200, u150, u100])),
                    'v':(['lev', 'lat', 'lon'], np.array([v1000, v850, v500, v200, v150, v100]))},
                    coords={'lev': [1000, 850, 500, 200, 150, 100], 'lat': info_z['lat'], 'lon': info_z['lon']})

# 计算涡度
heights = frc['z'] * units('m^2/s^2')
wind = uv['u'] * units('m/s'), uv['v'] * units('m/s')
dx, dy = mpcalc.lat_lon_grid_deltas(np.meshgrid(heights.lon, heights.lat)[0], np.meshgrid(heights.lon, heights.lat)[1])
vor = np.zeros(heights.shape)
p=0
for i in [1000, 850, 500, 200, 150, 100]:
    ug, vg = wind[0].sel(lev=i), wind[1].sel(lev=i)
    vor[p] = mpcalc.vorticity(ug, vg, dx=dx, dy=dy) # 计算水平风的垂直涡度
    p += 1

vor = xr.Dataset({'v':(['lev', 'lat', 'lon'], np.where(np.isnan(vor), 0, vor))},
                    coords={'lev': [1000, 850, 500, 200, 150, 100], 'lat': info_z['lat'], 'lon': info_z['lon']})

lon, lat = np.meshgrid(frc['lon'], frc['lat'])
########################################
mask = ((np.where(lon<= zone[1], 1, 0) * np.where(lon>= zone[0], 1, 0))
        * (np.where(lat>= zone[3], 1, 0) * np.where(lat<= zone[2], 1, 0))
        * np.where(vor['v'] != 0, 1, 0))
########################################
vor_mask = vor.where(mask != 0, 0)

frc_nc_sigma = interp3d_lbm(vor_mask)
frc_nc_p = interp3d_lbm(vor_mask, 'p')
# 绘图
# 图1
var = 'v' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
n = 10
extent1 = [-180, 180, -30, 80]
fig = plt.figure(figsize=(10, 5), constrained_layout=True)
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180-70))
ax1.coastlines(linewidths=0.3)
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
frc_vor_avg = frc_nc_p[var].mean(dim='lev')
frc_fill_white, lon_fill_white = add_cyclic(frc_vor_avg.sel(time=0), frc_nc_p[var]['lon'])
lev_range = np.linspace(-np.nanmax(np.abs(frc_vor_avg.sel(time=0).data)), np.nanmax(np.abs(frc_vor_avg.sel(time=0).data)), 10)
var200 = ax1.contourf(lon_fill_white, frc_nc_p[var]['lat'], frc_fill_white,
                    levels=lev_range, cmap=cmaps.BlueWhiteOrangeRed[40:-40], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
# 刻度线设置
xticks1 = np.arange(extent1[0], extent1[1] + 1, 10)
yticks1 = np.arange(extent1[2], extent1[3] + 1, 10)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
# ax1
ax1.set_xticks(xticks1, crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.set_yticks(yticks1, crs=ccrs.PlateCarree())
ax1.yaxis.set_major_formatter(lat_formatter)
xmajorLocator = MultipleLocator(60)  # 先定义xmajorLocator，再进行调用
xminorLocator = MultipleLocator(10)
ymajorLocator = MultipleLocator(30)
yminorLocator = MultipleLocator(10)
ax1.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
ax1.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
ax1.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
ax1.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
# ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
# 最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=11, width=2, color='darkgray')  # 最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=8, width=1.8, color='darkgray')  # 最小刻度长度，宽度设置
ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
# 调整刻度值字体大小
ax1.tick_params(axis='both', labelsize=12, colors='black')
# ax2 垂直层结
ax_ins = inset_axes(
    ax1,
    width="15%",  # width: 5% of parent_bbox width
    height="100%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(1.1, 0., 1, 1),
    bbox_transform=ax1.transAxes,
    borderpad=0,
    )

ax2 = ax_ins
S2D = 86400.
# 计算各层平均温度
avg_temp = vor_mask['v'].sel(lon=slice(zone[0], zone[1]), lat=slice(zone[2], zone[3])).mean(dim=['lat', 'lon']).values.squeeze()  # 按纬度和经度平均
avg_temp_frc_nc_np = frc_nc_p['v'].sel(lon=slice(zone[0], zone[1]), lat=slice(zone[2], zone[3])).mean(dim=['lat', 'lon']).values.squeeze() * S2D  # frc_nc_p 各层平均温度
pressure_levels = frc['lev'].values  # 气压层次
pressure_levels_frc_nc_np = frc_nc_p['lev'].values  # frc_nc_p 的气压层次

# 在 ax2 上绘制
ax2.plot(avg_temp, pressure_levels, marker='x', color='gray', label='Obs', alpha=0.7)
ax2.plot(avg_temp_frc_nc_np, pressure_levels_frc_nc_np, marker='.', color='r', label='Frc', alpha=0.7)
# 横轴零刻度线
ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
# 设置横纵坐标范围
ax2.set_ylim(100, 1000)  # 设置横轴范围
maxabs = np.nanmax([np.nanmax(np.abs(avg_temp)), np.nanmax(np.abs(avg_temp_frc_nc_np))]) * 1.05
ax2.set_xlim(-maxabs, maxabs)  # 设置横轴范围
# 设置纵轴为反转的气压坐标
ax2.set_yscale('log')  # 气压通常采用对数坐标
ax2.invert_yaxis()  # 倒置 y 轴，使高压在下，低压在上

# 设置坐标轴标签和标题
#ax2.set_xlabel('T/K', fontsize=14)
#ax2.set_ylabel('P/hPa', fontsize=14)

# 设置刻度和网格
ax2.yaxis.set_major_locator(FixedLocator([1000, 850, 700, 500, 300, 200, 100]))  # 使用线性刻度标注
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_minor_locator(FixedLocator([]))  # 设置次要刻度线
ax2.yaxis.set_minor_formatter(ScalarFormatter())
ax2.grid(which='both', linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', which='major', labelsize=12)

# 添加图例
ax2.legend(fontsize=12)
plt.savefig(r"D:\PyFile\p2\pic\frc_data_涡度试验.png", dpi=600, bbox_inches='tight')
plt.show()

if input("是否导出?(1/0)") == '1':
    r'''template = xr.open_dataset(r"D:\CODES\Python\Meteorological\frc_nc\Template.nc")
    template['v'] = frc_nc_sigma['v'].sel(lev2=0.995)
    template['d'] = frc_nc_sigma['d'].sel(lev2=0.995)
    template['t'] = frc_nc_sigma['t'].sel(lev2=0.995)
    template['p'] = frc_nc_sigma['v'].sel(lev2=0.995)'''
    frc_nc_sigma.to_netcdf(r'D:\lbm\main\data\Forcing\frc.t42l20.nc', format='NETCDF3_CLASSIC')
    frc_nc_p.to_netcdf(r'D:\lbm\main\data\Forcing\frc_p.t42l20.nc', format='NETCDF3_CLASSIC')

