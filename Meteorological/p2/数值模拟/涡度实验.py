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

K_type = xr.open_dataset(r"D:/PyFile/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc")
info_z = xr.open_dataset(r"D:/PyFile/p2/data/Z.nc")
info_u = xr.open_dataset(r"D:/PyFile/p2/data/U.nc")
info_v = xr.open_dataset(r"D:/PyFile/p2/data/V.nc")
Sst = xr.open_dataset(r"D:/PyFile/p2/data/SST.nc")

## 东部型
K_series = K_type.sel(type=1)['K'].data
K_series = K_series - np.polyval(np.polyfit(range(len(K_series)), K_series, 1), range(len(K_series)))
K_series = (K_series - np.mean(K_series))/np.std(K_series)
#### 北大西洋暖
corr_NPW = corr(K_series, Sst['sst'].sel(lon=slice(360-65, 360-10), lat=slice(65, 30)).data)
corr_NPW = (corr_NPW - np.mean(corr_NPW))/np.std(corr_NPW)
zone = [360-65, 360-10, 65, 30]


z200 = corr(K_series, info_z['z'].sel(level=200).data)
u200 = corr(K_series, info_u['u'].sel(level=200).data) * units('m/s')
v200 = corr(K_series, info_v['v'].sel(level=200).data) * units('m/s')

z500 = corr(K_series, info_z['z'].sel(level=500).data)
u500 = corr(K_series, info_u['u'].sel(level=500).data) * units('m/s')
v500 = corr(K_series, info_v['v'].sel(level=500).data) * units('m/s')

z600 = corr(K_series, info_z['z'].sel(level=600).data)
u600 = corr(K_series, info_u['u'].sel(level=600).data) * units('m/s')
v600 = corr(K_series, info_v['v'].sel(level=600).data) * units('m/s')

z700 = corr(K_series, info_z['z'].sel(level=700).data)
u700 = corr(K_series, info_u['u'].sel(level=700).data) * units('m/s')
v700 = corr(K_series, info_v['v'].sel(level=700).data) * units('m/s')

z850 = corr(K_series, info_z['z'].sel(level=850).data)
u850 = corr(K_series, info_u['u'].sel(level=850).data) * units('m/s')
v850 = corr(K_series, info_v['v'].sel(level=850).data) * units('m/s')
frc = xr.Dataset({'z':(['lev', 'lat', 'lon'], np.array([z850, z700, z600, z500, z200]))},
                 coords={'lev': [850, 700, 600, 500, 200], 'lat': info_z['lat'], 'lon': info_z['lon']})
uv = xr.Dataset({'u':(['lev', 'lat', 'lon'], np.array([u850, u700, u600, u500, u200])),
                    'v':(['lev', 'lat', 'lon'], np.array([v850, v700, v600, v500, v200]))},
                    coords={'lev': [850, 700, 600, 500, 200], 'lat': info_z['lat'], 'lon': info_z['lon']})
# 读取强迫场
# 选择45-90N，35W-35E的区域
ols = K_series  # 读取缓存

heights = frc['z'] * units('m^2/s^2')
wind = uv['u'] * units('m/s'), uv['v'] * units('m/s')
dx, dy = mpcalc.lat_lon_grid_deltas(np.meshgrid(heights.lon, heights.lat)[0], np.meshgrid(heights.lon, heights.lat)[1])


vor = np.zeros(heights.shape)
p=0
for i in [850, 700, 600, 500, 200]:
    ug, vg = wind[0].sel(lev=i)*(np.std(info_u['u'].sel(p=i), axis=0)/np.std(ols)), wind[1].sel(lev=i)*(np.std(info_v['v'].sel(p=i), axis=0)/np.std(ols))
    vor[p] = filters.gaussian_filter(mpcalc.vorticity(ug, vg, dx=dx, dy=dy), 3) # 计算水平风的垂直涡度
    p += 1

vor = xr.Dataset({'v':(['lev', 'lat', 'lon'], np.where(np.isnan(vor), 0, vor))},
                    coords={'lev': [850, 700, 600, 500, 200], 'lat': info_z['lat'], 'lon': info_z['lon']})

lon, lat = np.meshgrid(frc['lon'], frc['lat'])
########################################
mask = ((np.where(lon<= zone[1], 1, 0) * np.where(lon>= zone[0], 1, 0))
        * (np.where(lat>= zone[3], 1, 0) * np.where(lat<= zone[2], 1, 0))
        * np.where(vor['v'] <= 0, 1, 0) * corr_test(ols, frc['z'], alpha=0.05, other=0))
########################################
vor_mask = vor.where(mask != 0, 0)

frc_nc_sigma = interp3d_lbm(vor_mask)
frc_nc_p = interp3d_lbm(vor_mask, 'p')
# 绘图
# 图1
var = 'v' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
lev = 500
n = 10
extent1 = [-180, 180, -30, 80]
fig = plt.figure(figsize=(10, 5), constrained_layout=True)
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180-67.5))
ax1.coastlines(linewidths=0.3)
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
frc_fill_white, lon_fill_white = add_cyclic(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
lev_range = np.linspace(-np.max(np.abs(frc_nc_p[var].sel(lev=lev).data)), 0, 10)
var200 = ax1.contourf(lon_fill_white, frc_nc_p[var]['lat'], frc_fill_white,
                    levels=lev_range, cmap=cmaps.BlueWhiteOrangeRed[40:126], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
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
avg_temp = vor_mask['v'].sel(lon=slice(25, 87.5), lat=slice(78, 49)).mean(dim=['lat', 'lon']).values.squeeze()  # 按纬度和经度平均
avg_temp_frc_nc_np = frc_nc_p['v'].sel(lon=slice(25, 87.5), lat=slice(78, 49)).mean(dim=['lat', 'lon']).values.squeeze() * S2D  # frc_nc_p 各层平均温度
pressure_levels = frc['lev'].values  # 气压层次
pressure_levels_frc_nc_np = frc_nc_p['lev'].values  # frc_nc_p 的气压层次

# 在 ax2 上绘制
ax2.plot(avg_temp, pressure_levels, marker='x', color='gray', label='Obs', alpha=0.7)
ax2.plot(avg_temp_frc_nc_np, pressure_levels_frc_nc_np, marker='.', color='r', label='Frc', alpha=0.7)
# 横轴零刻度线
ax2.axvline(0, color='k', linestyle='-', linewidth=0.5)
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
plt.savefig(r"D:\PyFile\pic\frc_data_极涡.png", dpi=600, bbox_inches='tight')
plt.show()

if input("是否导出?(1/0)") == '1':
    r'''template = xr.open_dataset(r"D:\CODES\Python\Meteorological\frc_nc\Template.nc")
    template['v'] = frc_nc_sigma['v'].sel(lev2=0.995)
    template['d'] = frc_nc_sigma['d'].sel(lev2=0.995)
    template['t'] = frc_nc_sigma['t'].sel(lev2=0.995)
    template['p'] = frc_nc_sigma['v'].sel(lev2=0.995)'''
    frc_nc_sigma.to_netcdf(r'D:\lbm\main\data\Forcing\frc.t42l20.nc', format='NETCDF3_CLASSIC')
    frc_nc_p.to_netcdf(r'D:\lbm\main\data\Forcing\frc_p.t42l20.nc', format='NETCDF3_CLASSIC')

