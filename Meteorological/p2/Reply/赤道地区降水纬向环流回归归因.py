import xarray as xr
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.mpl.clip_path import bbox_to_path
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec
from metpy.calc import vertical_velocity
from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants
import cmaps

from climkit.TN_WaveActivityFlux import TN_WAF_3D
from climkit.Cquiver import *
from climkit.data_read import *
from climkit.lonlat_transform import *


PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"


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


K_type = xr.open_dataset(fr"{PYFILE}/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc")
Z = xr.open_dataset(fr"{PYFILE}/p2/data/Z.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(10, -10), lon=slice(0, 360)) / 9.8
U = xr.open_dataset(fr"{PYFILE}/p2/data/U.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(10, -10), lon=slice(0, 360))
V = xr.open_dataset(fr"{PYFILE}/p2/data/V.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(10, -10), lon=slice(0, 360))
T = xr.open_dataset(fr"{PYFILE}/p2/data/T.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(10, -10), lon=slice(0, 360))
Q = xr.open_dataset(fr"{PYFILE}/p2/data/Q.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(10, -10), lon=slice(0, 360))
W = xr.open_dataset(fr"{PYFILE}/p2/data/W.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(10, -10), lon=slice(0, 360))
# W = vertical_velocity(W['w'] * units('Pa/s') , W['level'] * units.hPa, T['t'] * units.degC)
Pre = xr.open_dataset(fr"{PYFILE}/p2/data/pre.nc").sel(lat=slice(10, -10), lon=slice(0, 360))
Sst = xr.open_dataset(fr"{PYFILE}/p2/data/sst.nc").sel(lat=slice(10, -10), lon=slice(0, 360))
Terrain = xr.open_dataset(fr"{DATA}/NOAA/ETOPO/ETOPO_2022_v1_30s_N90W180_bed.nc").sel(lat=slice(-10, 10), lon=slice(-180, 180))['z'].astype(np.float64).mean(dim='lat', skipna=True)
Z = transform(Z['z'], lon_name='lon', type='180->360')
U = transform(U['u'], lon_name='lon', type='180->360')
V = transform(V['v'], lon_name='lon', type='180->360')
Q = transform(Q['q'], lon_name='lon', type='180->360')
W = transform(W['w'], lon_name='lon', type='180->360')
Pre = transform(Pre['pre'], lon_name='lon', type='180->360')
Sst = transform(Sst['sst'], lon_name='lon', type='180->360')
Terrain = transform(Terrain, lon_name='lon', type='180->360')

Qu = Q * units('kg/kg') * U * units('m/s') / constants.g * 1000
Qv = Q * units('kg/kg') * V * units('m/s') / constants.g * 1000
dx, dy = mpcalc.lat_lon_grid_deltas(Qu.lon, Qu.lat)
# 计算水汽通量散度
Q_div = np.array([[mpcalc.divergence(Qu[iYear, iLev, :, :], Qv[iYear, iLev, :, :], dx=dx, dy=dy) for iLev in range(len(Qu['level']))] for iYear in range(len(Qu['year']))])


Terrain_ver = np.array(Terrain)
Terrain_ver = 1013 * (1 - 6.5/2.88e5 * Terrain_ver)**5.255
lon_Terrain = Terrain.lon


#### 拉尼娜型海温
zone = [120, 360-80, 5, -5]
### 合成分析
mid_lower_year = []
all_year = []
time_series = (Sst.sel(lon=slice(zone[0], zone[1]), lat=slice(zone[2], zone[3]))
               - Sst.sel(lon=slice(zone[0], zone[1]), lat=slice(zone[2], zone[3])).mean(['year']))
trend = regress(np.array(range(len(time_series))), time_series.data)[0]
time_series = time_series - np.arange(len(time_series))[:, np.newaxis, np.newaxis] * trend

## 中下游型
K_series = K_type.sel(type=1)['K'].data
K_series = (K_series - np.mean(K_series))/np.std(K_series)
corr_mid_lower = regress(K_series, Sst.sel(lon=slice(zone[0], zone[1]), lat=slice(zone[2], zone[3])).data)[0]

## 全域型
K_series = K_type.sel(type=2)['K'].data
K_series = K_series - np.polyval(np.polyfit(range(len(K_series)), K_series, 1), range(len(K_series)))
K_series = (K_series - np.mean(K_series))/np.std(K_series)
corr_all = regress(K_series, Sst.sel(lon=slice(zone[0], zone[1]), lat=slice(zone[2], zone[3])).data)[0]

for iyear in range(len(time_series)):
    corr1 = np.corrcoef(time_series[iyear].to_numpy().flatten()[~np.isnan(time_series[0].to_numpy().flatten())],
                corr_mid_lower.flatten()[~np.isnan(corr_mid_lower.flatten())])
    corr2 = np.corrcoef(time_series[iyear].to_numpy().flatten()[~np.isnan(time_series[0].to_numpy().flatten())],
                corr_all.flatten()[~np.isnan(corr_all.flatten())])
    if (corr1[0, 1] >=0.5) or (corr2[0, 1] >= 0.5):
        if np.abs(corr1[0, 1] - corr2[0, 1]) >= 0.15: #确保不是混合型年
            if corr1[0, 1] > corr2[0, 1]:
                mid_lower_year.append(time_series.year[iyear])
            else:
                all_year.append(time_series.year[iyear])

K_ = [mid_lower_year, all_year]

fig = plt.figure(figsize=(8, 4))
spec = gridspec.GridSpec(ncols=1, nrows=2, wspace=0, hspace=0, height_ratios=[2, 1])  # 设置子图比例
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
for i in range(1):
    Z_reg = Z.sel(year=time_series['year'].isin(K_[i])).mean('year').data - Z.mean('year').data
    U_reg = U.sel(year=time_series['year'].isin(K_[i])).mean('year').data - U.mean('year').data
    W_reg = W.sel(year=time_series['year'].isin(K_[i])).mean('year').data - W.mean('year').data
    Q_div_reg = Q_div[time_series['year'].isin(K_[i])].mean(axis=0) - Q_div.mean(axis=0)
    Pre_reg = Pre.sel(year=time_series['year'].isin(K_[i])).mean('year').data - Pre.mean('year').data
    Sst_reg = Sst.sel(year=time_series['year'].isin(K_[i])).mean('year').data - Sst.mean('year').data
    Z_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], Z_reg),},
                      coords={'level': Z['level'], 'lat': Z['lat'], 'lon': Z['lon']})
    U_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], U_reg)},
                      coords={'level': U['level'], 'lat': U['lat'], 'lon': U['lon']})
    W_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], W_reg)},
                      coords={'level': W['level'], 'lat': W['lat'], 'lon': W['lon']})
    Q_div_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], Q_div_reg)},
                            coords={'level': Qv['level'], 'lat': Qv['lat'], 'lon': Qv['lon']})
    Pre_nc = xr.Dataset({'reg': (['lat', 'lon'], Pre_reg)},
                        coords={'lat': Pre['lat'], 'lon': Pre['lon']})
    Sst_nc = xr.Dataset({'reg': (['lat', 'lon'], Sst_reg)},
                        coords={'lat': Sst['lat'], 'lon': Sst['lon']})
    W_nc['reg'] *= 1000   # Pa->0.001Pa

    f_ax = fig.add_subplot(spec[0])
    f_ax.set_title(r'(a) U$\omega$&moisture_flux_div anomaly of CEP$^{+}$ year', loc='left', fontsize=14, pad=12)
    f_ax.set_ylabel('Pressure(hPa)', fontsize=12)
    # f_ax.set_xlabel('Latitude', fontsize=18)
    f_ax.set_xlim(0, 360)
    f_ax.set_ylim(1000, 100)

    f_ax.fill_between(lon_Terrain, Terrain_ver, 1010, where=Terrain_ver < 1010, facecolor='#454545', clip_on=False, zorder=10) # 地形
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) != -999, facecolor='white', zorder=9) # 0海温异常
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) >= .2, facecolor='#fff4ad', zorder=9) # 暖海温异常
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) >= .3, facecolor='#ffdb86', zorder=9) # 暖海温异常
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) >= .4, facecolor='#ffb360', zorder=9) # 暖海温异常
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) >= .5, facecolor='#ff9b53', zorder=9) # 暖海温异常
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) >= .6, facecolor='#ff8448', zorder=9) # 暖海温异常
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) <= -.2, facecolor='#c9edf5', zorder=9) # 冷海温异常
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) <= -.3, facecolor='#afe3ef', zorder=9) # 冷海温异常
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) <= -.4, facecolor='#92d9e9', zorder=9) # 冷海温异常
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) <= -.5, facecolor='#7ac7e1', zorder=9) # 冷海温异常
    # f_ax.fill_between(Sst_nc['lon'], 1000, 975, where=Sst_nc['reg'].mean('lat', skipna=True) <= -.6, facecolor='#5db6d7', zorder=9) # 冷海温异常
    #
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) != -999, facecolor='white', zorder=9) # 0海温异常
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) >= .05, facecolor='#ddf191', zorder=9) # 正降水异常
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) >= .1, facecolor='#c5e67e', zorder=9)  # 正降水异常
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) >= .2, facecolor='#a9da6c', zorder=9)  # 正降水异常
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) >= .3, facecolor='#87cb67', zorder=9)  # 正降水异常
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) >= .4, facecolor='#63bc62', zorder=9)  # 正降水异常
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) <= -.05, facecolor='#f1d3d3', zorder=9) # 负降水异常
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) <= -.1, facecolor='#e6c5c5', zorder=9) # 负降水异常
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) <= -.2, facecolor='#daa9a9', zorder=9) # 负降水异常
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) <= -.3, facecolor='#cb8787', zorder=9) # 负降水异常
    # f_ax.fill_between(Pre_nc['lon'], 125, 100, where=Pre_nc['reg'].mean('lat', skipna=True) <= -.4, facecolor='#bc6363', zorder=9) # 负降水异常

    _U, _W = U_nc['reg'].mean('lat'), W_nc['reg'].mean('lat')
    vec = f_ax.Curlyquiver(U_nc['lon'], U_nc['level'], _U, _W, arrowsize=1.5, scale=20, linewidth=1, regrid=18, color='k', zorder=8.5, MinDistance=[0.1, 0.3], regrid_reso=9)
    vec.key(U=10, label='10 m/s (10$^{-2}$ Pa/s)', loc='upper right', edgewidth=0, edgecolor='none', arrowsize=10, linewidth=1, intetval=1, fontproperties={'size': 11}, bbox_to_anchor=[-0.02, 0.20, 1, 1])
    lev_z = np.array([-5, -4, -3, -2, -1, -.2, .2, 1, 2, 3, 4, 5])
    q_div = f_ax.contourf(Q_div_nc['lon'], Q_div_nc['level'], Q_div_nc['reg'].mean('lat', skipna=True), levels=lev_z * 10e-9,
                          cmap=cmaps.MPL_RdYlGn_r[22 + 0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn_r[72:106 - 0], zorder=8,
                          extend='both')
    q_div_nf = f_ax.contour(Q_div_nc['lon'], Q_div_nc['level'], Q_div_nc['reg'].mean('lat', skipna=True), levels=lev_z * 10e-9,
                            colors='w', linewidths=.2, zorder=8, linestyles='solid')


    f_pre = fig.add_subplot(spec[1])
    f_sst = f_pre.twinx()
    sst_lon_mean = Sst_nc['reg'].mean('lat', skipna=True).to_numpy()
    pre_lon_mean = Pre_nc['reg'].mean('lat', skipna=True).to_numpy()
    sst_lon = Sst_nc['lon'].to_numpy()
    pre_lon = Pre_nc['lon'].to_numpy()

    f_pre.axhline(y=0, color='#757575', linestyle='--', linewidth=1, zorder=5)
    f_pre.plot(pre_lon, pre_lon_mean, color='green', linewidth=1.5, label='PRE', zorder=5)
    f_sst.plot(sst_lon, sst_lon_mean, color='blue', linewidth=1.5, label='SST', zorder=5)


                                                                                                       
    # 对齐 x 轴与上方主图
    f_sst.set_xlim(0, 360)
    f_sst.set_xticks([0, 60, 120, 180, 240, 300, 360])
    f_sst.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
    f_pre.set_ylim(-2, 2)
    f_pre.set_yticks([-1.5, 0, 1.5])
    f_pre.set_yticklabels(['-1.5', ' 0 ', '1.5'])
    f_sst.set_ylim(-1.2, 1.2)     
    f_sst.set_yticks([-0.6, 0, 0.6])
    f_sst.set_yticklabels(['-0.6', ' 0 ', '0.6'])
    f_sst.tick_params(axis='y', labelsize=10, width=2, length=8, colors='blue')
    f_pre.tick_params(axis='x', labelsize=10, width=2, length=8, colors='black')
    f_pre.tick_params(axis='y', labelsize=10, width=2, length=8, colors='green')

    f_pre.spines['left'].set_color('green')
    f_pre.spines['right'].set_visible(False)
    f_sst.spines['right'].set_color('blue')
    f_sst.spines['left'].set_visible(False)

    # 视需要添加 y 轴标签与图例
    f_sst.set_ylabel('', fontsize=11)
    f_sst.legend(loc='lower right', bbox_to_anchor=(0.90, 0.00), fontsize=8, frameon=False)
    f_pre.legend(loc='lower right', fontsize=8, frameon=False)
    f_sst.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.5)

    for spine in f_ax.spines.values():
        spine.set_zorder(100)
        spine.set_linewidth(2)  # 设置边框线宽
    for spine in f_sst.spines.values():
        spine.set_linewidth(2)  # 设置边框线宽
    for spine in f_pre.spines.values():
        spine.set_linewidth(2)  # 设置边框线宽

# color bar位置
position = fig.add_axes([0.196, 0.00, 0.64, 0.03])
cb1 = plt.colorbar(q_div, cax=position, orientation='horizontal', drawedges=True)#orientation为水平或垂直
cb1.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
cb1.outline.set_linewidth(2.0)  # 加粗 colorbar 边框线宽
cb1.dividers.set_color('black') # 将colorbar内间隔线调为黑色
try:
    cb1.dividers.set_linewidth(2.0)  # 加粗隔断线宽
except Exception:
    # 某些 matplotlib 版本下 dividers 可能不是支持 set_linewidth 的对象，退回到设置每个 divider 的 linewidth
    for d in getattr(cb1, 'dividers', []):
        try:
            d.set_linewidth(2.0)
        except Exception:
            pass
cb1.ax.tick_params(length=0, labelsize=12, width=2)#length为刻度线的长度
cb1.locator = ticker.FixedLocator(lev_z) # colorbar上的刻度值个数
cb1.set_ticks(lev_z * 10e-9)
cb1.set_ticklabels([f'{i}' for i in lev_z])
# cb1.set_label('Divergence(10$^{-9}$ g·m$^{-2}$·s$^{-1}$)', fontsize=16)

# f_ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
# f_ax.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
f_ax.set_xticks([])
f_ax.tick_params(labelsize=10, width=2, length=8)  # length为刻度线的长度
#f_ax.set_yscale('log')
f_ax.set_yticks([1000, 850, 700, 600, 500, 400, 300, 200, 100])
f_ax.set_yticklabels(['1000','850', '700', '600', '500', '400', '300', '200', '100'])

plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r2赤道纬向环流合成归因.png", dpi=600, bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r2赤道纬向环流合成归因.pdf", bbox_inches='tight')
