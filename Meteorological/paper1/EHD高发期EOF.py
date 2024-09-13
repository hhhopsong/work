from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import pymannkendall as mk
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, FixedLocator
from matplotlib import gridspec
import matplotlib.colors as colors
from cnmaps import get_adm_maps, draw_maps
from eofs.standard import Eof
import cmaps
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
from toolbar.masked import masked  # 气象工具函数
from toolbar.significance_test import corr_test
from toolbar.sub_adjust import adjust_sub_axes
from toolbar.pre_whitening import ws2001
from toolbar.curved_quivers_master.modplot import velovect
import pandas as pd
import tqdm as tq
import seaborn as sns
import multiprocessing


# 数据读取
EHD = xr.open_dataset(r"cache\EHD.nc")  # 读取缓存
EHD = masked(EHD,
             r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
# 截取7月16日-8月19日数据
EHD_7 = EHD.sel(time=EHD['time.month'].isin([7]))
EHD_7 = EHD_7.sel(time=EHD_7['time.day'].isin(range(16, 32)))
EHD_8 = EHD.sel(time=EHD['time.month'].isin([8]))
EHD_8 = EHD_8.sel(time=EHD_8['time.day'].isin(range(1, 20)))
# 合并数据,并按时间排序
EHD_concat = xr.concat([EHD_7, EHD_8], dim='time').sortby('time')
EHD_concat.fillna(0)  # 数据二值化处理(1:极端高温,0:非极端高温)
EHD_concat = EHD_concat['tmax'].groupby('time.year').sum('time')  # 计算7月16日-8月19日累计极端高温日数
EHD_concat = masked(EHD_concat,
                    r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
# 计算EOF
eof = Eof(EHD_concat.to_numpy())  #进行eof分解
Modality = eof.eofs(eofscaling=2, neofs=1)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
PC = eof.pcs(pcscaling=1, npcs=1)  # 同上 npcs决定输出的时间序列个数
s = eof.varianceFraction(neigs=1)  # 得到前neig个模态的方差贡献
print('EOF计算完成')
# uvz
try:
    uv = xr.open_dataset(r"cache\ehd_eof\uv.nc")  # 读取缓存
except:
    uv = xr.open_dataset(r"E:\data\ERA5\ERA5_pressLev\era5_pressLev.nc").sel(
        date=slice('19790101', '20221231'),
        pressure_level=[850],
        latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])
    uv = xr.DataArray([uv['u'].data, uv['v'].data, uv['z'].data], coords=[('var', ['u', 'v', 'z']),
                                         ('time', pd.to_datetime(uv['date'], format="%Y%m%d")),
                                         ('p', uv['pressure_level'].data),
                                         ('lat', uv['latitude'].data),
                                         ('lon', uv['longitude'].data)]).to_dataset(name='uv')
    uv.to_netcdf(r"cache\ehd_eof\uv.nc")
uv = uv.sel(time=uv['time.month'].isin([7, 8]))
uv = uv.groupby('time.year').mean('time') # 两月平均
ols = np.load(r"cache\OLS_detrended.npy")  # 读取缓存

for i in ['u', 'v', 'z']:
    try:
        corr = np.load(fr"cache\ehd_eof\{i}_corr.npy")  # 读取缓存
    except:
        corr = np.array([[np.corrcoef(ols, uv['uv'].sel(var=i, p=850, lat=ilat, lon=ilon))[0, 1] for ilon in uv['lon']] for ilat in tq.tqdm(uv['lat'])])
        np.save(fr"cache\ehd_eof\{i}_corr.npy", corr)  # 保存缓存

u_r = np.load(fr"cache\ehd_eof\u_corr.npy")
v_r = np.load(fr"cache\ehd_eof\v_corr.npy")
z_r = np.load(fr"cache\ehd_eof\z_corr.npy")
print('相关系数加载完成')

u显著性检验结果 = corr_test(ols, u_r, alpha=0.10)
v显著性检验结果 = corr_test(ols, v_r, alpha=0.10)
z显著性检验结果 = corr_test(ols, z_r, alpha=0.10)
uv显著性检验结果 = np.where(np.where(u显著性检验结果 == 1, 1, 0) + np.where(v显著性检验结果 == 1, 1, 0) >= 1, 1, np.nan)
z显著性检验结果 = np.where(z显著性检验结果 == 1, 1, np.nan)
# 通过显著性检验结果进行筛选
u_corr = np.where(uv显著性检验结果 == 1, u_r, np.nan)
v_corr = np.where(uv显著性检验结果 == 1, v_r, np.nan)
u_np = np.where(uv显著性检验结果 != 1, u_r, np.nan)
v_np = np.where(uv显著性检验结果 != 1, v_r, np.nan)
u_np = np.where(u_np ** 2 + v_np ** 2 >= 0.15 ** 2, u_np, np.nan)
v_np = np.where(u_np ** 2 + v_np ** 2 >= 0.15 ** 2, v_np, np.nan)
print('显著性检验完成')

# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))  # 创建画布
spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], height_ratios=[1])  # 设置子图比例

extent_changjiang = [88, 124, 22, 38]  # 中国大陆经度范围，纬度范围
extent_CN = [85, 145, 12, 48]  # 中国大陆经度范围，纬度范围
proj = ccrs.PlateCarree()  # 投影方式
ax1 = fig.add_subplot(spec[0, 0], projection=proj)  # 添加子图
# 设置ax1 figsize=(9, 4)
ax1.set_extent(extent_CN, crs=proj)  # 设置地图范围
a1 = ax1.contourf(EHD['lon'], EHD['lat'], Modality[0], cmap=cmaps.BlueWhiteOrangeRed[140:],
                  levels=[0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
                  extend='both',transform=proj)
####
level2 = [-0.4 + i * 0.2 for i in range(5)]
z_r = gaussian_filter(z_r, 3)
a1_h = ax1.contour(uv['lon'], uv['lat'], z_r, transform=proj, levels=level2[:2], colors='blue', linewidths=1.5, linestyles='--',alpha=1)
a1_hm = ax1.contour(uv['lon'], uv['lat'], z_r, transform=proj, levels=[0], colors='gray', linewidths=1.5, linestyles='-', alpha=1)
a1_h1 = ax1.contour(uv['lon'], uv['lat'], z_r, transform=proj, levels=level2[3:], colors='red', linewidths=1.5, linestyles='-', alpha=1)
ax1.clabel(a1_h, inline=True, fontsize=14, fmt='%.01f', colors='blue')
ax1.clabel(a1_hm, inline=True, fontsize=14, fmt='%.01f', colors='gray')
ax1.clabel(a1_h1, inline=True, fontsize=14, fmt='%.01f', colors='red')
ax1.text(124.5, 31.6, 'A', fontsize=20, fontweight='bold', color='blue', zorder=20)
####
'''a1_uv = ax1.quiver(uv['lon'], uv['lat'], u_corr, v_corr, transform=proj, pivot='mid',scale=25,  regrid_shape=20,
                   headlength=3,headaxislength=3)'''
a1_uv = velovect(ax1, uv['lon'].data, uv['lat'].data[::-1], np.array(u_corr.tolist())[::-1, :], np.array(v_corr.tolist())[::-1, :], arrowstyle='fancy', scale = 1.25, grains = 100, color='black', transform=proj)
a1_uv_np_ = velovect(ax1, uv['lon'].data, uv['lat'].data[::-1], np.array(u_np.tolist())[::-1, :], np.array(v_np.tolist())[::-1, :], arrowstyle='fancy', scale = 1.25, grains = 100, color='gray', transform=proj)
a1_uv_np = ax1.quiver(uv['lon'], uv['lat'], u_np, v_np, color='gray', scale=15, regrid_shape=20, transform=proj)
ax1.quiverkey(a1_uv_np, X=0.90, Y=1.03, U=1,angle = 0,  label='1 m/s',
              labelpos='E', color='black',labelcolor = 'k',linewidth=0.8)  # linewidth=1为箭头的大小
cbar = plt.colorbar(a1, ax=ax1, orientation='horizontal', pad=0.05, aspect=50, shrink=0.8)
draw_maps(get_adm_maps(level='国'), linewidth=0.5)
ax1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray')  # 添加陆地并且陆地部分全部填充成浅灰色
ax1.add_geometries(Reader(
    r'C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(),
                   ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4)
ax1.add_geometries(Reader(r'D:\CODES\Python\Meteorological\maps\cnriver\长江\长江.shp').geometries(),
                   ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=0.2)
DBATP = r"D:\CODES\Python\PythonProject\map\DBATP\TP_2500m\TPBoundary_2500m.shp"
provinces = cfeature.ShapelyFeature(Reader(DBATP).geometries(), crs=ccrs.PlateCarree(), facecolor='gray', alpha=1)
# ax1.add_feature(provinces, lw=0.5, zorder=2)
# 设置坐标轴
xticks1=np.arange(extent_CN[0], extent_CN[1]+1, 10)
yticks1=np.arange(extent_CN[2], extent_CN[3]+1, 10)
ax1.set_xticks(xticks1, crs=proj)
ax1.set_yticks(yticks1, crs=proj)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
xmajorLocator = MultipleLocator(5)#先定义xmajorLocator，再进行调用
ax1.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
xminorLocator = MultipleLocator(2)
ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ymajorLocator = MultipleLocator(4)
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
yminorLocator = MultipleLocator(1)
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度


sns.set(style='ticks')
ax1_pc = fig.add_axes(ax1.get_position())
ax1_pc.set_title(f'{s[0] * 100:.2f}%', loc='right')
# 设置坐标轴
ax1_pc_xmajor = FixedLocator([1+i*5 for i in range(9)]) #先定义xmajorLocator，再进行调用
ax1_pc_xminor = MultipleLocator(1)
ax1_pc.xaxis.set_major_locator(ax1_pc_xmajor)  #x轴最大刻度
ax1_pc.xaxis.set_minor_locator(ax1_pc_xminor)  #x轴最小刻度
ax1_pc_ymajor = MultipleLocator(1)  #先定义xmajorLocator，再进行调用
ax1_pc_yminor = MultipleLocator(.5)
ax1_pc.yaxis.set_major_locator(ax1_pc_ymajor)  #x轴最大刻度
ax1_pc.yaxis.set_minor_locator(ax1_pc_yminor)  #x轴最小刻度
# 画条形图,正值为红色，负值为蓝色
a1_pc = sns.barplot(x=[i for i in range(1979, 2023)], y=PC[:, 0], ax=ax1_pc)
for i in range(44):
    if PC[i, 0] > 0:
        a1_pc.get_children()[i].set_color('#D85F4F')
    elif PC[i, 0] == 0:
        a1_pc.get_children()[i].set_color('#F7F7F7')
    else:
        a1_pc.get_children()[i].set_color('#1F6AA0')
ax1_pc.set_xlim(-.5, 43.5)
ax1_pc.set_ylim(-3, 3)
#设定子图ax2大小位置
adjust_sub_axes(ax1, ax1_pc, shrink=1, lr=-.1, ud=1.0)
ax1_pc_reg = ax1_pc.twinx()
k, b = mk.sens_slope(ws2001(PC[:, 0]))  # Theil-Sen 斜率, 截距
ax1_pc_reg = sns.regplot(x=[i for i in range(44)], y=PC[:, 0], ax=ax1_pc_reg, scatter=False, color='#74C476', ci=95)
#ax1_pc_sens = sns.lineplot(x=[i for i in range(44)], y=k * np.array([i for i in range(44)]) + b, ax=ax1_pc_reg, color='black')
ax1_pc_reg.set_ylim(-3, 3)
#ax1_pc_sens.set_ylim(-3, 3)
ax1_pc_reg.yaxis.set_visible(False)  # 隐藏y轴标签
ax1_pc_reg.spines['top'].set_visible(False)  # 隐藏上边框
ax1_pc_reg.spines['right'].set_visible(False)  # 隐藏右边框
ax1_pc_reg.spines['bottom'].set_visible(False)  # 显示下边框
ax1_pc_reg.spines['left'].set_visible(False)  # 显示左边框
#  pd.Series(PC[:, 0]).autocorr(2)  自相关计算=0.26


plt.savefig(r'C:\Users\10574\desktop\pic\EHD高发期EOF.png', dpi=1500, bbox_inches='tight')
plt.show()
