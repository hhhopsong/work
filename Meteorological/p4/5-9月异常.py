from climkit.average_filter import nanmean_filter
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec
from matplotlib import ticker, colors
from matplotlib.ticker import MultipleLocator
import cmaps
import pandas as pd


from climkit.masked import masked   # 气象工具函数

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

t2m = xr.open_dataset(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc")['t2m']
t2m = t2m.sel(date=slice('1979-01-01', '2022-12-31'))
t2m = xr.Dataset(
    {'t2m': (['time', 'lat', 'lon'], t2m.data)},
    coords={'time': pd.to_datetime(t2m['date'], format="%Y%m%d"),
            'lat': t2m['latitude'].data,
            'lon': t2m['longitude'].data})
t2m = t2m.sel(time=slice('1979-01-01', '2022-12-31'))
t2m = t2m.sel(time=t2m['time.month'].isin([5, 6, 7, 8, 9]))
t2m = t2m.transpose('time', 'lat', 'lon')

# 绘图
# ##地图要素设置
plt.rcParams['font.family'] = ['AVHershey Simplex', 'AVHershey Duplex', 'Helvetica']    # 字体为Hershey (安装字体后，清除.matplotlib的字体缓存即可生效)
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
spec = gridspec.GridSpec(ncols=3, nrows=2, wspace=0.13, hspace=0.25)  # 设置子图比例
extent_CN = [88, 124, 22, 38]  # 中国大陆经度范围，纬度范围
proj = ccrs.PlateCarree()   # 投影方式

#ax1
# 读取CN05.1逐日最高气温数据
t2m_clim = t2m.groupby('time.month').mean('time')  # 逐月气候态
t2m_ano = t2m.groupby('time.month') - t2m_clim  # 逐月距平
t2m_ano = masked(t2m_ano, fr"{PYFILE}/map/地图边界数据/长江区1：25万界线数据集（2002年）/长江区.shp")
t2m_ano = t2m_ano.assign_coords(year=t2m_ano["time"].dt.year, month=t2m_ano["time"].dt.month).set_index(time=["year", "month"]).unstack("time")


fig = plt.figure(figsize=(15, 6))   # 创建画布
for i in [5, 6, 7, 8, 9]:
    year = 1980
    ax = fig.add_subplot(spec[i-5], projection=proj)  # 添加子图
    # 统一加粗所有四个边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽

    ax.set_extent(extent_CN, crs=proj) # 设置地图范围
    ax.set_aspect('auto')  # 设置长宽比
    ax.set_title(f'({chr(ord('a')+i-5)}) {i}_T2m', fontsize=16, loc='left')
    a1 = ax.contourf(t2m['lon'], t2m['lat'], t2m_ano.sel(year=year, month=i).to_array()[0], cmap=cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4], levels=[-2, -1.6, -1.2, -0.8, -0.4, -0.1, 0.1, 0.4, 0.8, 1.2, 1.6, 2], extend='both', transform=proj)
    ax.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/长江区1：25万界线数据集（2002年）/长江区.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=.5)
    # ax.add_geometries(Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(), ccrs.PlateCarree(), facecolor='gray', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(fr'{PYFILE}/map/地图线路数据/长江/长江.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=0.6)
    ax.add_geometries(Reader(fr'{PYFILE}/map/地图线路数据/长江干流_lake/lake_wsg84.shp').geometries(), ccrs.PlateCarree(), facecolor='blue', edgecolor='blue', linewidth=0.2)

    # 刻度线设置
    xticks1 = np.arange(extent_CN[0], extent_CN[1] + 1, 10)
    yticks1 = np.arange(extent_CN[2], extent_CN[3] + 1, 10)
    ax.set_xticks(xticks1, crs=proj)
    ax.set_yticks(yticks1, crs=proj)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    xmajorLocator = MultipleLocator(5)  # 先定义xmajorLocator，再进行调用
    ax.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
    xminorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
    ymajorLocator = MultipleLocator(4)
    ax.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
    yminorLocator = MultipleLocator(1)
    ax.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax.tick_params(which='major', length=5, width=1, color='black')  # 最大刻度长度，宽度设置，
    ax.tick_params(which='minor', length=2, width=.9, color='black')  # 最小刻度长度，宽度设置
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    ax.grid(False)
    print(f'{i}月绘制完成')



# color bar位置
# position = fig.add_axes([0.296, 0.08, 0.44, 0.011])#位置[x, y, w, h]
ax_cbar = fig.add_axes([0.92, 0.23, 0.015, 0.54])
cb = plt.colorbar(a1, orientation='vertical', cax=ax_cbar, aspect=30, shrink=.6, drawedges=True)#orientation为水平或垂直
cb.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
cb.outline.set_linewidth(1.5) # 在这里设置你想要的粗细，例如 2.5
cb.dividers.set_color('black')  # 将colorbar内间隔线调为黑色
cb.dividers.set_linewidth(1.5)
cb.ax.tick_params(length=0, labelsize=12, color='black', direction='in')#length为刻度线的长度
tick_locator = ticker.FixedLocator([0, 5, 10, 15, 20, 25])  # colorbar上的刻度值个数

plt.savefig(fr'{PYFILE}/p4/pic/图1_{year}.pdf', bbox_inches='tight')
plt.savefig(fr'{PYFILE}/p4/pic/图1_{year}.png', dpi=600, bbox_inches='tight')