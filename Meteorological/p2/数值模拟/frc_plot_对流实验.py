import xarray as xr
import xgrads
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import scipy.ndimage as ndimage
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import geopandas as gpd


from toolbar.curved_quivers.modplot import Curlyquiver


def draw_frc():
    frc_nc_p = xr.open_dataset(r'D:\lbm\main\data\Forcing\frc_p.t42l20.nc') * 86400
    lbm = xr.open_dataset(r'D:\lbm\main\data\Output\Output_frc.t42l20.Tingyang.nc')
    u = lbm['u'][19:25].mean('time')
    v = lbm['v'][19:25].mean('time')
    t = lbm['t'][19:25].mean('time')
    z = lbm['z'][19:25].mean('time')
    lon = lbm['lon']
    lat = lbm['lat']
    # 绘图
    # 图1
    lev = 200
    extent1 = [-180, 180, -30, 80]
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(411, projection=ccrs.PlateCarree(central_longitude=100))
    ax1.set_title('a)Exp 200hPa UVZ', fontsize=6, loc='left')
    ax1.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax1.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    var = 't'
    frc_fill_white, lon_fill_white = add_cyclic(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)

    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    r'''z_fill_white, lon_fill_white = add_cyclic(Z, lon_Z)
    t200_low = ax1.contourf(lon_fill_white, lat, z_fill_white, levels=[-9e9, np.percentile(z_fill_white[z_fill_white<0], 50), 0],
                        colors='none', transform=ccrs.PlateCarree(central_longitude=0),
                        hatches=['xxxxxxxxxxxxxxxxxxxxxxxx', '////////////////////////'], alpha=0)
    t200_high = ax1.contourf(lon_fill_white, lat, z_fill_white, levels=[0, np.percentile(z_fill_white[z_fill_white>0], 50), 9e9],
                        colors='none', transform=ccrs.PlateCarree(central_longitude=0),
                        hatches=[r'\\\\\\\\\\\\\\\\\\\\\\\\', 'xxxxxxxxxxxxxxxxxxxxxxxx'], alpha=0)
    plt.rcParams['hatch.linewidth'] = 0.1
    for collection in t200_low.collections:
        collection.set_edgecolor('#4191c6')  # -----打点颜色设置
    for collection in t200_high.collections:
        collection.set_edgecolor('#ef3b2b')  # -----打点颜色设置'''
    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax1.contourf(lon_fill_white, frc_nc_p[var]['lat'], np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white, np.nan),
                        levels=lev_range, cmap=cmaps.MPL_RdYlGn[22+0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106-0], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    wind200 = Curlyquiver(ax1, lon_UV, lat, U, V, arrowsize=.3, scale=4, regrid=13, linewidth=.25,
                        color="#454545", center_lon=100, thinning=[.4, 'max'])
    wind200_ = Curlyquiver(ax1, lon_UV, lat, U, V, arrowsize=.3, scale=4, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                        color="black", center_lon=100, thinning=[.4, 'max_full'])
    wind200.key(fig, U=.4, label='0.4 m/s', ud=7.8, edgecolor='none', arrowsize=.5)
    wind200_.key(fig, U=.4, label='> 0.4 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k')
    # 图2
    lev = 500
    extent1 = extent1
    ax2 = fig.add_subplot(412, projection=ccrs.PlateCarree(central_longitude=100))
    ax2.set_title('b)Exp 500hPa UVZ', fontsize=6, loc='left')
    ax2.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax2.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    frc_fill_white, lon_fill_white = add_cyclic(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)
    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    '''z_fill_white, lon_fill_white = add_cyclic(Z, lon_Z)
    t500_low = ax2.contourf(lon_fill_white, lat, z_fill_white, levels=[-9e9, np.percentile(z_fill_white[z_fill_white<0], 50), 0],
                        colors='none', transform=ccrs.PlateCarree(central_longitude=0),
                        hatches=['xxxxxxxxxxxxxxxxxxxxxxxx', '////////////////////////'], alpha=0)
    t500_high = ax2.contourf(lon_fill_white, lat, z_fill_white, levels=[0, np.percentile(z_fill_white[z_fill_white>0], 50), 9e9],
                        colors='none', transform=ccrs.PlateCarree(central_longitude=0),
                        hatches=[r'\\\\\\\\\\\\\\\\\\\\\\\\', 'xxxxxxxxxxxxxxxxxxxxxxxx'], alpha=0)
    for collection in t500_low.collections:
        collection.set_edgecolor('#4191c6')  # -----打点颜色设置
    for collection in t500_high.collections:
        collection.set_edgecolor('#ef3b2b')  # -----打点颜色设置'''
    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax2.contourf(lon_fill_white, frc_nc_p[var]['lat'], np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white, np.nan),
                        levels=lev_range, cmap=cmaps.MPL_RdYlGn[22+0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106-0], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    wind500 = Curlyquiver(ax2, lon_UV, lat, U, V, arrowsize=.3, scale=2, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                           color="#454545", center_lon=100, thinning=[.2, 'max'])
    wind500_ = Curlyquiver(ax2, lon_UV, lat, U, V, arrowsize=.3, scale=2, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                        color="black", center_lon=100, thinning=[.2, 'max_full'])
    wind500.key(fig, U=.2, label='0.2 m/s', ud=7.8, edgecolor='none', arrowsize=.5)
    wind500_.key(fig, U=.2, label='> 0.2 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k')

    # 图1
    lev = 850
    extent1 = extent1
    ax4 = fig.add_subplot(413, projection=ccrs.PlateCarree(central_longitude=100))
    ax4.set_title('c)Exp 850hPa UVZ', fontsize=6, loc='left')
    ax4.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax4.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    frc_fill_white, lon_fill_white = add_cyclic(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)
    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    '''z_fill_white, lon_fill_white = add_cyclic(Z, lon_Z)
    t800_low = ax4.contourf(lon_fill_white, lat, z_fill_white, levels=[-9e9, np.percentile(z_fill_white[z_fill_white<0], 50), 0],
                        colors='none', transform=ccrs.PlateCarree(central_longitude=0),
                        hatches=['xxxxxxxxxxxxxxxxxxxxxxxx', '////////////////////////'], alpha=0)
    t800_high = ax4.contourf(lon_fill_white, lat, z_fill_white, levels=[0, np.percentile(z_fill_white[z_fill_white>0], 50), 9e9],
                        colors='none', transform=ccrs.PlateCarree(central_longitude=0),
                        hatches=[r'\\\\\\\\\\\\\\\\\\\\\\\\', 'xxxxxxxxxxxxxxxxxxxxxxxx'], alpha=0)
    for collection in t800_low.collections:
        collection.set_edgecolor('#4191c6')  # -----打点颜色设置
    for collection in t800_high.collections:
        collection.set_edgecolor('#ef3b2b')  # -----打点颜色设置'''
    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax4.contourf(lon_fill_white, frc_nc_p[var]['lat'], np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white, np.nan),
                        levels=lev_range, cmap=cmaps.MPL_RdYlGn[22+0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106-0], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    #z850 = ax4.contour(lon_Z, lat, Z, levels=4, colors='black', transform=ccrs.PlateCarree(central_longitude=0), linewidths=0.4)
    wind850 = Curlyquiver(ax4, lon_UV, lat, U, V, arrowsize=.3, scale=1, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                           color="#454545", center_lon=100, thinning=[.1, 'max'])
    wind850_ = Curlyquiver(ax4, lon_UV, lat, U, V, arrowsize=.3, scale=1, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                        color="black", center_lon=100, thinning=[.1, 'max_full'])
    wind850.key(fig, U=.1, label='0.1 m/s', ud=7.8, edgecolor='none', arrowsize=.5)
    wind850_.key(fig, U=.1, label='> 0.1 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k')
    DBATP = r"D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary_2500m\TPBoundary_2500m.shp"
    provinces = cfeature.ShapelyFeature(Reader(DBATP).geometries(), crs=ccrs.PlateCarree(), facecolor='gray', alpha=1)
    ax4.add_feature(provinces, lw=0.5, zorder=2)


    # 刻度线设置
    xticks1 = np.arange(extent1[0], extent1[1] + 1, 10)
    yticks1 = np.arange(extent1[2], extent1[3] + 1, 10)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    # ax1
    ax1.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax1.yaxis.set_major_formatter(lat_formatter)
    # ax2
    ax2.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax2.yaxis.set_major_formatter(lat_formatter)
    '''# ax3
    ax3.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax3.yaxis.set_major_formatter(lat_formatter)'''
    # ax4
    ax4.set_xticks(xticks1, crs=ccrs.PlateCarree())
    ax4.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax4.xaxis.set_major_formatter(lon_formatter)
    ax4.yaxis.set_major_formatter(lat_formatter)

    xmajorLocator = ticker.MultipleLocator(60)  # 先定义xmajorLocator，再进行调用
    xminorLocator = ticker.MultipleLocator(10)
    ymajorLocator = ticker.MultipleLocator(30)
    yminorLocator = ticker.MultipleLocator(10)
    ax4.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
    ax4.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度

    ax1.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
    ax2.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
    #ax3.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
    ax4.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度

    ax1.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    ax2.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    #ax3.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    ax4.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax1.tick_params(which='major', length=4, width=.3, color='darkgray')  # 最大刻度长度，宽度设置，
    ax1.tick_params(which='minor', length=2, width=.3, color='darkgray')  # 最小刻度长度，宽度设置
    ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    ax2.tick_params(which='major', length=4, width=.3, color='darkgray')  # 最大刻度长度，宽度设置，
    ax2.tick_params(which='minor', length=2, width=.3, color='darkgray')  # 最小刻度长度，宽度设置
    ax2.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    #ax3.tick_params(which='major', length=4, width=.3, color='darkgray')  # 最大刻度长度，宽度设置，
    #ax3.tick_params(which='minor', length=2, width=.3, color='darkgray')  # 最小刻度长度，宽度设置
    #ax3.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    ax4.tick_params(which='major', length=4, width=.3, color='darkgray')  # 最大刻度长度，宽度设置，
    ax4.tick_params(which='minor', length=2, width=.3, color='darkgray')  # 最小刻度长度，宽度设置
    ax4.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax1.tick_params(axis='both', labelsize=6, colors='black')
    ax2.tick_params(axis='both', labelsize=6, colors='black')
    #ax3.tick_params(axis='both', labelsize=6, colors='black')
    ax4.tick_params(axis='both', labelsize=6, colors='black')

    plt.savefig(r'D:\PyFile\p2\pic\Output_对流实验.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_frc()
