import xarray as xr
import xgrads
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import scipy.ndimage as ndimage
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import geopandas as gpd


from toolbar.curved_quivers.modplot import Curlyquiver


def draw_frc():
    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(wspace=0)
    # 字体为新罗马
    plt.rcParams['font.family'] = 'Times New Roman'
    frc_nc_p = xr.open_dataset(r'D:\PyFile\p2\lbm\type1_apre_frc_p.nc') * 86400
    lon_itp = np.arange(0, 360, 0.25)
    lat_itp = np.arange(-90, 90.25, 0.25)
    frc_nc_p = frc_nc_p.interp(lon=lon_itp, lat=lat_itp, kwargs={"fill_value": "extrapolate"})

    lbm = xr.open_dataset(r'D:\PyFile\p2\lbm\type1_apre.nc')
    u = lbm['u'][19:25].mean('time')
    v = lbm['v'][19:25].mean('time')
    t = lbm['t'][19:25].mean('time')
    z = lbm['z'][19:25].mean('time')
    lon = lbm['lon']
    lat = lbm['lat']
    extent1 = [0, 360, -20, 80]
    c_lon_1 = 60
    # 绘图
    # 图1
    lev = 200
    ax1 = fig.add_subplot(332, projection=ccrs.PlateCarree(central_longitude=c_lon_1))
    ax1.set_title('b) Exp TAP 200hPa UV&FRC', fontsize=10, loc='left')
    ax1.set_aspect('auto')
    ax1.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax1.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
    ax1.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    var = 't'
    frc_fill_white, lon_fill_white = add_cyclic_point(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)

    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon

    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax1.contourf(lon_fill_white, frc_nc_p[var]['lat'], np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white, np.nan),
                        levels=lev_range, cmap=cmaps.MPL_RdYlGn[22+0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106-0], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    wind200 = Curlyquiver(ax1, lon_UV, lat, U, V, arrowsize=.5, scale=4, regrid=13, linewidth=.25, nanmax=10,
                        color="black", center_lon=c_lon_1, thinning=[0.4, 'min'], MinDistance=[0.2, 0.1])
    # wind200_ = Curlyquiver(ax1, lon_UV, lat, U, V, arrowsize=.5, scale=8, regrid=13, linewidth=.5, nanmax=wind200.nanmax,
    #                     color="black", center_lon=c_lon_1, thinning=[1.5, 'max_full'])
    wind200.key(fig, U=1.5, label='1.5 m/s', ud=7.8, edgecolor='none', arrowsize=.5, linewidth=.5, fontproperties={'size': 8})
    # wind200_.key(fig, U=1.5, label='> 1.5 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k', linewidth=.75)

    # 图2
    lev = 500
    extent1 = extent1
    ax2 = fig.add_subplot(335, projection=ccrs.PlateCarree(central_longitude=c_lon_1))
    ax2.set_title('e) Exp TAP 500hPa UV&FRC', fontsize=10, loc='left')
    ax2.set_aspect('auto')
    ax2.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax2.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
    ax2.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    frc_fill_white, lon_fill_white = add_cyclic_point(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    # lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)
    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax2.contourf(lon_fill_white, frc_nc_p[var]['lat'], np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white, np.nan),
                        levels=lev_range, cmap=cmaps.MPL_RdYlGn[22+0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106-0], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    wind500 = Curlyquiver(ax2, lon_UV, lat, U, V, arrowsize=.5, scale=4, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                           color="black", center_lon=c_lon_1, thinning=[.25, 'min'], MinDistance=[0.2, 0.1])
    # wind500_ = Curlyquiver(ax2, lon_UV, lat, U, V, arrowsize=.5, scale=8, regrid=13, linewidth=.5, nanmax=wind200.nanmax,
    #                     color="black", center_lon=c_lon_1, thinning=[1.5, 'max_full'])
    wind500.key(fig, U=1.5, label='1.5 m/s', ud=7.8, edgecolor='none', arrowsize=.5, linewidth=.5, fontproperties={'size': 8})
    # wind500_.key(fig, U=1.5, label='> 1.5 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k', linewidth=.75)

    # 图1
    lev = 850
    extent1 = extent1
    ax4 = fig.add_subplot(338, projection=ccrs.PlateCarree(central_longitude=c_lon_1))
    ax4.set_title('i) Exp TAP 850hPa UV&FRC', fontsize=10, loc='left')
    ax4.set_aspect('auto')
    ax4.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax4.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
    ax4.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    frc_fill_white, lon_fill_white = add_cyclic_point(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    # lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)
    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax4.contourf(lon_fill_white, frc_nc_p[var]['lat'], np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white, np.nan),
                        levels=lev_range, cmap=cmaps.MPL_RdYlGn[22+0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106-0], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    #z850 = ax4.contour(lon_Z, lat, Z, levels=4, colors='black', transform=ccrs.PlateCarree(central_longitude=0), linewidths=0.4)
    wind850 = Curlyquiver(ax4, lon_UV, lat, U, V, arrowsize=.5, scale=2, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                           color="black", center_lon=c_lon_1, thinning=[.2, 'min'], MinDistance=[0.2, 0.1])
    # wind850_ = Curlyquiver(ax4, lon_UV, lat, U, V, arrowsize=.5, scale=4, regrid=13, linewidth=.5, nanmax=wind200.nanmax,
    #                     color="black", center_lon=c_lon_1, thinning=[.75, 'max_full'])
    wind850.key(fig, U=.75, label='0.75 m/s', ud=7.8, edgecolor='none', arrowsize=.5, linewidth=.5, fontproperties={'size': 8})
    # wind850_.key(fig, U=.75, label='> 0.75 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k', linewidth=.75)
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
    ax1.tick_params(which='major', length=4, width=.3, color='black')  # 最大刻度长度，宽度设置，
    ax1.tick_params(which='minor', length=2, width=.3, color='black')  # 最小刻度长度，宽度设置
    ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    ax2.tick_params(which='major', length=4, width=.3, color='black')  # 最大刻度长度，宽度设置，
    ax2.tick_params(which='minor', length=2, width=.3, color='black')  # 最小刻度长度，宽度设置
    ax2.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)

    ax4.tick_params(which='major', length=4, width=.3, color='black')  # 最大刻度长度，宽度设置，
    ax4.tick_params(which='minor', length=2, width=.3, color='black')  # 最小刻度长度，宽度设置
    ax4.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax1.tick_params(axis='both', labelsize=8, colors='black')
    ax2.tick_params(axis='both', labelsize=8, colors='black')
    ax4.tick_params(axis='both', labelsize=8, colors='black')


    frc_nc_p = xr.open_dataset(r'D:\PyFile\p2\lbm\type1_ppre_frc_p.nc') * 86400
    frc_nc_p = frc_nc_p.interp(lon=lon_itp, lat=lat_itp, kwargs={"fill_value": "extrapolate"})
    lbm = xr.open_dataset(r'D:\PyFile\p2\lbm\type1_ppre.nc')
    u = lbm['u'][19:25].mean('time')
    v = lbm['v'][19:25].mean('time')
    t = lbm['t'][19:25].mean('time')
    z = lbm['z'][19:25].mean('time')
    lon = lbm['lon']
    lat = lbm['lat']
    extent1 = [0, 360, -20, 80]
    c_lon_1 = 60
    # 绘图
    # 图1
    lev = 200
    ax1_ = fig.add_subplot(331, projection=ccrs.PlateCarree(central_longitude=c_lon_1))
    ax1_.set_title('a) Exp TCEP 200hPa UV&FRC', fontsize=10, loc='left')
    ax1_.set_aspect('auto')
    ax1_.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax1_.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
    ax1_.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    var = 't'
    frc_fill_white, lon_fill_white = add_cyclic_point(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    # lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)

    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon

    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax1_.contourf(lon_fill_white, frc_nc_p[var]['lat'], np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white, np.nan),
                        levels=lev_range, cmap=cmaps.MPL_RdYlGn[22+0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106-0], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    wind200 = Curlyquiver(ax1_, lon_UV, lat, U, V, arrowsize=.5, scale=4, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                        color="black", center_lon=c_lon_1, thinning=[.4, 'min'], MinDistance=[0.2, 0.1])
    # wind200_ = Curlyquiver(ax1_, lon_UV, lat, U, V, arrowsize=.5, scale=8, regrid=13, linewidth=.5, nanmax=wind200.nanmax,
    #                     color="black", center_lon=c_lon_1, thinning=[1.5, 'max_full'])
    wind200.key(fig, U=1.5, label='1.5 m/s', ud=7.8, edgecolor='none', arrowsize=.5, linewidth=.5, fontproperties={'size': 8})
    # wind200_.key(fig, U=1.5, label='> 1.5 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k', linewidth=.75)
    # 图2
    lev = 500
    extent1 = extent1
    ax2_ = fig.add_subplot(334, projection=ccrs.PlateCarree(central_longitude=c_lon_1))
    ax2_.set_title('d) Exp TCEP 500hPa UV&FRC', fontsize=10, loc='left')
    ax2_.set_aspect('auto')
    ax2_.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax2_.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
    ax2_.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    frc_fill_white, lon_fill_white = add_cyclic_point(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    # lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)
    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax2_.contourf(lon_fill_white, frc_nc_p[var]['lat'], np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white, np.nan),
                        levels=lev_range, cmap=cmaps.MPL_RdYlGn[22+0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106-0], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    wind500 = Curlyquiver(ax2_, lon_UV, lat, U, V, arrowsize=.5, scale=4, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                           color="black", center_lon=c_lon_1, thinning=[.25, 'min'], MinDistance=[0.2, 0.1])
    # wind500_ = Curlyquiver(ax2, lon_UV, lat, U, V, arrowsize=.5, scale=8, regrid=13, linewidth=.5, nanmax=wind200.nanmax,
    #                     color="black", center_lon=c_lon_1, thinning=[1.5, 'max_full'])
    wind500.key(fig, U=1.5, label='1.5 m/s', ud=7.8, edgecolor='none', arrowsize=.5, linewidth=.5, fontproperties={'size': 8})
    # wind500_.key(fig, U=1.5, label='> 1.5 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k', linewidth=.75)

    # 图1
    lev = 850
    extent1 = extent1
    ax4_ = fig.add_subplot(337, projection=ccrs.PlateCarree(central_longitude=c_lon_1))
    ax4_.set_title('g) Exp TCEP 850hPa UV&FRC', fontsize=10, loc='left')
    ax4_.set_aspect('auto')
    ax4_.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax4_.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
    ax4_.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    frc_fill_white, lon_fill_white = add_cyclic_point(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    # lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)
    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax4_.contourf(lon_fill_white, frc_nc_p[var]['lat'], np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white, np.nan),
                        levels=lev_range, cmap=cmaps.MPL_RdYlGn[22+0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106-0], transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    #z850 = ax4_.contour(lon_Z, lat, Z, levels=4, colors='black', transform=ccrs.PlateCarree(central_longitude=0), linewidths=0.4)
    wind850 = Curlyquiver(ax4_, lon_UV, lat, U, V, arrowsize=.5, scale=2, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                           color="black", center_lon=c_lon_1, thinning=[.2, 'min'], MinDistance=[0.2, 0.1])
    # wind850_ = Curlyquiver(ax4_, lon_UV, lat, U, V, arrowsize=.5, scale=4, regrid=13, linewidth=.5, nanmax=wind200.nanmax,
    #                     color="black", center_lon=c_lon_1, thinning=[.75, 'max_full'])
    wind850.key(fig, U=.75, label='0.75 m/s', ud=7.8, edgecolor='none', arrowsize=.5, linewidth=.5, fontproperties={'size': 8})
    # wind850_.key(fig, U=.75, label='> 0.75 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k', linewidth=.75)
    DBATP = r"D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary_2500m\TPBoundary_2500m.shp"
    provinces = cfeature.ShapelyFeature(Reader(DBATP).geometries(), crs=ccrs.PlateCarree(), facecolor='gray', alpha=1)
    ax4_.add_feature(provinces, lw=0.5, zorder=2)


    # 刻度线设置
    xticks1 = np.arange(extent1[0], extent1[1] + 1, 10)
    yticks1 = np.arange(extent1[2], extent1[3] + 1, 10)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    # ax1_
    ax1_.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax1_.yaxis.set_major_formatter(lat_formatter)
    # ax2_
    ax2_.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax2_.yaxis.set_major_formatter(lat_formatter)
    # ax4_
    ax4_.set_xticks(xticks1, crs=ccrs.PlateCarree())
    ax4_.set_yticks(yticks1, crs=ccrs.PlateCarree())
    ax4_.xaxis.set_major_formatter(lon_formatter)
    ax4_.yaxis.set_major_formatter(lat_formatter)

    xmajorLocator = ticker.MultipleLocator(60)  # 先定义xmajorLocator，再进行调用
    xminorLocator = ticker.MultipleLocator(10)
    ymajorLocator = ticker.MultipleLocator(30)
    yminorLocator = ticker.MultipleLocator(10)
    ax4_.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
    ax4_.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度

    ax1_.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
    ax2_.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
    #ax3_.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
    ax4_.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度

    ax1_.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    ax2_.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    #ax3_.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    ax4_.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度

    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax1_.tick_params(which='major', length=4, width=.3, color='darkgray')  # 最大刻度长度，宽度设置，
    ax1_.tick_params(which='minor', length=2, width=.3, color='darkgray')  # 最小刻度长度，宽度设置
    ax1_.tick_params(which='both', bottom=False, top=False, left=True, labelbottom=True, labeltop=False)
    ax2_.tick_params(which='major', length=4, width=.3, color='darkgray')  # 最大刻度长度，宽度设置，
    ax2_.tick_params(which='minor', length=2, width=.3, color='darkgray')  # 最小刻度长度，宽度设置
    ax2_.tick_params(which='both', bottom=False, top=False, left=True, labelbottom=True, labeltop=False)

    ax4_.tick_params(which='major', length=4, width=.3, color='black')  # 最大刻度长度，宽度设置，
    ax4_.tick_params(which='minor', length=2, width=.3, color='black')  # 最小刻度长度，宽度设置
    ax4_.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax1_.tick_params(axis='both', labelsize=8, colors='black')
    ax2_.tick_params(axis='both', labelsize=8, colors='black')

    ax4_.tick_params(axis='both', labelsize=8, colors='black')

    frc_nc_p = xr.open_dataset(r'D:\PyFile\p2\lbm\type1_frc_p.nc') * 86400
    frc_nc_p = frc_nc_p.interp(lon=lon_itp, lat=lat_itp, kwargs={"fill_value": "extrapolate"})
    lbm = xr.open_dataset(r'D:\PyFile\p2\lbm\type1_all.nc')
    u = lbm['u'][19:25].mean('time')
    v = lbm['v'][19:25].mean('time')
    t = lbm['t'][19:25].mean('time')
    z = lbm['z'][19:25].mean('time')
    lon = lbm['lon']
    lat = lbm['lat']
    extent1 = [0, 360, -20, 80]
    c_lon_1 = 60
    # 绘图
    # 图1
    lev = 200
    ax1 = fig.add_subplot(333, projection=ccrs.PlateCarree(central_longitude=c_lon_1))
    ax1.set_title('g) Exp Both 200hPa UV&FRC', fontsize=10, loc='left')
    ax1.set_aspect('auto')
    ax1.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax1.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
    ax1.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    var = 't'
    frc_fill_white, lon_fill_white = add_cyclic_point(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    # lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)

    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon

    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax1.contourf(lon_fill_white, frc_nc_p[var]['lat'],
                             np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white,
                                      np.nan),
                             levels=lev_range,
                             cmap=cmaps.MPL_RdYlGn[22 + 0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106 - 0],
                             transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    wind200 = Curlyquiver(ax1, lon_UV, lat, U, V, arrowsize=.5, scale=4, regrid=13, linewidth=.25, nanmax=wind200.nanmax,
                          color="black", center_lon=c_lon_1, thinning=[.4, 'min'], MinDistance=[0.2, 0.1])
    # wind200_ = Curlyquiver(ax1, lon_UV, lat, U, V, arrowsize=.5, scale=8, regrid=13, linewidth=.5,
    #                        nanmax=wind200.nanmax, color="black", center_lon=c_lon_1, thinning=[1.5, 'max_full'])
    wind200.key(fig, U=1.5, label='1.5 m/s', ud=7.8, edgecolor='none', arrowsize=.5, linewidth=.5, fontproperties={'size': 8})
    # wind200_.key(fig, U=1.5, label='> 1.5 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k', linewidth=.75)
    # 图2
    lev = 500
    extent1 = extent1
    ax2 = fig.add_subplot(336, projection=ccrs.PlateCarree(central_longitude=c_lon_1))
    ax2.set_title('h) Exp Both 500hPa UV&FRC', fontsize=10, loc='left')
    ax2.set_aspect('auto')
    ax2.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax2.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
    ax2.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    frc_fill_white, lon_fill_white = add_cyclic_point(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    # lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)
    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax2.contourf(lon_fill_white, frc_nc_p[var]['lat'],
                             np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white,
                                      np.nan),
                             levels=lev_range,
                             cmap=cmaps.MPL_RdYlGn[22 + 0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106 - 0],
                             transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    wind500 = Curlyquiver(ax2, lon_UV, lat, U, V, arrowsize=.5, scale=4, regrid=13, linewidth=.25,
                          nanmax=wind200.nanmax, color="black", center_lon=c_lon_1
                          , thinning=[.25, 'min'], MinDistance=[0.2, 0.1])
    # wind500_ = Curlyquiver(ax2, lon_UV, lat, U, V, arrowsize=.5, scale=8, regrid=13, linewidth=.5,
    #                        nanmax=wind200.nanmax,
    #                        color="black", center_lon=c_lon_1, thinning=[1.5, 'max_full'])
    wind500.key(fig, U=1.5, label='1.5 m/s', ud=7.8, edgecolor='none', arrowsize=.5, linewidth=.5, fontproperties={'size': 8})
    # wind500_.key(fig, U=1.5, label='> 1.5 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k', linewidth=.75)

    # 图1
    lev = 850
    extent1 = extent1
    ax4 = fig.add_subplot(339, projection=ccrs.PlateCarree(central_longitude=c_lon_1))
    ax4.set_title('i) Exp Both 850hPa UV&FRC', fontsize=10, loc='left')
    ax4.set_aspect('auto')
    ax4.add_feature(cfeature.LAND.with_scale('110m'), color='lightgray', lw=0.05)
    ax4.add_geometries(Reader(r'D:\PyFile\map\self\长江_TP\长江_tp.shp').geometries(), ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth=.5)
    ax4.set_extent(extent1, crs=ccrs.PlateCarree())
    # 强迫
    frc_fill_white, lon_fill_white = add_cyclic_point(frc_nc_p[var].sel(lev=lev, time=0), frc_nc_p[var]['lon'])
    # lev_range = np.linspace(-np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), np.nanmax(np.abs(frc_nc_p[var].sel(lev=lev).data)), 8)
    # 响应
    T, lon_T = t.sel(lev=lev), lon
    Z, lon_Z = ndimage.gaussian_filter(z.sel(lev=lev), 1), lon
    U, lon_UV = u.sel(lev=lev), lon
    V, lon_UV = v.sel(lev=lev), lon
    zero_mask = (lev_range[1] - lev_range[0]) / 2
    var_contr = ax4.contourf(lon_fill_white, frc_nc_p[var]['lat'],
                             np.where((frc_fill_white >= zero_mask) | (frc_fill_white <= -zero_mask), frc_fill_white,
                                      np.nan),
                             levels=lev_range,
                             cmap=cmaps.MPL_RdYlGn[22 + 0:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:106 - 0],
                             transform=ccrs.PlateCarree(central_longitude=0), extend='both')
    # z850 = ax4.contour(lon_Z, lat, Z, levels=4, colors='black', transform=ccrs.PlateCarree(central_longitude=0), linewidths=0.4)
    wind850 = Curlyquiver(ax4, lon_UV, lat, U, V, arrowsize=.5, scale=2, regrid=13, linewidth=.25,
                          nanmax=wind200.nanmax, color="black", center_lon=c_lon_1, thinning=[.2, 'min'], MinDistance=[0.2, 0.1])
    # wind850_ = Curlyquiver(ax4, lon_UV, lat, U, V, arrowsize=.5, scale=4, regrid=13, linewidth=.5,
    #                        nanmax=wind200.nanmax, color="black", center_lon=c_lon_1, thinning=[.75, 'max_full'])
    wind850.key(fig, U=.75, label='0.75 m/s', ud=7.8, edgecolor='none', arrowsize=.5, linewidth=.5, fontproperties={'size': 8})
    # wind850_.key(fig, U=.75, label='> 0.75 m/s', ud=7.8, lr=2, edgecolor='none', arrowsize=.5, color='k', linewidth=.75)
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
    # ax3.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度
    ax4.yaxis.set_major_locator(ymajorLocator)  # y轴最大刻度

    ax1.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    ax2.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    # ax3.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    ax4.yaxis.set_minor_locator(yminorLocator)  # y轴最小刻度
    ax1.axes.yaxis.set_ticklabels([]) ##隐藏刻度标签
    ax2.axes.yaxis.set_ticklabels([]) ##隐藏刻度标签
    ax4.axes.yaxis.set_ticklabels([]) ##隐藏刻度标签

    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax1.tick_params(which='major', length=0, width=.3, color='darkgray')  # 最大刻度长度，宽度设置，
    ax1.tick_params(which='minor', length=0, width=.3, color='darkgray')  # 最小刻度长度，宽度设置
    ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    ax2.tick_params(which='major', length=0, width=.3, color='darkgray')  # 最大刻度长度，宽度设置，
    ax2.tick_params(which='minor', length=0, width=.3, color='darkgray')  # 最小刻度长度，宽度设置
    ax2.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)

    ax4.tick_params(which='major', length=4, width=.3, color='black')  # 最大刻度长度，宽度设置，
    ax4.tick_params(which='minor', length=2, width=.3, color='black')  # 最小刻度长度，宽度设置
    ax4.tick_params(which='both', bottom=True, top=False, left=False, labelbottom=True, labeltop=False)
    plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
    # 调整刻度值字体大小
    ax1.tick_params(axis='both', labelsize=8, colors='black')
    ax2.tick_params(axis='both', labelsize=8, colors='black')

    ax4.tick_params(axis='both', labelsize=8, colors='black')

    plt.savefig(r'D:\PyFile\p2\pic\Output_对流实验_type1_single.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_frc()
