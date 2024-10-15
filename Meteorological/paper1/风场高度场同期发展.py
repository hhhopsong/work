from cartopy import crs as ccrs
import cartopy.feature as cfeature
import multiprocessing
import sys
import cartopy.feature as cfeature
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr
from cartopy import crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
from cartopy.util import add_cyclic_point
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import filters
from toolbar.significance_test import corr_test
from toolbar.TN_WaveActivityFlux import TN_WAF_3D
from toolbar.Cquiver import curly_vector
from toolbar.curved_quivers_master.modplot import velovect

# 多核计算部分函数
def multi_core(var, p, ols, sen):
    import numpy as np
    print(f"{p}hPa层{var}相关系数计算中...")
    if var == 'u' or var == 'v' or var == 'z':
        pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\{var}_same.nc")[var].sel(p=p).transpose('lat', 'lon', 'year')
    elif var == 'sst':
        pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\sst\sst_same.nc")['sst'].transpose('lat', 'lon', 'year')
    elif var == 'precip':
        pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\pre\pre_same.nc")['precip'].transpose('lat', 'lon', 'year')
    try:
        if var == 'u' or var == 'v' or var == 'z':
            corr_1 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_{var}{p}_same.npy")  # 读取缓存
        elif var == 'sst':
            corr_1 = np.load(fr"D:\PyFile\paper1\cache\sst\corr_sst_same.npy")
        elif var == 'precip':
            corr_1 = np.load(fr"D:\PyFile\paper1\cache\pre\corr_pre_same.npy")
    except:
        shape = pre_diff.shape
        pre_diff = pre_diff.data.reshape(shape[0] * shape[1], shape[2])
        corr_1 = np.array([np.corrcoef(d, ols)[0, 1] for d in tqdm.tqdm(pre_diff)]).reshape(shape[0], shape[1])
        if var == 'u' or var == 'v' or var == 'z':
            np.save(fr"D:\PyFile\paper1\cache\uvz\corr_{var}{p}_same.npy", corr_1)  # 保存缓存
        elif var == 'sst':
            np.save(fr"D:\PyFile\paper1\cache\sst\corr_sst_same.npy", corr_1)
        elif var == 'precip':
            np.save(fr"D:\PyFile\paper1\cache\pre\corr_pre_same.npy", corr_1)
    try:
        if var == 'u' or var == 'v' or var == 'z':
            corr_2 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr2_{var}{p}_same.npy")  # 读取缓存
        elif var == 'sst':
            corr_2 = np.load(fr"D:\PyFile\paper1\cache\sst\corr2_sst_same.npy")
        elif var == 'precip':
            corr_2 = np.load(fr"D:\PyFile\paper1\cache\pre\corr2_pre_same.npy")
    except:
        shape = pre_diff.shape
        pre_diff = pre_diff.data if isinstance(pre_diff, xr.DataArray) else pre_diff
        pre_diff = pre_diff.reshape(shape[0] * shape[1], shape[2])
        corr_2 = np.array([np.corrcoef(d, sen)[0, 1] for d in tqdm.tqdm(pre_diff)]).reshape(shape[0], shape[1])
        if var == 'u' or var == 'v' or var == 'z':
            np.save(fr"D:\PyFile\paper1\cache\uvz\corr2_{var}{p}_same.npy", corr_2)  # 保存缓存
        elif var == 'sst':
            np.save(fr"D:\PyFile\paper1\cache\sst\corr2_sst_same.npy", corr_2)
        elif var == 'precip':
            np.save(fr"D:\PyFile\paper1\cache\pre\corr2_pre_same.npy", corr_2)
    if var == 'z' and p == 200:
        try:
            reg_z = np.load(fr"D:\PyFile\paper1\cache\uvz\reg_{var}{p}_same.npy")
        except:
            shape = pre_diff.shape
            pre_diff = pre_diff.data if isinstance(pre_diff, xr.DataArray) else pre_diff
            pre_diff = pre_diff.data.reshape(shape[0] * shape[1], shape[2])
            reg_z = np.array([np.polyfit(ols, f, 1)[0] for f in tqdm.tqdm(pre_diff)]).reshape(shape[0], shape[1])
            np.save(fr"D:\PyFile\paper1\cache\uvz\reg_{var}{p}_same.npy", reg_z)
    print(f"{p}hPa层{var}相关系数完成。")


if __name__ == '__main__':
    # 数据读取
    ols = np.load(r"D:\PyFile\paper1\OLS35_detrended.npy")  # 读取缓存
    sen = np.load(r"D:\PyFile\paper1\SEN35_detrended.npy")  # 读取缓存
    M = 6  # 临界月
    # 多核计算
    if eval(input("是否进行相关系数计算(0/1):")):
        Ncpu = multiprocessing.cpu_count()
        data_pool = []
        for var in ['u', 'v', 'z', 'sst', 'precip']:
            if var == 'u' or var == 'v' or var == 'z':
                for p in [200, 500, 600, 700, 850]:
                    data_pool.append([var, p, ols, sen])
            else:
                data_pool.append([var, 0, ols, sen])
        p = multiprocessing.Pool()
        p.starmap(multi_core, data_pool)
        p.close()
        p.join()
        del data_pool

    # 绘图
    # ##地图要素设置
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(16, 9))  # 创建画布
    lev = 15
    select = 1
    p = [200, 500, 600, 700, 850]
    spec = gridspec.GridSpec(nrows=len(p), ncols=1)  # 设置子图比例
    col = -1
    alpha = 0.05
    for date in [0]:
        col += 1
        x = 0.92
        y = 1.04
        title_size = 8
        extent1 = [-67.5, 292.5, -30, 80]
        xticks1 = np.arange(extent1[0], extent1[1] + 1, 10)
        yticks1 = np.arange(extent1[2], extent1[3] + 1, 30)
        for p in [200, 500, 600, 700, 850]:
            u_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\u_same.nc")['u'].sel(p=p).transpose('lat', 'lon', 'year')
            u_corr_1 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_u{p}_same.npy")  # 读取缓存
            u_corr_2 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_u{p}_same.npy")  # 读取缓存
            v_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\v_same.nc")['v'].sel(p=p).transpose('lat', 'lon', 'year')
            v_corr_1 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_v{p}_same.npy")  # 读取缓存
            v_corr_2 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_v{p}_same.npy")  # 读取缓存
            z_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\z_same.nc")['z'].sel(p=p).transpose('lat', 'lon', 'year')
            z_corr_1 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_z{p}_same.npy")  # 读取缓存
            z_corr_2 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_z{p}_same.npy")  # 读取缓存
            if p == 200:
                if select == 1:
                    u_corr = u_corr_1
                    v_corr = v_corr_1
                    z_corr = z_corr_1
                    u显著性检验结果 = corr_test(ols, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(ols, v_corr, alpha=alpha)
                    z显著性检验结果 = corr_test(ols, z_corr, alpha=alpha)
                    pc = ols
                else:
                    u_corr = u_corr_2
                    v_corr = v_corr_2
                    z_corr = z_corr_2
                    u显著性检验结果 = corr_test(sen, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(sen, v_corr, alpha=alpha)
                    z显著性检验结果 = corr_test(sen, z_corr, alpha=alpha)
                    pc = sen
                # 计算TN波作用通量
                reg_z200 = np.load(fr"D:\PyFile\paper1\cache\uvz\reg_z200_same.npy")
                Geoc = xr.DataArray(z_diff.mean('year').data[np.newaxis, :, :],
                                    coords=[('level', [200]),
                                            ('lat', z_diff['lat'].data),
                                            ('lon', z_diff['lon'].data)])
                Uc = xr.DataArray(u_diff.mean('year').data[np.newaxis, :, :],
                                    coords=[('level', [200]),
                                            ('lat', u_diff['lat'].data),
                                            ('lon', u_diff['lon'].data)])
                Vc = xr.DataArray(v_diff.mean('year').data[np.newaxis, :, :],
                                    coords=[('level', [200]),
                                            ('lat', v_diff['lat'].data),
                                            ('lon', v_diff['lon'].data)])
                GEOa = xr.DataArray(reg_z200[np.newaxis, :, :],
                                    coords=[('level', [200]),
                                            ('lat', z_diff['lat'].data),
                                            ('lon', z_diff['lon'].data)])
                waf_x, waf_y, waf_streamf = TN_WAF_3D(Geoc, Uc, Vc, GEOa, return_streamf=True, u_threshold=0, filt=3)
                ax1 = fig.add_subplot(spec[0, col], projection=ccrs.PlateCarree(central_longitude=180+extent1[0]))
                waf, lon = add_cyclic_point(waf_streamf[0], coord=z_diff['lon'])
                streamf图层 = ax1.contourf(lon, z_diff['lat'], waf*10**-6, levels=[-2.5, -2, -1.5, -1, -0.5,-.25,.25, 0.5, 1, 1.5, 2, 2.5],
                                           cmap=cmaps.MPL_PuOr_r[11:106],
                                           extend='both',
                                           transform=ccrs.PlateCarree(central_longitude=0))
                # waf_x = np.where(waf_x**2 + waf_y**2>=0.05**2, waf_x, 0)
                # waf_y = np.where(waf_x**2 + waf_y**2>=0.05**2, waf_y, 0)
                WAF图层1 = ax1.quiver(z_diff['lon'][0:3], z_diff['lat'][0:3], waf_x[0:3, 0:3], waf_y[0:3, 0:3], scale=5, regrid_shape=30, transform=ccrs.PlateCarree(central_longitude=0))
                WAF图层 = velovect(ax1, z_diff['lon'], z_diff['lat'][:180],
                                  np.array(waf_x.tolist())[:180, :],
                                  np.array(waf_y.tolist())[:180, :], regrid=20, lon_trunc=-67.5,
                                  arrowstyle='fancy', arrowsize=.3, scale=5, grains=32, linewidth=0.75,
                                  color='gray', transform=ccrs.PlateCarree(central_longitude=0))

                '''WAF图层_ = curly_vector(ax1, z_diff['lon'], z_diff['lat'][:180], np.array(waf_x.tolist())[:180, :].T, np.array(waf_y.tolist())[:180, :].T,
                                      lon_trunc=-67.5, linewidth=0.5, arrowsize=3, scale=5, regrid=20, color='black', transform=ccrs.PlateCarree(central_longitude=0))'''
                ax1.quiverkey(WAF图层1, X=x-0.05, Y=y, U=0.25, angle=0, label='0.25 m$^2$/s$^2$',
                                  labelpos='E', color='green', fontproperties={'size': 5})  # linewidth=1为箭头的大小
                ax1.set_extent(extent1, crs=ccrs.PlateCarree(central_longitude=0))
                ax1.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.05)
                # 在赤道画一条纬线
                ax1.plot((extent1[0], extent1[1]), (0, 0), color='red', linewidth=1, linestyle=(0,(2, 1, 1, 1)),transform=ccrs.PlateCarree(central_longitude=0))
                ax1.add_geometries(Reader(r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").geometries(),
                                  ccrs.PlateCarree(central_longitude=0), facecolor='none',edgecolor='black',linewidth=.2)

                # 刻度线设置
                # ax1
                ax1.set_yticks(yticks1, crs=ccrs.PlateCarree())
                lon_formatter = LongitudeFormatter()
                lat_formatter = LatitudeFormatter()
                ax1.yaxis.set_major_formatter(lat_formatter)


                ymajorLocator = MultipleLocator(30)  # 先定义xmajorLocator，再进行调用
                ax1.yaxis.set_major_locator(ymajorLocator)  # x轴最大刻度
                yminorLocator = MultipleLocator(10)
                ax1.yaxis.set_minor_locator(yminorLocator)  # x轴最小刻度
                # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
                # 最大刻度、最小刻度的刻度线长短，粗细设置
                ax1.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
                ax1.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
                ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
                plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
                # 调整刻度值字体大小
                ax1.tick_params(axis='both', labelsize=title_size, colors='black')

                # 设置色标
                cbar = plt.colorbar(streamf图层, orientation='vertical', drawedges=True, ax=ax1)
                cbar.Location = 'eastoutside'
                cbar.locator = ticker.FixedLocator([-2.5, -2, -1.5, -1, -0.5, -.25, .25, 0.5, 1, 1.5, 2, 2.5])
                #cbar.ax.set_title('Proportion of EHT-Grids(%)', fontsize=5)
                cbar.ax.tick_params(length=0)  # 设置色标刻度长度
                cbar.ax.tick_params(labelsize=4)
                cbar.dividers.set_linewidth(.2)  # 设置分割线宽度
                cbar.outline.set_linewidth(.2)  # 设置色标轮廓宽度
            if p == 500:
                lev = [-.4, -.35, -.3, -.25, -.2, -.15, -.1, -.05, .05, .1, .15, .2, .25, .3, .35, .4]
                if select == 1:
                    u_corr = u_corr_1
                    v_corr = v_corr_1
                    z_corr = z_corr_1
                    u显著性检验结果 = corr_test(ols, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(ols, v_corr, alpha=alpha)
                    z显著性检验结果 = corr_test(ols, z_corr, alpha=alpha)
                    pc = ols
                else:
                    u_corr = u_corr_2
                    v_corr = v_corr_2
                    z_corr = z_corr_2
                    u显著性检验结果 = corr_test(sen, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(sen, v_corr, alpha=alpha)
                    z显著性检验结果 = corr_test(sen, z_corr, alpha=alpha)
                    pc = sen
                ax = fig.add_subplot(spec[1, col], projection=ccrs.PlateCarree(central_longitude=180+extent1[0]))
                ax.set_title('500hPa UVZ', fontsize=title_size, loc='left')
                z_corr, lon = add_cyclic_point(z_corr, coord=z_diff['lon'])
                相关系数图层 = ax.contourf(lon, z_diff['lat'], z_corr, levels=lev,
                                           cmap=cmaps.GMT_polar,
                                           extend='both',
                                           transform=ccrs.PlateCarree(central_longitude=0))
                显著性检验结果 = np.where(z显著性检验结果 == 1, 0, np.nan)
                显著性检验图层 = ax.quiver(z_diff['lon'], z_diff['lat'], 显著性检验结果, 显著性检验结果, scale=20,
                                           color='white', headlength=2, headaxislength=2, regrid_shape=60,
                                           transform=ccrs.PlateCarree(central_longitude=0))
                uv显著性检验结果 = np.where(np.where(u显著性检验结果 == 1, 1, 0) + np.where(v显著性检验结果 == 1, 1, 0) >= 1, 1, np.nan)
                u_np = np.where(uv显著性检验结果 != 1, u_corr, np.nan)
                v_np = np.where(uv显著性检验结果 != 1, v_corr, np.nan)
                u_np = np.where(u_np**2 + v_np**2 >= 0.15**2, u_np, np.nan)
                v_np = np.where(u_np**2 + v_np**2 >= 0.15**2, v_np, np.nan)
                u_corr = np.where(uv显著性检验结果 == 1, u_corr, np.nan)
                v_corr = np.where(uv显著性检验结果 == 1, v_corr, np.nan)
                uv = ax.quiver(u_diff['lon'][0:3], u_diff['lat'][0:3], u_corr[0:3, 0:3], v_corr[0:3, 0:3], scale=10, regrid_shape=40, transform=ccrs.PlateCarree(central_longitude=0))
                uv_np_ = velovect(ax, u_diff['lon'], u_diff['lat'],
                               np.array(np.where(np.isnan(u_np), 0, u_np).tolist()),
                               np.array(np.where(np.isnan(v_np), 0, v_np).tolist()),
                               arrowstyle='fancy', arrowsize=.3, scale=1.75, grains=32, linewidth=0.75,
                               color='gray', transform=ccrs.PlateCarree(central_longitude=0))
                uv_np_ = curly_vector(ax, u_diff['lon'], u_diff['lat'], u_np.T,  v_np.T,
                             lon_trunc=-67.5, linewidth=0.5, arrowsize=3, scale=5, regrid=20, color='red',
                             transform=ccrs.PlateCarree(central_longitude=0))
                '''uv_ = velovect(ax, u_diff['lon'], u_diff['lat'],
                               np.array(np.where(np.isnan(u_corr),0 , u_corr).tolist()),
                               np.array(np.where(np.isnan(v_corr),0 , v_corr).tolist()),
                               lon_trunc=-67.5, regrid=20, arrowstyle='fancy', arrowsize=.3, scale=1.73, grains=29,linewidth=0.75,
                               color='black', transform=ccrs.PlateCarree(central_longitude=0))'''
                ax.quiverkey(uv, X=x, Y=y, U=.5, angle=0, label='0.5', labelpos='E', fontproperties={'size': 5}, color='green')
                ax.set_extent(extent1, crs=ccrs.PlateCarree(central_longitude=0))
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.05)
                ax.plot((extent1[0], extent1[1]), (0, 0), color='red', linewidth=1, linestyle=(0,(2, 1, 1, 1)),transform=ccrs.PlateCarree(central_longitude=0))
                ax.add_geometries(Reader(r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").geometries(),
                                  ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.2)

                # 刻度线设置
                # ax1
                ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
                lon_formatter = LongitudeFormatter()
                lat_formatter = LatitudeFormatter()
                ax.yaxis.set_major_formatter(lat_formatter)


                ymajorLocator = MultipleLocator(30)  # 先定义xmajorLocator，再进行调用
                ax.yaxis.set_major_locator(ymajorLocator)  # x轴最大刻度
                yminorLocator = MultipleLocator(10)
                ax.yaxis.set_minor_locator(yminorLocator)  # x轴最小刻度
                # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
                # 最大刻度、最小刻度的刻度线长短，粗细设置
                ax.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
                ax.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
                ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
                plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
                # 调整刻度值字体大小
                ax.tick_params(axis='both', labelsize=title_size, colors='black')
                # 设置色标
                cbar = plt.colorbar(相关系数图层, orientation='vertical', drawedges=True, ax=ax)
                cbar.Location = 'eastoutside'
                cbar.locator = ticker.FixedLocator([-.4, -.3, -.2, -.1, .1, .2, .3, .4])
                #cbar.ax.set_title('Proportion of EHT-Grids(%)', fontsize=5)
                cbar.ax.tick_params(length=0)  # 设置色标刻度长度
                cbar.ax.tick_params(labelsize=4)
                cbar.dividers.set_linewidth(.2)  # 设置分割线宽度
                cbar.outline.set_linewidth(.2)  # 设置色标轮廓宽度
            if p == 700:
                lev = [-.4, -.35, -.3, -.25, -.2, -.15, -.1, -.05, .05, .1, .15, .2, .25, .3, .35, .4]
                level_z = [-.4, -.3, -.2, -.1, 0, .1, .2, .3, .4]
                if select == 1:
                    u_corr = u_corr_1
                    v_corr = v_corr_1
                    z_corr = z_corr_1
                    sst_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\sst\sst_same.nc")['sst'].transpose('lat', 'lon', 'year')
                    sst_corr = np.load(fr"D:\PyFile\paper1\cache\sst\corr_sst_same.npy")  # 读取缓存
                    u显著性检验结果 = corr_test(ols, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(ols, v_corr, alpha=alpha)
                    sst显著性检验结果 = corr_test(ols, sst_corr, alpha=alpha)
                    pc = ols
                else:
                    u_corr = u_corr_2
                    v_corr = v_corr_2
                    z_corr = z_corr_2
                    sst_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\sst\sst_same.nc")['sst'].sel(p=p).transpose('lat', 'lon', 'year')
                    sst_corr = np.load(fr"D:\PyFile\paper1\cache\sst\corr_sst_same.npy")  # 读取缓存
                    u显著性检验结果 = corr_test(sen, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(sen, v_corr, alpha=alpha)
                    sst显著性检验结果 = corr_test(sen, sst_corr, alpha=alpha)
                    pc = sen
                ax = fig.add_subplot(spec[2, col], projection=ccrs.PlateCarree(central_longitude=180+extent1[0]))
                ax.set_title('700hPa UVZ&SST', fontsize=title_size, loc='left')
                sst_corr, lon = add_cyclic_point(sst_corr, coord=sst_diff['lon'])
                sst相关系数图层 = ax.contourf(lon, sst_diff['lat'], sst_corr, levels=lev,
                                           cmap=cmaps.BlueWhiteOrangeRed,
                                           extend='both',
                                           transform=ccrs.PlateCarree(central_longitude=0))
                z_corr = filters.gaussian_filter(z_corr, 4)
                z_corr, lon = add_cyclic_point(z_corr, coord=z_diff['lon'])
                z相关系数图层_low = ax.contour(lon, z_diff['lat'], z_corr, cmap=cmaps.BlueDarkRed18[0], levels=level_z[:4],
                                     linewidths=.2, linestyles='--', alpha=1, transform=ccrs.PlateCarree(central_longitude=0))
                z相关系数图层_0 = ax.contour(lon, z_diff['lat'], z_corr, color='gray', levels=[0],
                                     linewidths=.2, linestyles='--', alpha=1, transform=ccrs.PlateCarree(central_longitude=0))
                z相关系数图层_high = ax.contour(lon, z_diff['lat'], z_corr, cmap=cmaps.BlueDarkRed18[17], levels=level_z[5:],
                                     linewidths=.2, linestyles='-', alpha=1, transform=ccrs.PlateCarree(central_longitude=0))
                plt.clabel(z相关系数图层_low, inline=True, fontsize=3, fmt='%.1f', inline_spacing=5)
                plt.clabel(z相关系数图层_0, inline=True, fontsize=3, fmt='%d', inline_spacing=5)
                plt.clabel(z相关系数图层_high, inline=True, fontsize=3, fmt='%.1f', inline_spacing=5)
                显著性检验结果 = np.where(sst显著性检验结果 == 1, 0, np.nan)
                显著性检验图层 = ax.quiver(sst_diff['lon'], sst_diff['lat'], 显著性检验结果, 显著性检验结果, scale=20,
                                           color='white', headlength=2, headaxislength=2, regrid_shape=60,
                                           transform=ccrs.PlateCarree(central_longitude=0))
                uv显著性检验结果 = np.where(np.where(u显著性检验结果 == 1, 1, 0) + np.where(v显著性检验结果 == 1, 1, 0) >= 1, 1, np.nan)
                u_np = np.where(uv显著性检验结果 != 1, u_corr, np.nan)
                v_np = np.where(uv显著性检验结果 != 1, v_corr, np.nan)
                u_np = np.where(u_np**2 + v_np**2 >= 0.15**2, u_np, np.nan)
                v_np = np.where(u_np**2 + v_np**2 >= 0.15**2, v_np, np.nan)
                u_corr = np.where(uv显著性检验结果 == 1, u_corr, np.nan)
                v_corr = np.where(uv显著性检验结果 == 1, v_corr, np.nan)
                uv = ax.quiver(u_diff['lon'][0:3], u_diff['lat'][0:3], u_corr[0:3, 0:3], v_corr[0:3, 0:3], scale=10, regrid_shape=40, transform=ccrs.PlateCarree(central_longitude=0))
                uv_np_ = velovect(ax, u_diff['lon'], u_diff['lat'],
                               np.array(np.where(np.isnan(u_np), 0, u_np).tolist()),
                               np.array(np.where(np.isnan(v_np), 0, v_np).tolist()),
                               arrowstyle='fancy', arrowsize=.3, scale=1.73, grains=31, linewidth=0.75,
                               color='gray', transform=ccrs.PlateCarree(central_longitude=0))
                uv_ = velovect(ax, u_diff['lon'], u_diff['lat'],
                               np.array(np.where(np.isnan(u_corr),0 , u_corr).tolist()),
                               np.array(np.where(np.isnan(v_corr),0 , v_corr).tolist()),
                               arrowstyle='fancy', arrowsize=.3, scale=1.73, grains=29,linewidth=0.75,
                               color='black', transform=ccrs.PlateCarree(central_longitude=0))
                ax.quiverkey(uv, X=x, Y=y, U=.5, angle=0, label='0.5', labelpos='E', fontproperties={'size': 5}, color='green')
                ax.set_extent(extent1, crs=ccrs.PlateCarree(central_longitude=0))
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.05)
                ax.plot((extent1[0], extent1[1]), (0, 0), color='red', linewidth=1, linestyle=(0,(2, 1, 1, 1)),transform=ccrs.PlateCarree(central_longitude=0))
                ax.add_geometries(Reader(r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").geometries(),
                                  ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.2)
                # 刻度线设置
                # ax1
                ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
                lon_formatter = LongitudeFormatter()
                lat_formatter = LatitudeFormatter()
                ax.yaxis.set_major_formatter(lat_formatter)


                ymajorLocator = MultipleLocator(30)  # 先定义xmajorLocator，再进行调用
                ax.yaxis.set_major_locator(ymajorLocator)  # x轴最大刻度
                yminorLocator = MultipleLocator(10)
                ax.yaxis.set_minor_locator(yminorLocator)  # x轴最小刻度
                # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
                # 最大刻度、最小刻度的刻度线长短，粗细设置
                ax.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
                ax.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
                ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
                plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
                # 调整刻度值字体大小
                ax.tick_params(axis='both', labelsize=title_size, colors='black')

                # 设置色标
                cbar = plt.colorbar(sst相关系数图层, orientation='vertical', drawedges=True, ax=ax)
                cbar.Location = 'eastoutside'
                cbar.locator = ticker.FixedLocator([-.4, -.3, -.2, -.1, .1, .2, .3, .4])
                #cbar.ax.set_title('Proportion of EHT-Grids(%)', fontsize=5)
                cbar.ax.tick_params(length=0)  # 设置色标刻度长度
                cbar.ax.tick_params(labelsize=4)
                cbar.dividers.set_linewidth(.2)  # 设置分割线宽度
                cbar.outline.set_linewidth(.2)  # 设置色标轮廓宽度
            if p == 850:
                lev = [-.4, -.35, -.3, -.25, -.2, -.15, -.1, .1, .15, .2, .25, .3, .35, .4]
                if select == 1:
                    u_corr = u_corr_1
                    v_corr = v_corr_1
                    pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\pre\pre_same.nc")['precip'].transpose('lat', 'lon', 'year')
                    pre_corr = np.load(fr"D:\PyFile\paper1\cache\pre\corr_pre_same.npy")  # 读取缓存
                    u显著性检验结果 = corr_test(ols, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(ols, v_corr, alpha=alpha)
                    pre显著性检验结果 = corr_test(ols, pre_corr, alpha=alpha)
                    pc = ols
                else:
                    u_corr = u_corr_2
                    v_corr = v_corr_2
                    pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\pre\pre_same.nc")['precip'].sel(p=p).transpose('lat', 'lon', 'year')
                    pre_corr = np.load(fr"D:\PyFile\paper1\cache\pre\corr_pre_same.npy")  # 读取缓存
                    u显著性检验结果 = corr_test(sen, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(sen, v_corr, alpha=alpha)
                    pre显著性检验结果 = corr_test(sen, pre_corr, alpha=alpha)
                    pc = sen
                ax = fig.add_subplot(spec[3, col], projection=ccrs.PlateCarree(central_longitude=180+extent1[0]))
                ax.set_title('850hPa UV&PRE', fontsize=title_size, loc='left')
                pre_corr, lon = add_cyclic_point(pre_corr, coord=pre_diff['lon'])
                pre相关系数图层 = ax.contourf(lon, pre_diff['lat'], pre_corr, levels=lev,
                                           cmap=cmaps.MPL_RdYlGn[32:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:96],
                                           extend='both',
                                           transform=ccrs.PlateCarree(central_longitude=0))
                显著性检验结果 = np.where(pre显著性检验结果 == 1, 0, np.nan)
                显著性检验图层 = ax.quiver(pre_diff['lon'], pre_diff['lat'], 显著性检验结果, 显著性检验结果, scale=20,
                                           color='white', headlength=2, headaxislength=2, regrid_shape=60,
                                           transform=ccrs.PlateCarree(central_longitude=0))
                uv显著性检验结果 = np.where(np.where(u显著性检验结果 == 1, 1, 0) + np.where(v显著性检验结果 == 1, 1, 0) >= 1, 1, np.nan)
                u_np = np.where(uv显著性检验结果 != 1, u_corr, np.nan)
                v_np = np.where(uv显著性检验结果 != 1, v_corr, np.nan)
                u_np = np.where(u_np**2 + v_np**2 >= 0.15**2, u_np, np.nan)
                v_np = np.where(u_np**2 + v_np**2 >= 0.15**2, v_np, np.nan)
                u_corr = np.where(uv显著性检验结果 == 1, u_corr, np.nan)
                v_corr = np.where(uv显著性检验结果 == 1, v_corr, np.nan)
                uv = ax.quiver(u_diff['lon'][0:3], u_diff['lat'][0:3], u_corr[0:3, 0:3], v_corr[0:3, 0:3], scale=10, regrid_shape=40, transform=ccrs.PlateCarree(central_longitude=0))
                uv_np_ = velovect(ax, u_diff['lon'], u_diff['lat'],
                               np.array(np.where(np.isnan(u_np), 0, u_np).tolist()),
                               np.array(np.where(np.isnan(v_np), 0, v_np).tolist()),
                               arrowstyle='fancy', arrowsize=.3, scale=1.73, grains=29, linewidth=0.75,
                               color='gray', transform=ccrs.PlateCarree(central_longitude=0))
                uv_ = velovect(ax, u_diff['lon'], u_diff['lat'],
                               np.array(np.where(np.isnan(u_corr),0 , u_corr).tolist()),
                               np.array(np.where(np.isnan(v_corr),0 , v_corr).tolist()),
                               arrowstyle='fancy', arrowsize=.3, scale=1.75, grains=32,linewidth=0.75,
                               color='black', transform=ccrs.PlateCarree(central_longitude=0))
                ax.quiverkey(uv, X=x, Y=y, U=.5, angle=0, label='0.5', labelpos='E', fontproperties={'size': 5}, color='green')
                ax.set_extent(extent1, crs=ccrs.PlateCarree(central_longitude=0))
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.05)
                ax.plot((extent1[0], extent1[1]), (0, 0), color='red', linewidth=1, linestyle=(0,(2, 1, 1, 1)),transform=ccrs.PlateCarree(central_longitude=0))
                ax.add_geometries(Reader(r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").geometries(),
                                  ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.2)
                # 刻度线设置
                # ax1
                ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
                lon_formatter = LongitudeFormatter()
                lat_formatter = LatitudeFormatter()
                ax.yaxis.set_major_formatter(lat_formatter)


                ymajorLocator = MultipleLocator(30)  # 先定义xmajorLocator，再进行调用
                ax.yaxis.set_major_locator(ymajorLocator)  # x轴最大刻度
                yminorLocator = MultipleLocator(10)
                ax.yaxis.set_minor_locator(yminorLocator)  # x轴最小刻度
                # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
                # 最大刻度、最小刻度的刻度线长短，粗细设置
                ax.tick_params(which='major', length=4, width=.5, color='black')  # 最大刻度长度，宽度设置，
                ax.tick_params(which='minor', length=2, width=.2, color='black')  # 最小刻度长度，宽度设置
                ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
                plt.rcParams['ytick.direction'] = 'out'  # 将x轴的刻度线方向设置向内或者外
                # 调整刻度值字体大小
                ax.tick_params(axis='both', labelsize=title_size, colors='black')

                # 设置色标
                cbar = plt.colorbar(pre相关系数图层, orientation='vertical', drawedges=True, ax=ax)
                cbar.Location = 'eastoutside'
                cbar.locator = ticker.FixedLocator([-.4, -.3, -.2, -.1, .1, .2, .3, .4])
                #cbar.ax.set_title('Proportion of EHT-Grids(%)', fontsize=5)
                cbar.ax.tick_params(length=0)  # 设置色标刻度长度
                cbar.ax.tick_params(labelsize=4)
                cbar.dividers.set_linewidth(.2)  # 设置分割线宽度
                cbar.outline.set_linewidth(.2)  # 设置色标轮廓宽度

    plt.savefig(fr"D:\PyFile\pic\uvz_corr_same.png", dpi=666, bbox_inches='tight')
    plt.show()
