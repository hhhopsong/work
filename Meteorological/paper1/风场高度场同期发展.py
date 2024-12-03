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
from matplotlib.pyplot import quiverkey
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import filters
from toolbar.significance_test import corr_test
from toolbar.TN_WaveActivityFlux import TN_WAF_3D
from toolbar.curved_quivers.modplot import velovect, velovect_key


# 多核计算部分函数
def multi_core(var, p, ols, sen):
    import numpy as np
    print(f"{p}hPa层{var}相关系数计算中...")
    if var == 'u' or var == 'v' or var == 'z':
        pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\{var}_same.nc")[var].sel(p=p).transpose('lat', 'lon', 'year')
    elif var == 'hu' or var == 'hv' or var == 'hz':
        pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\{var[1]}_same_high.nc")[var[1]].sel(p=p).transpose('lat', 'lon', 'year')
    elif var == 'sst':
        pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\sst\sst_same.nc")['sst'].transpose('lat', 'lon', 'year')
    elif var == 'precip':
        pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\pre\pre_same.nc")['precip'].transpose('lat', 'lon', 'year')
    elif var == 'olr':
        pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\olr\olr_same.nc")['olr'].transpose('lat', 'lon', 'year')
    shape = pre_diff.shape
    pre_diff = pre_diff.data if isinstance(pre_diff, xr.DataArray) else pre_diff
    pre_diff = pre_diff.reshape(shape[0] * shape[1], shape[2])
    corr_1 = np.array([np.corrcoef(d, ols)[0, 1] for d in tqdm.tqdm(pre_diff)]).reshape(shape[0], shape[1])
    if var == 'u' or var == 'v' or var == 'z':
        np.save(fr"D:\PyFile\paper1\cache\uvz\corr_{var}{p}_same.npy", corr_1)  # 保存缓存
    elif var == 'hu' or var == 'hv' or var == 'hz':
        np.save(fr"D:\PyFile\paper1\cache\uvz\corr_{var[1]}{p}_same.npy", corr_1)
    elif var == 'sst':
        np.save(fr"D:\PyFile\paper1\cache\sst\corr_sst_same.npy", corr_1)
    elif var == 'precip':
        np.save(fr"D:\PyFile\paper1\cache\pre\corr_pre_same.npy", corr_1)
    elif var == 'olr':
        np.save(fr"D:\PyFile\paper1\cache\olr\corr_olr_same.npy", corr_1)
    if var == 'z':
        reg_z = np.array([np.polyfit(ols, f, 1)[0] for f in tqdm.tqdm(pre_diff)]).reshape(shape[0], shape[1])
        np.save(fr"D:\PyFile\paper1\cache\uvz\reg_{var}{p}_same.npy", reg_z)
    elif var == 'hz':
        reg_z = np.array([np.polyfit(ols, f, 1)[0] for f in tqdm.tqdm(pre_diff)]).reshape(shape[0], shape[1])
        np.save(fr"D:\PyFile\paper1\cache\uvz\reg_{var[1]}{p}_same.npy", reg_z)
    r'''shape = pre_diff.shape
    pre_diff = pre_diff.data if isinstance(pre_diff, xr.DataArray) else pre_diff
    pre_diff = pre_diff.reshape(shape[0] * shape[1], shape[2])
    corr_2 = np.array([np.corrcoef(d, sen)[0, 1] for d in tqdm.tqdm(pre_diff)]).reshape(shape[0], shape[1])
    if var == 'u' or var == 'v' or var == 'z':
        np.save(fr"D:\PyFile\paper1\cache\uvz\corr2_{var}{p}_same.npy", corr_2)  # 保存缓存
    elif var == 'sst':
        np.save(fr"D:\PyFile\paper1\cache\sst\corr2_sst_same.npy", corr_2)
    elif var == 'precip':
        np.save(fr"D:\PyFile\paper1\cache\pre\corr2_pre_same.npy", corr_2)'''
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
        for var in ['u', 'v', 'z', 'hu', 'hv', 'hz', 'sst', 'precip', 'olr']:
            if var == 'u' or var == 'v' or var == 'z':
                for p in [200, 300, 400, 500, 600, 700, 850]:
                    data_pool.append([var, p, ols, sen])
            elif var == 'hu' or var == 'hv' or var == 'hz':
                for p in [10, 30, 50, 70, 100, 150]:
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
    p_all = [100, 150, 200, 300, 500, 600, 700, 850]
    spec = gridspec.GridSpec(nrows=5, ncols=1)  # 设置子图比例
    col = -1
    alpha = 0.05
    for date in [0]:
        col += 1
        x = 0.92
        y = 1.04
        title_size = 8
        extent1 = [-67.5, 292.5, -30, 80]
        xticks1 = np.arange(-180, 180, 10)
        yticks1 = np.arange(extent1[2], extent1[3] + 1, 30)
        for p in p_all:
            if p < 200:
                u_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\u_same_high.nc")['u'].sel(p=p).transpose('lat', 'lon', 'year')
                u_corr_1 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_u{p}_same.npy")  # 读取缓存
                v_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\v_same_high.nc")['v'].sel(p=p).transpose('lat', 'lon', 'year')
                v_corr_1 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_v{p}_same.npy")  # 读取缓存
                z_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\z_same_high.nc")['z'].sel(p=p).transpose('lat', 'lon', 'year')
                z_corr_1 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_z{p}_same.npy")  # 读取缓存
            else:
                u_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\u_same.nc")['u'].sel(p=p).transpose('lat', 'lon', 'year')
                u_corr_1 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_u{p}_same.npy")  # 读取缓存
                v_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\v_same.nc")['v'].sel(p=p).transpose('lat', 'lon', 'year')
                v_corr_1 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_v{p}_same.npy")  # 读取缓存
                z_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\z_same.nc")['z'].sel(p=p).transpose('lat', 'lon', 'year')
                z_corr_1 = np.load(fr"D:\PyFile\paper1\cache\uvz\corr_z{p}_same.npy")  # 读取缓存
            olr_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\olr\olr_same.nc")['olr'].transpose('lat', 'lon', 'year')
            olr_corr_1 = np.load(fr"D:\PyFile\paper1\cache\olr\corr_olr_same.npy")  # 读取缓存
            # 绘图
            if p == 200:
                if select == 1:
                    u_corr = u_corr_1
                    v_corr = v_corr_1
                    z_corr = z_corr_1
                    olr_corr = olr_corr_1
                    u显著性检验结果 = corr_test(ols, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(ols, v_corr, alpha=alpha)
                    z显著性检验结果 = corr_test(ols, z_corr, alpha=alpha)
                    olr显著性检验结果 = corr_test(ols, olr_corr, alpha=alpha)
                    pc = ols
                else:
                    pass
                    '''u_corr = u_corr_2
                    v_corr = v_corr_2
                    z_corr = z_corr_2
                    u显著性检验结果 = corr_test(sen, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(sen, v_corr, alpha=alpha)
                    z显著性检验结果 = corr_test(sen, z_corr, alpha=alpha)
                    pc = sen'''
                # 计算TN波作用通量
                reg_z150 = np.load(fr"D:\PyFile\paper1\cache\uvz\reg_z150_same.npy")
                reg_z200 = np.load(fr"D:\PyFile\paper1\cache\uvz\reg_z200_same.npy")
                reg_z300 = np.load(fr"D:\PyFile\paper1\cache\uvz\reg_z300_same.npy")
                reg_3d = np.array([reg_z150, reg_z200, reg_z300])
                z_diff_high = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\z_same_high.nc")['z'].sel(p=[150]).transpose('p', 'lat', 'lon', 'year')
                z_diff_low = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\z_same.nc")['z'].sel(p=[200, 300]).transpose('p', 'lat', 'lon', 'year')
                z_diff_3d = xr.concat([z_diff_high, z_diff_low], dim='p')
                u_diff_high = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\u_same_high.nc")['u'].sel(p=[150]).transpose('p', 'lat', 'lon', 'year')
                u_diff_low = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\u_same.nc")['u'].sel(p=[200, 300]).transpose('p', 'lat', 'lon', 'year')
                u_diff_3d = xr.concat([u_diff_high, u_diff_low], dim='p')
                v_diff_high = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\v_same_high.nc")['v'].sel(p=[150]).transpose('p', 'lat', 'lon', 'year')
                v_diff_low = xr.open_dataset(fr"D:\PyFile\paper1\cache\uvz\v_same.nc")['v'].sel(p=[200, 300]).transpose('p', 'lat', 'lon', 'year')
                v_diff_3d = xr.concat([v_diff_high, v_diff_low], dim='p')
                t_diff_high = xr.open_dataset(fr"D:\PyFile\paper1\cache\t\t_same_high.nc")['t'].sel(p=[150]).transpose('p', 'lat', 'lon', 'year')
                t_diff_low = xr.open_dataset(fr"D:\PyFile\paper1\cache\t\t_same.nc")['t'].sel(p=[200, 300]).transpose('p', 'lat', 'lon', 'year')
                t_diff_3d = xr.concat([t_diff_high, t_diff_low], dim='p')
                Geoc = xr.DataArray(z_diff_3d.mean('year').data,
                                    coords=[('level', [150, 200, 300]),
                                            ('lat', z_diff_3d['lat'].data),
                                            ('lon', z_diff_3d['lon'].data)])
                Uc = xr.DataArray(u_diff_3d.mean('year').data,
                                    coords=[('level', [150, 200, 300]),
                                            ('lat', u_diff_3d['lat'].data),
                                            ('lon', u_diff_3d['lon'].data)])
                Vc = xr.DataArray(v_diff_3d.mean('year').data,
                                    coords=[('level', [150, 200, 300]),
                                            ('lat', v_diff_3d['lat'].data),
                                            ('lon', v_diff_3d['lon'].data)])
                Tc = xr.DataArray(t_diff_3d.mean('year').data,
                                    coords=[('level', [150, 200, 300]),
                                            ('lat', t_diff_3d['lat'].data),
                                            ('lon', t_diff_3d['lon'].data)])
                GEOa = xr.DataArray(reg_3d,
                                    coords=[('level', [150, 200, 300]),
                                            ('lat', z_diff_3d['lat'].data),
                                            ('lon', z_diff_3d['lon'].data)])
                waf_x, waf_y, waf_z = TN_WAF_3D(Geoc, Uc, Vc, GEOa, Tc, u_threshold=0, filt=3)
                ax1 = fig.add_subplot(spec[1, col], projection=ccrs.PlateCarree(central_longitude=180+extent1[0]))
                ax1.set_title('200hPa UVZ&WAF&WAF_W', fontsize=title_size, loc='left')
                waf, lon = add_cyclic_point(waf_z[1], coord=z_diff['lon'])
                wafz图层 = ax1.contourf(lon, z_diff['lat'], waf,
                                           levels=np.array([-1., -.8, -.6, -.4, -.2, -.05, .05, .2, .4, .6, .8, 1.])*0.001,
                                           cmap=cmaps.MPL_PuOr_r[11+15:56]+ cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.MPL_PuOr_r[64:106-15],
                                           extend='both',
                                           transform=ccrs.PlateCarree(central_longitude=0))
                '''显著性检验结果 = np.where(olr显著性检验结果 == 1, 0, np.nan)
                显著性检验图层 = ax1.quiver(olr_diff['lon'], olr_diff['lat'], 显著性检验结果, 显著性检验结果, scale=20,
                                           color='white', headlength=2, headaxislength=2, regrid_shape=60,
                                           transform=ccrs.PlateCarree(central_longitude=0))'''
                # waf_x = np.where(waf_x**2 + waf_y**2>=0.05**2, waf_x, 0)
                # waf_y = np.where(waf_x**2 + waf_y**2>=0.05**2, waf_y, 0)
                uv显著性检验结果 = np.where(np.where(u显著性检验结果 == 1, 1, 0) + np.where(v显著性检验结果 == 1, 1, 0) >= 1, 1, np.nan)
                u_np = np.where(uv显著性检验结果 != 1, u_corr, np.nan)
                v_np = np.where(uv显著性检验结果 != 1, v_corr, np.nan)
                u_np = np.where(u_np**2 + v_np**2 >= 0.01**2, u_np, np.nan)
                v_np = np.where(u_np**2 + v_np**2 >= 0.01**2, v_np, np.nan)
                u_corr = np.where(uv显著性检验结果 == 1, u_corr, np.nan)
                v_corr = np.where(uv显著性检验结果 == 1, v_corr, np.nan)
                uv_p = velovect(ax1, u_diff['lon'], u_diff['lat'], u_corr, v_corr,
                                  lon_trunc=-67.5, arrowsize=.5, scale=5, linewidth=0.4, regrid=20,
                                  transform=ccrs.PlateCarree(central_longitude=0))
                velovect_key(fig, ax1, uv_p, U=.5, label='0.5 ', lr=-5.04)
                uv_np_ = velovect(ax1, u_diff['lon'], u_diff['lat'], u_np, v_np, color='gray', regrid=20,
                                  lon_trunc=-67.5, arrowsize=.5, scale=5, linewidth=0.4, transform=ccrs.PlateCarree(central_longitude=0))
                WAF图层 = velovect(ax1, z_diff['lon'], z_diff['lat'][:180],
                                  waf_x[1, :180, :], waf_y[1, :180, :],
                                  regrid=15, lon_trunc=-67.5, arrowsize=.3, scale=15, linewidth=0.4,
                                  color='blue', transform=ccrs.PlateCarree(central_longitude=0))
                velovect_key(fig, ax1, WAF图层, U=.1, label='0.1 m$^2$/s$^2$', lr=-6.04, color='blue')
                z_corr = filters.gaussian_filter(z_corr, 4)
                z_corr, lon = add_cyclic_point(z_corr, coord=z_diff['lon'])
                z_low = ax1.contour(lon, z_diff['lat'], z_corr, cmap=cmaps.BlueDarkRed18[0], levels=[-0.2, -0.1],
                                     linewidths=.5, linestyles='-', alpha=1, transform=ccrs.PlateCarree(central_longitude=0), zorder=1)
                z_high = ax1.contour(lon, z_diff['lat'], z_corr, cmap=cmaps.BlueDarkRed18[17], levels=[0.2, 0.4],
                                     linewidths=.5, linestyles='-', alpha=1, transform=ccrs.PlateCarree(central_longitude=0), zorder=1)
                plt.clabel(z_low, inline=True, fontsize=5, fmt='%.1f', inline_spacing=3)
                plt.clabel(z_high, inline=True, fontsize=5, fmt='%.1f', inline_spacing=3)
                ax1.set_extent(extent1, crs=ccrs.PlateCarree(central_longitude=0))
                ax1.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.2)
                # 在赤道画一条纬线
                ax1.plot((extent1[0], extent1[1]), (0, 0), color='red', linewidth=1, linestyle=(0,(2, 1, 1, 1)),transform=ccrs.PlateCarree(central_longitude=0))
                ax1.add_geometries(Reader(r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").geometries(),
                                  ccrs.PlateCarree(central_longitude=0), facecolor='none',edgecolor='black',linewidth=.4)

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
                cbar = plt.colorbar(wafz图层, orientation='vertical', drawedges=True, ax=ax1)
                cbar.Location = 'eastoutside'
                cbar.locator = ticker.FixedLocator(np.array([-1., -.8, -.6, -.4, -.2, -.05, .05, .2, .4, .6, .8, 1.])*0.001)
                cbar.set_ticklabels(['-1 ($×10^{-3}$)', '-0.8', '-0.6', '-0.4', '-0.2', '-0.05','0.05','0.2','0.4','0.6','0.8','1'])
                #cbar.ax.set_title('Proportion of EHT-Grids(%)', fontsize=5)
                cbar.ax.tick_params(length=0)  # 设置色标刻度长度
                cbar.ax.tick_params(labelsize=4)
                cbar.dividers.set_linewidth(.2)  # 设置分割线宽度
                cbar.outline.set_linewidth(.2)  # 设置色标轮廓宽度
            if p == 150:
                lev = [-.4, -.35, -.3, -.25, -.2, -.15, -.1, -.05, .05, .1, .15, .2, .25, .3, .35, .4]
                if select == 1:
                    u_corr = u_corr_1
                    v_corr = v_corr_1
                    z_corr = z_corr_1
                    olr_corr = olr_corr_1
                    olr显著性检验结果 = corr_test(ols, olr_corr, alpha=alpha)
                    u显著性检验结果 = corr_test(ols, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(ols, v_corr, alpha=alpha)
                    z显著性检验结果 = corr_test(ols, z_corr, alpha=alpha)
                    pc = ols
                else:
                    pass
                    '''u_corr = u_corr_2
                    v_corr = v_corr_2
                    z_corr = z_corr_2
                    u显著性检验结果 = corr_test(sen, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(sen, v_corr, alpha=alpha)
                    z显著性检验结果 = corr_test(sen, z_corr, alpha=alpha)
                    pc = sen'''
                # 计算TN波作用通量
                reg_z200 = np.load(fr"D:\PyFile\paper1\cache\uvz\reg_z200_same.npy")
                Geoc = xr.DataArray(z_diff.mean('year').data[np.newaxis, :, :],
                                        coords=[('level', [150]),
                                                ('lat', z_diff['lat'].data),
                                                ('lon', z_diff['lon'].data)])
                Uc = xr.DataArray(u_diff.mean('year').data[np.newaxis, :, :],
                                      coords=[('level', [150]),
                                              ('lat', u_diff['lat'].data),
                                              ('lon', u_diff['lon'].data)])
                Vc = xr.DataArray(v_diff.mean('year').data[np.newaxis, :, :],
                                      coords=[('level', [150]),
                                              ('lat', v_diff['lat'].data),
                                              ('lon', v_diff['lon'].data)])
                GEOa = xr.DataArray(reg_z200[np.newaxis, :, :],
                                        coords=[('level', [150]),
                                                ('lat', z_diff['lat'].data),
                                                ('lon', z_diff['lon'].data)])
                waf_x, waf_y, waf_streamf = TN_WAF_3D(Geoc, Uc, Vc, GEOa, return_streamf=True, u_threshold=0, filt=3)
                waf, lon = add_cyclic_point(waf_streamf[0], coord=z_diff['lon'])
                ax = fig.add_subplot(spec[0, col], projection=ccrs.PlateCarree(central_longitude=180+extent1[0]))
                ax.set_title('150hPa UV&OLR', fontsize=title_size, loc='left')
                z_corr, lon = add_cyclic_point(z_corr, coord=z_diff['lon'])
                显著性检验结果 = np.where(z显著性检验结果 == 1, 0, np.nan)
                显著性检验图层 = ax.quiver(z_diff['lon'], z_diff['lat'], 显著性检验结果, 显著性检验结果, scale=20,
                                           color='white', headlength=2, headaxislength=2, regrid_shape=60,
                                           transform=ccrs.PlateCarree(central_longitude=0))
                WAF图层 = velovect(ax, z_diff['lon'], z_diff['lat'][:180],
                                  waf_x[:180, :], waf_y[:180, :],
                                  regrid=15, lon_trunc=-67.5, arrowsize=.3, scale=30, linewidth=0.4,
                                  color='blue', transform=ccrs.PlateCarree(central_longitude=0))
                velovect_key(fig, ax, WAF图层, U=.1, label='0.1 m$^2$/s$^2$', lr=-6.04, color='blue')
                olr, lon = add_cyclic_point(olr_corr, coord=olr_diff['lon'])
                olr图层 = ax.contourf(lon, z_diff['lat'], olr,
                                           levels=[-.5, -.4, -.3, -.2, -0.1, -.05, .05, .1, .2, .3, .4, .5],
                                           cmap=cmaps.MPL_PuOr_r[11+15:56]+ cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.CBR_wet[0] + cmaps.MPL_PuOr_r[64:106-15],
                                           extend='both',
                                           transform=ccrs.PlateCarree(central_longitude=0))
                uv显著性检验结果 = np.where(np.where(u显著性检验结果 == 1, 1, 0) + np.where(v显著性检验结果 == 1, 1, 0) >= 1, 1, np.nan)
                u_np = np.where(uv显著性检验结果 != 1, u_corr, np.nan)
                v_np = np.where(uv显著性检验结果 != 1, v_corr, np.nan)
                u_np = np.where(u_np**2 + v_np**2 >= 0.01**2, u_np, np.nan)
                v_np = np.where(u_np**2 + v_np**2 >= 0.01**2, v_np, np.nan)
                u_corr = np.where(uv显著性检验结果 == 1, u_corr, np.nan)
                v_corr = np.where(uv显著性检验结果 == 1, v_corr, np.nan)
                uv_p = velovect(ax, u_diff['lon'], u_diff['lat'], u_corr, v_corr,
                                  lon_trunc=-67.5, arrowsize=.5, scale=5, linewidth=0.4, regrid=20,
                                  transform=ccrs.PlateCarree(central_longitude=0))
                uv_np_ = velovect(ax, u_diff['lon'], u_diff['lat'], u_np, v_np, color='gray', regrid=20,
                                  lon_trunc=-67.5, arrowsize=.5, scale=5, linewidth=0.4, transform=ccrs.PlateCarree(central_longitude=0))
                velovect_key(fig, ax, uv_np_, U=.25, label='0.25', lr=-5.04)
                ax.set_extent(extent1, crs=ccrs.PlateCarree(central_longitude=0))
                ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.2)
                ax.plot((extent1[0], extent1[1]), (0, 0), color='red', linewidth=1, linestyle=(0,(2, 1, 1, 1)),transform=ccrs.PlateCarree(central_longitude=0))
                ax.add_geometries(Reader(r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").geometries(),
                                  ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.4)

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
                cbar = plt.colorbar(olr图层, orientation='vertical', drawedges=True, ax=ax)
                cbar.Location = 'eastoutside'
                cbar.locator = ticker.FixedLocator([-.5, -.4, -.3, -.2, -0.1, .1, .2, .3, .4, .5])
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
                    pass
                    '''u_corr = u_corr_2
                    v_corr = v_corr_2
                    z_corr = z_corr_2
                    u显著性检验结果 = corr_test(sen, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(sen, v_corr, alpha=alpha)
                    z显著性检验结果 = corr_test(sen, z_corr, alpha=alpha)
                    pc = sen'''
                ax = fig.add_subplot(spec[2, col], projection=ccrs.PlateCarree(central_longitude=180+extent1[0]))
                ax.set_title('500hPa UVZ', fontsize=title_size, loc='left')
                z_corr, lon = add_cyclic_point(z_corr, coord=z_diff['lon'])
                相关系数图层 = ax.contourf(lon, z_diff['lat'], z_corr, levels=lev,
                                           cmap=cmaps.GMT_polar[4:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-4],
                                           extend='both',
                                           transform=ccrs.PlateCarree(central_longitude=0))
                显著性检验结果 = np.where(z显著性检验结果 == 1, 0, np.nan)
                显著性检验图层 = ax.quiver(z_diff['lon'], z_diff['lat'], 显著性检验结果, 显著性检验结果, scale=20,
                                           color='white', headlength=2, headaxislength=2, regrid_shape=60,
                                           transform=ccrs.PlateCarree(central_longitude=0))
                uv显著性检验结果 = np.where(np.where(u显著性检验结果 == 1, 1, 0) + np.where(v显著性检验结果 == 1, 1, 0) >= 1, 1, np.nan)
                u_np = np.where(uv显著性检验结果 != 1, u_corr, np.nan)
                v_np = np.where(uv显著性检验结果 != 1, v_corr, np.nan)
                u_np = np.where(u_np**2 + v_np**2 >= 0.01**2, u_np, np.nan)
                v_np = np.where(u_np**2 + v_np**2 >= 0.01**2, v_np, np.nan)
                u_corr = np.where(uv显著性检验结果 == 1, u_corr, np.nan)
                v_corr = np.where(uv显著性检验结果 == 1, v_corr, np.nan)
                uv_p = velovect(ax, u_diff['lon'], u_diff['lat'], u_corr, v_corr,
                                  lon_trunc=-67.5, arrowsize=.5, scale=5, linewidth=0.4, regrid=20,
                                  transform=ccrs.PlateCarree(central_longitude=0))
                uv_np_ = velovect(ax, u_diff['lon'], u_diff['lat'], u_np, v_np, color='gray', regrid=20,
                                  lon_trunc=-67.5, arrowsize=.5, scale=5, linewidth=0.4, transform=ccrs.PlateCarree(central_longitude=0))
                velovect_key(fig, ax, uv_np_, U=.25, label='0.25', lr=-6.04)
                ax.set_extent(extent1, crs=ccrs.PlateCarree(central_longitude=0))
                ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.2)
                ax.plot((extent1[0], extent1[1]), (0, 0), color='red', linewidth=1, linestyle=(0,(2, 1, 1, 1)),transform=ccrs.PlateCarree(central_longitude=0))
                ax.add_geometries(Reader(r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").geometries(),
                                  ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.4)

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
                    pass
                    '''u_corr = u_corr_2
                    v_corr = v_corr_2
                    z_corr = z_corr_2
                    sst_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\sst\sst_same.nc")['sst'].sel(p=p).transpose('lat', 'lon', 'year')
                    sst_corr = np.load(fr"D:\PyFile\paper1\cache\sst\corr_sst_same.npy")  # 读取缓存
                    u显著性检验结果 = corr_test(sen, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(sen, v_corr, alpha=alpha)
                    sst显著性检验结果 = corr_test(sen, sst_corr, alpha=alpha)
                    pc = sen'''
                ax = fig.add_subplot(spec[3, col], projection=ccrs.PlateCarree(central_longitude=180+extent1[0]))
                ax.set_title('700hPa UVZ&SST', fontsize=title_size, loc='left')
                sst_corr, lon = add_cyclic_point(sst_corr, coord=sst_diff['lon'])
                sst相关系数图层 = ax.contourf(lon, sst_diff['lat'], sst_corr, levels=lev,
                                           cmap=cmaps.BlueWhiteOrangeRed[40:-40],
                                           extend='both',
                                           transform=ccrs.PlateCarree(central_longitude=0))
                z_corr = filters.gaussian_filter(z_corr, 4)
                z_corr, lon = add_cyclic_point(z_corr, coord=z_diff['lon'])
                z相关系数图层_low = ax.contour(lon, z_diff['lat'], z_corr, cmap=cmaps.BlueDarkRed18[0], levels=level_z[:4:2],
                                     linewidths=.5, linestyles='-', alpha=1, transform=ccrs.PlateCarree(central_longitude=0), zorder=1)
                z相关系数图层_high = ax.contour(lon, z_diff['lat'], z_corr, cmap=cmaps.BlueDarkRed18[17], levels=level_z[5::2],
                                     linewidths=.5, linestyles='-', alpha=1, transform=ccrs.PlateCarree(central_longitude=0), zorder=1)
                plt.clabel(z相关系数图层_low, inline=True, fontsize=3, fmt='%.1f', inline_spacing=5)
                plt.clabel(z相关系数图层_high, inline=True, fontsize=3, fmt='%.1f', inline_spacing=5)
                显著性检验结果 = np.where(sst显著性检验结果 == 1, 0, np.nan)
                显著性检验图层 = ax.quiver(sst_diff['lon'], sst_diff['lat'], 显著性检验结果, 显著性检验结果, scale=20,
                                           color='white', headlength=2, headaxislength=2, regrid_shape=60,
                                           transform=ccrs.PlateCarree(central_longitude=0))
                uv显著性检验结果 = np.where(np.where(u显著性检验结果 == 1, 1, 0) + np.where(v显著性检验结果 == 1, 1, 0) >= 1, 1, np.nan)
                u_np = np.where(uv显著性检验结果 != 1, u_corr, np.nan)
                v_np = np.where(uv显著性检验结果 != 1, v_corr, np.nan)
                u_np = np.where(u_np**2 + v_np**2 >= 0.01**2, u_np, np.nan)
                v_np = np.where(u_np**2 + v_np**2 >= 0.01**2, v_np, np.nan)
                u_corr = np.where(uv显著性检验结果 == 1, u_corr, np.nan)
                v_corr = np.where(uv显著性检验结果 == 1, v_corr, np.nan)
                uv_np_ = velovect(ax, u_diff['lon'], u_diff['lat'], u_np, v_np,
                               arrowsize=.5, scale=5,lon_trunc=-67.5, linewidth=0.4, regrid=20,
                               color='gray', transform=ccrs.PlateCarree(central_longitude=0))
                uv_ = velovect(ax, u_diff['lon'], u_diff['lat'] ,u_corr, v_corr,
                               arrowsize=.5, scale=5,lon_trunc=-67.5, linewidth=0.4, regrid=20,
                               color='black', transform=ccrs.PlateCarree(central_longitude=0))
                velovect_key(fig, ax, uv_np_, U=.25, label='0.25', lr=-6.04)
                ax.set_extent(extent1, crs=ccrs.PlateCarree(central_longitude=0))
                ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.2)
                ax.plot((extent1[0], extent1[1]), (0, 0), color='red', linewidth=1, linestyle=(0,(2, 1, 1, 1)),transform=ccrs.PlateCarree(central_longitude=0))
                ax.add_geometries(Reader(r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").geometries(),
                                  ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.4)
                ax.add_geometries(Reader(
                    r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary_2500m\TPBoundary_2500m.shp').geometries(),
                                   ccrs.PlateCarree(), facecolor='gray', edgecolor='gray', linewidth=.1, hatch='.', zorder=2)
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
                    pass
                    '''u_corr = u_corr_2
                    v_corr = v_corr_2
                    pre_diff = xr.open_dataset(fr"D:\PyFile\paper1\cache\pre\pre_same.nc")['precip'].sel(p=p).transpose('lat', 'lon', 'year')
                    pre_corr = np.load(fr"D:\PyFile\paper1\cache\pre\corr_pre_same.npy")  # 读取缓存
                    u显著性检验结果 = corr_test(sen, u_corr, alpha=alpha)
                    v显著性检验结果 = corr_test(sen, v_corr, alpha=alpha)
                    pre显著性检验结果 = corr_test(sen, pre_corr, alpha=alpha)
                    pc = sen'''
                ax = fig.add_subplot(spec[4, col], projection=ccrs.PlateCarree(central_longitude=180+extent1[0]))
                ax.set_title('850hPa UV&PRE', fontsize=title_size, loc='left')
                pre_corr, lon = add_cyclic_point(pre_corr, coord=pre_diff['lon'])
                pre相关系数图层 = ax.contourf(lon, pre_diff['lat'], pre_corr, levels=lev,
                                           cmap=cmaps.MPL_RdYlGn[32+10:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:96-10],
                                           extend='both',
                                           transform=ccrs.PlateCarree(central_longitude=0))
                显著性检验结果 = np.where(pre显著性检验结果 == 1, 0, np.nan)
                显著性检验图层 = ax.quiver(pre_diff['lon'], pre_diff['lat'], 显著性检验结果, 显著性检验结果, scale=20,
                                           color='white', headlength=2, headaxislength=2, regrid_shape=60,
                                           transform=ccrs.PlateCarree(central_longitude=0))
                uv显著性检验结果 = np.where(np.where(u显著性检验结果 == 1, 1, 0) + np.where(v显著性检验结果 == 1, 1, 0) >= 1, 1, np.nan)
                u_np = np.where(uv显著性检验结果 != 1, u_corr, np.nan)
                v_np = np.where(uv显著性检验结果 != 1, v_corr, np.nan)
                u_np = np.where(u_np**2 + v_np**2 >= 0.01**2, u_np, np.nan)
                v_np = np.where(u_np**2 + v_np**2 >= 0.01**2, v_np, np.nan)
                u_corr = np.where(uv显著性检验结果 == 1, u_corr, np.nan)
                v_corr = np.where(uv显著性检验结果 == 1, v_corr, np.nan)
                uv_np_ = velovect(ax, u_diff['lon'], u_diff['lat'],
                               np.array(np.where(np.isnan(u_np), 0, u_np).tolist()),
                               np.array(np.where(np.isnan(v_np), 0, v_np).tolist()),
                               arrowsize=.5, scale=5,lon_trunc=-67.5, linewidth=0.4, regrid=20,
                               color='gray', transform=ccrs.PlateCarree(central_longitude=0))
                uv_ = velovect(ax, u_diff['lon'], u_diff['lat'],
                               np.array(np.where(np.isnan(u_corr),0 , u_corr).tolist()),
                               np.array(np.where(np.isnan(v_corr),0 , v_corr).tolist()),
                               arrowsize=.5, scale=5, lon_trunc=-67.5, linewidth=0.4, regrid=20,
                               color='black', transform=ccrs.PlateCarree(central_longitude=0))
                velovect_key(fig, ax, uv_, U=.5, label='0.5', lr=-6.04)
                ax.set_extent(extent1, crs=ccrs.PlateCarree(central_longitude=0))
                ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.2)
                ax.plot((extent1[0], extent1[1]), (0, 0), color='red', linewidth=1, linestyle=(0,(2, 1, 1, 1)),transform=ccrs.PlateCarree(central_longitude=0))
                ax.add_geometries(Reader(r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").geometries(),
                                  ccrs.PlateCarree(central_longitude=0), facecolor='none', edgecolor='black', linewidth=.4)
                ax.add_geometries(Reader(
                    r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary_2500m\TPBoundary_2500m.shp').geometries(),
                                  ccrs.PlateCarree(), facecolor='gray', edgecolor='gray', linewidth=.1, hatch='.',
                                  zorder=2)

                # 刻度线设置
                ax.set_xticks(xticks1, crs=ccrs.PlateCarree())
                ax.set_yticks(yticks1, crs=ccrs.PlateCarree())
                lat_formatter = LatitudeFormatter()
                ax.yaxis.set_major_formatter(lat_formatter)
                # 起始经度设为0
                lon_formatter = LongitudeFormatter()
                ax.xaxis.set_major_formatter(lon_formatter)

                xmajorLocator = MultipleLocator(60)  # 先定义xmajorLocator，再进行调用
                ax.xaxis.set_major_locator(xmajorLocator)  # x轴最大刻度
                xminorLocator = MultipleLocator(10)
                ax.xaxis.set_minor_locator(xminorLocator)  # x轴最小刻度
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

    plt.savefig(fr"D:\PyFile\pic\uvz_corr_same.png", dpi=2000, bbox_inches='tight')
    plt.show()
