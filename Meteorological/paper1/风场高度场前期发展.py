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
import cmaps
from toolbar.masked import masked  # 气象工具函数
from toolbar.sub_adjust import adjust_sub_axes
from toolbar.pre_whitening import ws2001
from toolbar.significance_test import corr_test
from toolbar.TN_WaveActivityFlux import TN_WAF
import seaborn as sns
import tqdm
import multiprocessing


# 多核计算部分函数
def multi_core(num, m1, m2, var, p, ols, sen):
    print(f"{p}hPa层{var}相关系数计算中...")
    pre_diff = xr.open_dataset(fr"cache\uvz\{var}\diff\{var}_{num}_{m1}_{m2}.nc")[var].sel(p=p).transpose('lat', 'lon', 'time')  # 读取缓存
    try:
        corr_1 = np.load(fr"cache\uvz\{var}\corr1\corr_{p}_{num}_{m1}_{m2}.npy")  # 读取缓存
    except:
        corr_1 = np.array([[np.corrcoef(ols, pre_diff.sel(lat=ilat, lon=ilon))[0, 1] for ilon in pre_diff['lon']] for ilat in pre_diff['lat']])
        np.save(fr"cache\uvz\{var}\corr1\corr_{p}_{num}_{m1}_{m2}.npy", corr_1)  # 保存缓存
    try:
        corr_2 = np.load(fr"cache\uvz\{var}\corr2\corr_{p}_{num}_{m1}_{m2}.npy")  # 读取缓存
    except:
        corr_2 = np.array([[np.corrcoef(sen, pre_diff.sel(lat=ilat, lon=ilon))[0, 1] for ilon in pre_diff['lon']] for ilat in pre_diff['lat']])
        np.save(fr"cache\uvz\{var}\corr2\corr_{p}_{num}_{m1}_{m2}.npy", corr_2)
    if var == 'z' and p == 200:
        try:
            reg_z = np.load(fr"cache\uvz\{var}\reg\{var}_{num}_{m1}_{m2}.npy")
        except:
            reg_z = np.array([[np.polyfit(ols, pre_diff.sel(lon=ilon, lat=ilat), 1)[0] for ilon in pre_diff['lon']] for ilat in tqdm.tqdm((pre_diff['lat']), desc=f'计算reg {m1} {p}{var}', position=0, leave=True)])
            np.save(fr"cache\uvz\{var}\reg\{var}_{num}_{m1}_{m2}.npy", reg_z)
    print(f"{p}hPa层{var}相关系数完成。")


if __name__ == '__main__':
    # 数据读取
    ols = np.load(r"cache\OLS_detrended.npy")  # 读取缓存
    sen = np.load(r"cache\SEN_detrended.npy")  # 读取缓存
    M = 6  # 临界月
    # 多核计算
    if eval(input("是否进行相关系数计算(0/1):")):
        Ncpu = multiprocessing.cpu_count()
        data_pool = []
        for p in [200, 500, 600, 700, 850]:
            for var in ['u', 'v', 'z']:
                num = 0
                for x in range(11, -1, -1):
                    m1 = M + x + 1
                    if m1 > 12:
                        m1 -= 12
                    for y in range(11 - x, 12):
                        num += 1
                        m2 = M - y
                        if m2 <= 0:
                            m2 += 12
                        if m1 == m2:
                            data_pool.append([num, m1, m2, var, p, ols, sen])

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
    select = eval(input("选择回归方案(1 OLS 2 SEN):"))
    num = 0
    p = [200, 500, 600, 700, 850]
    time = [6] ## 选择时间
    spec = gridspec.GridSpec(nrows=len(p), ncols=len(time))  # 设置子图比例
    date_pool = []
    for date in time:
        for x in range(11, -1, -1):
            m1 = M + x + 1
            if m1 > 12:
                m1 -= 12
            for y in range(11 - x, 12):
                num += 1
                m2 = M - y
                if m2 <= 0:
                    m2 += 12
                if m1 == m2:
                    if m1 == date:
                        date_pool.append([num, m1, m2]) # 保存文件名称索引
                        break
    col = -1
    for date in date_pool:
        num, m1, m2 = date
        col += 1
        for p in [200, 500, 600, 700, 850]:
            u_diff = xr.open_dataset(fr"cache\uvz\u\diff\u_{num}_{m1}_{m2}.nc")['u'].sel(p=p).transpose('lat', 'lon', 'time')
            u_corr_1 = np.load(fr"cache\uvz\u\corr1\corr_{p}_{num}_{m1}_{m2}.npy")  # 读取缓存
            u_corr_2 = np.load(fr"cache\uvz\u\corr2\corr_{p}_{num}_{m1}_{m2}.npy")  # 读取缓存
            v_diff = xr.open_dataset(fr"cache\uvz\v\diff\v_{num}_{m1}_{m2}.nc")['v'].sel(p=p).transpose('lat', 'lon', 'time')
            v_corr_1 = np.load(fr"cache\uvz\v\corr1\corr_{p}_{num}_{m1}_{m2}.npy")  # 读取缓存
            v_corr_2 = np.load(fr"cache\uvz\v\corr2\corr_{p}_{num}_{m1}_{m2}.npy")  # 读取缓存
            z_diff = xr.open_dataset(fr"cache\uvz\z\diff\z_{num}_{m1}_{m2}.nc")['z'].sel(p=p).transpose('lat', 'lon', 'time')
            z_corr_1 = np.load(fr"cache\uvz\z\corr1\corr_{p}_{num}_{m1}_{m2}.npy")  # 读取缓存
            z_corr_2 = np.load(fr"cache\uvz\z\corr2\corr_{p}_{num}_{m1}_{m2}.npy")  # 读取缓存
            if p == 200:
                if select == 1:
                    u_corr = u_corr_1
                    v_corr = v_corr_1
                    z_corr = z_corr_1
                    u显著性检验结果 = corr_test(ols, u_corr, alpha=0.05)
                    v显著性检验结果 = corr_test(ols, v_corr, alpha=0.05)
                    z显著性检验结果 = corr_test(ols, z_corr, alpha=0.05)
                    pc = ols
                else:
                    u_corr = u_corr_2
                    v_corr = v_corr_2
                    z_corr = z_corr_2
                    u显著性检验结果 = corr_test(sen, u_corr, alpha=0.05)
                    v显著性检验结果 = corr_test(sen, v_corr, alpha=0.05)
                    z显著性检验结果 = corr_test(sen, z_corr, alpha=0.05)
                    pc = sen
                # 计算TN波作用通量
                reg_z200 = np.load(fr"cache\uvz\z\reg\z_{num}_{m1}_{m2}.npy")
                waf = np.array(TN_WAF(z_diff.mean('time'), u_diff.mean('time'), v_diff.mean('time'), reg_z200+z_diff.mean('time'), z_diff['lon'], z_diff['lat']))
                ax = fig.add_subplot(spec[0, col], projection=ccrs.PlateCarree(central_longitude=180))
                相关系数图层 = ax.contourf(z_diff['lon'], z_diff['lat'], z_corr, levels=lev,
                                           cmap=cmaps.MPL_RdYlGn[32:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn[72:96],
                                           extend='both',
                                           transform=ccrs.PlateCarree())
                显著性检验结果 = np.where(z显著性检验结果 == 1, 0, np.nan)
                显著性检验图层 = ax.quiver(z_diff['lon'], z_diff['lat'], 显著性检验结果, 显著性检验结果, scale=20,
                                           color='black', headlength=2, headaxislength=2, regrid_shape=60,
                                           transform=ccrs.PlateCarree(central_longitude=0))
                WAF图层 = ax.quiver(z_diff['lon'], z_diff['lat'], waf[0], waf[1], scale=1000,
                                           color='black', headlength=2, headaxislength=2, regrid_shape=60,
                                           transform=ccrs.PlateCarree(central_longitude=0))
                ax.quiverkey(WAF图层, X=0.946, Y=1.03, U=25, angle=0, label='25 m$^2$/s$^2$',
                              labelpos='N', color='black', labelcolor='k',
                              linewidth=0.8)  # linewidth=1为箭头的大小
                ax.set_extent([-180, 180, -30, 80], crs=ccrs.PlateCarree(central_longitude=180))
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.05)
                draw_maps(get_adm_maps(level='国'), linewidth=0.15)

    plt.savefig(fr"C:\Users\10574\Desktop\pic\uvz_corr{select}.png", dpi=2000, bbox_inches='tight')
    plt.show()
