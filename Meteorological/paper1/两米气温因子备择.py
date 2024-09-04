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
import seaborn as sns
import tqdm
import multiprocessing


# 多核计算部分函数
def multi_core(num, m1, m2, ols, sen):
    print(f"第{num}个相关系数计算中...")
    pre_diff = xr.open_dataset(fr"cache\2mT\diff\2mT_{num}_{m1}_{m2}.nc")['precip'].transpose('lat', 'lon', 'time')  # 读取缓存
    try:
        corr_1 = np.load(fr"cache\2mT\corr1\corr_{num}_{m1}_{m2}.npy")  # 读取缓存
    except:
        corr_1 = np.array([[np.corrcoef(ols, pre_diff.sel(lat=ilat, lon=ilon))[0, 1] for ilon in pre_diff['lon']] for ilat in pre_diff['lat']])
        np.save(fr"cache\2mT\corr1\corr_{num}_{m1}_{m2}.npy", corr_1)  # 保存缓存
    try:
        corr_2 = np.load(fr"cache\2mT\corr2\corr_{num}_{m1}_{m2}.npy")  # 读取缓存
    except:
        corr_2 = np.array([[np.corrcoef(sen, pre_diff.sel(lat=ilat, lon=ilon))[0, 1] for ilon in pre_diff['lon']] for ilat in pre_diff['lat']])
        np.save(fr"cache\2mT\corr2\corr_{num}_{m1}_{m2}.npy", corr_2)
    print(f"第{num}个相关系数完成。")


if __name__ == '__main__':
    # 数据读取
    ols = np.load(r"cache\OLS_detrended.npy")  # 读取缓存
    sen = np.load(r"cache\SEN_detrended.npy")  # 读取缓存
    num = 0
    M = 6  # 临界月
    Ncpu = multiprocessing.cpu_count()
    data_pool = []
    # 多核计算
    for x in range(11, -1, -1):
        m1 = M + x + 1
        if m1 > 12:
            m1 -= 12
        for y in range(11 - x, 12):
            num += 1
            m2 = M - y
            if m2 <= 0:
                m2 += 12
            data_pool.append([num, m1, m2, ols, sen])

    p = multiprocessing.Pool()
    p.starmap(multi_core, data_pool)
    p.close()
    p.join()

    # 绘图
    # ##地图要素设置
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(16, 9))  # 创建画布
    spec = gridspec.GridSpec(nrows=12, ncols=12)  # 设置子图比例
    lev = [-.8, -.6, -.4, -.2, .2, .4, .6, .8]
    select = eval(input("选择回归方案(1 OLS 2 SEN):"))
    num = 0
    draw_pool = []
    for x in range(11, -1, -1):
        m1 = M + x + 1
        if m1 > 12:
            m1 -= 12
        for y in tqdm.trange(11 - x, 12):
            num += 1
            m2 = M - y
            if m2 <= 0:
                m2 += 12
            pre_diff = xr.open_dataset(fr"cache\2mT\diff\2mT_{num}_{m1}_{m2}.nc")['t2m'].transpose('lat', 'lon', 'time')
            corr_1 = np.load(fr"cache\2mT\corr1\corr_{num}_{m1}_{m2}.npy")  # 读取缓存
            corr_2 = np.load(fr"cache\2mT\corr2\corr_{num}_{m1}_{m2}.npy")  # 读取缓存
            if select == 1:
                corr = corr_1
                显著性检验结果 = corr_test(ols, corr, alpha=0.05)
            elif select == 2:
                corr = corr_2
                显著性检验结果 = corr_test(sen, corr, alpha=0.05)
            ax = fig.add_subplot(spec[y, x], projection=ccrs.PlateCarree(central_longitude=180))
            相关系数图层 = ax.contourf(pre_diff['lon'], pre_diff['lat'], corr, levels=lev,
                                       cmap=cmaps.WhiteBlueGreenYellowRed,
                                       extend='both',
                                       transform=ccrs.PlateCarree())
            显著性检验结果 = np.where(显著性检验结果 == 1, 0, np.nan)
            显著性检验图层 = ax.quiver(pre_diff['lon'], pre_diff['lat'], 显著性检验结果, 显著性检验结果, scale=20,
                                       color='black', headlength=2, headaxislength=2, regrid_shape=60,
                                       transform=ccrs.PlateCarree(central_longitude=0))
            ax.set_extent([-180, 180, -30, 80], crs=ccrs.PlateCarree(central_longitude=180))
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.05)
            draw_maps(get_adm_maps(level='国'), linewidth=0.15)

    plt.savefig(fr"C:\Users\10574\Desktop\pic\t2m_corr{select}.png", dpi=2000, bbox_inches='tight')
    plt.show()
