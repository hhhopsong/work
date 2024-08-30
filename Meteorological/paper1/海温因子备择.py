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
import seaborn as sns
import tqdm


# 数据读取
ols = np.load(r"cache\OLS_detrended.npy")  # 读取缓存
sen = np.load(r"cache\SEN_detrended.npy")  # 读取缓存
# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))  # 创建画布
spec = gridspec.GridSpec(nrows=12, ncols=12)  # 设置子图比例
num = 0
M = 6  # 临界月
lev = [i*.05 for i in range(-10, 11, 1)]
for x in range(11, -1, -1):
    m1 = M + x + 1
    if m1 > 12:
        m1 -= 12
    for y in tqdm.trange(11 - x, 12):
        num += 1
        m2 = M - y
        if m2 <= 0:
            m2 += 12
        sst_diff = xr.open_dataset(fr"cache\sst_diff\sst_{num}_{m1}_{m2}.nc")['sst'].transpose('lat','lon','time')  # 读取缓存
        try:
            corr_1 = np.load(fr"cache\corr_sst_1\corr_{num}_{m1}_{m2}.npy")  # 读取缓存
        except:
            corr_1 = np.array([[np.corrcoef(ols, sst_diff.sel(lat=ilat, lon=ilon))[0, 1] for ilon in sst_diff['lon']] for ilat in sst_diff['lat']])
            np.save(fr"cache\corr_sst_1\corr_{num}_{m1}_{m2}.npy", corr_1)  # 保存缓存
        try:
            corr_2 = np.load(fr"cache\corr_sst_2\corr_{num}_{m1}_{m2}.npy")  # 读取缓存
        except:
            corr_2 = np.array([[np.corrcoef(sen, sst_diff.sel(lat=ilat, lon=ilon))[0, 1] for ilon in sst_diff['lon']] for ilat in sst_diff['lat']])
            np.save(fr"cache\corr_sst_2\corr_{num}_{m1}_{m2}.npy", corr_2)
        ax = fig.add_subplot(spec[y, x], projection=ccrs.PlateCarree(central_longitude=180))
        相关系数图层 = ax.contourf(sst_diff['lon'], sst_diff['lat'], corr_2, levels=lev, cmap=cmaps.WhiteBlueGreenYellowRed, extend='both', transform=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -30, 80], crs=ccrs.PlateCarree(central_longitude=180))
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.05)
        draw_maps(get_adm_maps(level='国'), linewidth=0.15)

plt.savefig(r"C:\Users\10574\Desktop\SEN_SST_corr.png", dpi=2000, bbox_inches='tight')
plt.show()

