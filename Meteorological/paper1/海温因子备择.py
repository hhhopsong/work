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
for x in range(11, -1, -1):
    m1 = M + x + 1
    if m1 > 12:
        m1 -= 12
    for y in range(11 - x, 12):
        num += 1
        m2 = M - y
        if m2 <= 0:
            m2 += 12
        sst_diff = xr.open_dataset(fr"cache\sst_diff\sst_{num}_{m1}_{m2}.nc")  # 读取缓存
        corr = np.corrcoef(np.array([[ols for j in range(180)] for i in range(89)]), sst_diff['sst'].transpose('lat','lon','time'))[0, 1]
        ax = fig.add_subplot(spec[y, x], projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([-180, 180, -30, 80], crs=ccrs.PlateCarree(central_longitude=180))
        ax.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray')
        draw_maps(get_adm_maps(level='国'), linewidth=0.4)

