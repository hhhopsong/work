import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from climkit.masked import masked



def type_of_WEHT(UP, LOW, ALL, EHCI):
    """
    不同类型WEHT事件划分
    输入:
        UP: 上游极端高温格点数时间序列
        LOW: 中下游极端高温格点数时间序列
        ALL: 全流域极端高温格点数时间序列
    输出:
        WEHT_type: 不同类型WEHT事件时间序列
    """
    EHCI = EHCI.where(EHCI > 0.3, np.nan)
    threshold = 1/5
    low_up = LOW - UP
    stations_threshold = ALL * threshold
    sign = xr.full_like(ALL, fill_value=1, dtype=np.int8) # 正矩阵

    low_type = sign.where(low_up > stations_threshold, 0)  # 1:中下游型 赋值为1
    up_type = -sign.where(low_up < -stations_threshold, 0) # -1:上游型 赋值为-1

    # 0:全流域型 赋值为0
    WEHT_type = low_type + up_type
    up_year = (WEHT_type.where(WEHT_type == -1, np.nan) + 2).sum(dim='day')
    low_year = (WEHT_type.where(WEHT_type == 1, np.nan) + 0).sum(dim='day')
    all_year = (WEHT_type.where(WEHT_type == 0, np.nan) + 1).sum(dim='day')
    WEHT_year = np.zeros(WEHT_type.shape[0], dtype=np.int8)
    for i in range(WEHT_type.shape[0]):
        if up_year[i] >= low_year[i] and up_year[i] >= all_year[i]:
            WEHT_year[i] = -1  # 上游型
        elif low_year[i] >= up_year[i] and low_year[i] >= all_year[i]:
            WEHT_year[i] = 1   # 中下游型
        else:
            WEHT_year[i] = 0   # 全流域型
    WEHT_year = xr.DataArray(WEHT_year, coords=[WEHT_type.year], dims=['year'])
    return WEHT_type, WEHT_year



# 文件路径
PYFILE = r"/volumes/sty/PyFile"
DATA = r"/volumes/sty/data"


# 数据读取
time = [1961, 2022]
EHDstations_zone = xr.open_dataarray(fr"{PYFILE}/p2/data/Tmax_5Day_filt.nc")
T_th = 0.90
t95 = masked(EHDstations_zone.sel(day=slice('1', '88')), fr"{PYFILE}/map/地图边界数据/长江区1：25万界线数据集（2002年）/长江区.shp").mean(dim=['year', 'day']).quantile(T_th)  # 夏季内 长江中下游流域 分位数
EHD = EHDstations_zone - t95
EHD = EHD.where(EHD > 0, 0)  # 极端高温日温度距平
EHD = EHD.where(EHD == 0, 1)  # 数据二值化处理(1:极端高温, 0:非极端高温)
EHD = masked(EHD, fr"{PYFILE}/map/self/长江_TP/长江_tp.shp")  # 掩膜处理得长江流域EHD温度距平
CN051_2 = xr.open_dataset(fr"{DATA}/CN05.1/2022/CN05.1_Tmax_2022_daily_025x025.nc")
zone_stations = masked((CN051_2 - CN051_2 + 1).sel(time='2022-01-01'), fr"{PYFILE}/map/self/长江_TP/长江_tp.shp").sum()['tmax'].data
EHD = EHD.sel(day=slice('29', '88'))
EHCI = EHD.sum(dim=['lat', 'lon']) / zone_stations  # 长江流域逐日极端高温格点占比
EHT_up_stations  = masked(EHD, fr"{PYFILE}/map/self/WYTR/长江_tp.shp").sum(dim=['lat', 'lon'])  # 长江上游极端高温格点数
EHT_low_stations = masked(EHD, fr"{PYFILE}/map/self/EYTR/长江_tp.shp").sum(dim=['lat', 'lon'])  # 长江中下游极端高温格点数
EHT_all_stations = EHT_up_stations + EHT_low_stations  # 长江流域极端高温格点数

Daily_WEHT_type, Yearly_WEHT_type = type_of_WEHT(EHT_up_stations, EHT_low_stations, EHT_all_stations, EHCI)



plt.rcParams['font.family'] = ['AVHershey Simplex', 'AVHershey Duplex', 'Helvetica']    # 字体为Hershey (安装字体后，清除.matplotlib的字体缓存即可生效)
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
fig = plt.figure(figsize=(16*0.3, 9*0.3))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-1, 63)
ax.set_xticks([0.5, 9.5, 19.5, 29.5, 39.5, 49.5, 59.5])
ax.set_xticklabels(["1961", "1970", "1980", "1990", "2000",  "2010", "2020"], rotation=0, fontsize=12)
ax.set_ylim(-1.5, 1.5)
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(["UR-type", "AR-type", "MLR-type"], fontsize=12)
ax.plot(Yearly_WEHT_type.year, Yearly_WEHT_type, color='k', linestyle='-', linewidth=2.5)
for t, c in [(-1, 'blue'), (0, 'purple'), (1, 'green')]:
    sel = Yearly_WEHT_type.where(Yearly_WEHT_type == t)
    ax.scatter(sel.year, sel, color=c, s=10, marker='o', zorder=3)
plt.savefig(f'{PYFILE}/p3/pic/不同类型WEHT事件划分.png', dpi=600, bbox_inches='tight')
plt.show()
