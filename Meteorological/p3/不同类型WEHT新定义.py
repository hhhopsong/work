import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from climkit.masked import masked



def type_of_WEHT(EHD):
    """
    不同类型WEHT事件划分
    输入:
        UP: 上游极端高温格点数时间序列
        LOW: 中下游极端高温格点数时间序列
        ALL: 全流域极端高温格点数时间序列
    输出:
        WEHT_type: 不同类型WEHT事件时间序列
    """
    type_weight = xr.open_dataset(fr"{PYFILE}/p2/data/Tavg_weight.nc")
    type1 = EHD * ((type_weight.sel(type=1) - type_weight.sel(type=1).mean())/type_weight.sel(type=1).std())
    type2 = EHD * ((type_weight.sel(type=2) - type_weight.sel(type=2).mean())/type_weight.sel(type=2).std())
    type3 = EHD * ((type_weight.sel(type=3) - type_weight.sel(type=3).mean())/type_weight.sel(type=3).std())
    type1, type2, type3 = type1.sum(dim=['lat', 'lon']), type2.sum(dim=['lat', 'lon']), type3.sum(dim=['lat', 'lon'])
    for iyear in range(len(type1.year)):
        for day in range(len(type1.day)):
            if type1.W[iyear, day] > type2.W[iyear, day]:
                if type3.W[iyear, day] > type1.W[iyear, day]:
                    type1.W[iyear, day], type2.W[iyear, day], type3.W[iyear, day] = 0, 0, 1
                else:
                    type1.W[iyear, day], type2.W[iyear, day], type3.W[iyear, day] = 1, 0, 0
            else:
                if type3.W[iyear, day] > type2.W[iyear, day]:
                    type1.W[iyear, day], type2.W[iyear, day], type3.W[iyear, day] = 0, 0, 1
                else:
                    type1.W[iyear, day], type2.W[iyear, day], type3.W[iyear, day] = 0, 1, 0

    type1, type2, type3 = type1.sum(dim=['day']), type2.sum(dim=['day']), type3.sum(dim=['day'])
    type1, type2, type3 = (type1 - type1.mean())/ type1.std(), (type2 - type2.mean())/ type2.std(), (type3 - type3.mean())/ type3.std() # 标准化
    WEHT_days = xr.Dataset({'I': (['type', 'year'], [type1.W, type2.W, type3.W])}, coords={'type': [1, 2, 3], 'year': type1.year})
    return WEHT_days



# 文件路径
PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"


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

EHCI_thes = np.where(EHCI > 0.3, 1, 0)
# EHT_up_stations  = masked(EHD, fr"{PYFILE}/map/self/WYTR/长江_tp.shp").sum(dim=['lat', 'lon']) * EHCI_thes  # 长江上游极端高温格点数
# EHT_low_stations = masked(EHD, fr"{PYFILE}/map/self/EYTR/长江_tp.shp").sum(dim=['lat', 'lon']) * EHCI_thes  # 长江中下游极端高温格点数
# low_type = (EHT_low_stations - EHT_up_stations).sum(dim='day')
# up_type = (EHT_up_stations - EHT_low_stations).sum(dim='day')
# low_type = (low_type - low_type.min()) / (low_type.max() - low_type.min()) * 2 - 1
Daily_WEHT_type = type_of_WEHT(EHD)


# xr -1～1 标准化


type1_days = xr.open_dataset(fr"{PYFILE}/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc").sel(type=1)['K']
type1_days = (type1_days - type1_days.min()) / (type1_days.max() - type1_days.min()) * 2 - 1
type1_days_ = Daily_WEHT_type.sel(type=1)
type1_days_ = (type1_days_ - type1_days_.min()) / (type1_days_.max() - type1_days_.min()) * 2 - 1

type2_days = xr.open_dataset(fr"{PYFILE}/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc").sel(type=2)['K']
type2_days = (type2_days - type2_days.min()) / (type2_days.max() - type2_days.min()) * 2 - 1
type2_days_ = Daily_WEHT_type.sel(type=2)
type2_days_ = (type2_days_ - type2_days_.min()) / (type2_days_.max() - type2_days_.min()) * 2 - 1

type3_days = xr.open_dataset(fr"{PYFILE}/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc").sel(type=3)['K']
type3_days = (type3_days - type3_days.min()) / (type3_days.max() - type3_days.min()) * 2 - 1
type3_days_ = Daily_WEHT_type.sel(type=3)
type3_days_ = (type3_days_ - type3_days_.min()) / (type3_days_.max() - type3_days_.min()) * 2 - 1

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
ax.plot([i for i in range(62)], type1_days, color='red', linestyle='-', linewidth=2.5)
ax.plot([i for i in range(62)], type2_days, color='blue', linestyle='-', linewidth=2.5)
ax.plot([i for i in range(62)], type3_days, color='green', linestyle='-', linewidth=2.5)

ax.plot([i for i in range(62)], type1_days_.I, color='red', linestyle='-', linewidth=1)
ax.plot([i for i in range(62)], type2_days_.I, color='blue', linestyle='--', linewidth=1)
ax.plot([i for i in range(62)], type3_days_.I, color='green', linestyle=':', linewidth=1)
# for t, c in [(-1, 'blue'), (0, 'purple'), (1, 'green')]:
#     sel = Yearly_WEHT_type.where(Yearly_WEHT_type == t)
#     ax.scatter(sel.year, sel, color=c, s=10, marker='o', zorder=3)
plt.savefig(f'{PYFILE}/p3/pic/不同类型WEHT事件划分.png', dpi=600, bbox_inches='tight')
plt.show()

corr_1 = np.corrcoef(type1_days, type1_days_.I)[0, 1]
corr_2 = np.corrcoef(type2_days, type2_days_.I)[0, 1]
corr_3 = np.corrcoef(type3_days, type3_days_.I)[0, 1]

print(corr_1, corr_2, corr_3)