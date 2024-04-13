import pprint
import numpy
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def E(T, RH=100):
    # 求水汽压;T单位:K; RH单位:%
    a = 17.27
    b = 35.86
    return 6.1078 * np.exp(a * (T - 273.16) / (T - b)) * RH / 100


def q_(P, T, RH=100):
    # 求比湿; T单位:K; RH单位:%
    e = E(T, RH)
    return 0.622 * e / (P - 0.378 * e)


# 数据读取
q = xr.open_dataset(r"D:\dxtq\rhum.2022.nc")     # 相对湿度(eg:78,67...)
t = xr.open_dataset(r"D:\dxtq\air.2022.nc")  # 温度(单位:K)
# 调出2020年7月18日8:00数据(时次:6月13日08时   [层次:5(500hPa) 10(200hPa)]  范围:lat12-32;lon28-56   格点:25 * 41)
time = eval(input('输入时次索引(0-8时 1-14时 2-20时 3-1天2时):'))    # 时次(从6月13日08时开始，每隔6小时取一次)
lon = q['lon'][24:65]
lat = q['lat'][8:33]
Q = q['rhum'][48 + time, :10, 8:33, 24:65]
T = t['air'][48 + time, :10, 8:33, 24:65]
PW = np.zeros((10, 25, 41))
for i in range(10):
    for j in range(25):
        for k in range(41):
            if i != 0:
                PW[i, j, k] = 1 / 9.8 * (q_(Q[i]['level'], T[i, j, k], Q[i, j, k]) * (Q[i - 1]['level'] - Q[i]['level']) * 100)
PW = np.sum(PW, axis=0)
# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
# ##修正前
level1 = 500
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax1.set_title(f'2022-06-13 {8 + time * 6}:00 大气可降水量分布图(单位:mm)', color='black', fontsize=15)
ax1.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
a1 = ax1.contourf(lon, lat, PW, cmap='YlGnBu', levels=level1, extend='both', transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle='-')
ax1.add_feature(cfeature.RIVERS, lw=0.25)
fig.colorbar(orientation='horizontal', shrink=0.8, pad=0.05, aspect=50, mappable=a1)
plt.savefig(f'D:/dxtq/2022-6-13-{8 + time * 6}大气可降水量.png', dpi=1000, bbox_inches='tight')
plt.show()
