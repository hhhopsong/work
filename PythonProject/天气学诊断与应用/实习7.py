import pprint
import numpy
from cartopy import crs as ccrs
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
q = xr.open_dataset("D:/tz_shixi/rhum.2020.nc")     # 相对湿度(eg:78,67...)
U = xr.open_dataset("D:/tz_shixi/uwnd.2020.nc")
V = xr.open_dataset("D:/tz_shixi/vwnd.2020.nc")
t = xr.open_dataset("D:/tz_shixi/air.2020.nc")  # 温度(单位:K)
# 调出2020年7月19日8:00数据(时次:800 范围:lat12-32;lon28-56    格点:25 * 41)
lon = q['lon'][23:66]
lat = q['lat'][7:34]
qq = q['rhum'][801][:, 7:34, 23:66]
tt = t['air'][801][:, 7:34, 23:66]
uu = U['uwnd'][801][:, 7:34, 23:66]
vv = V['vwnd'][801][:, 7:34, 23:66]
R = 6371 * 1000  # 地球半径
# 计算比湿
q = q_(qq['level'], tt, qq)
# 计算水汽通量
qu_g = q * uu / 9.8
qv_g = q * vv / 9.8
# 计算整层水汽通量
Vq = np.zeros((qq.shape[1], qq.shape[2]))
Uq = np.zeros((qq.shape[1], qq.shape[2]))
D = np.zeros((qq.shape[1] - 2, qq.shape[2] - 2))
for i in range(len(qq['level']) - 1):
    for j in range(len(qq['lat'])):
        for k in range(len(qq['lon'])):
            Uq[j, k] += (qu_g[i, j, k] + qu_g[i + 1, j, k]) / 2 * (qq['level'][i] - qq['level'][i + 1])
            Vq[j, k] += (qv_g[i, j, k] + qv_g[i + 1, j, k]) / 2 * (qq['level'][i] - qq['level'][i + 1])
    print('%dhPa水汽通量计算完毕' % np.array(qq['level'][i]).tolist())
for j in range(qq.shape[1] - 2):
    for k in range(qq.shape[2] - 2):
        D[j, k] = (Uq[j + 1, k + 2] - Uq[j + 1, k]) / (R * np.cos(lat[i + 1] * np.pi / 180) * 5 * np.pi / 180) + (Vq[j, k + 1] - Vq[j + 2, k + 1]) / (R * 5 * np.pi / 180)
# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
level1 = 10
# ##整层水汽通量图
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
ax1.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
ax1.set_title('2020-07-19 8:00 整层水汽通量', color='black', fontsize=15)
a1 = ax1.contourf(lon, lat, np.sqrt(Uq ** 2 + Vq ** 2), cm0ap='Spectral_r', levels=level1, extend='both', transform=ccrs.PlateCarree())
fig.colorbar(orientation='horizontal', shrink=0.8, pad=0.05, aspect=50, mappable=a1)
ax1.coastlines()
ax1.quiver(lon, lat, Uq, Vq)
# ##整层水汽通量散度图
ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
ax2.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
ax2.set_title('2020-07-19 8:00 整层水汽通量散度', color='black', fontsize=15)
a2 = ax2.contourf(lon[1:42], lat[1:26], D, cm0ap='Spectral_r', levels=level1, extend='both', transform=ccrs.PlateCarree())
fig.colorbar(orientation='horizontal', shrink=0.8, pad=0.05, aspect=50, mappable=a2)
ax2.coastlines()
plt.savefig('D:/tz_shixi/实习7/实习7.png', dpi=1500, bbox_inches='tight')
plt.show()
