import pprint
import numpy
from cartopy import crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature


# 角度转弧度
def pi(x):
    return numpy.pi / 180 * x


# 计算散度
def D(U, V, lat):
    R = 6371 * 10 ** 3  # m
    du_dx = np.zeros((25, 41))
    dv_dy = np.zeros((25, 41))
    for i in range(25):
        for ii in range(41):
            du_dx[i, ii] = (U[i + 2, ii + 1] - U[i, ii + 1]) / (R * np.cos(lat[i + 1] * np.pi / 180) * pi(5))
            dv_dy[i, ii] = (V[i + 1, ii] - V[i + 1, ii + 2]) / (R * pi(5))
    return du_dx + dv_dy


# 理查逊法求势函数
def ψ_1(D_, lat, m=25, n=41):
    R = 6371 * 10 ** 3  # m
    ψ_ = np.zeros((m, n))
    ψ_1 = np.zeros((m, n))
    R = np.zeros((m, n))
    while True:
        for i in range(m):
            for ii in range(n):
                R[i, ii] = (ψ_[i + 1, ii] + ψ_[i - 1, ii]) / (R * np.cos(lat[i + 1] * np.pi / 180) * pi(5)) ** 2\
                            + (ψ_[i, ii - 1] + ψ_[i, ii + 1]) / (R * pi(5)) ** 2 \
                            - (2 / (R * np.cos(lat[i + 1] * np.pi / 180) * pi(5)) ** 2 + 2 / (R * pi(5)) ** 2) * ψ_[i, ii] \
                            + D_[i, ii]
                ψ_1[i, ii] = ψ_[i, ii] + R[i, ii] / (2 / (R * np.cos(lat[i + 1] * np.pi / 180) * pi(5)) ** 2 + 2 / (R * pi(5)) ** 2)
        if np.max(np.abs(ψ_1 - ψ_)) < 1000:
            break
        ψ_ = ψ_1
    return ψ_1


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
U = xr.open_dataset("D:/tz_shixi/uwnd.2020.nc")
V = xr.open_dataset("D:/tz_shixi/vwnd.2020.nc")
q = xr.open_dataset("D:/tz_shixi/rhum.2020.nc")
t = xr.open_dataset("D:/tz_shixi/air.2020.nc")
# 调出2020年7月18日8:00数据(时次:796   [层次:2(850hPa)]  范围:lat12-32;lon28-56   格点:25 * 41)
lon = U['lon'][23:66]
lat = U['lat'][7:34]
U = U['uwnd'][801][2, 7:34, 23:66]
V = V['vwnd'][801][2, 7:34, 23:66]
qq = q['rhum'][801][:, 7:34, 23:66]
tt = t['air'][801][:, 7:34, 23:66]
R = 6371 * 1000  # 地球半径
# 计算比湿
q = q_(qq['level'], tt, qq)
# 计算水汽通量
qu_g = q * U / 9.8
qv_g = q * V / 9.8
# 计算整层水汽通量
Vq = np.zeros((qq.shape[1], qq.shape[2]))
Uq = np.zeros((qq.shape[1], qq.shape[2]))
D = np.zeros((qq.shape[1] - 2, qq.shape[2] - 2))
for i in range(2, 3):
    for j in range(len(qq['lat'])):
        for k in range(len(qq['lon'])):
            Uq[j, k] += (qu_g[i, j, k] + qu_g[i + 1, j, k]) / 2 * (qq['level'][i] - qq['level'][i + 1])
            Vq[j, k] += (qv_g[i, j, k] + qv_g[i + 1, j, k]) / 2 * (qq['level'][i] - qq['level'][i + 1])
    print('%dhPa水汽通量计算完毕' % np.array(qq['level'][i]).tolist())
# 计算散度
for j in range(qq.shape[1] - 2):
    for k in range(qq.shape[2] - 2):
        D[j, k] = (Uq[j + 1, k + 2] - Uq[j + 1, k]) / (R * np.cos(lat[i + 1] * np.pi / 180) * 5 * np.pi / 180) + (Vq[j, k + 1] - Vq[j + 2, k + 1]) / (R * 5 * np.pi / 180)
print(f'散度计算完毕!请稍后...')
# 计算地转涡度
dvg_dx = np.zeros((25, 41))
dug_dy = np.zeros((25, 41))
for i in range(25):
    for ii in range(41):
        dvg_dx[i, ii] = (Vq[i + 1, ii + 2] - Vq[i + 1, ii]) / (R * np.cos(lat[i + 1] * np.pi / 180) * 5 * np.pi / 180)
        dug_dy[i, ii] = (Uq[i, ii + 1] - Uq[i + 2, ii + 1]) / (R * 5 * np.pi / 180)
# 数据整合
ξ850 = dvg_dx - dug_dy
D850 = D
# 计算850hPa的理查逊流函数
ψ850 = ψ_1(D850, lat)
print(f'理查逊流函数计算完毕!请稍后...')
# 计算850hPa的理查逊势函数
Φ850 = ψ_1(ξ850, lat)
print(f'理查逊势函数计算完毕!请稍后...')
# 画图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
level1 = 500
# ##画图
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
ax1.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
ax1.set_title('2020-07-19 8:00 850hPa水汽通量流函数', color='black', fontsize=15)
a1 = ax1.contourf(lon[1:42], lat[1:26], ψ850, cmap='YlGnBu', levels=level1, extend='both', transform=ccrs.PlateCarree())
fig.colorbar(orientation='horizontal', shrink=0.8, pad=0.05, aspect=50, mappable=a1)
ax1.add_feature(cfeature.BORDERS, linestyle='-')
ax1.add_feature(cfeature.RIVERS, lw=0.25)
ax1.coastlines()
ax1.quiver(lon, lat, Uq, Vq)
# ##整层水汽通量散度图
ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
ax2.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
ax2.set_title('2020-07-19 8:00 850hPa水汽通量势函数', color='black', fontsize=15)
a2 = ax2.contourf(lon[1:42], lat[1:26], Φ850, cmap='Spectral_r', levels=level1, extend='both', transform=ccrs.PlateCarree())
fig.colorbar(orientation='horizontal', shrink=0.8, pad=0.05, aspect=50, mappable=a2)
ax2.add_feature(cfeature.BORDERS, linestyle='-')
ax2.add_feature(cfeature.RIVERS, lw=0.25)
ax2.coastlines()
plt.savefig('D:/tz_shixi/实习8/实习8.png', dpi=1500, bbox_inches='tight')
plt.show()
