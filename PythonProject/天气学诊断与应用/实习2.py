import pprint
from cartopy import crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def E(T, RH=100):
    # 求水汽压;T单位:K; RH单位:%
    a = 17.27
    b = 35.86
    return 6.1078 * np.exp(a * (T - 273.16) / (T - b)) * RH / 100


def Td(T, RH, index=2):
    # 求露点温度; T单位:K; RH单位:%
    a = 17.27
    b = 35.86
    T_d = T  # 单位:K
    e = E(T_d, RH)
    while True:
        es = E(T_d)
        if e < es:
            T_d -= index
        else:
            break
    index /= 2
    if index == 0.25:
        return T_d
    else:
        return Td(T_d + index * 2, RH, index)


def TL(T, Td, P, z, P_start=300, index=128):
    # 求凝结高度; T:K; Td:K; P:hPa; z:m
    Tdz = Td
    Tz = T
    Pz = P
    ez = E(Tdz)
    θz = Tz * ((1000 / Pz) ** 0.286)
    qz = 0.622 * ez / (Pz - 0.378 * ez)  # 单位:kg/kg
    Cp = 1005  # 单位:J/...
    PL = P_start
    while True:
        Tl = θz * (PL / 1000) ** 0.286
        eL = E(Tl)
        qL = 0.622 * eL / (PL - 0.378 * eL)
        if qL - qz < 0:
            PL += index
        else:
            ZL = (9.8 * z + Cp * Tz - Cp * Tl) / 9.8
            break
    index /= 2
    if index == 0.5:
        return θz, Tl, PL, ZL, qL
    else:
        return TL(T, Td, P, z, PL - index * 2, index)


def θse(θ, Tl, ql):
    Cp = 1005  # 等压比容,单位:J / (kg * K)
    L = 2.5 * 10 ** 6  # 水汽凝结潜热,单位:J/kg
    ws = ql / (1 - ql)
    θse = θ * np.exp((L * ws) / (Cp * Tl))
    return θse


# 数据读取
H = xr.open_dataset("D:/tz_shixi/hgt.2020.nc")  # 高度场(单位:m)
T = xr.open_dataset("D:/tz_shixi/air.2020.nc")  # 温度(单位:K)
q = xr.open_dataset("D:/tz_shixi/rhum.2020.nc")  # 相对湿度(eg:78,67...)
lon = H['lon'][28:57]
lat = H['lat'][12:33]
# 调出2020年7月18日8:00数据(时次:796   层次:5(500hPa);2(850hPa)  范围:lat12-32;lon28-56)
H500 = H['hgt'][796][5, 12:33, 28:57]
H850 = H['hgt'][796][2, 12:33, 28:57]
T500 = T['air'][796][5, 12:33, 28:57]
T850 = T['air'][796][2, 12:33, 28:57]
q500 = q['rhum'][796][5, 12:33, 28:57]
q850 = q['rhum'][796][2, 12:33, 28:57]
# 求500hPa假相当位温
θse500 = [[None for i in range(29)] for ii in range(21)]
for i in range(21):
    for ii in range(29):
        bridge = TL(T500[i, ii], Td(T500[i, ii], q500[i, ii]), 500, H500[i, ii])
        θse500[i][ii] = θse(bridge[0], bridge[1], bridge[4])
θse500 = np.array(θse500)
# 地图要素设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
levels = 15
# 绘制500hPa假相当位温分布图
ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
ax1.set_title('2020-07-18 500hPa假相当位温场(单位:K)', color='black', fontsize=15)
ax1.set_extent([70, 140, 10, 60], crs=ccrs.PlateCarree())
ax1.coastlines()
ax1.stock_img()
a1 = ax1.contourf(lon, lat, θse500, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both')
b1 = ax1.contour(lon, lat, θse500, levels=levels, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b1, inline=True, fontsize=5, fmt='%.00f', colors='black')
# 求850hPa假相当位温
θse850 = [[None for i in range(29)] for ii in range(21)]
for i in range(21):
    for ii in range(29):
        bridge = TL(T850[i, ii], Td(T850[i, ii], q850[i, ii]), 850, H850[i, ii])
        θse850[i][ii] = θse(bridge[0], bridge[1], bridge[4])
θse850 = np.array(θse850)
# 绘制850hPa假相当位温分布图
ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
ax2.set_title('2020-07-18 850hPa假相当位温场(单位:K)', color='black', fontsize=15)
ax2.set_extent([70, 140, 10, 60], crs=ccrs.PlateCarree())
ax2.coastlines()
ax2.stock_img()
a2 = ax2.contourf(lon, lat, θse850, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both')
b2 = ax2.contour(lon, lat, θse850, levels=levels, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b2, inline=True, fontsize=5, fmt='%.00f', colors='black')
# 绘制500hPa - 850hPa假相当位温差值分布图
ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
ax3.set_title('2020-07-18 500hPa、850hPa假相当位温差值场(单位:K)', color='black', fontsize=15)
ax3.set_extent([70, 140, 10, 60], crs=ccrs.PlateCarree())
ax3.coastlines()
ax3.stock_img()
a3 = ax3.contourf(lon, lat, θse500 - θse850, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=10, extend='both')
b3 = ax3.contour(lon, lat, θse500 - θse850, levels=10, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b3, inline=True, fontsize=5, fmt='%+.00f', colors='black')
plt.savefig(f'D:/tz_shixi/实习2/实习2.png', dpi=1500, bbox_inches='tight')
plt.show()
