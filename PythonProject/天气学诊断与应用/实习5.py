import numpy
from cartopy import crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


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


# 速度修正
def V_re(N, ωk, k, ωn):
    M = 0.5 * N * (N + 1)
    return ωk - k * (k + 1) / 2 / M * (ωn - np.zeros(ωn.shape))


# 数据读取
U = xr.open_dataset("D:/tz_shixi/uwnd.2020.nc")
V = xr.open_dataset("D:/tz_shixi/vwnd.2020.nc")
# 调出2020年7月18日8:00数据(时次:796   [层次:5(500hPa)]  范围:lat12-32;lon28-56   格点:25 * 41)
lon = U['lon'][23:66]
lat = U['lat'][7:34]
U = U['uwnd'][797][:, 7:34, 23:66]
V = V['vwnd'][797][:, 7:34, 23:66]
# 计算各层散度
D_all = np.zeros((17, 25, 41))
for i in range(17):
    D_all[i] = D(U[i], V[i], lat)
    print(f'{np.array(U[i]["level"]).tolist():.0f}hPa散度计算完毕!请稍后...')
ω_all = np.zeros((17, 25, 41))
for i in range(17):
    if i != 0:
        ω_all[i] = ω_all[i - 1] + 0.5 * (D_all[i] + D_all[i - 1]) * np.array(U[i - 1]['level'] - U[i]['level']).tolist()
    else:
        ω_all[i] = np.zeros((25, 41))
    print(f'{np.array(U[i]["level"]).tolist():.0f}hPa修正前涡度计算完毕!请稍后...')
ω500_re = V_re(17, ω_all[5], 6, ω_all[16])
# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
# ##修正前
level1 = 10
ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
ax1.set_title(f'2020-07-18 8:00 垂直速度场(×10$^-$$^3$)  修正前', color='black', fontsize=15)
ax1.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
a1 = ax1.contourf(lon[1:42], lat[1:26], ω_all[5], cmap='Spectral_r', levels=level1, extend='both', transform=ccrs.PlateCarree())
b1 = ax1.contour(lon[1:42], lat[1:26], ω_all[5] * 10 ** 3, levels=level1, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b1, inline=True, fontsize=5, fmt='%+.0f', colors='black')
ax1.coastlines()
# ##修正后
level2 = 10
ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
ax2.set_title(f'2020-07-18 8:00 垂直速度场(×10$^-$$^3$)  修正后', color='black', fontsize=15)
ax2.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
a2 = ax2.contourf(lon[1:42], lat[1:26], ω500_re, cmap='Spectral_r', levels=level2, extend='both', transform=ccrs.PlateCarree())
b2 = ax2.contour(lon[1:42], lat[1:26], ω500_re * 10 ** 3, levels=level2, linewidths=0.5, transform=ccrs.PlateCarree())
plt.clabel(b2, inline=True, fontsize=5, fmt='%+.0f', colors='black')
ax2.coastlines()
plt.savefig(f'D:/tz_shixi/实习5/实习5.png', dpi=1500, bbox_inches='tight')
plt.show()
