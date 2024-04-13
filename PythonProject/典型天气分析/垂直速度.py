import numpy
from cartopy import crs as ccrs
import cartopy.feature as cfeature
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
U = xr.open_dataset(r"D:\dxtq\uwnd.2022.nc")
V = xr.open_dataset(r"D:\dxtq\vwnd.2022.nc")
t = xr.open_dataset(r"D:\dxtq\air.2022.nc")  # 温度(单位:K)
# 调出2020年7月18日8:00数据(时次:796   [层次:5(500hPa)]  范围:lat12-32;lon28-56   格点:25 * 41)
time = eval(input('输入时次索引(0-8时 1-14时 2-20时 3-1天2时):'))    # 时次(从6月13日08时开始，每隔6小时取一次)
lev = eval(input('输入层数(0-1000hPa 1-925hPa 2-850hPa 5-500hPa):'))    # 时次(从6月13日08时开始，每隔6小时取一次)
lon = U['lon'][23:66]
lat = U['lat'][7:34]
U = U['uwnd'][16 + time, :, 7:34, 23:66]
V = V['vwnd'][16 + time, :, 7:34, 23:66]
tt = t['air'][48 + time, :, 8:33, 24:65]
# 计算各层散度
D_all = np.zeros((17, 25, 41))
for i in range(17):
    D_all[i] = D(U[i], V[i], lat)
    print(f'{np.array(U[i]["level"]).tolist():.0f}\thPa散度计算完毕!请稍后...')
ω_all = np.zeros((17, 25, 41))
for i in range(17):
    if i != 0:
        ω_all[i] = ω_all[i - 1] + 0.5 * (D_all[i] + D_all[i - 1]) * np.array(U[i - 1]['level'] - U[i]['level']).tolist()
    else:
        ω_all[i] = np.zeros((25, 41))
    print(f'{np.array(U[i]["level"]).tolist():.0f}\thPa修正前涡度计算完毕!请稍后...')
ω_re = np.zeros((17, 25, 41))
for i in range(17):
    if i != 16:
        ω_re[i, :, :] = V_re(17, ω_all[i], i+1, ω_all[16]) * 100 * 8.314 * tt[i] / 9.8 / (tt[i]['level'] * 100)
    else:
        ω_re[i, :, :] = 0
    print(f'{np.array(U[i]["level"]).tolist():.0f}\thPa修正后涡度计算完毕!请稍后...')
# 绘图
# ##地图要素设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
# ##修正后
level1 = np.arange(-np.fabs(ω_re[lev]).max(), np.fabs(ω_re[lev]).max(), np.fabs(ω_re[lev]).max() * 2 / 500)
ax2 = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax2.set_title(f'2022-06-13 {8 + time * 6}:00 {np.array(U[lev]["level"]).tolist():.0f}hPa垂直速度场(单位:m/s)', color='black', fontsize=15)
ax2.set_extent([60, 160, 10, 70], crs=ccrs.PlateCarree())
a2 = ax2.contourf(lon[1:42], lat[1:26], ω_re[lev], cmap='Spectral_r', levels=level1, extend='both', transform=ccrs.PlateCarree())
fig.colorbar(orientation='horizontal', shrink=0.8, pad=0.05, aspect=50, mappable=a2)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle='-')
ax2.add_feature(cfeature.RIVERS, lw=0.25)
plt.savefig(f'D:/dxtq/2022-6-13-{8 + time * 6}垂直速度场.png', dpi=1000, bbox_inches='tight')
plt.show()
