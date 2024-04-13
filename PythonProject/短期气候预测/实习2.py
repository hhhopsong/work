import pprint
import numpy as np
from eofs.standard import Eof
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import xarray as xr
from xgrads import open_CtlDataset


def transf(data, var, KS):
    # This Subroutine Provides Initial F By KS
    # KS=-1:Self; KS=0:Departure; KS=1:Standardized Departure
    print('数据预处理中...')
    GridPointsNum = len(data['lon']) * len(data['lat'])
    TimeSeriesLength = len(data['time'])
    H = data[var]
    if KS == -1:
        return data, len(data['lat']), len(data['lon']), TimeSeriesLength
    elif KS == 0 or KS == 1:
        avg = np.mean(H, axis=0)  # 平均高度场
        Departure = np.zeros((TimeSeriesLength, len(data['lat']), len(data['lon'])))
        for i in range(TimeSeriesLength):
            Departure[i] = H[i] - avg  # 高度距平场
        if KS == 0:
            return Departure, len(data['lat']), len(data['lon']), TimeSeriesLength
        Std = np.zeros((TimeSeriesLength, len(data['lat']), len(data['lon'])))
        for y in range(len(data['lat'])):
            for x in range(len(data['lon'])):
                Sx = np.sqrt((1 / TimeSeriesLength) * np.sum(Departure[:, y, x] ** 2))  # 标准差
                Std[:, y, x] = Departure[:, y, x] / Sx  # 标准化
        return Std, len(data['lat']), len(data['lon']), TimeSeriesLength



data = xr.open_dataset('D:/short_term_pre/task2/data/hgt4821-500.nc')
H = transf(data, 'hgt', 1)[0]
eof_data = np.array(H)
lon = data['lon']
lat = data['lat']
# eof分解
eof = Eof(eof_data)   #进行eof分解
U = eof.eofs(eofscaling=0, neofs=3)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
PC = eof.pcs(pcscaling=0, npcs=3)  # 同上 npcs决定输出的时间序列个数
s = eof.varianceFraction(neigs=3)   # 得到前neig个模态的方差贡献
# Map-Setting
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 9))
# Map_1
ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_title(f'1月500hPa特征向量场1 方差贡献:{s[0]*100:.2f}%', color='black', fontsize=15)
ax1.set_extent([40, 140, 10, 70], crs=ccrs.PlateCarree())
ax1.coastlines()
ax1.stock_img()
#a1 = ax1.contourf(lon, lat, H_ave, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=15, extend='both')
b1 = ax1.contour(lon, lat, U[0, :, :], transform=ccrs.PlateCarree(), lw=0.3, levels=20, extend='both')
plt.clabel(b1, inline=True, fontsize=5, colors='black')
# Map_2
ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree(central_longitude=180))
ax2.set_title(f'1月500hPa特征向量场2 方差贡献:{s[1]*100:.2f}%', color='black', fontsize=15)
ax2.set_extent([40, 140, 10, 70], crs=ccrs.PlateCarree())
ax2.coastlines()
ax2.stock_img()
#a2 = ax2.contourf(lon, lat, H_A, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=15, extend='both')
b2 = ax2.contour(lon, lat, U[1, :, :], transform=ccrs.PlateCarree(), lw=0.3, levels=20, extend='both')
plt.clabel(b2, inline=True, fontsize=5, colors='black')
# Map_3
ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree(central_longitude=180))
ax3.set_title(f'1月500hPa特征向量场3 方差贡献:{s[2]*100:.2f}%', color='black', fontsize=15)
ax3.set_extent([40, 140, 10, 70], crs=ccrs.PlateCarree())
ax3.coastlines()
ax3.stock_img()
#a3 = ax3.contourf(lon, lat, H_LA, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=15, extend='both')
b3 = ax3.contour(lon, lat, U[2, :, :], transform=ccrs.PlateCarree(), lw=0.3, levels=20, extend='both')
plt.clabel(b3, inline=True, fontsize=5, colors='black')
# Print
plt.savefig(f'D:/short_term_pre/task2/实习2.png', dpi=1500)
plt.show()
