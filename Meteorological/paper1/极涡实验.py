import cartopy.crs as ccrs
import cmaps

from matplotlib import pyplot as plt

from LBM.force_file import horizontal_profile as hp
from LBM.force_file import vertical_profile as vp
from LBM.force_file import mk_grads, mk_wave, interp3d_lbm

import metpy.calc as mpcalc

import xarray as xr
import numpy as np



frc = xr.open_dataset(r"D:\CODES\Python\Meteorological\paper1\cache\uvz\z\diff\z_24_4_4.nc")# 读取强迫场
frc_zone = frc.sel(lon=slice(-35, 35), lat=slice(45, 80))  # 选择区域

heights = frc['z']  # 等压位势高度
f = mpcalc.coriolis_parameter(frc['lat'])  # 包含运动学参数的计算(例如，散度或涡度)
dx, dy = mpcalc.grid_deltas_from_dataarray(heights)
ug, vg = mpcalc.geostrophic_wind(heights, f, dx, dy)  # 计算从高度或重力势给出的地转风
vert_abs_vort = f + mpcalc.vorticity(ug, vg, dx, dy)  # 计算水平风的垂直涡度

frc_nc_sigma = interp3d_lbm(frc)
frc_nc_p = interp3d_lbm(frc, 'p')
# 绘图
# 图1
lev = 500
n = 10
extent1 = [-180, 180, -80, 80]
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
ax1.coastlines(linewidths=0.3)
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
lev_range = np.linspace(-np.max(np.abs(frc_nc_p['t'].sel(lev=lev).data)), np.max(np.abs(frc_nc_p['t'].sel(lev=lev).data)), 10)
t200 = ax1.contourf(frc_nc_p['t']['lon'], frc_nc_p['t']['lat'], frc_nc_p['t'].sel(lev=lev),
                    levels=lev_range, cmap=cmaps.GMT_polar, transform=ccrs.PlateCarree(central_longitude=0), extend='both')
plt.show()

if input("是否导出?(y/n)") == 'y':
    frc_nc_sigma.to_netcdf(r'D:\CODES\Python\Meteorological\frc_nc\frc.t42l20.Tingyang.nc')

