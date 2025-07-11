import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from toolbar.curved_quivers.modplot import *

# 读取表格数据
df = pd.read_csv(r"D:\PyFile\test\1_average_uv_wind_per_grid_1961.csv")

# 筛选1961年6月上旬的数据
df = df[(df['year'] == 1961) & (df['month'] == 6) & (df['period'] == 'early')]

# 获取经纬度和风场数据
latitudes = df['latitude'].values
longitudes = df['longitude'].values
u_wind = df['average_u_wind'].values
v_wind = df['average_v_wind'].values

# 绘制矢量图
fig = plt.figure(figsize=(10, 8))
proj = ccrs.PlateCarree()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=proj)

# 设置绘图范围
ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=proj)

# 添加地理特征
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

# 设置经纬度刻度间隔为10并格式化显示
ax.set_xticks(np.arange(longitudes.min(), longitudes.max() + 10, 10), crs=proj)
ax.set_yticks(np.arange(latitudes.min(), latitudes.max() + 10, 10), crs=proj)
lon_formatter = plt.matplotlib.ticker.FormatStrFormatter('%g°E')
lat_formatter = plt.matplotlib.ticker.FormatStrFormatter('%g°')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

# 绘制风场矢量（使用三角箭头）
arrow_style = FancyArrowPatch((0, 0), (1, 1), arrowstyle='->', mutation_scale=20)
#q = ax.quiver(longitudes[::10], latitudes[::10], u_wind[::10], v_wind[::10], color='k',
#              scale=200, zorder=5, width=0.002,regrid_shape = 30, headwidth=3, headlength=4.5, transform=ccrs.PlateCarree())
cq = Curlyquiver(ax,longitudes.reshape(281, 441)[0], latitudes.reshape(281, 441).T[0], u_wind.reshape(281, 441), v_wind.reshape(281, 441),
            arrowsize=1, linewidth=1.5, scale=20, regrid=15, arrowstyle='fancy')
cq.key(fig, U=30, label="30 m/s")
# 添加矢量图的key0
#plt.quiverkey(q, X=0.9, Y=1.05, U=10, label='Wind: 10 m/s', labelpos='E')

ax.set_title('Wind Field Vector Plot')
plt.savefig(r"D:\PyFile\pic\1961_6_early.png", dpi=1000)
plt.show()