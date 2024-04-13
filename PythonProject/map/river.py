
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.io.shapereader import Reader
from matplotlib import rcParams
config = {"font.family":'Times New Roman',"font.size": 16,"mathtext.fontset":'stix'}
rcParams.update(config)
fig = plt.figure(dpi=500)
proj=ccrs.PlateCarree()
ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
extent = [70,140,15,55]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.5,zorder=2,color='k')# 添加海岸线
ax.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\cnriver\2级河流.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='b',linewidth=0.2)
ax.set_xticks(np.arange(extent[0], extent[1] + 1, 10), crs = proj)
ax.set_yticks(np.arange(extent[-2], extent[-1] + 1,10), crs = proj)
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
ax.yaxis.set_major_formatter(LatitudeFormatter())
plt.show()