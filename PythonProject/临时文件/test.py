from cartopy import crs as ccrs
import matplotlib.pyplot as plt

# 画极地投影地图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
ax.coastlines()
ax.gridlines()
plt.show()
