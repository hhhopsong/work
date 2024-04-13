#%matplotlib inline

import matplotlib.path as mpath
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
ax.coastlines()
ax.stock_img()

star_path = mpath.Path.unit_regular_star(200, 1.5)
star_path = mpath.Path(star_path.vertices.copy() * 80,
                       star_path.codes.copy())

ax.set_boundary(star_path, transform = ccrs.PlateCarree())

plt.show()