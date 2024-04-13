import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.stock_img()
ax.set_extent([60, 140, 5, 70], crs=ccrs.PlateCarree())


ny_lon, ny_lat = -75, 43
delhi_lon, delhi_lat = 77.23, 28.61

plt.plot([ny_lon, delhi_lon], [ny_lat, delhi_lat],
            color='blue', linewidth=2, marker='o',
            transform=ccrs.Geodetic(),
            )

plt.plot([ny_lon, delhi_lon], [ny_lat, delhi_lat],
            color='red', linewidth=2,
            linestyle='--',
            transform=ccrs.PlateCarree(),
            )


plt.show()
