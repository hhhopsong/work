import numpy
from cartopy import crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from numba import njit


# 角度转弧度
def pi(x):
    return numpy.pi / 180 * x

# 理查逊法求势函数
def ψ_1(U, V, lat):
    U = np.asarray(U)
    V = np.asarray(V)
    lat = np.asarray(lat)
    Re = 6371e3

    m = U.shape[0] - 2
    n = U.shape[1] - 2

    psi = np.zeros((m, n), dtype=float)
    psi_new = np.zeros((m, n), dtype=float)

    dlon = pi(5.0)
    dlat = pi(5.0)

    # 计算地转涡度
    dvg_dx = np.zeros((m, n))
    dug_dy = np.zeros((m, n))
    for i in range(m):
        for ii in range(n):
            dvg_dx[i, ii] = (V[i + 1, ii + 2] - V[i + 1, ii]) / (
                        Re * np.cos(lat[i + 1] * np.pi / 180) * 5 * np.pi / 180)
            dug_dy[i, ii] = (U[i, ii + 1] - U[i + 2, ii + 1]) / (Re * 5 * np.pi / 180)
    # 数据整合
    e = dvg_dx - dug_dy

    while True:
        max_diff = jacobi_step(psi, psi_new, e, lat, Re, dlon, dlat)
        print(f"Max difference: {max_diff}")
        if max_diff < 1000:
            break

        psi[:, :] = psi_new

    return psi_new

@njit(cache=True, fastmath=True)
def jacobi_step(psi, psi_new, D, lat1d, Re, dlon, dlat):
    m, n = psi.shape
    max_diff = 0.0
    DEG2RAD = np.pi / 180.0

    for i in range(1, m - 1):
        coslat = np.cos(lat1d[i + 1] * DEG2RAD)  # lat1d 必须是 1D
        ax = 1.0 / (Re * coslat * dlon) ** 2
        ay = 1.0 / (Re * dlat) ** 2
        denom = 2.0 * (ax + ay)

        for j in range(1, n - 1):
            res = ax * (psi[i + 1, j] + psi[i - 1, j]) + ay * (psi[i, j + 1] + psi[i, j - 1]) \
                  - denom * psi[i, j] + D[i, j]

            newv = psi[i, j] + res / denom
            psi_new[i, j] = newv

            diff = abs(newv - psi[i, j])
            if diff > max_diff:
                max_diff = diff

    return max_diff

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

u = xr.open_dataset(fr"{PYFILE}/p2/data/U.nc").sel(level=500, year=1961)
v = xr.open_dataset(fr"{PYFILE}/p2/data/V.nc").sel(level=500, year=1961)
lon = u['lon']
lat = v['lat']

psi = ψ_1(u.u, v.v, lat)

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
ax.contourf(lon[1:-1], lat[1:-1], psi*1e-6, levels=10, extend='both', transform=ccrs.PlateCarree())
ax.quiver(lon, lat, u.u, v.v, regrid_shape=20, color='gray', transform=ccrs.PlateCarree())
plt.savefig('test.png', dpi=600)