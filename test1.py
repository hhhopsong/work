import numpy
from cartopy import crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from numba import njit

Re = 6371e3
DEG2RAD = np.pi / 180.0
# Earth's radius in meters
def _dx_atmos(longitude, latitude):
    lon = np.asarray(longitude, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)

    # lon: (nx,), lat: (ny,)
    dlon = np.gradient(lon) * DEG2RAD              # (nx,)
    coslat = np.cos(lat * DEG2RAD)                 # (ny,)

    dx = Re * coslat[:, None] * dlon[None, :]      # (ny, nx)
    return dx

def _dy_atmos(longitude, latitude):
    lon = np.asarray(longitude, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)

    dlat = np.gradient(lat) * DEG2RAD              # (ny,)
    ny, nx = lat.size, lon.size

    dy = Re * dlat[:, None] * np.ones((ny, nx), dtype=np.float64)  # (ny, nx)
    return dy

def _divh_atmos(longitude, latitude, u, v):
    lon = np.asarray(longitude, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # dx, dy: (ny,nx) in meters
    dx = _dx_atmos(lon, lat)
    dy = _dy_atmos(lon, lat)

    # du/dx, dv/dy
    du = np.gradient(u, axis=1)   # along lon
    dv = np.gradient(v, axis=0)   # along lat

    dudx = du / dx
    dvdy = dv / dy

    # metric term: - v * tan(phi) / R
    tanlat = np.tan(lat[:, None] * DEG2RAD)  # (ny,1) -> broadcast to (ny,nx)

    divh = dudx + dvdy - v * tanlat / Re
    divh[np.abs(latitude)==90] = np.nan
    return divh

def _curlz_atmos(longitude, latitude, u, v):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    dx = _dx_atmos(longitude, latitude)  # (ny,nx)
    dy = _dy_atmos(longitude, latitude)  # (ny,nx)

    dVdx = np.gradient(v, axis=1) / dx   # axis=1: lon
    dUdy = np.gradient(u, axis=0) / dy   # axis=0: lat

    return dVdx - dUdy

def _grad_atmos(longitude, latitude, H):
    dx = _dx_atmos(longitude, latitude)
    dy = _dy_atmos(longitude, latitude)
    grad_y, grad_x = np.gradient(H)
    grad_x = grad_x / dx
    grad_y = grad_y / dy
    return grad_x, grad_y


@njit(cache=True, fastmath=True)
def _sor_poisson(phi, rhs, dx2, dy2, omega, tol, max_iter):
    ny, nx = phi.shape

    for _ in range(max_iter):
        max_diff = 0.0

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                old = phi[i, j]

                dx2ij = dx2[i, j]
                dy2ij = dy2[i, j]

                denom = 2.0 / dx2ij + 2.0 / dy2ij
                numer = (phi[i, j+1] + phi[i, j-1]) / dx2ij + (phi[i+1, j] + phi[i-1, j]) / dy2ij - rhs[i, j]

                newv = (1.0 - omega) * old + omega * (numer / denom)
                phi[i, j] = newv

                diff = abs(newv - old)
                if diff > max_diff:
                    max_diff = diff

        if max_diff < tol:
            break

    return phi


def velocity_potential(longitude, latitude, u, v, loop_max=1e6, epsilon=1e-7, sor_index=0.2):
    """
    Calculate velocity potential using Richardson iterative method.

    Parameters
    ----------
    lon : (M, N) `array`
        longitude array.
    lat : (M, N) `array`
        latitude array.
    u : (M, N) `array`
        x component of the wind.
    v : (M, N) `array`
        y component of the wind.
    loop_max : `int`
        Maximum iteration loop number. Default is `1e6`.
    epsilon : `float`
        Minimum error value. Default is `1e-7`
    sor_index : `float`
        Super relaxation coefficient [0.2 - 0.5]. Default is `0.2`.

    Returns
    -------
    A 3-item tuple of arrays
        velocity potential and divergence wind component
    """
    if isinstance(loop_max, float):
        loop_max = int(loop_max)

    divh = _divh_atmos(longitude, latitude, u, v)  # vertical divergence
    dx2 = _dx_atmos(longitude, latitude)**2        # square of latitude gradient
    dy2 = _dy_atmos(longitude, latitude)**2                   # square of longitude gradient
    phi0 = np.zeros_like(divh, dtype=np.float64)   # 初值
    phi = _sor_poisson(phi0,divh,dx2,dy2,omega=sor_index,tol=epsilon,max_iter=loop_max)
    phi = -phi

    # divergence wind
    DphiDx, DphiDy = _grad_atmos(longitude, latitude, phi)
    Uphi = DphiDx
    Vphi = DphiDy

    # make boundary NaN
    phi[0,:] = np.nan; Uphi[0,:] = np.nan; Vphi[0,:] = np.nan;
    phi[-1,:] = np.nan; Uphi[-1,:] = np.nan; Vphi[-1,:] = np.nan;
    phi[:,0] = np.nan; Uphi[:,0] = np.nan; Vphi[:,0] = np.nan;
    phi[:,-1] = np.nan; Uphi[:,-1] = np.nan; Vphi[:,-1] = np.nan;

    return phi, Uphi, Vphi

def stream_function(longitude, latitude, u, v, loop_max=int(1e10), epsilon=1e-10, sor_index=0.2):
    """
    Calculate stream function using Richardson iterative method.

    Parameters
    ----------
    lon : (M, N) `array`
        longitude array
    lat : (M, N) `array`
        latitude array
    u : (M, N) `array`
        x component of the wind
    v : (M, N) `array`
        y component of the wind
    loop_max : `int`
        Maximum iteration loop number. Default is `1e6`.
    epsilon : `float`
        Minimum error value. Default is `1e-7`
    sor_index : `float`
        Super relaxation coefficient [0.2 - 0.5]. Default is `0.2`.

    Returns
    -------
    A 3-item tuple of arrays
        stream function and vorticity wind component
    """
    if isinstance(loop_max, float):
        loop_max = int(loop_max)

    curlz = _curlz_atmos(longitude, latitude, u, v)  # vorticity
    dx2 = _dx_atmos(longitude, latitude)**2        # square of latitude gradient
    dy2 = _dy_atmos(longitude, latitude)**2                   # square of longitude gradient
    phi0 = np.zeros_like(curlz, dtype=np.float64)
    psi = _sor_poisson(phi0, curlz, dx2, dy2, omega=sor_index, tol=epsilon, max_iter=loop_max)
    psi = -psi

    # vorticity wind
    DpsiDx, DpsiDy = _grad_atmos(longitude, latitude, psi)
    Upsi = -DpsiDy
    Vpsi = DpsiDx

    # make boundary NaN
    psi[0,:] = np.nan; Upsi[0,:] = np.nan; Vpsi[0,:] = np.nan;
    psi[-1,:] = np.nan; Upsi[-1,:] = np.nan; Vpsi[-1,:] = np.nan;
    psi[:,0] = np.nan; Upsi[:,0] = np.nan; Vpsi[:,0] = np.nan;
    psi[:,-1] = np.nan; Upsi[:,-1] = np.nan; Vpsi[:,-1] = np.nan;

    return psi, Upsi, Vpsi



PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

u = xr.open_dataset(fr"{PYFILE}/p2/data/U.nc").sel(level=500, year=2022)
v = xr.open_dataset(fr"{PYFILE}/p2/data/V.nc").sel(level=500, year=2022)
lon = u['lon']
lat = v['lat']
psi, Upsi, Vpsi = velocity_potential(lon, lat, u.u, v.v, 1000, 1e-5) #速度势
fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
ax.contourf(lon, lat, psi*1e-6, levels=10, extend='both', transform=ccrs.PlateCarree(), cmap='RdBu_r')
ax.quiver(lon, lat, Upsi, Vpsi, regrid_shape=20, color='gray', transform=ccrs.PlateCarree())
plt.savefig('test.png', dpi=600)