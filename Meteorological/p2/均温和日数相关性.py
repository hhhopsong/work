import xarray as xr
import numpy as np
from scipy import signal

t2m = xr.open_dataset(r"D:\PyFile\p2\data\t2m_678.nc")
weight = xr.open_dataset(r"D:\PyFile\p2\data\Tavg_weight.nc")
t2m_interp = t2m.interp(lon=weight.lon, lat=weight.lat)
# t2m去趋势
t2m_detrend = t2m_interp - t2m_interp.mean('year')
t2m_detrend = signal.detrend(t2m_detrend['t2m'], axis=0, type='linear')
t2m_weight = np.array([(t2m_detrend * weight['W'].sel(type=1))['t2m'].data,
                       (t2m_detrend * weight['W'].sel(type=2))['t2m'].data,
                       (t2m_detrend * weight['W'].sel(type=3))['t2m'].data
                       ])