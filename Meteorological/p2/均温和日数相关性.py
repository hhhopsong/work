import xarray as xr
import numpy as np
from scipy import signal, stats
from toolbar.masked import masked
from toolbar.significance_test import r_test

t2m = xr.open_dataset(r"D:\PyFile\p2\data\t2m_78.nc")
weight = xr.open_dataset(r"D:\PyFile\p2\data\Tavg_weight.nc")
weight = masked(weight, r"D:\PyFile\map\self\长江_TP\长江_tp.shp")
weight = weight.where(weight['W'] > 0, np.nan)
weight_anomaly = weight - weight.mean(['lat', 'lon'])
t2m_interp = t2m.interp(lon=weight.lon, lat=weight.lat)
# t2m去趋势
t2m_detrend = t2m_interp - t2m_interp.mean('year')
t2m_detrend = signal.detrend(t2m_detrend['t2m'], axis=0, type='linear')
t2m_weight = np.array([(t2m_detrend * weight_anomaly['W'].sel(type=1).to_numpy()),
                       (t2m_detrend * weight_anomaly['W'].sel(type=2).to_numpy()),
                       (t2m_detrend * weight_anomaly['W'].sel(type=3).to_numpy())
                       ])
t2m_weight = xr.Dataset({'t2m': (['type', 'year', 'lat', 'lon'], t2m_weight)},
                        coords={'type': [1, 2, 3],
                                'year': t2m['year'].data,
                                'lat': weight['lat'].data,
                                'lon': weight['lon'].data})
t2m_weight = masked(t2m_weight, r"D:\PyFile\map\self\长江_TP\长江_tp.shp")
t2m_weight_avg = t2m_weight.mean(['lat', 'lon'])
time_ser = xr.open_dataset(r"D:\PyFile\p2\data\Time_type_AverFiltAll0.9%_0.3%_3.nc").transpose('type', 'year')

corr = np.corrcoef(time_ser['K'], t2m_weight_avg['t2m'])