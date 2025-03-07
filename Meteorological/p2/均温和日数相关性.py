import xarray as xr
import numpy as np
from scipy import signal, stats
from toolbar.masked import masked
from toolbar.significance_test import r_test


def detrend(data):
    return data - np.polyval(np.polyfit(range(len(data)), data, 1), range(len(data)))


t2m = xr.open_dataset(r"D:\PyFile\p2\data\t2m_78.nc")
t2m_all_region = masked(t2m, r"D:\PyFile\map\self\长江_TP\长江_tp.shp")['t2m'].mean(['lat', 'lon'], skipna=True)
t2m_region1 = masked(t2m, r"D:\Code\work\Meteorological\p2\map\WYTR\长江_tp.shp")['t2m'].mean(['lat', 'lon'], skipna=True)
t2m_region2 = masked(t2m, r"D:\Code\work\Meteorological\p2\map\EYTR\长江_tp.shp")['t2m'].mean(['lat', 'lon'], skipna=True)
# t2m去趋势
t2m_weight = np.array([detrend(t2m_region2.to_numpy()),
                       detrend(t2m_all_region.to_numpy()),
                       detrend(t2m_region1.to_numpy())
                       ])
t2m_weight = xr.Dataset({'t2m': (['type', 'year'], t2m_weight)},
                        coords={'type': [1, 2, 3],
                                'year': t2m['year'].data,})
time_ser = xr.open_dataset(r"D:\PyFile\p2\data\Time_type_AverFiltAll0.9%_0.3%_3.nc")
type_1 = time_ser.sel(type=1)['K'].data
type_2 = detrend(time_ser.sel(type=2)['K'].data)
type_3 = time_ser.sel(type=3)['K'].data
time_ser = xr.Dataset({'K': (['type', 'year'], np.array([type_1, type_2, type_3]))},
                      coords={'type': [1, 2, 3],
                              'year': t2m['year'].data})
corr = np.corrcoef(time_ser['K'], t2m_weight['t2m'])