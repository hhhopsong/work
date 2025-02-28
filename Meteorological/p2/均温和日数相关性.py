import xarray as xr
import numpy as np
from scipy import signal, stats
from toolbar.masked import masked
from toolbar.significance_test import r_test
import salem

t2m = xr.open_dataset(r"D:\PyFile\p2\data\t2m_78.nc")
# 区域1选择 (25, 100) (35, 100) (35, 120)构成的三角区域
triangle_geojson = {
    "type": "Polygon",
    "coordinates": [[[100, 25], [100, 35], [120, 35], [100, 25]]]
}
lon_grid, lat_grid = np.meshgrid(t2m.lon, t2m.lat)
mask = xr.DataArray(
    t_path.contains_points(np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T).reshape(lon_grid.shape),
    dims=['lat', 'lon'],
    coords={'lat': t2m.lat, 'lon': t2m.lon}
)
t2m_region1 = t2m.where(mask)


# t2m去趋势
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