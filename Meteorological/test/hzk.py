from climkit.significance_test import r_test
import numpy as np
import xarray as xr
from climkit.masked import masked

print(r_test(45, 0.01))
print(r_test(45, 0.05))
print(r_test(45, 0.1))

break_time = xr.open_dataset("/Volumes/TiPlus7100/PyFile/p6/data/standardized_first_large_scale_extreme_heat_onset_JJA_1961_2022.nc")['standardized_onset']
# time_ser = xr.open_dataset(r"/Volumes/TiPlus7100/PyFile/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc")
# type_1 = time_ser.sel(type=1)['K'].data
# type_2 = time_ser.sel(type=2)['K'].data
# type_3 = time_ser.sel(type=3)['K'].data
# days = type_1 + type_2 + type_3

t2m = xr.open_dataset(r"/Volumes/TiPlus7100/PyFile/p2/data/t2m_78.nc")
t2m_all_region = masked(t2m, r"/Volumes/TiPlus7100/PyFile/map/self/长江_TP/长江_tp.shp")['t2m'].mean(['lat', 'lon'], skipna=True)


print(np.corrcoef(t2m_all_region, break_time))
