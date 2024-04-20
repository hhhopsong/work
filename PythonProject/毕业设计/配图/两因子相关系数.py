import os
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as path_effects
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from cnmaps import get_adm_maps, draw_maps
from matplotlib import ticker
import cmaps
from matplotlib.ticker import MultipleLocator, FixedLocator
from eofs.standard import Eof
from scipy.ndimage import filters
from tqdm import tqdm
import geopandas as gpd
import salem
from tools.TN_WaveActivityFlux import TN_WAF
import pprint



# 时间范围
time = ['1979', '2014']
# 地理范围
tas_lonlat = [36, 62, 54, 70]
tos_lonlat = [50, 100, -12, 6]
# 数据路径
mlp_T2m = xr.open_dataset(r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\ERA5\ERA5_2mTemperature_MeanSlp.nc")
sst = xr.open_dataset(r"C:\Users\10574\Desktop\data\sst.mnmean.nc")  # NetCDF-4文件路径不可含中文
tas_dataurl = r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\CMIP6\historical\CMIP6_historical_tas\amon"#数据路径
tos_dataurl = r"D:/CODES/Python/PythonProject/cache/CMIP6_tos/interp_global"#数据路径
Model_Name = os.listdir(tas_dataurl)
Model_Name_tos = os.listdir(tos_dataurl)

T2m = mlp_T2m['t2m'].sel(time=slice('1979-01-01', '2014-12-31'))
slp_2mT_78 = T2m.sel(time=T2m.time.dt.month.isin([7, 8]))
slp_2mT_78 = slp_2mT_78.sel(longitude=slice(tas_lonlat[0], tas_lonlat[1]), latitude=slice(tas_lonlat[3], tas_lonlat[2]))
sst = sst['sst'].sel(time=slice('1979-01-01', '2014-12-31'))
sst = sst.sel(time=sst.time.dt.month.isin([7, 8]))
sst = sst.sel(lon=slice(tos_lonlat[0], tos_lonlat[1]), lat=slice(tos_lonlat[3], tos_lonlat[2]))
slp_2mT_78 = slp_2mT_78.groupby('time.year').mean('time').mean(['latitude', 'longitude'])
sst_78 = sst.groupby('time.year').mean('time').mean(['lat', 'lon'])
slp_2mT_78 = np.array(slp_2mT_78)
sst_78 = np.array(sst_78)

# 进行显著性0.05检验
from scipy.stats import t
# 计算自由度
N = len(sst_78)
# 计算临界值
t_critical = t.ppf(0.99, N - 2)

# 总存
r = np.corrcoef(slp_2mT_78, sst_78)[0, 1]
pas = 0
t_ = r * np.sqrt((N - 2) / (1 - r ** 2))
if t_ > t_critical:
    pas = 1
print(f"{r:.2f}, {pas}(98%显著性)")
