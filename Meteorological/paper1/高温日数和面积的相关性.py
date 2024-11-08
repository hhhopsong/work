from cartopy import crs as ccrs
import cartopy.feature as cfeature
import multiprocessing
import sys
import cartopy.feature as cfeature
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr

from cartopy import crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
from cartopy.util import add_cyclic_point
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.pyplot import quiverkey
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import filters
from toolbar.masked import masked
from toolbar.significance_test import corr_test
from toolbar.TN_WaveActivityFlux import TN_WAF_3D
from toolbar.curved_quivers.modplot import velovect, velovect_key

EHD = xr.open_dataset(fr"D:\PyFile\paper1\EHD35.nc")
EHD = masked(EHD, r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
EHD = EHD.sel(time=EHD['time.month'].isin([7, 8]))  # 截取7-8月数据
EHD = masked(EHD.groupby('time.year').sum('time'), r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").mean(['lat', 'lon'])['tmax'] # 计算长江流域EHD年平均
stations = np.load("D:\PyFile\paper1\OLS35.npy") # 读取站点数据
corr = np.corrcoef(EHD, stations)[0, 1] # 进行显著性检验
