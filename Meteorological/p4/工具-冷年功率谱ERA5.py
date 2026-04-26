import os
import numpy as np
import pandas as pd
import xarray as xr
from climkit.masked import masked
from climkit.filter import *
import matplotlib.pyplot as plt
import tqdm

from climkit.wavelet import WaveletAnalysis

def region_mean_series(da, shp_path):
    vals = []
    for i in range(da.sizes['time']):
        da_clip = masked(da.isel(time=i), shp_path)
        vals.append(da_clip.mean(dim=('lat', 'lon'), skipna=True))
    return xr.concat(vals, dim='time').assign_coords(time=da['time'])

# =========================================================
# ====================== User settings =====================
# =========================================================

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 合并后的夏季 t2m 数据
NC_FILE = "/Volumes/TiPlus7100/p4/data/ERA5_daily_t2m_sum.nc"

# 长江流域 shp
SHP_FILE = fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"

# 输出目录
OUT_DIR = fr"{PYFILE}/p4/pic/"
os.makedirs(OUT_DIR, exist_ok=True)

# 输出图片
OUT_FIG = os.path.join(OUT_DIR, r"图3_t2m逐日实际场_三线")

# 参考气候态年份
CLIM_START = 1961
CLIM_END = 2022

# 各要素按“月-日”逐日气候态（不是 dayofyear）
CLIM = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_clim_sum.nc")
u_clim = CLIM["u_clim"]
v_clim = CLIM["v_clim"]
z_clim = CLIM["z_clim"]
t_clim = CLIM["t_clim"]
w_clim = CLIM["w_clim"]
olr_clim = CLIM["olr_clim"]
t2m_clim = CLIM["t2m_clim"]

time = [1961, 2022]
COOL_YEAR = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014, 2015]


ds = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_ano_all_years_MJJAS_1961_2022.nc")['t2m_ano']



# 小波分析
global_signif = []
fft_power = []
global_power =[]
for i in tqdm.tqdm(COOL_YEAR):
    # 生成每个冷年的 6-8 月逐日时间标签
    # 用 sel 按标签选择
    ANO = ds.sel(time=pd.date_range(f"{i}-06-01", f"{i}-08-31", freq="D").values)
    t2m_ano = ANO
    t2m_ano = t2m_ano.sel(time=t2m_ano.time.dt.month.isin([6, 7, 8]))
    t2m_ano = region_mean_series(t2m_ano, SHP_FILE)

    wavelet_analysis = WaveletAnalysis(t2m_ano.data, wave='Morlet', dt=1, detrend=False, normal=False, signal=.95, J=7)
    period = wavelet_analysis.period # np.log2(period)
    global_signif.append(wavelet_analysis.global_signif)
    fft_power.append(wavelet_analysis.fft_power * wavelet_analysis.var)
    global_power.append(wavelet_analysis.global_power * wavelet_analysis.var)
#%%
wavelet = xr.Dataset(
    {'global_signif': (['time', 'period'], global_signif),
     'global_power': (['time', 'period'], global_power)
     },
    coords={'time': COOL_YEAR, 'period': period}
)

wavelet.to_netcdf("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_ano_all_JJA_wavelet.nc")

#%%
# 绘制瀑布图展示功率谱
