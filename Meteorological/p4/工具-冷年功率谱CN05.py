import os
import numpy as np
import pandas as pd
import xarray as xr
from climkit.masked import masked
from climkit.filter import *
import matplotlib.pyplot as plt
import tqdm

from climkit.wavelet import WaveletAnalysis


# =========================================================
# ====================== User settings =====================
# =========================================================

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

SHP_FILE = fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"

OUT_DIR = fr"{PYFILE}/p4/pic/"
os.makedirs(OUT_DIR, exist_ok=True)

COOL_YEAR = np.arange(1961, 2023)


# =========================================================
# ====================== Read data =========================
# =========================================================

ds = xr.open_dataset(
    "/Volumes/TiPlus7100/p4/data/CN05_daily_ano_all_years_MJJAS_1961_2022.nc"
)['t2m_ano'].sel(lat=slice(40, 20), lon=slice(90, 130))


# =========================================================
# =========== Build region mask only once ==================
# =========================================================

def build_region_mask(da2d, shp_path):
    """
    只做一次 shp 掩膜。
    用全 1 的 DataArray 生成 mask，避免原数据 NaN 影响区域范围判断。
    """
    ones = xr.ones_like(da2d, dtype=np.float32)
    mask_da = masked(ones, shp_path)
    return mask_da.notnull()


region_mask = build_region_mask(ds.isel(time=0), SHP_FILE)


# =========================================================
# ======= Select all cool-year JJA data at once ============
# =========================================================

time_mask = (
    ds.time.dt.year.isin(COOL_YEAR)
    & ds.time.dt.month.isin([6, 7, 8])
)

ds_jja_cool = ds.sel(time=time_mask)


# =========================================================
# ======= Region mean for all days in one operation ========
# =========================================================

t2m_series = ds_jja_cool.where(region_mask).mean(
    dim=('lat', 'lon'),
    skipna=True
)

# 如果数据较大或后端是懒加载，这里提前加载区域平均后的 1D 序列，后面小波会快很多
t2m_series = t2m_series.load()


# =========================================================
# ===================== Wavelet analysis ===================
# =========================================================

global_signif = []
global_power = []
period = None

for year in tqdm.tqdm(COOL_YEAR):
    x = t2m_series.sel(
        time=slice(f"{year}-06-01", f"{year}-08-31")
    ).values

    x = np.asarray(x, dtype=np.float64)

    # 可选：检查是否有缺测
    if np.isnan(x).any():
        x = pd.Series(x).interpolate(limit_direction="both").values

    wavelet_analysis = WaveletAnalysis(
        x,
        wave='Morlet',
        dt=1,
        detrend=False,
        normal=False,
        signal=.99,
        J=7
    )

    if period is None:
        period = wavelet_analysis.period

    global_signif.append(wavelet_analysis.global_signif)
    global_power.append(wavelet_analysis.global_power * wavelet_analysis.var)


# =========================================================
# ====================== Save result =======================
# =========================================================

wavelet = xr.Dataset(
    {
        'global_signif': (['time', 'period'], np.asarray(global_signif)),
        'global_power': (['time', 'period'], np.asarray(global_power)),
    },
    coords={
        'time': COOL_YEAR,
        'period': period
    }
)

wavelet.to_netcdf(
    "/Volumes/TiPlus7100/p4/data/CN05_daily_ano_all_JJA_wavelet.nc"
)
