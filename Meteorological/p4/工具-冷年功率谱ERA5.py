import os
import numpy as np
import pandas as pd
import xarray as xr
from climkit.masked import masked
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

OUT_NC = "/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_ano_all_JJA_wavelet.nc"


# 如果要算全部年份，改成：
years = np.arange(1961, 2023)
# years = np.array(COOL_YEAR)


def region_mean_series_fast(da, shp_path, lat_name="lat", lon_name="lon"):
    """
    只生成一次空间 mask，然后对所有 time 一次性做区域平均。
    比逐日 masked 快很多。
    """

    # 用全 1 的二维场生成一次 mask
    template = xr.ones_like(da.isel(time=0))

    template_clip = masked(template, shp_path)

    # True 表示在 shp 内
    mask2d = np.isfinite(template_clip)

    # 对所有时间直接套用同一个 mask
    da_region = da.where(mask2d)

    return da_region.mean(dim=(lat_name, lon_name), skipna=True)


# =========================================================
# ====================== Read data =========================
# =========================================================

ds = xr.open_dataset(
    "/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_ano_all_years_MJJAS_1961_2022.nc"
)["t2m_ano"].sel(lat=slice(20, 40), lon=slice(90, 130))

# 只保留 JJA
ds_jja = ds.where(ds.time.dt.month.isin([6, 7, 8]), drop=True)

# 只保留需要的年份，减少后续计算量
ds_jja = ds_jja.where(ds_jja.time.dt.year.isin(years), drop=True)

# 一次性计算长江流域平均
t2m_region_mean = region_mean_series_fast(ds_jja, SHP_FILE)

# 区域平均之后数据量很小，直接 load 到内存
t2m_region_mean = t2m_region_mean.load()


# =========================================================
# ====================== Wavelet ===========================
# =========================================================

global_signif_list = []
global_power_list = []
valid_years = []
period = None

for year in tqdm.tqdm(years):
    one_year = t2m_region_mean.where(
        t2m_region_mean.time.dt.year == year,
        drop=True
    )

    # 理论上 JJA 应该是 92 天
    if one_year.sizes["time"] == 0:
        print(f"Skip {year}: no data")
        continue

    arr = np.asarray(one_year.data, dtype=float)

    # 如果有缺测，小波一般不能直接吃 NaN
    if np.isnan(arr).any():
        s = pd.Series(arr)
        arr = (
            s.interpolate(limit_direction="both")
             .to_numpy()
        )

    wavelet_analysis = WaveletAnalysis(
        arr,
        wave="Morlet",
        dt=1,
        detrend=False,
        normal=False,
        signal=.99,
        J=7
    )

    if period is None:
        period = wavelet_analysis.period

    global_signif_list.append(wavelet_analysis.global_signif)
    global_power_list.append(wavelet_analysis.global_power * wavelet_analysis.var)
    valid_years.append(year)


# =========================================================
# ====================== Save ==============================
# =========================================================

wavelet = xr.Dataset(
    {
        "global_signif": (["time", "period"], np.asarray(global_signif_list)),
        "global_power": (["time", "period"], np.asarray(global_power_list)),
    },
    coords={
        "time": np.asarray(valid_years),
        "period": period,
    }
)

wavelet.to_netcdf(OUT_NC)

print(f"Saved to: {OUT_NC}")
