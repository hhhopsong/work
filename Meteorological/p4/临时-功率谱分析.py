import os
import numpy as np
import pandas as pd
import xarray as xr
from climkit.masked import masked
from climkit.filter import *
import matplotlib.pyplot as plt

from climkit.wavelet import WaveletAnalysis

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

# 合成年份
COMPOSITE_YEARS = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014]

# 单独分析年份
TARGET_YEAR = 2015


# =========================================================
# ====================== Helper funcs ======================
# =========================================================

def detect_main_var(ds: xr.Dataset) -> str:
    """自动识别主变量名"""
    preferred = ["t2m", "2m_temperature", "tem"]
    for v in preferred:
        if v in ds.data_vars:
            return v
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"无法自动识别温度变量，请检查变量名：{list(ds.data_vars)}")


def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """统一经纬度坐标名为 lon/lat"""
    rename_dict = {}
    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if rename_dict:
        ds = ds.rename(rename_dict)
    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("数据中未找到 lon/lat 或 longitude/latitude 坐标。")
    return ds


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """流域平均"""
    mean_da = da.mean(dim=("lat", "lon"))
    return mean_da


def build_summer_day_index(time_index: pd.DatetimeIndex) -> np.ndarray:
    """
    为夏季 6/7/8 月构造统一 day index:
    6月1日 -> 1
    ...
    8月31日 -> 92
    """
    month = time_index.month
    day = time_index.day
    offsets = np.where(month == 6, 0, np.where(month == 7, 30, 61))
    return offsets + day


def ensure_celsius(da: xr.DataArray) -> xr.DataArray:
    """如果像 Kelvin，就转成 Celsius"""
    vmin = float(da.min().values)
    vmax = float(da.max().values)
    if vmin > 150 and vmax < 400:
        print("检测到温度可能为 Kelvin，自动转换为 Celsius。")
        da = da - 273.15
        da.attrs["units"] = "degC"
    return da


# =========================================================
# =========================== Main =========================
# =========================================================
print("1) 读取 nc 数据...")
ds = xr.open_dataset(NC_FILE)
ds = standardize_latlon(ds)

var_name = detect_main_var(ds)
print(f"识别到变量名：{var_name}")

da = ds[var_name]

if "time" not in da.coords:
    if "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})
    else:
        raise ValueError("数据中没有 time 或 valid_time 坐标。")

# 只保留 1961-2023 夏季
da = da.sel(time=slice(f"{CLIM_START}-06-01", f"{CLIM_END}-08-31"))

# 单位检查
da = ensure_celsius(da)

print("2) 裁剪长江流域...")
da_clip = masked(da, SHP_FILE)

print("3) 计算流域平均...")
ts = area_weighted_mean(da_clip)

# 转 DataFrame

df = ts.to_dataframe(name="tem").reset_index()
df["time"] = pd.to_datetime(df["time"], format="%Y%m%d")

df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day

# 只保留夏季
df = df[df["month"].isin([6, 7, 8])].copy()

# 构造夏季日序
df["summer_day"] = build_summer_day_index(pd.DatetimeIndex(df["time"]))

# ================= 1) 1961-2023 气候态实际场 =================
print("4) 计算 1961-2023 夏季逐日气候态实际场...")
clim_df = (
    df[(df["year"] >= CLIM_START) & (df["year"] <= CLIM_END)]
    .groupby("summer_day", as_index=False)["tem"]
    .mean()
    .rename(columns={"tem": "climatology_actual"})
)

# ================= 2) 合成年份逐日实际场 =================
print("5) 计算合成年份逐日实际场...")
comp_df = (
    df[df["year"].isin(COMPOSITE_YEARS)]
    .groupby("summer_day", as_index=False)["tem"]
    .mean()
    .rename(columns={"tem": "composite_actual"})
)

print("5) 计算合成年份逐日实际场及逐日标准差...")
comp_stat_df = (
    df[df["year"].isin(COMPOSITE_YEARS)]
    .groupby("summer_day")["tem"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={
        "mean": "composite_actual",
        "std": "composite_std"
    })
)
# 如果某些 summer_day 恰好只有一个值，std 会是 NaN，这里补成 0
comp_stat_df["composite_std"] = comp_stat_df["composite_std"].fillna(0.0)

# ================= 3) 2015 逐日实际场 =================
print(f"6) 计算 {TARGET_YEAR} 年逐日实际场...")
y2015_df = (
    df[df["year"] == TARGET_YEAR][["summer_day", "tem"]]
    .groupby("summer_day", as_index=False)
    .mean()
    .rename(columns={"tem": f"{TARGET_YEAR}_actual"})
)

# 横坐标
tick_positions = [1, 16, 31, 46, 62, 77, 92]
tick_labels = ["Jun-01", "Jun-16", "Jul-01", "Jul-16", "Aug-01", "Aug-16", "Aug-31"]

# ================= 作图：三根线一张图 =================
print("7) 绘图...")

# 先按 summer_day 合并，确保三条线逐日对齐
plot_df = clim_df.merge(comp_stat_df, on="summer_day", how="inner")
plot_df = plot_df.merge(y2015_df, on="summer_day", how="inner")

x = plot_df["summer_day"].values
clim_y = plot_df["climatology_actual"].values
comp_y = plot_df["composite_actual"].values
comp_std = plot_df["composite_std"].values

y2015_y = plot_df[f"{TARGET_YEAR}_actual"].values
# 滤波
y2015_y_filt = LanczosFilter(y2015_y-clim_y, 'bandpass', [10, 30], nwts=9).filted()
y2015_y_nan = np.full(len(y2015_y), np.nan)

comp_y_filt = LanczosFilter(comp_y-clim_y, 'bandpass', [10, 30], nwts=9).filted()
comp_y_nan = np.full(len(comp_y), np.nan)

# 小波分析
wavelet_analysis = WaveletAnalysis(y2015_y-clim_y, wave='Morlet', dt=1, detrend=False, normal=False, signal=.95, J=7)
wavelet_analysis.plot(unit="K", figpath=fr"{PYFILE}/p4/pic/功率谱分析_2015.pdf")
wavelet_analysis.plot(unit="K", figpath=fr"{PYFILE}/p4/pic/功率谱分析_2015.png")