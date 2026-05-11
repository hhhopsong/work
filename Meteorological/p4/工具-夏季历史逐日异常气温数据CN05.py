import os
import warnings

import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import dask

from dask.array import PerformanceWarning

from climkit.filter import *
from climkit.masked import masked
from climkit.Cquiver import *
from climkit.TN_WaveActivityFlux import *


# =========================
# 基础设置
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

warnings.filterwarnings("ignore", category=PerformanceWarning)
dask.config.set({"array.slicing.split_large_chunks": False})

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
OUT_DIR = r"/Volumes/TiPlus7100/p4/data"


# =========================
# 工具函数
# =========================
def standardize_time_da(da: xr.DataArray) -> xr.DataArray:
    if "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})

    if "time" not in da.coords:
        raise ValueError(f"{da.name or 'DataArray'} 中没有 time 或 valid_time 坐标。")

    time_vals = da["time"].values

    # 情况 1：已经被错误读成 1970-01-01 00:00:00.019610101 这种
    if np.issubdtype(da["time"].dtype, np.datetime64):
        years = pd.DatetimeIndex(time_vals).year

        if np.all(years == 1970):
            raw = pd.DatetimeIndex(time_vals).nanosecond.astype(str).str.zfill(9)

            # 0.019610101 秒对应 nanosecond = 19610101
            dates = pd.to_datetime(raw, format="%Y%m%d")
            da = da.assign_coords(time=dates)

    # 情况 2：time 本身是 19610101 这种整数或字符串
    else:
        raw = pd.Series(time_vals).astype(str).str.extract(r"(\d{8})")[0]
        dates = pd.to_datetime(raw, format="%Y%m%d")
        da = da.assign_coords(time=dates.values)

    return da



def smart_chunk(da: xr.DataArray, time_full: bool = True) -> xr.DataArray:
    chunk_map = {}

    if "time" in da.dims:
        chunk_map["time"] = -1 if time_full else 32
    if "level" in da.dims:
        chunk_map["level"] = 1
    if "lat" in da.dims:
        chunk_map["lat"] = 45
    if "lon" in da.dims:
        chunk_map["lon"] = 90
    if "mmdd" in da.dims:
        chunk_map["mmdd"] = -1

    if chunk_map:
        da = da.chunk(chunk_map)
    return da


def select_months(da: xr.DataArray, months=(5, 6, 7, 8, 9)) -> xr.DataArray:
    """只保留指定月份"""
    return da.where(da.time.dt.month.isin(months), drop=True)


def daily_clim_by_mmdd(da: xr.DataArray, time_dim: str = "time", drop_feb29: bool = True) -> xr.DataArray:
    """按 MM-DD 计算逐日气候态"""
    if time_dim not in da.dims:
        raise ValueError(f"{da.name or 'DataArray'} 缺少时间维 {time_dim}。")

    out = da

    # Ensure time is in datetime format
    out[time_dim] = pd.to_datetime(out[time_dim].values)

    if drop_feb29:
        leap_mask = (out[time_dim].dt.month == 2) & (out[time_dim].dt.day == 29)
        out = out.where(~leap_mask, drop=True)

    mmdd = out[time_dim].dt.strftime("%m-%d")
    out = out.assign_coords(mmdd=(time_dim, mmdd.data))
    out = out.groupby("mmdd").mean(time_dim)

    out = smart_chunk(out, time_full=True)
    return out



def anomaly_by_mmdd_for_months(
    da: xr.DataArray,
    clim: xr.DataArray,
    start: str,
    end: str,
    months=(5, 6, 7, 8, 9)
) -> xr.DataArray:
    """计算指定年份范围内、指定月份的逐日异常"""
    sub = da.sel(time=slice(start, end))
    sub = select_months(sub, months=months)
    sub = smart_chunk(sub, time_full=True)
    clim = smart_chunk(clim, time_full=True)

    mmdd_indexer = xr.DataArray(
        sub.time.dt.strftime("%m-%d").values,
        coords={"time": sub.time},
        dims="time"
    )

    clim_on_time = clim.sel(mmdd=mmdd_indexer)

    # 删除辅助坐标 mmdd，避免后面 xr.Dataset 合并时报冲突
    if "mmdd" in clim_on_time.coords:
        clim_on_time = clim_on_time.drop_vars("mmdd")

    out = sub - clim_on_time

    # 再保险：异常场中如果还带着 mmdd，也删除
    if "mmdd" in out.coords:
        out = out.drop_vars("mmdd")

    out = smart_chunk(out, time_full=True)
    return out



def make_encoding(ds: xr.Dataset, complevel: int = 1) -> dict:
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            "zlib": True,
            "complevel": complevel,
            "shuffle": True
        }
        if np.issubdtype(ds[var].dtype, np.floating):
            encoding[var]["dtype"] = "float32"
    return encoding


def prepare_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def regrid_to_target_grid(
    da: xr.DataArray,
    target_lon: xr.DataArray,
    target_lat: xr.DataArray,
    method: str = "linear"
) -> xr.DataArray:
    da = normalize_lonlat_da(da)

    clean_lon = xr.DataArray(
        target_lon.values,
        dims=("lon",),
        coords={"lon": target_lon.values}
    )
    clean_lat = xr.DataArray(
        target_lat.values,
        dims=("lat",),
        coords={"lat": target_lat.values}
    )

    out = da.interp(lon=clean_lon, lat=clean_lat, method=method)
    out = smart_chunk(out, time_full=True)
    return out


def assert_same_horizontal_grid(ref: xr.DataArray, da: xr.DataArray, name: str):
    if "lat" not in ref.coords or "lon" not in ref.coords:
        raise ValueError("参考场缺少 lat/lon 坐标。")
    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(f"{name} 缺少 lat/lon 坐标。")

    lat_ok = np.array_equal(ref["lat"].values, da["lat"].values)
    lon_ok = np.array_equal(ref["lon"].values, da["lon"].values)

    if not lat_ok or not lon_ok:
        raise ValueError(
            f"{name} 的水平网格与参考网格不一致，不能直接合并。\n"
            f"lat_ok={lat_ok}, lon_ok={lon_ok}\n"
            f"ref lat/lon size=({ref.lat.size}, {ref.lon.size}), "
            f"{name} lat/lon size=({da.lat.size}, {da.lon.size})"
        )


def print_grid_info(name: str, da: xr.DataArray):
    msg = [f"{name}: dims={da.dims}, shape={da.shape}"]
    if "lat" in da.coords:
        msg.append(f"lat={da.lat.size}, [{float(da.lat.min()):.3f}, {float(da.lat.max()):.3f}]")
    if "lon" in da.coords:
        msg.append(f"lon={da.lon.size}, [{float(da.lon.min()):.3f}, {float(da.lon.max()):.3f}]")
    print(" | ".join(msg))


# =========================
# 参数区
# =========================
clim_start = "1991"
clim_end = "2020"

target_start = "1961-05-01"
target_end = "2022-09-30"
target_months = (5, 6, 7, 8, 9)


# =========================
# 读取 T2M，并插值到 ERA5 网格
# =========================
print("读取 T2M...")

t2m = xr.open_dataset(
    "/Volumes/TiPlus7100/data/CHN_GRID_DATA/tem/Tmean_19510101-20250714_0.5.nc",
    chunks="auto"
)["tem"]

t2m = standardize_time_da(t2m)
t2m = t2m.sel(time=slice(clim_start, clim_end))


# =========================
# 计算逐日气候态
# =========================
print("计算 MM-DD 气候态...")

t2m_clim = daily_clim_by_mmdd(t2m)

# =========================
# 计算所有年份 5-9 月异常
# =========================
print("计算所有年份 5-9 月异常场...")

t2m_ano = anomaly_by_mmdd_for_months(t2m, t2m_clim, target_start, target_end, months=target_months)

# =========================
# 合并前网格一致性检查
# =========================
print("检查合并前网格一致性...")

print("网格检查通过。")


# =========================
# 导出所有年份 5-9 月异常到 NetCDF
# =========================
print("准备导出 NetCDF...")

prepare_output_dir(OUT_DIR)

ano_ds = xr.Dataset({
    "t2m_ano": t2m_ano
})

ano_ds.attrs["description"] = (
    f"Daily anomaly by MM-DD for all years, May to September only. "
    f"Target period: {target_start} to {target_end}. "
    f"Climatology period: {clim_start}-{clim_end}."
)

ano_all_file = f"{OUT_DIR}/CN05_daily_ano_all_years_MJJAS_{clim_start}_{clim_end}.nc"

print("写出所有年份 5-9 月 anomaly...")
ano_ds.to_netcdf(
    ano_all_file,
    encoding=make_encoding(ano_ds, complevel=1)
)

print("导出完成：")
print(ano_all_file)
