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
def standardize_latlon_ds(ds: xr.Dataset) -> xr.Dataset:
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


def standardize_latlon_da(da: xr.DataArray) -> xr.DataArray:
    rename_dict = {}
    if "longitude" in da.coords:
        rename_dict["longitude"] = "lon"
    if "latitude" in da.coords:
        rename_dict["latitude"] = "lat"
    if rename_dict:
        da = da.rename(rename_dict)

    if "lon" not in da.coords or "lat" not in da.coords:
        raise ValueError(f"{da.name or 'DataArray'} 中未找到 lon/lat 或 longitude/latitude 坐标。")
    return da


def standardize_time_ds(ds: xr.Dataset) -> xr.Dataset:
    if "time" in ds.coords:
        return ds
    if "valid_time" in ds.coords:
        return ds.rename({"valid_time": "time"})
    raise ValueError("数据中没有 time 或 valid_time 坐标。")


def standardize_time_da(da: xr.DataArray) -> xr.DataArray:
    if "time" in da.coords:
        return da
    if "valid_time" in da.coords:
        return da.rename({"valid_time": "time"})
    raise ValueError(f"{da.name or 'DataArray'} 中没有 time 或 valid_time 坐标。")


def normalize_lonlat_da(da: xr.DataArray) -> xr.DataArray:
    da = standardize_latlon_da(da)

    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=((da.lon + 360) % 360))

    da = da.sortby("lon")
    da = da.sortby("lat")
    return da


def normalize_lonlat_ds(ds: xr.Dataset) -> xr.Dataset:
    ds = standardize_latlon_ds(ds)

    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=((ds.lon + 360) % 360))

    ds = ds.sortby("lon")
    ds = ds.sortby("lat")
    return ds


def ensure_celsius(da: xr.DataArray) -> xr.DataArray:
    vmin = float(da.min().compute().values)
    vmax = float(da.max().compute().values)
    if vmin > 150 and vmax < 400:
        print("检测到温度可能为 Kelvin，自动转换为 Celsius。")
        da = da - 273.15
        da.attrs["units"] = "degC"
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
clim_start = "1961"
clim_end = "2022"

target_start = "1961-05-01"
target_end = "2022-09-30"
target_months = (5, 6, 7, 8, 9)


# =========================
# 读取 ERA5 主数据
# =========================
print("读取 ERA5 主数据...")

ds = xr.open_zarr(
    r"/Volumes/TiPlus7100/p4/data/ERA5_daily_uvwztq_sum.zarr",
    chunks="auto"
)

ds = standardize_time_ds(ds)
ds = normalize_lonlat_ds(ds)

required_vars = ["u", "v", "z", "t", "w"]
missing_vars = [v for v in required_vars if v not in ds.data_vars]
if missing_vars:
    raise ValueError(f"数据中缺少变量: {missing_vars}；当前变量有: {list(ds.data_vars)}")

ds = ds[required_vars].sel(time=slice(clim_start, clim_end))

chunk_map_ds = {}
if "time" in ds.dims:
    chunk_map_ds["time"] = -1
if "level" in ds.dims:
    chunk_map_ds["level"] = 1
if "lat" in ds.dims:
    chunk_map_ds["lat"] = 45
if "lon" in ds.dims:
    chunk_map_ds["lon"] = 90

ds = ds.chunk(chunk_map_ds)

u = ds["u"]
v = ds["v"]
z = ds["z"]
t = ds["t"]
w = ds["w"]

era5_lon = ds["lon"]
era5_lat = ds["lat"]

print_grid_info("u", u.isel(time=0, level=0))


# =========================
# 读取 OLR，并插值到 ERA5 网格
# =========================
print("读取 OLR...")

olr = xr.open_dataset(
    fr"{DATA}/NOAA/CPC/olr.cbo-1deg.day.mean.nc",
    chunks="auto"
)["olr"]

olr = standardize_time_da(olr)
olr = normalize_lonlat_da(olr)
olr = olr.sel(time=slice(clim_start, clim_end))

print_grid_info("olr_raw", olr.isel(time=0))

print("将 OLR 插值到 ERA5 网格...")
olr = regrid_to_target_grid(olr, era5_lon, era5_lat, method="linear")

print_grid_info("olr_regridded", olr.isel(time=0))


# =========================
# 读取 T2M，并插值到 ERA5 网格
# =========================
print("读取 T2M...")

t2m = xr.open_dataset(
    "/Volumes/TiPlus7100/p4/data/ERA5_daily_t2m_sum.nc",
    chunks="auto"
)["t2m"]

t2m = standardize_time_da(t2m)
t2m = normalize_lonlat_da(t2m)
t2m = ensure_celsius(t2m)
t2m = t2m.sel(time=slice(clim_start, clim_end))

print_grid_info("t2m_raw", t2m.isel(time=0))

print("将 T2M 插值到 ERA5 网格...")
t2m = regrid_to_target_grid(t2m, era5_lon, era5_lat, method="linear")

print_grid_info("t2m_regridded", t2m.isel(time=0))


# =========================
# 读取 single level 数据，并插值到 ERA5 网格
# =========================
print("读取 single level 数据...")

ds_sl = xr.open_zarr(
    r"/Volumes/TiPlus7100/p4/data/single_level_sum.zarr",
    chunks="auto"
)

ds_sl = standardize_time_ds(ds_sl)
ds_sl = normalize_lonlat_ds(ds_sl)

required_vars = ["ssr", "str", "slhf", "sshf", "tcc", "tp"]
missing_vars = [v for v in required_vars if v not in ds_sl.data_vars]
if missing_vars:
    raise ValueError(f"数据中缺少变量: {missing_vars}；当前变量有: {list(ds_sl.data_vars)}")

ds_sl = ds_sl[required_vars].sel(time=slice(clim_start, clim_end))

chunk_map_ds = {}
if "time" in ds_sl.dims:
    chunk_map_ds["time"] = -1
if "lat" in ds_sl.dims:
    chunk_map_ds["lat"] = 45
if "lon" in ds_sl.dims:
    chunk_map_ds["lon"] = 90

ds_sl = ds_sl.chunk(chunk_map_ds)

ssr = ds_sl["ssr"]
str_ = ds_sl["str"]
slhf = ds_sl["slhf"]
sshf = ds_sl["sshf"]
tcc = ds_sl["tcc"]
tp = ds_sl["tp"]

print("将 single level 数据插值到 ERA5 网格...")

ssr = regrid_to_target_grid(ssr, era5_lon, era5_lat, method="linear")
str_ = regrid_to_target_grid(str_, era5_lon, era5_lat, method="linear")
slhf = regrid_to_target_grid(slhf, era5_lon, era5_lat, method="linear")
sshf = regrid_to_target_grid(sshf, era5_lon, era5_lat, method="linear")
tcc = regrid_to_target_grid(tcc, era5_lon, era5_lat, method="linear")
tp = regrid_to_target_grid(tp, era5_lon, era5_lat, method="linear")

print_grid_info("single_level_regridded", ssr.isel(time=0))


# =========================
# 计算逐日气候态
# =========================
print("计算 MM-DD 气候态...")

u_clim = daily_clim_by_mmdd(u)
v_clim = daily_clim_by_mmdd(v)
z_clim = daily_clim_by_mmdd(z)
t_clim = daily_clim_by_mmdd(t)
w_clim = daily_clim_by_mmdd(w)

olr_clim = daily_clim_by_mmdd(olr)
t2m_clim = daily_clim_by_mmdd(t2m)

ssr_clim = daily_clim_by_mmdd(ssr)
str_clim = daily_clim_by_mmdd(str_)
slhf_clim = daily_clim_by_mmdd(slhf)
sshf_clim = daily_clim_by_mmdd(sshf)
tcc_clim = daily_clim_by_mmdd(tcc)
tp_clim = daily_clim_by_mmdd(tp)


# =========================
# 计算所有年份 5-9 月异常
# =========================
print("计算所有年份 5-9 月异常场...")

u_ano = anomaly_by_mmdd_for_months(u, u_clim, target_start, target_end, months=target_months)
v_ano = anomaly_by_mmdd_for_months(v, v_clim, target_start, target_end, months=target_months)
z_ano = anomaly_by_mmdd_for_months(z, z_clim, target_start, target_end, months=target_months)
t_ano = anomaly_by_mmdd_for_months(t, t_clim, target_start, target_end, months=target_months)
w_ano = anomaly_by_mmdd_for_months(w, w_clim, target_start, target_end, months=target_months)

olr_ano = anomaly_by_mmdd_for_months(olr, olr_clim, target_start, target_end, months=target_months)
t2m_ano = anomaly_by_mmdd_for_months(t2m, t2m_clim, target_start, target_end, months=target_months)

ssr_ano = anomaly_by_mmdd_for_months(ssr, ssr_clim, target_start, target_end, months=target_months)
str_ano = anomaly_by_mmdd_for_months(str_, str_clim, target_start, target_end, months=target_months)
slhf_ano = anomaly_by_mmdd_for_months(slhf, slhf_clim, target_start, target_end, months=target_months)
sshf_ano = anomaly_by_mmdd_for_months(sshf, sshf_clim, target_start, target_end, months=target_months)
tcc_ano = anomaly_by_mmdd_for_months(tcc, tcc_clim, target_start, target_end, months=target_months)
tp_ano = anomaly_by_mmdd_for_months(tp, tp_clim, target_start, target_end, months=target_months)


# =========================
# 合并前网格一致性检查
# =========================
print("检查合并前网格一致性...")

ref_ano = u_ano.isel(level=0)

assert_same_horizontal_grid(ref_ano, olr_ano, "olr_ano")
assert_same_horizontal_grid(ref_ano, t2m_ano, "t2m_ano")
assert_same_horizontal_grid(ref_ano, ssr_ano, "ssr_ano")
assert_same_horizontal_grid(ref_ano, str_ano, "str_ano")
assert_same_horizontal_grid(ref_ano, slhf_ano, "slhf_ano")
assert_same_horizontal_grid(ref_ano, sshf_ano, "sshf_ano")
assert_same_horizontal_grid(ref_ano, tcc_ano, "tcc_ano")
assert_same_horizontal_grid(ref_ano, tp_ano, "tp_ano")

print("网格检查通过。")


# =========================
# 导出所有年份 5-9 月异常到 NetCDF
# =========================
print("准备导出 NetCDF...")

prepare_output_dir(OUT_DIR)

ano_ds = xr.Dataset({
    "u_ano": u_ano,
    "v_ano": v_ano,
    "z_ano": z_ano,
    "t_ano": t_ano,
    "w_ano": w_ano,
    "olr_ano": olr_ano,
    "t2m_ano": t2m_ano,
    "ssr_ano": ssr_ano,
    "str_ano": str_ano,
    "slhf_ano": slhf_ano,
    "sshf_ano": sshf_ano,
    "tcc_ano": tcc_ano,
    "tp_ano": tp_ano,
})

ano_ds.attrs["description"] = (
    f"Daily anomaly by MM-DD for all years, May to September only. "
    f"Target period: {target_start} to {target_end}. "
    f"Climatology period: {clim_start}-{clim_end}."
)

ano_all_file = f"{OUT_DIR}/ERA5_CPC_daily_ano_all_years_MJJAS_{clim_start}_{clim_end}.nc"

print("写出所有年份 5-9 月 anomaly...")
ano_ds.to_netcdf(
    ano_all_file,
    encoding=make_encoding(ano_ds, complevel=1)
)

print("导出完成：")
print(ano_all_file)
