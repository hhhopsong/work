import os
import warnings

import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import dask

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.geometry import Point
from dask.array import PerformanceWarning

from climkit.Cquiver import *
from climkit.masked import masked
from climkit.filter import *
from climkit.TN_WaveActivityFlux import *
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import dry_air_gas_constant as R
from metpy.constants import dry_air_spec_heat_press as cp


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
    """统一 Dataset 经纬度坐标名为 lon/lat"""
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
    """统一 DataArray 经纬度坐标名为 lon/lat"""
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
    """统一 Dataset 时间坐标名为 time"""
    if "time" in ds.coords:
        return ds
    if "valid_time" in ds.coords:
        return ds.rename({"valid_time": "time"})
    raise ValueError("数据中没有 time 或 valid_time 坐标。")


def standardize_time_da(da: xr.DataArray) -> xr.DataArray:
    """统一 DataArray 时间坐标名为 time"""
    if "time" in da.coords:
        return da
    if "valid_time" in da.coords:
        return da.rename({"valid_time": "time"})
    raise ValueError(f"{da.name or 'DataArray'} 中没有 time 或 valid_time 坐标。")


def normalize_lonlat_da(da: xr.DataArray) -> xr.DataArray:
    """
    统一 DataArray 经纬度：
    1. 坐标名为 lon/lat
    2. lon 转为 0~360
    3. lat/lon 排序
    """
    da = standardize_latlon_da(da)

    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=((da.lon + 360) % 360))

    da = da.sortby("lon")
    da = da.sortby("lat")
    return da


def normalize_lonlat_ds(ds: xr.Dataset) -> xr.Dataset:
    """
    统一 Dataset 经纬度：
    1. 坐标名为 lon/lat
    2. lon 转为 0~360
    3. lat/lon 排序
    """
    ds = standardize_latlon_ds(ds)

    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=((ds.lon + 360) % 360))

    ds = ds.sortby("lon")
    ds = ds.sortby("lat")
    return ds


def ensure_celsius(da: xr.DataArray) -> xr.DataArray:
    """如果像 Kelvin，就转成 Celsius"""
    vmin = float(da.min().compute().values)
    vmax = float(da.max().compute().values)
    if vmin > 150 and vmax < 400:
        print("检测到温度可能为 Kelvin，自动转换为 Celsius。")
        da = da - 273.15
        da.attrs["units"] = "degC"
    return da


def smart_chunk(da: xr.DataArray, time_full: bool = True) -> xr.DataArray:
    """
    按常见气象数据维度统一 chunk。
    time_full=True: 时间维合并成单块，适合 groupby / anomaly / filter
    """
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


def daily_clim_by_mmdd(da: xr.DataArray, time_dim: str = "time", drop_feb29: bool = True) -> xr.DataArray:
    """按 MM-DD 计算逐日气候态（避免闰年 dayofyear 错位）"""
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


def anomaly_by_mmdd(da: xr.DataArray, clim: xr.DataArray, start: str, end: str) -> xr.DataArray:
    """
    根据 MM-DD 气候态计算异常。
    """
    sub = da.sel(time=slice(start, end))
    sub = smart_chunk(sub, time_full=True)
    clim = smart_chunk(clim, time_full=True)

    mmdd_indexer = xr.DataArray(
        sub.time.dt.strftime("%m-%d").values,
        coords={"time": sub.time},
        dims="time"
    )

    clim_on_time = clim.sel(mmdd=mmdd_indexer)
    out = sub - clim_on_time
    out = smart_chunk(out, time_full=True)
    return out


def make_encoding(ds: xr.Dataset, complevel: int = 4) -> dict:
    """为输出 nc 构造压缩 encoding"""
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


def regrid_to_target_grid(da: xr.DataArray, target_lon: xr.DataArray, target_lat: xr.DataArray,
                          method: str = "linear") -> xr.DataArray:
    """
    将 da 插值到目标 lon/lat 网格。
    这里只传纯净的 1D lon/lat，避免 time/level 标量坐标冲突。
    """
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
    """
    检查水平网格是否完全一致，防止 Dataset 合并时自动对齐出错。
    """
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
    """打印网格信息便于排查"""
    msg = [f"{name}: dims={da.dims}, shape={da.shape}"]
    if "lat" in da.coords:
        msg.append(f"lat={da.lat.size}, [{float(da.lat.min()):.3f}, {float(da.lat.max()):.3f}]")
    if "lon" in da.coords:
        msg.append(f"lon={da.lon.size}, [{float(da.lon.min()):.3f}, {float(da.lon.max()):.3f}]")
    print(" | ".join(msg))


# =========================
# 参数区
# =========================
year = 2015

# 气候态基期
clim_start = "1961"
clim_end = "2022"

# 为 61 点滤波预留边界
filter_start = f"{year}-05-01"
filter_end = f"{year}-09-30"

# 最终分析期
plot_start = f"{year}-06-01"
plot_end = f"{year}-08-31"


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
# 读取 ssr str slhf sshf tcc tp，并插值到 ERA5 网格
# =========================
ds = xr.open_zarr(
    r"/Volumes/TiPlus7100/p4/data/single_level_sum.zarr",
    chunks="auto"
)

ds = standardize_time_ds(ds)
ds = normalize_lonlat_ds(ds)

required_vars = ["ssr", "str", "slhf", "sshf", "tcc", "tp"]
missing_vars = [v for v in required_vars if v not in ds.data_vars]
if missing_vars:
    raise ValueError(f"数据中缺少变量: {missing_vars}；当前变量有: {list(ds.data_vars)}")

ds = ds[required_vars].sel(time=slice(clim_start, clim_end))

chunk_map_ds = {}
if "time" in ds.dims:
    chunk_map_ds["time"] = -1
if "lat" in ds.dims:
    chunk_map_ds["lat"] = 45
if "lon" in ds.dims:
    chunk_map_ds["lon"] = 90
ds = ds.chunk(chunk_map_ds)

ssr = ds['ssr']
str = ds['str']
slhf = ds['slhf']
sshf = ds['sshf']
tcc = ds['tcc']
tp = ds['tp']

print("将 single level 插值到 ERA5 网格...")
ssr = regrid_to_target_grid(ssr, era5_lon, era5_lat, method="linear")
str = regrid_to_target_grid(str, era5_lon, era5_lat, method="linear")
slhf = regrid_to_target_grid(slhf, era5_lon, era5_lat, method="linear")
sshf = regrid_to_target_grid(sshf, era5_lon, era5_lat, method="linear")
tcc = regrid_to_target_grid(tcc, era5_lon, era5_lat, method="linear")
tp = regrid_to_target_grid(tp, era5_lon, era5_lat, method="linear")

print_grid_info("single level_regridded", ssr.isel(time=0))

# =========================
# 各要素按“月-日”计算逐日气候态
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
str_clim = daily_clim_by_mmdd(str)
slhf_clim = daily_clim_by_mmdd(slhf)
sshf_clim = daily_clim_by_mmdd(sshf)
tcc_clim = daily_clim_by_mmdd(tcc)
tp_clim = daily_clim_by_mmdd(tp)


# =========================
# 计算 5-9 月异常
# =========================
print("计算异常场...")
u_ano = anomaly_by_mmdd(u, u_clim, filter_start, filter_end)
v_ano = anomaly_by_mmdd(v, v_clim, filter_start, filter_end)
z_ano = anomaly_by_mmdd(z, z_clim, filter_start, filter_end)
t_ano = anomaly_by_mmdd(t, t_clim, filter_start, filter_end)
w_ano = anomaly_by_mmdd(w, w_clim, filter_start, filter_end)
olr_ano = anomaly_by_mmdd(olr, olr_clim, filter_start, filter_end)
t2m_ano = anomaly_by_mmdd(t2m, t2m_clim, filter_start, filter_end)
ssr_ano = anomaly_by_mmdd(ssr, ssr_clim, filter_start, filter_end)
str_ano = anomaly_by_mmdd(str, str_clim, filter_start, filter_end)
slhf_ano = anomaly_by_mmdd(slhf, slhf_clim, filter_start, filter_end)
sshf_ano = anomaly_by_mmdd(sshf, sshf_clim, filter_start, filter_end)
tcc_ano = anomaly_by_mmdd(tcc, tcc_clim, filter_start, filter_end)
tp_ano = anomaly_by_mmdd(tp, tp_clim, filter_start, filter_end)


# =========================
# 61 点 Lanczos 滤波后，再裁剪回 6-8 月
# =========================
print("进行 10-30 天 Lanczos 带通滤波...")
u_bp = LanczosFilter(u_ano,   'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
v_bp = LanczosFilter(v_ano,   'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
z_bp = LanczosFilter(z_ano,   'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
t_bp = LanczosFilter(t_ano,   'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
w_bp = LanczosFilter(w_ano,   'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
olr_bp = LanczosFilter(olr_ano, 'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
t2m_bp = LanczosFilter(t2m_ano, 'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
ssr_bp = LanczosFilter(ssr_ano, 'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
str_bp = LanczosFilter(str_ano, 'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
slhf_bp = LanczosFilter(slhf_ano, 'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
sshf_bp = LanczosFilter(sshf_ano, 'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
tcc_bp = LanczosFilter(tcc_ano, 'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))
tp_bp = LanczosFilter(tp_ano, 'bandpass', period=[10, 30], nwts=61).filted().sel(time=slice(plot_start, plot_end))

u_bp = smart_chunk(u_bp, time_full=True)
v_bp = smart_chunk(v_bp, time_full=True)
z_bp = smart_chunk(z_bp, time_full=True)
t_bp = smart_chunk(t_bp, time_full=True)
w_bp = smart_chunk(w_bp, time_full=True)
olr_bp = smart_chunk(olr_bp, time_full=True)
t2m_bp = smart_chunk(t2m_bp, time_full=True)
ssr_bp = smart_chunk(ssr_bp, time_full=True)
str_bp = smart_chunk(str_bp, time_full=True)
slhf_bp = smart_chunk(slhf_bp, time_full=True)
sshf_bp = smart_chunk(sshf_bp, time_full=True)
tcc_bp = smart_chunk(tcc_bp, time_full=True)
tp_bp = smart_chunk(tp_bp, time_full=True)


# =========================
# 合并前网格一致性检查
# =========================
print("检查合并前网格一致性...")
ref_clim = u_clim.isel(level=0)
ref_ano = u_ano.isel(level=0)
ref_bp = u_bp.isel(level=0)

assert_same_horizontal_grid(ref_clim, olr_clim, "olr_clim")
assert_same_horizontal_grid(ref_clim, t2m_clim, "t2m_clim")
assert_same_horizontal_grid(ref_clim, ssr_clim, "ssr_clim")
assert_same_horizontal_grid(ref_clim, str_clim, "str_clim")
assert_same_horizontal_grid(ref_clim, slhf_clim, "slhf_clim")
assert_same_horizontal_grid(ref_clim, sshf_clim, "sshf_clim")
assert_same_horizontal_grid(ref_clim, tcc_clim, "tcc_clim")
assert_same_horizontal_grid(ref_clim, tp_clim, "tp_clim")

assert_same_horizontal_grid(ref_ano, olr_ano, "olr_ano")
assert_same_horizontal_grid(ref_ano, t2m_ano, "t2m_ano")
assert_same_horizontal_grid(ref_ano, ssr_ano, "ssr_ano")
assert_same_horizontal_grid(ref_ano, str_ano, "str_ano")
assert_same_horizontal_grid(ref_ano, slhf_ano, "slhf_ano")
assert_same_horizontal_grid(ref_ano, sshf_ano, "sshf_ano")
assert_same_horizontal_grid(ref_ano, tcc_ano, "tcc_ano")
assert_same_horizontal_grid(ref_ano, tp_ano, "tp_ano")

assert_same_horizontal_grid(ref_bp, olr_bp, "olr_bp")
assert_same_horizontal_grid(ref_bp, t2m_bp, "t2m_bp")
assert_same_horizontal_grid(ref_bp, ssr_bp, "ssr_bp")
assert_same_horizontal_grid(ref_bp, str_bp, "str_bp")
assert_same_horizontal_grid(ref_bp, slhf_bp, "slhf_bp")
assert_same_horizontal_grid(ref_bp, sshf_bp, "sshf_bp")
assert_same_horizontal_grid(ref_bp, tcc_bp, "tcc_bp")
assert_same_horizontal_grid(ref_bp, tp_bp, "tp_bp")

print("网格检查通过。")


# =========================
# 导出 clim / ano / bp 到 nc
# =========================
print("准备导出 NetCDF...")
prepare_output_dir(OUT_DIR)

# climatology（按 mmdd）
clim_ds = xr.Dataset({
    "u_clim": u_clim,
    "v_clim": v_clim,
    "z_clim": z_clim,
    "t_clim": t_clim,
    "w_clim": w_clim,
    "olr_clim": olr_clim,
    "t2m_clim": t2m_clim,
    "ssr_clim": ssr_clim,
    "str_clim": str_clim,
    "slhf_clim": slhf_clim,
    "sshf_clim": sshf_clim,
    "tcc_clim": tcc_clim,
    "tp_clim": tp_clim,
})

# anomaly（5-9月异常）
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
ano_ds.attrs["description"] = f"Daily anomaly by MM-DD, from {filter_start} to {filter_end}"

# bandpass（10-30天，61点Lanczos，最终裁剪到6-8月）
bp_ds = xr.Dataset({
    "u_bp": u_bp,
    "v_bp": v_bp,
    "z_bp": z_bp,
    "t_bp": t_bp,
    "w_bp": w_bp,
    "olr_bp": olr_bp,
    "t2m_bp": t2m_bp,
    "ssr_bp": ssr_bp,
    "str_bp": str_bp,
    "slhf_bp": slhf_bp,
    "sshf_bp": sshf_bp,
    "tcc_bp": tcc_bp,
    "tp_bp": tp_bp,
})
bp_ds.attrs["description"] = f"Lanczos bandpass filtered anomaly (10-30 days, nwts=61), from {plot_start} to {plot_end}"

# 文件名
clim_file = f"{OUT_DIR}/ERA5_CPC_daily_clim_{clim_start}_{clim_end}.nc"
ano_file  = f"{OUT_DIR}/ERA5_CPC_daily_ano_{year}_MJJAS.nc"
bp_file   = f"{OUT_DIR}/ERA5_CPC_daily_bp_{year}_JJA_10-30d.nc"

print("写出 climatology...")
clim_ds.to_netcdf(clim_file, encoding=make_encoding(clim_ds, complevel=1))

print("写出 anomaly...")
ano_ds.to_netcdf(ano_file, encoding=make_encoding(ano_ds, complevel=1))

print("写出 bandpass...")
bp_ds.to_netcdf(bp_file, encoding=make_encoding(bp_ds, complevel=1))

print("导出完成：")
print(clim_file)
print(ano_file)
print(bp_file)
