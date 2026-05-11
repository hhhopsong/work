import os
import shutil
import warnings
import builtins

import numpy as np
import xarray as xr
import pandas as pd
import dask

from dask.array import PerformanceWarning
from dask.diagnostics import ProgressBar
from climkit.filter import *


# =========================
# 基础设置
# =========================
warnings.filterwarnings("ignore", category=PerformanceWarning)
dask.config.set({"array.slicing.split_large_chunks": False})

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
OUT_DIR = r"/Volumes/TiPlus7100/p4/data"


# =========================
# 参数区
# =========================
PROCESS_START_YEAR = None
PROCESS_END_YEAR = None

FILTER_PERIOD = [10, 30]
NWTS = 61

FILTER_START_MMDD = "05-01"
FILTER_END_MMDD = "09-30"

SAVE_START_MMDD = "06-01"
SAVE_END_MMDD = "08-31"

# 输出模式：
# "nc"   = 逐变量写 NetCDF
# "zarr" = 逐变量写 Zarr，通常更快
OUTPUT_MODE = "nc"

# NetCDF 压缩等级：
# 0 = 不压缩，最快
# 1 = 轻微压缩，慢一些
NC_COMPLEVEL = 0

# 只测试单个变量时可改成 "olr_bp" / "t2m_bp" / "u_bp"
# 全部导出则保持 None
EXPORT_ONLY_VAR = None

# 导出线程数
NUM_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))


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
    """
    计算阶段 chunk。
    注意：这里仍保持 lat/lon 为中等块，避免计算阶段单块过大。
    导出阶段会单独重分块。
    """
    chunk_map = {}

    if "time" in da.dims:
        chunk_map["time"] = -1 if time_full else 92
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


def daily_clim_by_mmdd(
    da: xr.DataArray,
    time_dim: str = "time",
    drop_feb29: bool = True
) -> xr.DataArray:
    if time_dim not in da.dims:
        raise ValueError(f"{da.name or 'DataArray'} 缺少时间维 {time_dim}。")

    out = da

    if drop_feb29:
        leap_mask = (out[time_dim].dt.month == 2) & (out[time_dim].dt.day == 29)
        out = out.where(~leap_mask, drop=True)

    if out.sizes.get(time_dim, 0) == 0:
        raise ValueError(f"{da.name or 'DataArray'} 去除 2 月 29 日后没有时间点。")

    mmdd = out[time_dim].dt.strftime("%m-%d")
    out = out.assign_coords(mmdd=(time_dim, mmdd.data))
    out = out.groupby("mmdd").mean(time_dim)

    out = smart_chunk(out, time_full=True)
    return out


def anomaly_by_mmdd(
    da: xr.DataArray,
    clim: xr.DataArray,
    start: str,
    end: str
) -> xr.DataArray:
    sub = da.sel(time=slice(start, end))

    ntime = sub.sizes.get("time", 0)
    if ntime == 0:
        raise ValueError(
            f"{da.name or 'DataArray'} 在 {start} 到 {end} 没有数据，无法计算异常。"
        )

    if ntime < NWTS:
        raise ValueError(
            f"{da.name or 'DataArray'} 在 {start} 到 {end} 只有 {ntime} 个时间点，"
            f"小于 NWTS={NWTS}，无法进行 Lanczos 滤波。"
        )

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


def prepare_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def regrid_to_target_grid(
    da: xr.DataArray,
    target_lon: xr.DataArray,
    target_lat: xr.DataArray,
    method: str = "linear"
) -> xr.DataArray:
    da = normalize_lonlat_da(da)

    same_lon = (
        da.lon.size == target_lon.size
        and np.allclose(da.lon.values, target_lon.values)
    )
    same_lat = (
        da.lat.size == target_lat.size
        and np.allclose(da.lat.values, target_lat.values)
    )

    if same_lon and same_lat:
        print(f"{da.name or 'DataArray'} 已经是目标网格，跳过插值。")
        return smart_chunk(da, time_full=True)

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
        msg.append(
            f"lat={da.lat.size}, "
            f"[{float(da.lat.min()):.3f}, {float(da.lat.max()):.3f}]"
        )

    if "lon" in da.coords:
        msg.append(
            f"lon={da.lon.size}, "
            f"[{float(da.lon.min()):.3f}, {float(da.lon.max()):.3f}]"
        )

    if "time" in da.coords:
        time_values = da.time.values

        if np.ndim(time_values) == 0:
            msg.append(
                f"time={pd.to_datetime(time_values).strftime('%Y-%m-%d')}"
            )
        else:
            if len(time_values) > 0:
                msg.append(
                    "time="
                    f"{pd.to_datetime(time_values[0]).strftime('%Y-%m-%d')} to "
                    f"{pd.to_datetime(time_values[-1]).strftime('%Y-%m-%d')}"
                )
            else:
                msg.append("time=empty")

    print(" | ".join(msg))


def print_time_range(name: str, da: xr.DataArray):
    if "time" not in da.coords or da.sizes.get("time", 0) == 0:
        print(f"{name}: 没有 time 坐标或 time 为空")
        return

    t0 = pd.to_datetime(da.time.values[0]).strftime("%Y-%m-%d")
    t1 = pd.to_datetime(da.time.values[-1]).strftime("%Y-%m-%d")
    years = np.unique(da.time.dt.year.values).astype(int)
    print(f"{name}: time {t0} to {t1}, years {years[0]}-{years[-1]}, count={len(years)}")


def optional_year_slice(da: xr.DataArray) -> xr.DataArray:
    start = None
    end = None

    if PROCESS_START_YEAR is not None:
        start = f"{PROCESS_START_YEAR}-01-01"
    if PROCESS_END_YEAR is not None:
        end = f"{PROCESS_END_YEAR}-12-31"

    if start is None and end is None:
        return da

    return da.sel(time=slice(start, end))


def get_filter_years_for_var(
    da: xr.DataArray,
    min_filter_days: int = NWTS
) -> list[int]:
    if "time" not in da.dims:
        raise ValueError(f"{da.name or 'DataArray'} 没有 time 维。")

    if da.sizes.get("time", 0) == 0:
        raise ValueError(f"{da.name or 'DataArray'} time 为空。")

    all_years = np.unique(da.time.dt.year.values).astype(int)

    valid_years = []
    for yy in all_years:
        filter_start = f"{yy}-{FILTER_START_MMDD}"
        filter_end = f"{yy}-{FILTER_END_MMDD}"
        save_start = f"{yy}-{SAVE_START_MMDD}"
        save_end = f"{yy}-{SAVE_END_MMDD}"

        sub_filter = da.sel(time=slice(filter_start, filter_end))
        sub_save = da.sel(time=slice(save_start, save_end))

        n_filter = sub_filter.sizes.get("time", 0)
        n_save = sub_save.sizes.get("time", 0)

        if n_filter >= min_filter_days and n_save > 0:
            valid_years.append(int(yy))
        else:
            print(
                f"跳过 {da.name or 'DataArray'} {yy} 年："
                f"5-9月时间点={n_filter}, 6-8月时间点={n_save}"
            )

    return valid_years


# =========================
# 导出相关函数
# =========================
def clean_da_before_export(da: xr.DataArray) -> xr.DataArray:
    """
    导出前清理不需要写入文件的辅助坐标。
    重点删除 mmdd，避免 object dtype 坐标触发 SerializationWarning。
    """
    da = da.copy()

    drop_coords = []

    for cname in list(da.coords):
        # mmdd 是异常计算时用的辅助坐标，最终结果不需要
        if cname == "mmdd" and cname not in da.dims:
            drop_coords.append(cname)

        # 其他 object 类型的非维度坐标也不建议写入 NetCDF
        elif cname not in da.dims and da.coords[cname].dtype == object:
            drop_coords.append(cname)

    if drop_coords:
        print(f"{da.name}: 导出前删除辅助坐标 {drop_coords}")
        da = da.reset_coords(drop_coords, drop=True)

    return da


def calc_one_var_bp(
    da: xr.DataArray,
    clim: xr.DataArray,
    out_name: str,
) -> xr.DataArray:
    da = smart_chunk(da, time_full=True)
    clim = smart_chunk(clim, time_full=True)

    years_this_var = get_filter_years_for_var(da, min_filter_days=NWTS)

    if len(years_this_var) == 0:
        raise ValueError(f"{out_name} 没有任何可用于 NWTS={NWTS} Lanczos 滤波的年份。")

    print(
        f"{out_name}: 使用自身可用年份 {years_this_var[0]}-{years_this_var[-1]}，"
        f"共 {len(years_this_var)} 年"
    )

    bp_list = []

    for yy in years_this_var:
        print(f"处理 {out_name} {yy} 年...")

        filter_start = f"{yy}-{FILTER_START_MMDD}"
        filter_end = f"{yy}-{FILTER_END_MMDD}"
        save_start = f"{yy}-{SAVE_START_MMDD}"
        save_end = f"{yy}-{SAVE_END_MMDD}"

        ano = anomaly_by_mmdd(da, clim, filter_start, filter_end)

        bp = (
            LanczosFilter(
                ano,
                "bandpass",
                period=FILTER_PERIOD,
                nwts=NWTS
            )
            .filted()
            .sel(time=slice(save_start, save_end))
        )

        bp = bp.rename(out_name)

        # 关键：删除 mmdd 辅助坐标，避免导出时 object dtype 警告
        bp = clean_da_before_export(bp)

        bp = smart_chunk(bp, time_full=False)
        bp_list.append(bp)

    out = xr.concat(bp_list, dim="time")
    out = out.sortby("time")

    # 再清理一次，防止 concat 后又带回辅助坐标
    out = clean_da_before_export(out)

    out = smart_chunk(out, time_full=False)

    out.attrs["description"] = (
        f"{out_name}: daily anomaly Lanczos bandpass filtered result. "
        f"Each year uses May-Sep data for filtering and saves JJA only."
    )
    out.attrs["filter"] = "Lanczos bandpass"
    out.attrs["filter_period_days"] = "10-30"
    out.attrs["filter_nwts"] = NWTS
    out.attrs["filter_input_months"] = "May-September"
    out.attrs["saved_months"] = "June-August"

    return out


def make_export_chunks_for_da(da: xr.DataArray) -> dict:
    """
    导出阶段 chunk。
    关键：
    lat/lon 用整块，避免 lat=(180, 1) 这种坏 chunk。
    """
    chunk_map = {}

    if "time" in da.dims:
        chunk_map["time"] = min(92, da.sizes["time"])
    if "level" in da.dims:
        chunk_map["level"] = 1
    if "lat" in da.dims:
        chunk_map["lat"] = -1
    if "lon" in da.dims:
        chunk_map["lon"] = -1

    return chunk_map


def get_export_chunksizes(ds: xr.Dataset, var: str) -> tuple:
    chunksizes = []

    for dim in ds[var].dims:
        if dim == "time":
            chunksizes.append(min(92, ds.sizes["time"]))
        elif dim == "level":
            chunksizes.append(1)
        elif dim == "lat":
            chunksizes.append(ds.sizes["lat"])
        elif dim == "lon":
            chunksizes.append(ds.sizes["lon"])
        else:
            chunksizes.append(ds.sizes[dim])

    return tuple(chunksizes)


def make_nc_encoding_one_var(ds: xr.Dataset, complevel: int = 0) -> dict:
    encoding = {}

    for var in ds.data_vars:
        enc = {
            "chunksizes": get_export_chunksizes(ds, var),
        }

        if np.issubdtype(ds[var].dtype, np.floating):
            enc["dtype"] = "float32"
            enc["_FillValue"] = np.float32(np.nan)

        if complevel > 0:
            enc["zlib"] = True
            enc["complevel"] = complevel
            enc["shuffle"] = True
        else:
            enc["zlib"] = False

        encoding[var] = enc

    return encoding


def make_zarr_encoding_one_var(ds: xr.Dataset) -> dict:
    encoding = {}

    for var in ds.data_vars:
        enc = {
            "chunks": get_export_chunksizes(ds, var),
        }

        if np.issubdtype(ds[var].dtype, np.floating):
            enc["dtype"] = "float32"
            enc["_FillValue"] = np.float32(np.nan)

        encoding[var] = enc

    return encoding


def print_dask_info_ds(ds: xr.Dataset):
    print("预计逻辑大小 GB:", ds.nbytes / 1024**3)

    for v in ds.data_vars:
        data = ds[v].data
        if hasattr(data, "npartitions"):
            print(v, "npartitions =", data.npartitions, "chunks =", ds[v].chunks)
        else:
            print(v, "not dask array")


# =========================
# 单文件 NetCDF 流式写出函数
# 带 level 的变量会保留为一个 4D nc 文件: level, time, lat, lon
# =========================

SKIP_EXISTING = True

# 每次写多少天。92 天大约等于一个 JJA。
STREAM_TIME_CHUNK = 92

# 写 4D 变量时建议小一点，避免内存爆掉。
NUM_WORKERS_STREAM = 2

# 如果还是被系统杀掉，就改成 1。
# NUM_WORKERS_STREAM = 1


def _safe_set_attrs(ncobj, attrs: dict):
    """
    安全写入 NetCDF 属性。
    某些 Python 对象类型不能直接作为 NetCDF attr，这里转成字符串。

    注意：这里使用 builtins.str，而不是 str。
    因为 ERA5 single level 变量里有一个变量名叫 str，脚本中如果误用 str 作为变量名，
    会覆盖 Python 内置 str 类型，导致 isinstance 和 str(value) 报错。
    """
    skip_keys = {"_FillValue", "missing_value", "scale_factor", "add_offset"}

    for key, value in attrs.items():
        if key in skip_keys:
            continue

        try:
            if value is None:
                continue
            elif isinstance(value, (builtins.str, int, float, np.integer, np.floating)):
                ncobj.setncattr(key, value)
            elif isinstance(value, (list, tuple)):
                ncobj.setncattr(key, ", ".join(map(builtins.str, value)))
            else:
                ncobj.setncattr(key, builtins.str(value))
        except Exception:
            ncobj.setncattr(key, builtins.str(value))


def _expected_nc_shape(da: xr.DataArray) -> tuple:
    """
    返回写入文件后的变量 shape。
    有 level: (level, time, lat, lon)
    无 level: (time, lat, lon)
    """
    if "level" in da.dims:
        return (
            da.sizes["level"],
            da.sizes["time"],
            da.sizes["lat"],
            da.sizes["lon"],
        )

    return (
        da.sizes["time"],
        da.sizes["lat"],
        da.sizes["lon"],
    )


def _nc_file_is_valid(path: str, var_name: str, expected_shape: tuple) -> bool:
    """
    判断已有 nc 文件是否完整可用。
    不只看文件大小，而是实际打开检查变量 shape。
    """
    if not os.path.exists(path):
        return False

    try:
        from netCDF4 import Dataset

        with Dataset(path, "r") as nc:
            if var_name not in nc.variables:
                return False
            if tuple(nc.variables[var_name].shape) != tuple(expected_shape):
                return False

        return True

    except Exception:
        return False


def _write_time_coord(nc, time_values):
    """
    写 time 坐标，使用 CF convention。
    xarray 重新 open_dataset 时可以自动 decode 成 datetime64。
    """
    from netCDF4 import date2num

    time_var = nc.createVariable("time", "f8", ("time",))

    units = "days since 1900-01-01 00:00:00"
    calendar = "proleptic_gregorian"

    time_pd = pd.to_datetime(time_values)
    time_py = time_pd.to_pydatetime()

    time_var[:] = date2num(time_py, units=units, calendar=calendar)
    time_var.units = units
    time_var.calendar = calendar
    time_var.standard_name = "time"
    time_var.axis = "T"


def _write_basic_coords(nc, da: xr.DataArray):
    """
    写入 time / level / lat / lon 坐标。
    """
    # time
    _write_time_coord(nc, da["time"].values)

    # lat
    lat_var = nc.createVariable("lat", "f8", ("lat",))
    lat_var[:] = da["lat"].values
    lat_var.units = "degrees_north"
    lat_var.standard_name = "latitude"
    lat_var.long_name = "latitude"
    lat_var.axis = "Y"

    # lon
    lon_var = nc.createVariable("lon", "f8", ("lon",))
    lon_var[:] = da["lon"].values
    lon_var.units = "degrees_east"
    lon_var.standard_name = "longitude"
    lon_var.long_name = "longitude"
    lon_var.axis = "X"

    # level
    if "level" in da.dims:
        level_values = da["level"].values

        if np.issubdtype(level_values.dtype, np.integer):
            level_dtype = "i4"
        else:
            level_dtype = "f8"

        lev_var = nc.createVariable("level", level_dtype, ("level",))
        lev_var[:] = level_values
        lev_var.long_name = "pressure_level"
        lev_var.units = "hPa"
        lev_var.positive = "down"
        lev_var.axis = "Z"


def _prepare_da_for_stream_export(da: xr.DataArray) -> xr.DataArray:
    """
    清理并统一维度顺序。
    有 level: level, time, lat, lon
    无 level: time, lat, lon
    """
    da = clean_da_before_export(da)

    required = ["time", "lat", "lon"]
    for dim in required:
        if dim not in da.dims:
            raise ValueError(f"{da.name} 缺少必要维度 {dim}，当前 dims={da.dims}")

    if "mmdd" in da.coords and "mmdd" not in da.dims:
        da = da.reset_coords("mmdd", drop=True)

    if "level" in da.dims:
        da = da.transpose("level", "time", "lat", "lon")
        da = da.chunk({
            "level": 1,
            "time": STREAM_TIME_CHUNK,
            "lat": -1,
            "lon": -1,
        })
    else:
        da = da.transpose("time", "lat", "lon")
        da = da.chunk({
            "time": STREAM_TIME_CHUNK,
            "lat": -1,
            "lon": -1,
        })

    return da


def export_one_var_nc_stream_single_file(
    da: xr.DataArray,
    out_dir: str,
    complevel: int = 0,
):
    """
    核心导出函数。

    特点：
    1. 所有 level 保存在同一个 nc 文件。
    2. 变量维度为 level, time, lat, lon。
    3. 按 level + time chunk 流式计算和写入，避免一次性爆内存。
    4. 不依赖 xarray.to_netcdf 写 4D 大变量。
    """
    from netCDF4 import Dataset

    name = da.name
    out_file = f"{out_dir}/{name}_JJA_10-30d.nc"
    tmp_file = f"{out_file}.tmp"

    print("=" * 100)
    print(f"开始流式写出 {name} -> 单个 NetCDF")
    print(f"输出文件: {out_file}")

    da = _prepare_da_for_stream_export(da)
    expected_shape = _expected_nc_shape(da)

    print(da)
    print(f"{name} 预计逻辑大小 GB: {da.nbytes / 1024**3:.3f}")
    print(f"{name} 输出 shape: {expected_shape}")
    print(f"{name} chunks: {da.chunks}")

    if SKIP_EXISTING and _nc_file_is_valid(out_file, name, expected_shape):
        size_gb = os.path.getsize(out_file) / 1024**3
        print(f"文件已存在且 shape 正确，跳过: {out_file}, size={size_gb:.2f} GB")
        return

    if os.path.exists(out_file):
        print(f"删除已有不完整文件: {out_file}")
        os.remove(out_file)

    if os.path.exists(tmp_file):
        print(f"删除已有临时文件: {tmp_file}")
        os.remove(tmp_file)

    has_level = "level" in da.dims
    ntime = da.sizes["time"]
    nlat = da.sizes["lat"]
    nlon = da.sizes["lon"]

    if has_level:
        nlev = da.sizes["level"]
        var_dims = ("level", "time", "lat", "lon")
        chunksizes = (1, min(STREAM_TIME_CHUNK, ntime), nlat, nlon)
    else:
        nlev = None
        var_dims = ("time", "lat", "lon")
        chunksizes = (min(STREAM_TIME_CHUNK, ntime), nlat, nlon)

    print(f"{name}: chunksizes in nc = {chunksizes}")
    print(f"{name}: complevel={complevel}, num_workers={NUM_WORKERS_STREAM}")

    with Dataset(tmp_file, "w", format="NETCDF4") as nc:
        # -------------------------
        # 创建维度
        # -------------------------
        if has_level:
            nc.createDimension("level", nlev)

        nc.createDimension("time", ntime)
        nc.createDimension("lat", nlat)
        nc.createDimension("lon", nlon)

        # -------------------------
        # 写坐标
        # -------------------------
        _write_basic_coords(nc, da)

        # -------------------------
        # 全局属性
        # -------------------------
        nc.description = (
            "Single-variable export. "
            "Daily anomaly Lanczos bandpass filtered result. "
            "May-Sep data are used for filtering and JJA data are saved."
        )
        nc.filter = "Lanczos bandpass, 10-30 days, nwts=61"
        nc.filter_period_days = "10-30"
        nc.filter_nwts = NWTS
        nc.filter_input_months = "May-September"
        nc.saved_months = "June-August"
        nc.output_note = (
            "Written by streaming level/time chunks into one NetCDF file "
            "to avoid high memory use."
        )

        if PROCESS_START_YEAR is None and PROCESS_END_YEAR is None:
            nc.processing_year_range = "all available years for each variable"
        else:
            nc.processing_year_range = f"{PROCESS_START_YEAR}-{PROCESS_END_YEAR}"

        # -------------------------
        # 创建数据变量
        # -------------------------
        zlib_flag = complevel > 0

        out_var = nc.createVariable(
            name,
            "f4",
            var_dims,
            zlib=zlib_flag,
            complevel=complevel if zlib_flag else 0,
            shuffle=zlib_flag,
            chunksizes=chunksizes,
            fill_value=np.float32(np.nan),
        )

        _safe_set_attrs(out_var, da.attrs)

        out_var.long_name = name
        out_var.coordinates = " ".join(var_dims)

        # -------------------------
        # 开始流式写入
        # -------------------------
        if has_level:
            level_values = da["level"].values

            for ilev, lev in enumerate(level_values):
                print("-" * 100)
                print(f"{name}: 写入 level={lev} ({ilev + 1}/{nlev})")

                da_lev = da.isel(level=ilev)

                for t0 in range(0, ntime, STREAM_TIME_CHUNK):
                    t1 = min(t0 + STREAM_TIME_CHUNK, ntime)

                    print(
                        f"{name}: level={lev}, "
                        f"time {t0}:{t1} / {ntime}"
                    )

                    block = da_lev.isel(time=slice(t0, t1))

                    with dask.config.set(
                        scheduler="threads",
                        num_workers=NUM_WORKERS_STREAM
                    ):
                        arr = block.astype("float32").compute().values

                    out_var[ilev, t0:t1, :, :] = arr

                    del arr

        else:
            for t0 in range(0, ntime, STREAM_TIME_CHUNK):
                t1 = min(t0 + STREAM_TIME_CHUNK, ntime)

                print(f"{name}: time {t0}:{t1} / {ntime}")

                block = da.isel(time=slice(t0, t1))

                with dask.config.set(
                    scheduler="threads",
                    num_workers=NUM_WORKERS_STREAM
                ):
                    arr = block.astype("float32").compute().values

                out_var[t0:t1, :, :] = arr

                del arr

    os.replace(tmp_file, out_file)

    size_gb = os.path.getsize(out_file) / 1024**3
    print(f"完成写出 {name}: {out_file}")
    print(f"文件大小: {size_gb:.2f} GB")


def export_one_var_nc(
    da: xr.DataArray,
    out_dir: str,
    complevel: int = 0,
):
    """
    替代原 export_one_var_nc。
    有 level 的变量不会再拆成多个文件，而是写成一个 4D nc。
    """
    export_one_var_nc_stream_single_file(
        da=da,
        out_dir=out_dir,
        complevel=complevel,
    )


def export_one_var(da: xr.DataArray, out_dir: str):
    """
    替代原 export_one_var。
    这里专注 NetCDF 单文件流式写出。
    """
    if OUTPUT_MODE != "nc":
        raise ValueError(
            "当前这套函数是为了写单个 NetCDF 文件。请设置 OUTPUT_MODE = 'nc'。"
        )

    export_one_var_nc(
        da=da,
        out_dir=out_dir,
        complevel=NC_COMPLEVEL,
    )

def output_file_exists_quick(out_dir: str, var_name: str) -> bool:
    """
    如果输出 nc 已经存在，并且里面有对应变量，则认为已经生成过，直接跳过。
    这里不强制检查 shape，目的是避免为了检查 shape 先触发大量计算。
    """
    if not SKIP_EXISTING:
        return False

    path = f"{out_dir}/{var_name}_JJA_10-30d.nc"

    if not os.path.exists(path):
        return False

    try:
        from netCDF4 import Dataset

        with Dataset(path, "r") as nc:
            if var_name not in nc.variables:
                print(f"已有文件缺少变量 {var_name}，将重新生成: {path}")
                return False

            var = nc.variables[var_name]

            required_dims = {"time", "lat", "lon"}
            if not required_dims.issubset(set(var.dimensions)):
                print(f"已有文件维度不完整，将重新生成: {path}, dims={var.dimensions}")
                return False

            if var.size == 0:
                print(f"已有文件变量为空，将重新生成: {path}")
                return False

        size_gb = os.path.getsize(path) / 1024**3
        print(f"{var_name}: 已存在输出文件，跳过计算和导出: {path}, size={size_gb:.2f} GB")
        return True

    except Exception as e:
        print(f"{var_name}: 已有文件无法打开或检查失败，将重新生成: {path}")
        print(f"原因: {e}")
        return False


# =========================
# 读取 ERA5 主数据
# =========================
print("读取 ERA5 主数据...")

ds_era5 = xr.open_zarr(
    r"/Volumes/TiPlus7100/p4/data/ERA5_daily_uvwztq_sum.zarr",
    chunks="auto"
)

ds_era5 = standardize_time_ds(ds_era5)
ds_era5 = normalize_lonlat_ds(ds_era5)

required_vars = ["u", "v", "z", "t", "w"]
missing_vars = [v for v in required_vars if v not in ds_era5.data_vars]
if missing_vars:
    raise ValueError(f"ERA5 主数据中缺少变量: {missing_vars}；当前变量有: {list(ds_era5.data_vars)}")

ds_era5 = ds_era5[required_vars]

chunk_map_ds = {}
if "time" in ds_era5.dims:
    chunk_map_ds["time"] = -1
if "level" in ds_era5.dims:
    chunk_map_ds["level"] = 1
if "lat" in ds_era5.dims:
    chunk_map_ds["lat"] = 45
if "lon" in ds_era5.dims:
    chunk_map_ds["lon"] = 90

ds_era5 = ds_era5.chunk(chunk_map_ds)

u = optional_year_slice(ds_era5["u"])
v = optional_year_slice(ds_era5["v"])
z = optional_year_slice(ds_era5["z"])
t = optional_year_slice(ds_era5["t"])
w = optional_year_slice(ds_era5["w"])

era5_lon = ds_era5["lon"]
era5_lat = ds_era5["lat"]

print_grid_info("u", u.isel(time=0, level=0))
print_time_range("u", u)
print_time_range("v", v)
print_time_range("z", z)
print_time_range("t", t)
print_time_range("w", w)


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
olr = optional_year_slice(olr)

print_grid_info("olr_raw", olr.isel(time=0))
print_time_range("olr", olr)

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
t2m = optional_year_slice(t2m)

print_grid_info("t2m_raw", t2m.isel(time=0))
print_time_range("t2m", t2m)

print("将 T2M 插值到 ERA5 网格...")
t2m = regrid_to_target_grid(t2m, era5_lon, era5_lat, method="linear")
print_grid_info("t2m_regridded", t2m.isel(time=0))


# =========================
# 读取 ssr str slhf sshf tcc tp，并插值到 ERA5 网格
# =========================
print("读取 single level 数据...")

ds_sl = xr.open_zarr(
    r"/Volumes/TiPlus7100/p4/data/single_level_sum.zarr",
    chunks="auto"
)

ds_sl = standardize_time_ds(ds_sl)
ds_sl = normalize_lonlat_ds(ds_sl)

required_vars = ["ssr", "str", "slhf", "sshf", "tcc", "tp", "ttr", "tsr"]
missing_vars = [v for v in required_vars if v not in ds_sl.data_vars]
if missing_vars:
    raise ValueError(f"single level 数据中缺少变量: {missing_vars}；当前变量有: {list(ds_sl.data_vars)}")

ds_sl = ds_sl[required_vars]

chunk_map_ds = {}
if "time" in ds_sl.dims:
    chunk_map_ds["time"] = -1
if "lat" in ds_sl.dims:
    chunk_map_ds["lat"] = 45
if "lon" in ds_sl.dims:
    chunk_map_ds["lon"] = 90

ds_sl = ds_sl.chunk(chunk_map_ds)

ssr = optional_year_slice(ds_sl["ssr"])
str_da = optional_year_slice(ds_sl["str"])
slhf = optional_year_slice(ds_sl["slhf"])
sshf = optional_year_slice(ds_sl["sshf"])
tcc = optional_year_slice(ds_sl["tcc"])
tp = optional_year_slice(ds_sl["tp"])
ttr = optional_year_slice(ds_sl["ttr"])
tsr = optional_year_slice(ds_sl["tsr"])


print_time_range("ssr", ssr)
print_time_range("str", str_da)
print_time_range("slhf", slhf)
print_time_range("sshf", sshf)
print_time_range("tcc", tcc)
print_time_range("tp", tp)
print_time_range("ttr", ttr)
print_time_range("tsr", tsr)


print("将 single level 插值到 ERA5 网格...")
ssr = regrid_to_target_grid(ssr, era5_lon, era5_lat, method="linear")
str_da = regrid_to_target_grid(str_da, era5_lon, era5_lat, method="linear")
slhf = regrid_to_target_grid(slhf, era5_lon, era5_lat, method="linear")
sshf = regrid_to_target_grid(sshf, era5_lon, era5_lat, method="linear")
tcc = regrid_to_target_grid(tcc, era5_lon, era5_lat, method="linear")
tp = regrid_to_target_grid(tp, era5_lon, era5_lat, method="linear")
ttr = regrid_to_target_grid(ttr, era5_lon, era5_lat, method="linear")
tsr = regrid_to_target_grid(tsr, era5_lon, era5_lat, method="linear")


print_grid_info("single level_regridded", ssr.isel(time=0))


# =========================
# 按变量逐个计算 + 导出
# 已存在输出文件的变量会在计算前直接跳过
# =========================
print("准备逐变量计算和导出...")
prepare_output_dir(OUT_DIR)

var_tasks = [
    ("olr_bp", olr),
    ("t2m_bp", t2m),

    ("ssr_bp", ssr),
    ("str_bp", str_da),
    ("slhf_bp", slhf),
    ("sshf_bp", sshf),
    ("tcc_bp", tcc),
    ("tp_bp", tp),
    ("ttr_bp", ttr),
    ("tsr_bp", tsr),

    ("u_bp", u),
    ("v_bp", v),
    ("z_bp", z),
    ("t_bp", t),
    ("w_bp", w),
]

if EXPORT_ONLY_VAR is not None:
    var_tasks = [(name, da) for name, da in var_tasks if name == EXPORT_ONLY_VAR]
    if len(var_tasks) == 0:
        raise ValueError(f"EXPORT_ONLY_VAR={EXPORT_ONLY_VAR} 没有匹配到任何变量。")

print("将处理以下变量：")
for name, _ in var_tasks:
    print(" -", name)

# 用 u 的水平网格作为参考
ref_grid = u.isel(time=0, level=0)

for out_name, da in var_tasks:
    print("=" * 100)
    print(f"准备处理 {out_name}")

    # 关键：文件已经生成过则直接跳过，不计算 clim / anomaly / filter
    if output_file_exists_quick(OUT_DIR, out_name):
        continue

    print(f"{out_name}: 输出不存在或不完整，开始计算。")

    # 水平网格检查
    if "level" in da.dims:
        check_da = da.isel(time=0, level=0)
    else:
        check_da = da.isel(time=0)

    assert_same_horizontal_grid(ref_grid, check_da, out_name)

    # 计算该变量自身气候态
    print(f"{out_name}: 计算 MM-DD 气候态...")
    clim = daily_clim_by_mmdd(da)

    # 计算滤波结果
    print(f"{out_name}: 计算 JJA 10-30 天带通结果...")
    bp = calc_one_var_bp(da, clim, out_name)

    # 导出
    print(f"{out_name}: 开始导出...")
    export_one_var(bp, OUT_DIR)

    # 主动释放引用
    del clim
    del bp

print("全部变量处理完成。")
