import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.units import units

from climkit.masked import masked
from climkit.filter import LanczosFilter

# -----------------------------
# Matplotlib settings
# -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

# -----------------------------
# Paths
# -----------------------------
PYFILE = r"/Volumes/TiPlus7100/PyFile"
DATA = r"/Volumes/TiPlus7100/data"

NC_FILE = r"/Volumes/TiPlus7100/p4/data/ERA5_daily_uvwztq_sum.zarr"
YANGTZE_SHP = fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"
OUT_DIR = fr"{PYFILE}/p4/pic"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG = os.path.join(OUT_DIR, "Temperature_Budget_Daily_4lines_2015_JJA_Yangtze_925hPa")

# -----------------------------
# Basic settings
# -----------------------------
CLIM_START = "1961-06-01"
CLIM_END   = "2022-08-31"

TARGET_START = "2015-06-01"
TARGET_END   = "2015-08-31"

# 为了适配 nwts=61，滤波计算扩展到 5–9 月
FILTER_START = "2015-05-01"
FILTER_END   = "2015-09-30"

LEVEL_SEL = 925

EVENT1_START = "2015-07-01"
EVENT1_END   = "2015-07-11"

EVENT2_START = "2015-07-15"
EVENT2_END   = "2015-07-26"


# =========================================================
# Utilities
# =========================================================
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


def standardize_time(ds: xr.Dataset) -> xr.Dataset:
    """统一时间坐标为 time"""
    if "time" in ds.coords:
        return ds
    if "valid_time" in ds.coords:
        return ds.rename({"valid_time": "time"})
    raise ValueError("数据中没有 time 或 valid_time 坐标。")


def daily_clim_by_mmdd(da: xr.DataArray, time_dim: str = "time", drop_feb29: bool = True) -> xr.DataArray:
    """按 MM-DD 计算逐日气候态，避免闰年 dayofyear 错位。"""
    if time_dim not in da.dims:
        raise ValueError(f"{da.name or 'DataArray'} 缺少时间维 {time_dim}。")

    out = da
    if drop_feb29:
        leap_mask = (out[time_dim].dt.month == 2) & (out[time_dim].dt.day == 29)
        out = out.where(~leap_mask, drop=True)

    mmdd = out[time_dim].dt.strftime("%m-%d")
    out = out.assign_coords(mmdd=(time_dim, mmdd.data))
    return out.groupby("mmdd").mean(time_dim)


def clim_to_time(clim: xr.DataArray, time_coord: xr.DataArray) -> xr.DataArray:
    """把按 mmdd 的逐日气候态映射回具体 time 维"""
    mmdd_indexer = xr.DataArray(
        pd.to_datetime(time_coord.values).strftime("%m-%d"),
        coords={"time": time_coord.values},
        dims="time"
    )

    out = clim.sel(mmdd=mmdd_indexer)
    out = out.assign_coords(time=time_coord.values)

    if "mmdd" in out.coords:
        out = out.drop_vars("mmdd")

    return out


def anomaly_by_mmdd(da: xr.DataArray, clim: xr.DataArray, start: str, end: str) -> xr.DataArray:
    """相对 MM-DD 逐日气候态的异常"""
    sub = da.sel(time=slice(start, end))
    mmdd_indexer = xr.DataArray(
        sub.time.dt.strftime("%m-%d").values,
        coords={"time": sub.time},
        dims="time"
    )
    clim_on_time = clim.sel(mmdd=mmdd_indexer)
    return sub - clim_on_time


def region_mean_series(da: xr.DataArray, shp_path: str) -> xr.DataArray:
    """
    对每个时次做 shp mask 后求区域平均
    输入 da 维度应至少包含 (time, lat, lon)
    """
    vals = []
    for i in range(da.sizes["time"]):
        da_clip = masked(da.isel(time=i), shp_path)
        vals.append(da_clip.mean(dim=("lat", "lon"), skipna=True))
    return xr.concat(vals, dim="time").assign_coords(time=da["time"])


def calc_horizontal_advection_2d(T2d: xr.DataArray, u2d: xr.DataArray, v2d: xr.DataArray) -> xr.DataArray:
    """
    计算单层二维水平温度平流
    返回单位：K/s
    输入 dims = (lat, lon)
    """
    T2d = T2d.copy()
    u2d = u2d.copy()
    v2d = v2d.copy()

    if "units" not in T2d.attrs:
        T2d.attrs["units"] = "K"
    if "units" not in u2d.attrs:
        u2d.attrs["units"] = "m/s"
    if "units" not in v2d.attrs:
        v2d.attrs["units"] = "m/s"

    Tq = T2d.metpy.quantify()
    uq = u2d.metpy.quantify()
    vq = v2d.metpy.quantify()

    lons = T2d["lon"].values
    lats = T2d["lat"].values
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

    adv = mpcalc.advection(
        Tq,
        u=uq,
        v=vq,
        dx=dx,
        dy=dy,
        x_dim=-1,
        y_dim=-2
    )

    out = xr.DataArray(
        adv.metpy.dequantify().values,
        coords=T2d.coords,
        dims=T2d.dims,
        name="adv"
    )
    out.attrs["units"] = "K/s"
    return out


def calc_daily_temperature_budget(t_da: xr.DataArray,
                                  u_da: xr.DataArray,
                                  v_da: xr.DataArray,
                                  w_da: xr.DataArray) -> xr.Dataset:
    """
    计算逐日温度方程各项：
    dTdt, adv_T, ver, Q, sigma

    输入维度必须为:
        (time, level, lat, lon)

    t_da: 温度, K
    u_da: 纬向风, m/s
    v_da: 经向风, m/s
    w_da: 压力坐标垂直速度 omega, Pa/s
    """
    from metpy.constants import dry_air_gas_constant as R
    from metpy.constants import dry_air_spec_heat_press as cp

    t_da = t_da.copy()
    u_da = u_da.copy()
    v_da = v_da.copy()
    w_da = w_da.copy()

    if "units" not in t_da.attrs:
        t_da.attrs["units"] = "K"
    if "units" not in u_da.attrs:
        u_da.attrs["units"] = "m/s"
    if "units" not in v_da.attrs:
        v_da.attrs["units"] = "m/s"
    if "units" not in w_da.attrs:
        w_da.attrs["units"] = "Pa/s"

    t_q = t_da.metpy.quantify()
    u_q = u_da.metpy.quantify()
    v_q = v_da.metpy.quantify()
    w_q = w_da.metpy.quantify()

    times = t_da["time"].values
    levs_hpa = t_da["level"].values
    lats = t_da["lat"].values
    lons = t_da["lon"].values

    nt = len(times)
    nz = len(levs_hpa)
    ny = len(lats)
    nx = len(lons)

    p_pa = (levs_hpa * 100.0) * units.Pa
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

    dTdt_arr = np.full((nt, nz, ny, nx), np.nan)
    adv_arr = np.full((nt, nz, ny, nx), np.nan)
    ver_arr = np.full((nt, nz, ny, nx), np.nan)
    Q_arr = np.full((nt, nz, ny, nx), np.nan)
    sigma_arr = np.full((nt, nz, ny, nx), np.nan)

    for i in range(nt):
        # 1) 时间差分 dTdt
        if i == 0:
            dt_seconds = (times[i + 1] - times[i]) / np.timedelta64(1, "s")
            dTdt = (t_q.isel(time=i + 1) - t_q.isel(time=i)) / (dt_seconds * units.s)
        elif i == nt - 1:
            dt_seconds = (times[i] - times[i - 1]) / np.timedelta64(1, "s")
            dTdt = (t_q.isel(time=i) - t_q.isel(time=i - 1)) / (dt_seconds * units.s)
        else:
            dt_seconds = (times[i + 1] - times[i - 1]) / np.timedelta64(1, "s")
            dTdt = (t_q.isel(time=i + 1) - t_q.isel(time=i - 1)) / (dt_seconds * units.s)

        # 2) 水平温度平流 adv_T
        adv_each_level = []
        for k in range(nz):
            adv_k = mpcalc.advection(
                t_q.isel(time=i, level=k),
                u=u_q.isel(time=i, level=k),
                v=v_q.isel(time=i, level=k),
                dx=dx,
                dy=dy,
                x_dim=-1,
                y_dim=-2
            )
            adv_each_level.append(adv_k)

        adv_T = xr.concat(
            [
                xr.DataArray(
                    a.metpy.dequantify().values,
                    coords={"lat": lats, "lon": lons},
                    dims=("lat", "lon")
                )
                for a in adv_each_level
            ],
            dim="level"
        )
        adv_T = adv_T.assign_coords(level=levs_hpa)
        adv_T = adv_T * units("K/s")

        # 3) sigma
        T_now = t_q.isel(time=i)
        T_vals = T_now.metpy.dequantify().values
        dTdp_vals = np.gradient(T_vals, p_pa.magnitude, axis=0)

        p_3d = p_pa.magnitude[:, None, None]
        term1 = (R.magnitude * T_vals) / (cp.magnitude * p_3d)

        sigma_vals = term1 - dTdp_vals
        sigma = xr.DataArray(
            sigma_vals,
            coords=T_now.coords,
            dims=T_now.dims
        ) * units("K/Pa")

        # 4) 垂直运动项
        ver = w_q.isel(time=i) * sigma

        # 5) 非绝热加热
        Q = dTdt - adv_T - ver

        dTdt_arr[i] = dTdt.metpy.dequantify().values
        adv_arr[i] = adv_T.metpy.dequantify().values
        ver_arr[i] = ver.metpy.dequantify().values
        Q_arr[i] = Q.metpy.dequantify().values
        sigma_arr[i] = sigma.metpy.dequantify().values

    coords = {
        "time": times,
        "level": levs_hpa,
        "lat": lats,
        "lon": lons
    }

    ds_out = xr.Dataset(
        {
            "dTdt": xr.DataArray(dTdt_arr, coords=coords, dims=("time", "level", "lat", "lon")),
            "adv_T": xr.DataArray(adv_arr, coords=coords, dims=("time", "level", "lat", "lon")),
            "ver": xr.DataArray(ver_arr, coords=coords, dims=("time", "level", "lat", "lon")),
            "Q": xr.DataArray(Q_arr, coords=coords, dims=("time", "level", "lat", "lon")),
            "sigma": xr.DataArray(sigma_arr, coords=coords, dims=("time", "level", "lat", "lon")),
        }
    )

    ds_out["dTdt"].attrs["units"] = "K/s"
    ds_out["adv_T"].attrs["units"] = "K/s"
    ds_out["ver"].attrs["units"] = "K/s"
    ds_out["Q"].attrs["units"] = "K/s"
    ds_out["sigma"].attrs["units"] = "K/Pa"

    return ds_out


def decompose_temperature_advection(u: xr.DataArray,
                                    v: xr.DataArray,
                                    t: xr.DataArray,
                                    u_clim: xr.DataArray,
                                    v_clim: xr.DataArray,
                                    t_clim: xr.DataArray,
                                    start: str,
                                    end: str,
                                    level=None) -> xr.Dataset:
    """
    水平温度平流分解:
    total = -(u·∇T)

    分解为:
    adv_clim = -(uc · ∇Tc)
    adv_a_wind_on_climT = -(u' · ∇Tc)
    adv_clim_wind_on_aT = -(uc · ∇T')
    adv_nonlinear = -(u' · ∇T')
    """
    u_sub = u.sel(time=slice(start, end))
    v_sub = v.sel(time=slice(start, end))
    t_sub = t.sel(time=slice(start, end))

    if level is not None:
        u_sub = u_sub.sel(level=level)
        v_sub = v_sub.sel(level=level)
        t_sub = t_sub.sel(level=level)

        uc_base = u_clim.sel(level=level)
        vc_base = v_clim.sel(level=level)
        Tc_base = t_clim.sel(level=level)
    else:
        uc_base = u_clim
        vc_base = v_clim
        Tc_base = t_clim

    uc = clim_to_time(uc_base, u_sub.time)
    vc = clim_to_time(vc_base, v_sub.time)
    Tc = clim_to_time(Tc_base, t_sub.time)

    ua = u_sub - uc
    va = v_sub - vc
    Ta = t_sub - Tc

    total_list = []
    clim_list = []
    awind_climT_list = []
    climwind_aT_list = []
    nl_list = []

    for i in range(u_sub.sizes["time"]):
        total_i = calc_horizontal_advection_2d(
            t_sub.isel(time=i), u_sub.isel(time=i), v_sub.isel(time=i)
        )
        clim_i = calc_horizontal_advection_2d(
            Tc.isel(time=i), uc.isel(time=i), vc.isel(time=i)
        )
        awind_climT_i = calc_horizontal_advection_2d(
            Tc.isel(time=i), ua.isel(time=i), va.isel(time=i)
        )
        climwind_aT_i = calc_horizontal_advection_2d(
            Ta.isel(time=i), uc.isel(time=i), vc.isel(time=i)
        )
        nl_i = calc_horizontal_advection_2d(
            Ta.isel(time=i), ua.isel(time=i), va.isel(time=i)
        )

        total_list.append(total_i)
        clim_list.append(clim_i)
        awind_climT_list.append(awind_climT_i)
        climwind_aT_list.append(climwind_aT_i)
        nl_list.append(nl_i)

    adv_total = xr.concat(total_list, dim="time").assign_coords(time=u_sub.time)
    adv_clim = xr.concat(clim_list, dim="time").assign_coords(time=u_sub.time)
    adv_a_wind_on_climT = xr.concat(awind_climT_list, dim="time").assign_coords(time=u_sub.time)
    adv_clim_wind_on_aT = xr.concat(climwind_aT_list, dim="time").assign_coords(time=u_sub.time)
    adv_nonlinear = xr.concat(nl_list, dim="time").assign_coords(time=u_sub.time)

    adv_anom = adv_total - adv_clim

    ds_out = xr.Dataset({
        "adv_total": adv_total,
        "adv_clim": adv_clim,
        "adv_anom": adv_anom,
        "adv_a_wind_on_climT": adv_a_wind_on_climT,
        "adv_clim_wind_on_aT": adv_clim_wind_on_aT,
        "adv_nonlinear": adv_nonlinear
    })

    for vname in ds_out.data_vars:
        ds_out[vname].attrs["units"] = "K/s"

    return ds_out


def build_summer_day_index(dt_index: pd.DatetimeIndex) -> np.ndarray:
    """6/1 -> 1, 8/31 -> 92"""
    out = []
    for tt in dt_index:
        if tt.month == 6:
            out.append(tt.day)
        elif tt.month == 7:
            out.append(30 + tt.day)
        elif tt.month == 8:
            out.append(61 + tt.day)
        else:
            out.append(np.nan)
    return np.array(out)


def date_to_summer_day(date_str: str) -> int:
    tt = pd.to_datetime(date_str)
    if tt.month == 6:
        return tt.day
    elif tt.month == 7:
        return 30 + tt.day
    elif tt.month == 8:
        return 61 + tt.day
    else:
        raise ValueError("Only summer months are supported.")


# =========================================================
# Main
# =========================================================
print("1) 读取数据...")
ds = xr.open_dataset(NC_FILE)
ds = standardize_latlon(ds)
ds = standardize_time(ds)

required_vars = ["u", "v", "z", "t", "w"]
missing_vars = [v for v in required_vars if v not in ds.data_vars]
if missing_vars:
    raise ValueError(f"数据中缺少变量: {missing_vars}；当前变量有: {list(ds.data_vars)}")

print("2) 选取 1961-2022 夏季资料...")
ds = ds[required_vars].sel(time=slice(CLIM_START, CLIM_END))

u = ds["u"]
v = ds["v"]
z = ds["z"]
t = ds["t"]
w = ds["w"]

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
YEAR = [2015]

# =========================
# 为了适配 nwts=61 的 Lanczos 滤波，
# 先取 5–9 月做异常和滤波，
# 再裁剪回 6–8 月，保证 6–8 月结果完整
# =========================
analysis_start = TARGET_START
analysis_end   = TARGET_END

print("3) 读取 2015 年异常场（5–9 月）...")
ANO = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_ano_2015_MJJAS.nc")
u_ano_full = ANO['u_ano']
v_ano_full = ANO['v_ano']
z_ano_full = ANO['z_ano']
t_ano_full = ANO['t_ano']
w_ano_full = ANO['w_ano']
olr_ano_full = ANO['olr_ano']
t2m_ano_full = ANO['t2m_ano']

print("4) 读取 2015 年带通滤波场（5–9 月）...")
BP = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_bp_2015_JJA_10-30d.nc")
u_bp_full = BP['u_bp']
v_bp_full = BP['v_bp']
z_bp_full = BP['z_bp']
t_bp_full = BP['t_bp']
w_bp_full = BP['w_bp']
olr_bp_full = BP['olr_bp']
t2m_bp_full = BP['t2m_bp']

print("5) 将异常场和带通场裁回 6–8 月备用...")
u_ano = u_ano_full.sel(time=slice(analysis_start, analysis_end))
v_ano = v_ano_full.sel(time=slice(analysis_start, analysis_end))
z_ano = z_ano_full.sel(time=slice(analysis_start, analysis_end))
t_ano = t_ano_full.sel(time=slice(analysis_start, analysis_end))
w_ano = w_ano_full.sel(time=slice(analysis_start, analysis_end))
olr_ano = olr_ano_full.sel(time=slice(analysis_start, analysis_end))
t2m_ano = t2m_ano_full.sel(time=slice(analysis_start, analysis_end))

u_bp = u_bp_full.sel(time=slice(analysis_start, analysis_end))
v_bp = v_bp_full.sel(time=slice(analysis_start, analysis_end))
z_bp = z_bp_full.sel(time=slice(analysis_start, analysis_end))
t_bp = t_bp_full.sel(time=slice(analysis_start, analysis_end))
w_bp = w_bp_full.sel(time=slice(analysis_start, analysis_end))
olr_bp = olr_bp_full.sel(time=slice(analysis_start, analysis_end))
t2m_bp = t2m_bp_full.sel(time=slice(analysis_start, analysis_end))

print("6) 计算 2015 原始场温度收支（5–9 月，用于滤波保边界）...")
budget_ds_ori = calc_daily_temperature_budget(
    t.sel(time=slice(FILTER_START, FILTER_END)),
    u.sel(time=slice(FILTER_START, FILTER_END)),
    v.sel(time=slice(FILTER_START, FILTER_END)),
    w.sel(time=slice(FILTER_START, FILTER_END)),
)

print("7) 将气候态映射到 2015 时间轴（5–9 月）...")
target_time = t.sel(time=slice(FILTER_START, FILTER_END)).time

t_clim_2015 = clim_to_time(t_clim, target_time)
u_clim_2015 = clim_to_time(u_clim, target_time)
v_clim_2015 = clim_to_time(v_clim, target_time)
w_clim_2015 = clim_to_time(w_clim, target_time)

print("8) 计算对应气候态温度收支（5–9 月）...")
budget_ds_clim = calc_daily_temperature_budget(
    t_clim_2015,
    u_clim_2015,
    v_clim_2015,
    w_clim_2015
)

print("9) 计算 925 hPa 收支异常（5–9 月）...")
dTdt = budget_ds_ori["dTdt"].sel(level=LEVEL_SEL) - budget_ds_clim["dTdt"].sel(level=LEVEL_SEL)
adv  = budget_ds_ori["adv_T"].sel(level=LEVEL_SEL) - budget_ds_clim["adv_T"].sel(level=LEVEL_SEL)
ver  = budget_ds_ori["ver"].sel(level=LEVEL_SEL) - budget_ds_clim["ver"].sel(level=LEVEL_SEL)
Q    = budget_ds_ori["Q"].sel(level=LEVEL_SEL) - budget_ds_clim["Q"].sel(level=LEVEL_SEL)

print("10) 计算长江流域区域平均并转为 K/day（5–9 月）...")
dTdt_yz_925_full = region_mean_series(dTdt, YANGTZE_SHP) * 86400.0
adv_yz_925_full  = region_mean_series(adv, YANGTZE_SHP) * 86400.0
ver_yz_925_full  = region_mean_series(ver, YANGTZE_SHP) * 86400.0
Q_yz_925_full    = region_mean_series(Q, YANGTZE_SHP) * 86400.0

print("10.1) 对 5–9 月区域平均序列做 10–30 天带通滤波...")
dTdt_yz_925_bp_full = xr.DataArray(
    LanczosFilter(dTdt_yz_925_full.values, "bandpass", period=[10, 30], nwts=61).filted(),
    coords=dTdt_yz_925_full.coords,
    dims=dTdt_yz_925_full.dims,
    name="dTdt_bp"
)

adv_yz_925_bp_full = xr.DataArray(
    LanczosFilter(adv_yz_925_full.values, "bandpass", period=[10, 30], nwts=61).filted(),
    coords=adv_yz_925_full.coords,
    dims=adv_yz_925_full.dims,
    name="adv_bp"
)

ver_yz_925_bp_full = xr.DataArray(
    LanczosFilter(ver_yz_925_full.values, "bandpass", period=[10, 30], nwts=61).filted(),
    coords=ver_yz_925_full.coords,
    dims=ver_yz_925_full.dims,
    name="ver_bp"
)

Q_yz_925_bp_full = xr.DataArray(
    LanczosFilter(Q_yz_925_full.values, "bandpass", period=[10, 30], nwts=61).filted(),
    coords=Q_yz_925_full.coords,
    dims=Q_yz_925_full.dims,
    name="Q_bp"
)

print("10.2) 再裁剪回 6–8 月...")
dTdt_yz_925 = dTdt_yz_925_bp_full.sel(time=slice(TARGET_START, TARGET_END))
adv_yz_925  = adv_yz_925_bp_full.sel(time=slice(TARGET_START, TARGET_END))
ver_yz_925  = ver_yz_925_bp_full.sel(time=slice(TARGET_START, TARGET_END))
Q_yz_925    = Q_yz_925_bp_full.sel(time=slice(TARGET_START, TARGET_END))

# ---------------------------------------------------------
# 若你还要保留平流分解结果，可继续算；这部分不参与最后四线图
# ---------------------------------------------------------
print("11) 计算水平温度平流分解（5–9 月，可选保留）...")
adv925 = decompose_temperature_advection(
    u=u.sel(time=slice(FILTER_START, FILTER_END)),
    v=v.sel(time=slice(FILTER_START, FILTER_END)),
    t=t.sel(time=slice(FILTER_START, FILTER_END)),
    u_clim=u_clim,
    v_clim=v_clim,
    t_clim=t_clim,
    start=FILTER_START,
    end=FILTER_END,
    level=LEVEL_SEL
)

adv925 = adv925 * 86400.0

adv925_series_full = xr.Dataset({
    "adv_clim": region_mean_series(adv925["adv_clim"], YANGTZE_SHP),
    "adv_a_wind_on_climT": region_mean_series(adv925["adv_a_wind_on_climT"], YANGTZE_SHP),
    "adv_clim_wind_on_aT": region_mean_series(adv925["adv_clim_wind_on_aT"], YANGTZE_SHP),
    "adv_nonlinear": region_mean_series(adv925["adv_nonlinear"], YANGTZE_SHP),
    "adv_ano_out_subseason": region_mean_series(
        adv925["adv_anom"]
        - adv925["adv_a_wind_on_climT"].data
        - adv925["adv_clim_wind_on_aT"].data
        - adv925["adv_nonlinear"].data,
        YANGTZE_SHP
    ),
    "adv_anom": region_mean_series(
        adv925["adv_total"] - adv925["adv_clim"].data,
        YANGTZE_SHP
    )
})

adv925_series = adv925_series_full.sel(time=slice(TARGET_START, TARGET_END))

print("12) 整理四线图数据...")
df_plot = pd.DataFrame({
    "time": pd.to_datetime(dTdt_yz_925.time.values),
    "dTdt": dTdt_yz_925.values,
    "adv": adv_yz_925.values,
    "ver": ver_yz_925.values,
    "Q": Q_yz_925.values,
    "uT": adv925_series['adv_a_wind_on_climT'].values,
    "Ut": adv925_series['adv_clim_wind_on_aT'].values,
    "eddy": adv925_series['adv_nonlinear'].values,
})

df_plot = df_plot[(df_plot["time"] >= TARGET_START) & (df_plot["time"] <= TARGET_END)].copy()
df_plot["summer_day"] = build_summer_day_index(pd.DatetimeIndex(df_plot["time"]))
x = df_plot["summer_day"].values

#%%
print("13) 绘制四线图...")
fig, ax = plt.subplots(figsize=(8, 4))

# 背景高亮区
ax.axvspan(
    date_to_summer_day(EVENT1_START),
    date_to_summer_day(EVENT1_END),
    color="#959595",
    alpha=0.25,
    zorder=0
)
ax.axvspan(
    date_to_summer_day(EVENT2_START),
    date_to_summer_day(EVENT2_END),
    color="#959595",
    alpha=0.25,
    zorder=0
)

# 四条线：这里直接画“先 5–9 月滤波、再裁到 6–8 月”的结果
ax.plot(x, df_plot["dTdt"], color="black",     lw=1.8, label=r"$\partial T/\partial t$", zorder=3)
ax.plot(x, df_plot["adv"],  color="royalblue", lw=1.8, label="Adv", zorder=3)
ax.plot(x, df_plot["ver"],  color="seagreen",  lw=1.8, label="Ver", zorder=3)
ax.plot(x, df_plot["Q"],    color="orangered", lw=1.8, label="Q", zorder=3)

# 如果后面要画平流分解项，可取消注释
# ax.plot(x, df_plot["uT"],   color="orange", lw=1.8, label=r"$-({u}^{\prime} \cdot \nabla \overline{T})$", zorder=3)
# ax.plot(x, df_plot["Ut"],   color="purple", lw=1.8, label=r"$-({\overline{u}} \cdot \nabla T^{\prime})$", zorder=3)
# ax.plot(x, df_plot["eddy"], color="cyan",   lw=1.8, label=r"$-({u}^{\prime} \cdot \nabla T^{\prime})$", zorder=3)

# 零线
ax.axhline(0, color="gray", lw=1.0, ls="--", zorder=1)

# x 轴
tick_positions = [1, 16, 31, 46, 62, 77, 92]
tick_labels = ["Jun-01", "Jun-16", "Jul-01", "Jul-16", "Aug-01", "Aug-16", "Aug-31"]

ax.set_xlim(1, 92)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

# labels / title
ax.set_xlabel("")
ax.set_ylabel(r"Temperature tendency (K day$^{-1}$)")
ax.set_title(f"Daily Temperature Budget over Yangtze Basin ({LEVEL_SEL} hPa, 2015)", loc="left", fontsize=14)

# grid / legend
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(frameon=False, ncol=4, loc="upper right")

# spine
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig(OUT_FIG + ".png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_FIG + ".pdf", bbox_inches="tight")
plt.close()

print(f"绘图完成，已保存：{OUT_FIG}")

ANO.close()
BP.close()
CLIM.close()
ds.close()

print("All done.")
