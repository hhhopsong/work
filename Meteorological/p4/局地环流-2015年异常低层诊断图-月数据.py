import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from climkit.data_read import era5_p
from climkit.masked import masked
from climkit.filter import *

from metpy.units import units
import metpy.calc as mpcalc


# =========================
# 字体设置
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"


# =========================================================
# 基础函数
# =========================================================
def standardize_coords(obj):
    """
    统一坐标名：
    longitude / latitude -> lon / lat
    pressure_level / isobaricInhPa -> level
    valid_time / date -> time
    """
    rename_dict = {}

    for old, new in [
        ("longitude", "lon"),
        ("latitude", "lat"),
        ("pressure_level", "level"),
        ("isobaricInhPa", "level"),
        ("valid_time", "time"),
    ]:
        if old in obj.coords or old in obj.dims:
            rename_dict[old] = new

    if rename_dict:
        obj = obj.rename(rename_dict)

    # 处理 ERA5 monthly 里可能出现的 date 坐标
    if "date" in obj.coords or "date" in obj.dims:
        date_vals = obj["date"].values

        try:
            time_vals = pd.to_datetime(date_vals.astype(str), format="%Y%m%d")
        except Exception:
            time_vals = pd.to_datetime(date_vals)

        obj = obj.assign_coords(date=time_vals)
        obj = obj.rename({"date": "time"})

    if "time" in obj.coords:
        obj = obj.assign_coords(time=pd.to_datetime(obj["time"].values))

    if "lon" not in obj.coords or "lat" not in obj.coords:
        raise ValueError(
            f"数据中未找到 lon/lat 或 longitude/latitude 坐标，当前坐标为: {list(obj.coords)}"
        )

    return obj


def get_first_var(ds, candidates, label):
    """
    从候选变量名中自动识别变量
    """
    for name in candidates:
        if name in ds.data_vars:
            print(f"{label} 使用变量名: {name}")
            return ds[name]

    raise KeyError(
        f"未在数据中找到 {label}。\n"
        f"候选变量名为: {candidates}\n"
        f"当前文件变量为: {list(ds.data_vars)}"
    )


def to_dataarray(obj, candidates, label):
    """
    Dataset / DataArray 统一转为 DataArray
    """
    if isinstance(obj, xr.DataArray):
        return obj

    if isinstance(obj, xr.Dataset):
        return get_first_var(obj, candidates, label)

    raise TypeError(f"{label} 既不是 Dataset 也不是 DataArray。")


def subset_region(da, lon_min=50, lon_max=170, lat_min=-10, lat_max=70):
    """
    为了减少计算量，只保留绘图区域附近。
    绘图区是 60–160E, 0–60N，这里额外留 10 度缓冲。
    """
    da = standardize_coords(da)

    if da["lon"].values[0] > da["lon"].values[-1]:
        da = da.sortby("lon")

    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.sortby("lat")

    da = da.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max)
    )

    return da


def prepare_monthly_pressure_da(
    da,
    levels,
    start_year=1961,
    end_year=2022,
    lon_min=50,
    lon_max=170,
    lat_min=-10,
    lat_max=70
):
    """
    逐月 pressure-level 数据预处理：
    统一坐标、截取时间、截取层次、截取区域、整理维度。
    """
    da = standardize_coords(da)

    if "level" not in da.coords and "level" not in da.dims:
        raise ValueError(f"{da.name} 没有 level 维度，当前维度为: {da.dims}")

    da = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    try:
        da = da.sel(level=levels)
    except Exception:
        raise ValueError(
            f"{da.name} 无法选取 levels={levels}。\n"
            f"当前可用 level 为: {da['level'].values}"
        )

    da = subset_region(
        da,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max
    )

    da = da.transpose("time", "level", "lat", "lon")

    return da


def to_latlon_2d(da):
    """
    压缩多余维度，并统一成 lat, lon 顺序
    """
    da = standardize_coords(da)
    da = da.squeeze()

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"{da.name} 维度中不包含 lat/lon，当前维度为: {da.dims}")

    da = da.transpose("lat", "lon")

    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.sortby("lat")

    if da["lon"].values[0] > da["lon"].values[-1]:
        da = da.sortby("lon")

    return da


def region_mean_scalar_2d(da, shp_path):
    """
    对二维 lat-lon 场做 shp 区域平均。
    输入 da 单位保持不变，例如 K/day。
    """
    da = to_latlon_2d(da)
    da_clip = masked(da, shp_path)
    out = da_clip.mean(dim=("lat", "lon"), skipna=True)
    return float(out.values)


# =========================================================
# 逐月温度诊断方程：支持多层平均
# =========================================================
def calc_monthly_temperature_budget_selected_level(
    t_da,
    u_da,
    v_da,
    w_da,
    level_plot=(950, 850),
    target_month=7
):
    """
    使用逐月数据计算指定月份、指定层次的温度诊断方程各项。

    支持：
    - level_plot=950：计算单层
    - level_plot=[950, 850]：分别计算 950 和 850 hPa，然后对两层求平均

    dT/dt 使用相邻月份中心差分：
        dT/dt = [T(t+1 month) - T(t-1 month)] / dt

    Returns
    -------
    ds_out : xr.Dataset
        dims: time, lat, lon
        dTdt  : K/s
        adv_T : K/s
        ver   : K/s
        Q     : K/s
        sigma : K/Pa
    """
    from metpy.constants import dry_air_gas_constant as R
    from metpy.constants import dry_air_spec_heat_press as cp

    t_da = standardize_coords(t_da).transpose("time", "level", "lat", "lon")
    u_da = standardize_coords(u_da).transpose("time", "level", "lat", "lon")
    v_da = standardize_coords(v_da).transpose("time", "level", "lat", "lon")
    w_da = standardize_coords(w_da).transpose("time", "level", "lat", "lon")

    # 对齐四个变量，防止坐标有极小差异
    t_da, u_da, v_da, w_da = xr.align(
        t_da,
        u_da,
        v_da,
        w_da,
        join="inner"
    )

    if "units" not in t_da.attrs:
        t_da.attrs["units"] = "K"
    if "units" not in u_da.attrs:
        u_da.attrs["units"] = "m/s"
    if "units" not in v_da.attrs:
        v_da.attrs["units"] = "m/s"
    if "units" not in w_da.attrs:
        w_da.attrs["units"] = "Pa/s"

    times = pd.to_datetime(t_da["time"].values)
    levs_hpa = t_da["level"].values
    lats = t_da["lat"].values
    lons = t_da["lon"].values

    # ----------------------------
    # 支持单层或多层平均
    # ----------------------------
    if np.isscalar(level_plot):
        level_list = [int(level_plot)]
    else:
        level_list = [int(x) for x in level_plot]

    missing_levels = [lv for lv in level_list if lv not in levs_hpa]

    if missing_levels:
        raise ValueError(
            f"level_plot={level_list} 中有层次不在数据 level 中: {missing_levels}\n"
            f"当前可用 level 为: {levs_hpa}"
        )

    level_indices = [
        int(np.where(levs_hpa == lv)[0][0])
        for lv in level_list
    ]

    print(f"将计算层次: {level_list} hPa，并对这些层求平均。")

    # 只计算所有目标月份，例如所有 7 月
    target_indices = np.where(times.month == target_month)[0]

    if len(target_indices) == 0:
        raise ValueError(f"没有找到 target_month={target_month} 的数据。")

    # 删除不能做中心差分的边界月
    target_indices = target_indices[
        (target_indices > 0) & (target_indices < len(times) - 1)
    ]

    if len(target_indices) == 0:
        raise ValueError(
            f"target_month={target_month} 的样本都在时间边界，无法做中心差分。"
        )

    print(f"将计算 {len(target_indices)} 个 {target_month:02d} 月样本的温度诊断项。")

    t_q = t_da.metpy.quantify()
    u_q = u_da.metpy.quantify()
    v_q = v_da.metpy.quantify()
    w_q = w_da.metpy.quantify()

    p_pa = (levs_hpa * 100.0) * units.Pa
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

    nt_sel = len(target_indices)
    ny = len(lats)
    nx = len(lons)

    dTdt_arr = np.full((nt_sel, ny, nx), np.nan)
    adv_arr = np.full((nt_sel, ny, nx), np.nan)
    ver_arr = np.full((nt_sel, ny, nx), np.nan)
    Q_arr = np.full((nt_sel, ny, nx), np.nan)
    sigma_arr = np.full((nt_sel, ny, nx), np.nan)

    for n, i in enumerate(target_indices):

        if (n + 1) % 10 == 0 or n == 0:
            print(f"计算进度: {n + 1}/{nt_sel}，时间: {times[i].strftime('%Y-%m')}")

        # ----------------------------
        # 月尺度中心差分 dTdt 的时间间隔
        # ----------------------------
        dt_seconds = (times[i + 1] - times[i - 1]) / np.timedelta64(1, "s")

        # ----------------------------
        # 用所有 pressure levels 计算 sigma
        # sigma = R*T/(cp*p) - dT/dp
        # ----------------------------
        T_now = t_q.isel(time=i)
        T_vals = T_now.metpy.dequantify().values

        dTdp_vals = np.gradient(
            T_vals,
            p_pa.magnitude,
            axis=0
        )

        p_3d = p_pa.magnitude[:, None, None]

        term1 = (
            R.magnitude * T_vals
        ) / (
            cp.magnitude * p_3d
        )

        sigma_vals = term1 - dTdp_vals

        # ----------------------------
        # 分别计算指定层次，然后求平均
        # ----------------------------
        dTdt_level_list = []
        adv_level_list = []
        ver_level_list = []
        Q_level_list = []
        sigma_level_list = []

        for level_index in level_indices:

            level_now = levs_hpa[level_index]

            # ----------------------------
            # (1) 月尺度中心差分 dTdt
            # ----------------------------
            dTdt = (
                t_q.isel(time=i + 1, level=level_index)
                - t_q.isel(time=i - 1, level=level_index)
            ) / (dt_seconds * units.s)

            # ----------------------------
            # (2) 水平温度平流 adv_T
            # MetPy advection 返回的是 -u*dT/dx - v*dT/dy
            # ----------------------------
            adv_T = mpcalc.advection(
                t_q.isel(time=i, level=level_index),
                u=u_q.isel(time=i, level=level_index),
                v=v_q.isel(time=i, level=level_index),
                dx=dx,
                dy=dy,
                x_dim=-1,
                y_dim=-2
            )

            # ----------------------------
            # (3) 当前层 sigma
            # ----------------------------
            sigma_level = xr.DataArray(
                sigma_vals[level_index, :, :],
                coords={
                    "lat": lats,
                    "lon": lons
                },
                dims=("lat", "lon")
            ) * units("K/Pa")

            # ----------------------------
            # (4) 垂直运动项 ver
            # ----------------------------
            ver = w_q.isel(time=i, level=level_index) * sigma_level

            # ----------------------------
            # (5) 非绝热加热项 Q
            # 保持原定义：
            # Q = dTdt - adv_T - ver
            # ----------------------------
            Q = dTdt - adv_T - ver

            dTdt_level_list.append(dTdt.metpy.dequantify().values)
            adv_level_list.append(adv_T.metpy.dequantify().values)
            ver_level_list.append(ver.metpy.dequantify().values)
            Q_level_list.append(Q.metpy.dequantify().values)
            sigma_level_list.append(sigma_level.metpy.dequantify().values)

        # ----------------------------
        # 对指定层次求平均
        # 例如 950 和 850 hPa 平均
        # ----------------------------
        dTdt_arr[n] = np.nanmean(np.stack(dTdt_level_list, axis=0), axis=0)
        adv_arr[n] = np.nanmean(np.stack(adv_level_list, axis=0), axis=0)
        ver_arr[n] = np.nanmean(np.stack(ver_level_list, axis=0), axis=0)
        Q_arr[n] = np.nanmean(np.stack(Q_level_list, axis=0), axis=0)
        sigma_arr[n] = np.nanmean(np.stack(sigma_level_list, axis=0), axis=0)

    out_times = times[target_indices]

    coords = {
        "time": out_times,
        "lat": lats,
        "lon": lons
    }

    ds_out = xr.Dataset(
        {
            "dTdt": xr.DataArray(
                dTdt_arr,
                coords=coords,
                dims=("time", "lat", "lon")
            ),
            "adv_T": xr.DataArray(
                adv_arr,
                coords=coords,
                dims=("time", "lat", "lon")
            ),
            "ver": xr.DataArray(
                ver_arr,
                coords=coords,
                dims=("time", "lat", "lon")
            ),
            "Q": xr.DataArray(
                Q_arr,
                coords=coords,
                dims=("time", "lat", "lon")
            ),
            "sigma": xr.DataArray(
                sigma_arr,
                coords=coords,
                dims=("time", "lat", "lon")
            ),
        }
    )

    ds_out["dTdt"].attrs["units"] = "K/s"
    ds_out["adv_T"].attrs["units"] = "K/s"
    ds_out["ver"].attrs["units"] = "K/s"
    ds_out["Q"].attrs["units"] = "K/s"
    ds_out["sigma"].attrs["units"] = "K/Pa"

    return ds_out


def monthly_budget_anom_to_kday(budget_da, year, month, clim_start=1991, clim_end=2020):
    """
    从某个诊断项中取：
    指定年月 - 气候态同月平均
    并从 K/s 转为 K/day。
    """
    da = standardize_coords(budget_da)

    da = da.sel(time=slice(f"{clim_start}-01-01", f"{clim_end}-12-31"))

    target = da.where(
        (da["time"].dt.year == year) &
        (da["time"].dt.month == month),
        drop=True
    )

    if target.sizes.get("time", 0) == 0:
        raise ValueError(f"{da.name} 中没有找到 {year}-{month:02d}。")

    target = target.mean("time", skipna=True)

    clim = da.where(
        da["time"].dt.month == month,
        drop=True
    ).mean("time", skipna=True)

    out = (target - clim) * 86400.0
    out.attrs["units"] = "K/day"

    return to_latlon_2d(out)


# =========================================================
# 绘图函数：空间图
# =========================================================
def plot_scalar_map(
    fig,
    pic_loc,
    scalar_plot,
    scalar_levels,
    title,
    cbar_label,
    cmap_use,
    add_tp_hatch=True
):
    """
    仅绘制填色场，不叠加 UV / Z。
    scalar_plot 使用自己的 lon/lat。
    """
    ax = fig.add_subplot(
        pic_loc,
        projection=ccrs.PlateCarree(central_longitude=180 - 70)
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_aspect("auto")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

    lon = scalar_plot["lon"]
    lat = scalar_plot["lat"]

    # -------------------------
    # 填色
    # -------------------------
    contf = ax.contourf(
        lon,
        lat,
        scalar_plot,
        levels=scalar_levels,
        cmap=cmap_use,
        extend="both",
        transform=ccrs.PlateCarree()
    )

    # -------------------------
    # 底图
    # -------------------------
    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=0.4
    )

    ax.add_geometries(
        Reader(fr"{PYFILE}/map/self/长江_TP/长江_tp.shp").geometries(),
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=0.5
    )

    ax.add_geometries(
        Reader(
            fr"{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp"
        ).geometries(),
        ccrs.PlateCarree(),
        facecolor="gray",
        edgecolor="black",
        linewidth=0.5
    )

    if add_tp_hatch:
        ax.add_geometries(
            Reader(
                fr"{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp"
            ).geometries(),
            ccrs.PlateCarree(),
            facecolor="#909090",
            edgecolor="#909090",
            linewidth=0,
            hatch=".",
            zorder=10
        )

    # -------------------------
    # 经纬度刻度
    # -------------------------
    xticks1 = np.arange(60, 161, 20)
    yticks1 = np.arange(0, 61, 15)

    ax.set_xticks(xticks1, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks1, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.tick_params(which="major", length=4, width=0.5, color="black")
    ax.tick_params(which="minor", length=2, width=0.2, color="black")
    ax.tick_params(
        which="both",
        bottom=True,
        top=False,
        left=True,
        labelbottom=True,
        labeltop=False
    )
    ax.tick_params(axis="both", labelsize=8, colors="black")

    # -------------------------
    # colorbar
    # -------------------------
    ax_colorbar = inset_axes(
        ax,
        width="3.5%",
        height="92%",
        loc="center right",
        bbox_to_anchor=(0.075, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    cb = plt.colorbar(
        contf,
        cax=ax_colorbar,
        orientation="vertical",
        drawedges=True
    )

    cb.locator = ticker.FixedLocator(scalar_levels)
    cb.update_ticks()
    cb.ax.tick_params(length=0, labelsize=8, direction="in")
    cb.dividers.set_linewidth(0.8)
    cb.outline.set_linewidth(1.0)
    cb.set_label(cbar_label, fontsize=8)

    return contf, ax


# =========================================================
# 绘图函数：温度收支 bar 图
# =========================================================
def plot_monthly_budget_bar(
    values,
    labels,
    title,
    out_pdf,
    out_png,
    ylim=None,
    ytick_step=0.05
):
    """
    仿照日尺度代码格式，画长江_TP 温度收支柱状图。
    values 单位：K/day
    """
    fig = plt.figure(figsize=(3.2, 1.6))
    ax = fig.add_subplot(111)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_aspect("auto")
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, linestyle="--", zorder=0, axis="y")

    values = np.array(values, dtype=float)

    # 红=正，蓝=负
    colors = ["#ff7373" if val > 0 else "#7373ff" for val in values]

    bars = ax.bar(
        range(len(values)),
        values,
        width=0.4,
        color=colors,
        edgecolor="black",
        zorder=2
    )

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_xlim(-0.5, len(values) - 0.5)

    # 自动对称 y 轴
    if ylim is None:
        max_abs = np.nanmax(np.abs(values))
        if not np.isfinite(max_abs) or max_abs == 0:
            ylim = 0.15
        else:
            ylim = np.ceil(max_abs * 1.25 / ytick_step) * ytick_step
            ylim = max(ylim, 0.15)

    ax.set_ylim(-ylim, ylim)
    ax.axhline(0, color="#454545", lw=0.7)

    yticks = np.arange(-ylim, ylim + ytick_step * 0.5, ytick_step)
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [f"{v:.2f}" if abs(v) > 1e-8 else "0" for v in yticks],
        fontsize=10,
        color="#000000"
    )

    ax.tick_params(axis="y", labelsize=9, color="#000000")
    ax.tick_params(axis="x", labelsize=10, color="#000000")

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.show()

    return fig, ax


# =========================================================
# 主程序：逐月资料
# =========================================================

# -------------------------
# 分析年月与层次
# -------------------------
analysis_year = 2015
analysis_month = 7

# 绘制 950 和 850 hPa 两层平均
level_plot = [1000, 950, 900]

clim_start = 1991
clim_end = 2020

# 用于读取和计算的层次
budget_levels = [1000, 950, 900]

# 计算区域，绘图区域是 60–160E, 0–60N
# 这里额外留缓冲，避免梯度边界太贴近图边
calc_lon_min = 50
calc_lon_max = 170
calc_lat_min = -10
calc_lat_max = 70


# -------------------------
# 读取逐月 ERA5 pressure level 数据
# -------------------------
t = era5_p(
    f"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc",
    clim_start,
    clim_end,
    budget_levels,
    "t"
)

u = era5_p(
    f"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc",
    clim_start,
    clim_end,
    budget_levels,
    "u"
)

v = era5_p(
    f"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc",
    clim_start,
    clim_end,
    budget_levels,
    "v"
)

w = era5_p(
    f"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc",
    clim_start,
    clim_end,
    budget_levels,
    "w"
)

t = to_dataarray(t, ["t", "T"], "T")
u = to_dataarray(u, ["u", "U"], "U")
v = to_dataarray(v, ["v", "V"], "V")
w = to_dataarray(w, ["w", "omega", "OMEGA"], "W/Omega")


# -------------------------
# 预处理
# -------------------------
t = prepare_monthly_pressure_da(
    t,
    budget_levels,
    clim_start,
    clim_end,
    calc_lon_min,
    calc_lon_max,
    calc_lat_min,
    calc_lat_max
)

u = prepare_monthly_pressure_da(
    u,
    budget_levels,
    clim_start,
    clim_end,
    calc_lon_min,
    calc_lon_max,
    calc_lat_min,
    calc_lat_max
)

v = prepare_monthly_pressure_da(
    v,
    budget_levels,
    clim_start,
    clim_end,
    calc_lon_min,
    calc_lon_max,
    calc_lat_min,
    calc_lat_max
)

w = prepare_monthly_pressure_da(
    w,
    budget_levels,
    clim_start,
    clim_end,
    calc_lon_min,
    calc_lon_max,
    calc_lat_min,
    calc_lat_max
)


# -------------------------
# 计算所有 7 月的温度诊断项
# 输出已经是 950 和 850 hPa 两层平均
# -------------------------
budget_ds = calc_monthly_temperature_budget_selected_level(
    t_da=t,
    u_da=u,
    v_da=v,
    w_da=w,
    level_plot=level_plot,
    target_month=analysis_month
)


# -------------------------
# 计算 2015 年 7 月异常，并转为 K/day
# -------------------------
adv_plot = monthly_budget_anom_to_kday(
    budget_ds["adv_T"],
    analysis_year,
    analysis_month,
    clim_start,
    clim_end
)

ver_plot = monthly_budget_anom_to_kday(
    budget_ds["ver"],
    analysis_year,
    analysis_month,
    clim_start,
    clim_end
)

Q_plot = monthly_budget_anom_to_kday(
    budget_ds["Q"],
    analysis_year,
    analysis_month,
    clim_start,
    clim_end
)


# -------------------------
# 维度检查
# -------------------------
print("====== 数据维度检查 ======")
print("adv_plot:", adv_plot.shape, "lon:", len(adv_plot["lon"]), "lat:", len(adv_plot["lat"]))
print("ver_plot:", ver_plot.shape, "lon:", len(ver_plot["lon"]), "lat:", len(ver_plot["lat"]))
print("Q_plot:", Q_plot.shape, "lon:", len(Q_plot["lon"]), "lat:", len(Q_plot["lat"]))
print("==========================")


# =========================================================
# 长江_TP 区域平均温度收支诊断
# 单位：K/day
# =========================================================
yangtze_shp = fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"

adv_yz = region_mean_scalar_2d(adv_plot, yangtze_shp)
ver_yz = region_mean_scalar_2d(ver_plot, yangtze_shp)
Q_yz = region_mean_scalar_2d(Q_plot, yangtze_shp)

print("====== 长江_TP 温度收支区域平均，单位 K/day ======")
print("horizontal advection:", adv_yz)
print("vertical motion    :", ver_yz)
print("diabatic heating   :", Q_yz)
print("sum adv + ver + Q  :", adv_yz + ver_yz + Q_yz)
print("===============================================")


# =========================================================
# 色标范围
# 单位：K/day
# =========================================================
lev_budget = np.array([
    -1.6, -1.2, -0.8, -0.4, -0.2,
     0.2,  0.4,  0.8,  1.2,  1.6
])


# =========================================================
# 配色
# =========================================================
cmap_a = plt.get_cmap("RdBu_r")   # 平流项
cmap_b = plt.get_cmap("PuOr_r")   # 垂直运动项
cmap_c = plt.get_cmap("PiYG_r")   # 非绝热加热项


# =========================================================
# 作空间图
# =========================================================
fig = plt.figure(figsize=(5.2, 8.4))
plt.subplots_adjust(wspace=0.15, hspace=0.28)

title_head = f"{analysis_year} Jul"


# -------------------------
# (a) 平流项
# 画青藏高原阴影
# -------------------------
contf1, ax1 = plot_scalar_map(
    fig=fig,
    pic_loc=311,
    scalar_plot=adv_plot,
    scalar_levels=lev_budget,
    title=f"(a) {title_head} horizontal advection term",
    cbar_label="K/day",
    cmap_use=cmap_a,
    add_tp_hatch=True
)


# -------------------------
# (b) 垂直运动项
# 画青藏高原阴影
# -------------------------
contf2, ax2 = plot_scalar_map(
    fig=fig,
    pic_loc=312,
    scalar_plot=ver_plot,
    scalar_levels=lev_budget,
    title=f"(b) {title_head} vertical motion term",
    cbar_label="K/day",
    cmap_use=cmap_b,
    add_tp_hatch=True
)


# -------------------------
# (c) 非绝热加热项
# 不画青藏高原阴影
# -------------------------
contf3, ax3 = plot_scalar_map(
    fig=fig,
    pic_loc=313,
    scalar_plot=Q_plot,
    scalar_levels=lev_budget,
    title=f"(c) {title_head} diabatic heating term",
    cbar_label="K/day",
    cmap_use=cmap_c,
    add_tp_hatch=False
)


# -------------------------
# 强制裁剪
# -------------------------
for ax_ in fig.axes:
    for artist in ax_.get_children():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(True)


# -------------------------
# 保存空间图
# -------------------------
out_dir = fr"{PYFILE}/p4/pic"
os.makedirs(out_dir, exist_ok=True)

out_pdf = fr"{out_dir}/{analysis_year}07_逐月温度诊断方程_平流_垂直运动_非绝热加热.pdf"
out_png = fr"{out_dir}/{analysis_year}07_逐月温度诊断方程_平流_垂直运动_非绝热加热.png"

plt.savefig(
    out_pdf,
    bbox_inches="tight"
)

plt.savefig(
    out_png,
    bbox_inches="tight",
    dpi=600
)

plt.show()


# =========================================================
# 另画一个长江_TP 温度收支诊断 bar 图
# =========================================================
bar_values = [
    adv_yz,
    ver_yz,
    Q_yz
]

bar_labels = [
    r"$-(\mathbf{V} \cdot \nabla T)^{\prime}$",
    r"$(\omega \sigma)^{\prime}$",
    r"${Q}^{\prime}$"
]

out_bar_pdf = fr"{out_dir}/{analysis_year}07_长江TP_逐月温度收支诊断_bar.pdf"
out_bar_png = fr"{out_dir}/{analysis_year}07_长江TP_逐月温度收支诊断_bar.png"

fig_bar, ax_bar = plot_monthly_budget_bar(
    values=bar_values,
    labels=bar_labels,
    title=f"2015 Jul temp. budget",
    out_pdf=out_bar_pdf,
    out_png=out_bar_png,
    ylim=None,
    ytick_step=0.05
)
