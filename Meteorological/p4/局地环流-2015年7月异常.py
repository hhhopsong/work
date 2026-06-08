import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import calendar

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from climkit.Cquiver import *
from climkit.data_read import *
from climkit.masked import masked
from climkit.lonlat_transform import *
from climkit.filter import *
from climkit.corr_reg import *

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

    # ERA5 single level 可能是 date 维度，例如 20150101
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
            f"数据中未找到 lon/lat 或 longitude/latitude 坐标。当前坐标为: {list(obj.coords)}"
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


def subset_time(da, start_year=1961, end_year=2022):
    """
    截取时间段
    """
    da = standardize_coords(da)
    return da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))


def select_month_value(da, year, month):
    """
    选取某一年某一月的逐月值。
    如果该月存在多个时间点，则对 time 求平均。
    """
    da = standardize_coords(da)

    out = da.where(
        (da["time"].dt.year == year) &
        (da["time"].dt.month == month),
        drop=True
    )

    if out.sizes.get("time", 0) == 0:
        raise ValueError(f"{da.name} 中没有找到 {year}-{month:02d} 的数据。")

    return out.mean("time", skipna=True)


def monthly_clim_anom(da, year, month, clim_start=1961, clim_end=2022):
    """
    逐月异常：
    anomaly = 指定年月值 - 气候态同月平均

    例如：
    2015年7月异常 = 2015年7月 - 1961–2022年所有7月平均
    """
    da = standardize_coords(da)
    da = da.sel(time=slice(f"{clim_start}-01-01", f"{clim_end}-12-31"))

    target = select_month_value(da, year, month)

    clim_sample = da.where(da["time"].dt.month == month, drop=True)

    if clim_sample.sizes.get("time", 0) == 0:
        raise ValueError(f"{da.name} 中没有找到 {month:02d} 月气候态样本。")

    clim = clim_sample.mean("time", skipna=True)

    out = target - clim
    out.attrs = da.attrs.copy()
    out.name = f"{da.name}_ano"

    return out


def sel_level_monthly_anom(
    da,
    level,
    year,
    month,
    clim_start=1961,
    clim_end=2022
):
    """
    选取指定气压层并计算逐月异常
    """
    da = standardize_coords(da)

    if "level" not in da.dims and "level" not in da.coords:
        raise ValueError(f"{da.name} 没有 level 维度，不能选 {level} hPa。")

    da_level = da.sel(level=level)

    return monthly_clim_anom(
        da_level,
        year=year,
        month=month,
        clim_start=clim_start,
        clim_end=clim_end
    )


def to_latlon_2d(da):
    """
    压缩多余维度，并统一成 lat, lon 顺序
    """
    da = standardize_coords(da)
    da = da.squeeze()

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"{da.name} 维度中不包含 lat/lon，当前维度为: {da.dims}")

    da = da.transpose("lat", "lon")

    # 为了 contourf / contour 稳定，统一按 lat、lon 升序
    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.sortby("lat")

    if da["lon"].values[0] > da["lon"].values[-1]:
        da = da.sortby("lon")

    return da


def maybe_z_to_gpm(z_da):
    """
    ERA5 z 通常是 geopotential，单位 m^2 s^-2。
    画图时转为位势高度 gpm。
    如果已经是 gpm，则保持不变。
    """
    units = str(z_da.attrs.get("units", "")).lower()

    if (
        "m**2" in units
        or "m^2" in units
        or "m2" in units
        or "geopotential" in units
        or units in ["m2 s-2", "m2/s2", "m**2 s**-2"]
    ):
        out = z_da / 9.80665
        out.attrs["units"] = "gpm"
        return out

    return z_da


def maybe_tcc_to_percent(tcc_da):
    """
    ERA5 TCC 常为 0–1。
    若异常量绝对值小于等于 2，则认为是比例，转为百分比异常。
    """
    vmax = float(np.nanmax(np.abs(tcc_da.values)))

    if vmax <= 2:
        out = tcc_da * 100.0
        out.attrs["units"] = "%"
        return out

    return tcc_da


def maybe_pre_to_mmday(pre_da):
    """
    PRE 单位判断：
    - 若单位是 m 或 m/day，则转成 mm/day
    - 若已经是 mm/day，则保持不变
    - 若无单位，则保持原值
    """
    units = str(pre_da.attrs.get("units", "")).lower()

    if units in ["m", "meter", "metre"] or "m/day" in units or "m d" in units:
        out = pre_da * 1000.0
        out.attrs["units"] = "mm/day"
        return out

    return pre_da


def maybe_pre_to_month_total(pre_da, year, month, use_month_total=False):
    """
    如果想把 mm/day 转成月累计异常，可打开 use_month_total=True。
    默认保持 mm/day。
    """
    if not use_month_total:
        return pre_da

    days = calendar.monthrange(year, month)[1]
    out = pre_da * days
    out.attrs["units"] = "mm/month"
    return out


# =========================================================
# 绘图函数
# =========================================================
def pic_anom(
    fig,
    pic_loc,
    u_plot,
    v_plot,
    z_plot,
    scalar_plot,
    scalar_levels,
    z_levels,
    title,
    cbar_label,
    cmap_use,
    qkey=2,
    qlabel="2 m/s",
    nanmax=5,
    wind_scale=4,
    add_tp_hatch=True
):
    ax = fig.add_subplot(
        pic_loc,
        projection=ccrs.PlateCarree(central_longitude=180 - 70)
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_aspect("auto")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())

    # -------------------------
    # 填色变量
    # 注意：填色变量使用自己的 lon/lat
    # -------------------------
    roi_shape = ((60, 0), (160, 60))

    try:
        scalar_roi = scalar_plot.salem.roi(corners=roi_shape)
    except Exception:
        scalar_roi = scalar_plot

    scalar_lon = scalar_roi["lon"]
    scalar_lat = scalar_roi["lat"]

    contf = ax.contourf(
        scalar_lon,
        scalar_lat,
        scalar_roi,
        levels=scalar_levels,
        cmap=cmap_use,
        extend="both",
        alpha=0.75,
        transform=ccrs.PlateCarree()
    )

    # -------------------------
    # Z 异常等值线
    # 注意：Z 使用自己的 lon/lat
    # -------------------------
    z_lon = z_plot["lon"]
    z_lat = z_plot["lat"]

    z_neg = [lv for lv in z_levels if lv < 0]
    z_pos = [lv for lv in z_levels if lv > 0]

    if len(z_pos) > 0:
        ax.contour(
            z_lon,
            z_lat,
            z_plot,
            levels=z_pos,
            colors="red",
            linewidths=1.2,
            transform=ccrs.PlateCarree()
        )

    if len(z_neg) > 0:
        ax.contour(
            z_lon,
            z_lat,
            z_plot,
            levels=z_neg,
            colors="blue",
            linestyles="--",
            linewidths=1.2,
            transform=ccrs.PlateCarree()
        )

    # -------------------------
    # UV 风矢量
    # 注意：风场使用自己的 lon/lat
    # -------------------------
    u_lon = u_plot["lon"]
    u_lat = u_plot["lat"]

    Cq = ax.Curlyquiver(
        u_lon,
        u_lat,
        u_plot,
        v_plot,
        center_lon=110,
        scale=wind_scale,
        linewidth=1,
        arrowsize=1.0,
        transform=ccrs.PlateCarree(),
        MinDistance=[0.2, 0.5],
        regrid=12,
        color="#454545",
        nanmax=nanmax,
        thinning=["10%", "min"]
    )

    Cq.key(
        U=qkey,
        label=qlabel,
        color="k",
        fontproperties={"size": 8},
        linewidth=0.7,
        arrowsize=3.0,
        facecolor="#FFFFFF"
    )

    # -------------------------
    # 地图底图
    # -------------------------
    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=1.2,
        color="#AAAAAA"
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
    # Colorbar
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

    return contf, ax


# =========================================================
# 主程序：逐月数据
# =========================================================

# =========================
# 目标年月与气候态时段
# =========================
analysis_year = 2015
analysis_month = 7

clim_start = 1991
clim_end = 2020

# 降水是否转成月累计异常
# False：保持原资料单位，例如 mm/day
# True ：乘以当月天数，变成 mm/month
PRE_USE_MONTH_TOTAL = False


# =========================
# 读取逐月 ERA5 pressure level 数据
# 这里只需要 200 / 500 / 850 hPa
# =========================
z = era5_p(
    f"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc",
    clim_start,
    clim_end,
    [200, 500, 850],
    "z"
)

u = era5_p(
    f"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc",
    clim_start,
    clim_end,
    [200, 500, 850],
    "u"
)

v = era5_p(
    f"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc",
    clim_start,
    clim_end,
    [200, 500, 850],
    "v"
)

z = to_dataarray(z, ["z", "Z"], "Z")
u = to_dataarray(u, ["u", "U"], "U")
v = to_dataarray(v, ["v", "V"], "V")

z = subset_time(z, clim_start, clim_end)
u = subset_time(u, clim_start, clim_end)
v = subset_time(v, clim_start, clim_end)


# =========================
# 读取逐月 ERA5 single level 数据：T2M / TCC
# =========================
sg = xr.open_dataset(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc")

t2m = get_first_var(
    sg,
    ["t2m", "T2M"],
    "T2M"
)

tcc = get_first_var(
    sg,
    ["tcc", "TCC"],
    "TCC"
)

t2m = subset_time(t2m, clim_start, clim_end)
tcc = subset_time(tcc, clim_start, clim_end)


# =========================
# 读取逐月降水异常
# precip.mon.anom.nc 默认已经是异常场
# 所以直接取 2015 年 7 月，不再减气候态
# =========================
pre = prec(
    f"{DATA}/NOAA/PREC/precip.mon.anom.nc",
    clim_start,
    clim_end
)

pre = to_dataarray(
    pre,
    ["pre", "PRE", "precip", "precipitation"],
    "PRE/降水异常"
)

pre = subset_time(pre, clim_start, clim_end)


# =========================================================
# 计算 2015 年 7 月异常
# =========================================================

# =========================
# 200 hPa UVZ anomaly
# =========================
u200 = to_latlon_2d(
    sel_level_monthly_anom(
        u,
        200,
        analysis_year,
        analysis_month,
        clim_start,
        clim_end
    )
)

v200 = to_latlon_2d(
    sel_level_monthly_anom(
        v,
        200,
        analysis_year,
        analysis_month,
        clim_start,
        clim_end
    )
)

z200 = to_latlon_2d(
    maybe_z_to_gpm(
        sel_level_monthly_anom(
            z,
            200,
            analysis_year,
            analysis_month,
            clim_start,
            clim_end
        )
    )
)


# =========================
# 500 hPa UVZ anomaly
# =========================
u500 = to_latlon_2d(
    sel_level_monthly_anom(
        u,
        500,
        analysis_year,
        analysis_month,
        clim_start,
        clim_end
    )
)

v500 = to_latlon_2d(
    sel_level_monthly_anom(
        v,
        500,
        analysis_year,
        analysis_month,
        clim_start,
        clim_end
    )
)

z500 = to_latlon_2d(
    maybe_z_to_gpm(
        sel_level_monthly_anom(
            z,
            500,
            analysis_year,
            analysis_month,
            clim_start,
            clim_end
        )
    )
)


# =========================
# 850 hPa UVZ anomaly
# =========================
u850 = to_latlon_2d(
    sel_level_monthly_anom(
        u,
        850,
        analysis_year,
        analysis_month,
        clim_start,
        clim_end
    )
)

v850 = to_latlon_2d(
    sel_level_monthly_anom(
        v,
        850,
        analysis_year,
        analysis_month,
        clim_start,
        clim_end
    )
)

z850 = to_latlon_2d(
    maybe_z_to_gpm(
        sel_level_monthly_anom(
            z,
            850,
            analysis_year,
            analysis_month,
            clim_start,
            clim_end
        )
    )
)


# =========================
# 填色变量 anomaly
# =========================
t2m_plot = to_latlon_2d(
    monthly_clim_anom(
        t2m,
        analysis_year,
        analysis_month,
        clim_start,
        clim_end
    )
)

tcc_plot = to_latlon_2d(
    maybe_tcc_to_percent(
        monthly_clim_anom(
            tcc,
            analysis_year,
            analysis_month,
            clim_start,
            clim_end
        )
    )
)

pre_plot = to_latlon_2d(
    maybe_pre_to_month_total(
        maybe_pre_to_mmday(
            select_month_value(
                pre,
                analysis_year,
                analysis_month
            )
        ),
        analysis_year,
        analysis_month,
        use_month_total=PRE_USE_MONTH_TOTAL
    )
)


# =========================================================
# 检查维度
# =========================================================
print("====== 数据维度检查 ======")
print("u200:", u200.shape, "lon:", len(u200["lon"]), "lat:", len(u200["lat"]))
print("v200:", v200.shape, "lon:", len(v200["lon"]), "lat:", len(v200["lat"]))
print("z200:", z200.shape, "lon:", len(z200["lon"]), "lat:", len(z200["lat"]))
print("t2m:", t2m_plot.shape, "lon:", len(t2m_plot["lon"]), "lat:", len(t2m_plot["lat"]))
print("tcc:", tcc_plot.shape, "lon:", len(tcc_plot["lon"]), "lat:", len(tcc_plot["lat"]))
print("pre:", pre_plot.shape, "lon:", len(pre_plot["lon"]), "lat:", len(pre_plot["lat"]))
print("==========================")


# =========================================================
# 色阶设置
# =========================================================
lev_t2m = np.array([
    -3, -2, -1, -0.5, -0.2,
     0.2, 0.5, 1, 2, 3
])

lev_tcc = np.array([
    -15, -10, -5, -2,
      2,   5,  10, 15
])

lev_pre = np.array([
    -6, -4, -2, -1,
     1,  2,  4,  6
])


# =========================================================
# 配色
# =========================================================
from matplotlib import ticker, colors
from matplotlib.ticker import MultipleLocator
cmap_a = cmaps.GMT_polar[2:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-2]
cmap_b = colors.LinearSegmentedColormap.from_list(
    "PuOr_light",
    plt.get_cmap("PuOr")(np.linspace(0.15, 0.85, 256))
)
cmap_c = plt.get_cmap("BrBG")


# =========================================================
# 正式绘图
# =========================================================
fig = plt.figure(figsize=(4.3, 8.4))
plt.subplots_adjust(wspace=0.15, hspace=0.28)

title_head = f"{analysis_year} Jul"


# -------------------------
# (a) 200UVZ & T2M
# -------------------------
contf1, ax1 = pic_anom(
    fig=fig,
    pic_loc=311,
    u_plot=u200,
    v_plot=v200,
    z_plot=z200,
    scalar_plot=t2m_plot,
    scalar_levels=lev_t2m,
    z_levels=np.array([-160, -40, 140, 280]),
    title=f"(a) {title_head} 200UVZ & T2M anomaly",
    cbar_label="T2M anomaly",
    cmap_use=cmap_a,
    qkey=2,
    qlabel="2 m/s",
    nanmax=5,
    wind_scale=4,
    add_tp_hatch=False
)


# -------------------------
# (b) 500UVZ & TCC
# -------------------------
contf2, ax2 = pic_anom(
    fig=fig,
    pic_loc=312,
    u_plot=u500,
    v_plot=v500,
    z_plot=z500,
    scalar_plot=tcc_plot,
    scalar_levels=lev_tcc,
    z_levels=np.array([-80, -20, 100, 200]),
    title=f"(b) {title_head} 500UVZ & TCC anomaly",
    cbar_label="TCC anomaly (%)",
    cmap_use=cmap_b,
    qkey=2,
    qlabel="2 m/s",
    nanmax=5,
    wind_scale=4,
    add_tp_hatch=False
)


# -------------------------
# (c) 850UVZ & PRE
# -------------------------
contf3, ax3 = pic_anom(
    fig=fig,
    pic_loc=313,
    u_plot=u850,
    v_plot=v850,
    z_plot=z850,
    scalar_plot=pre_plot,
    scalar_levels=lev_pre,
    z_levels=np.array([-100, -60, -20, 80, 120]),
    title=f"(c) {title_head} 850UVZ & PRE anomaly",
    cbar_label="PRE anomaly",
    cmap_use=cmap_c,
    qkey=2,
    qlabel="2 m/s",
    nanmax=5,
    wind_scale=8,
    add_tp_hatch=True
)


# -------------------------
# 强制裁剪
# -------------------------
for ax_ in fig.axes:
    for artist in ax_.get_children():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(True)


# -------------------------
# 保存
# -------------------------
out_pdf = fr"{PYFILE}/p4/pic/{analysis_year}07_月异常场_200UVZT2M_500UVZTCC_850UVZPRE.pdf"
out_png = fr"{PYFILE}/p4/pic/{analysis_year}07_月异常场_200UVZT2M_500UVZTCC_850UVZPRE.png"

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