import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from climkit.Cquiver import *
from climkit.masked import masked
from climkit.lonlat_transform import *
from climkit.filter import *
from climkit.corr_reg import *

# =========================
# 字体设置
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"


# =========================================================
# 基础函数
# =========================================================
def standardize_latlon(ds):
    """
    统一经纬度坐标名为 lon/lat
    """
    rename_dict = {}

    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"

    if "longitude" in ds.dims:
        rename_dict["longitude"] = "lon"
    if "latitude" in ds.dims:
        rename_dict["latitude"] = "lat"

    if rename_dict:
        ds = ds.rename(rename_dict)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("数据中未找到 lon/lat 或 longitude/latitude 坐标。")

    return ds


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


def mean_anom(da, start, end):
    """
    对指定时间段求时间平均
    """
    return da.sel(time=slice(start, end)).mean("time", skipna=True)


def sel_level_mean(da, level, start, end):
    """
    选取指定层次并做时间平均
    """
    if "level" not in da.dims:
        raise ValueError(f"{da.name} 没有 level 维度，不能选 {level} hPa。")

    return da.sel(level=level).sel(time=slice(start, end)).mean("time", skipna=True)


def to_latlon_2d(da):
    """
    压缩多余维度，并统一成 lat, lon 顺序
    """
    da = da.squeeze()

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"{da.name} 维度中不包含 lat/lon，当前维度为: {da.dims}")

    return da.transpose("lat", "lon")


def maybe_z_to_gpm(z_da):
    """
    ERA5 z 通常是 geopotential，单位 m^2 s^-2。
    画图时通常转为位势高度 gpm。
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
    PRE/TP 单位判断：
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


# =========================================================
# 绘图函数
# =========================================================
def pic_anom(
    fig,
    pic_loc,
    lon,
    lat,
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
    # 填色变量裁剪区域
    # -------------------------
    roi_shape = ((60, 0), (160, 60))

    try:
        scalar_roi = scalar_plot.salem.roi(corners=roi_shape)
    except Exception:
        scalar_roi = scalar_plot

    contf = ax.contourf(
        lon,
        lat,
        scalar_roi,
        levels=scalar_levels,
        cmap=cmap_use,
        extend="both",
        alpha=0.75,
        transform=ccrs.PlateCarree()
    )

    # -------------------------
    # Z 异常等值线
    # -------------------------
    z_neg = [lv for lv in z_levels if lv < 0]
    z_pos = [lv for lv in z_levels if lv > 0]

    if len(z_pos) > 0:
        cont_pos = ax.contour(
            lon,
            lat,
            z_plot,
            levels=z_pos,
            colors="red",
            linewidths=1.2,
            transform=ccrs.PlateCarree()
        )

    if len(z_neg) > 0:
        cont_neg = ax.contour(
            lon,
            lat,
            z_plot,
            levels=z_neg,
            colors="blue",
            linestyles="--",
            linewidths=1.2,
            transform=ccrs.PlateCarree()
        )

    # -------------------------
    # UV 风矢量
    # -------------------------
    Cq = ax.Curlyquiver(
        lon,
        lat,
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

    # 是否添加青藏高原填色
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
    cb.set_label(cbar_label, fontsize=8)

    return contf, ax


# =========================================================
# 主程序
# =========================================================

# =========================
# 时间范围
# =========================
analysis_start = "2015-06-15"
analysis_end = "2015-07-31"

# =========================
# 读取 2015 年异常场
# =========================
ANO = xr.open_dataset(
    "/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_ano_2015_MJJAS.nc"
)

ANO = standardize_latlon(ANO)

if "time" not in ANO.coords:
    if "valid_time" in ANO.coords:
        ANO = ANO.rename({"valid_time": "time"})
    else:
        raise ValueError("ANO 数据中没有 time 或 valid_time 坐标。")

# =========================
# 读取变量
# =========================
u_ano_full = get_first_var(
    ANO,
    ["u_ano", "u_anom", "u"],
    "U 异常"
)

v_ano_full = get_first_var(
    ANO,
    ["v_ano", "v_anom", "v"],
    "V 异常"
)

z_ano_full = get_first_var(
    ANO,
    ["z_ano", "z_anom", "z"],
    "Z 异常"
)

t2m_ano_full = get_first_var(
    ANO,
    ["t2m_ano", "t2m_anom", "t2m", "T2M_ano", "T2M"],
    "T2M 异常"
)

tcc_ano_full = get_first_var(
    ANO,
    ["tcc_ano", "tcc_anom", "tcc", "TCC_ano", "TCC"],
    "TCC 异常"
)

pre_ano_full = get_first_var(
    ANO,
    [
        "pre_ano",
        "pre_anom",
        "pre",
        "PRE_ano",
        "PRE",
        "precip_ano",
        "precip_anom",
        "precip",
        "tp_ano",
        "tp_anom",
        "tp"
    ],
    "PRE/降水异常"
)*1000*31

# =========================
# 坐标
# =========================
lon = ANO["lon"]
lat = ANO["lat"]

# =========================
# 200 hPa UVZ
# =========================
u200 = to_latlon_2d(
    sel_level_mean(u_ano_full, 200, analysis_start, analysis_end)
)

v200 = to_latlon_2d(
    sel_level_mean(v_ano_full, 200, analysis_start, analysis_end)
)

z200 = to_latlon_2d(
    maybe_z_to_gpm(
        sel_level_mean(z_ano_full, 200, analysis_start, analysis_end)
    )
)

# =========================
# 500 hPa UVZ
# =========================
u500 = to_latlon_2d(
    sel_level_mean(u_ano_full, 500, analysis_start, analysis_end)
)

v500 = to_latlon_2d(
    sel_level_mean(v_ano_full, 500, analysis_start, analysis_end)
)

z500 = to_latlon_2d(
    maybe_z_to_gpm(
        sel_level_mean(z_ano_full, 500, analysis_start, analysis_end)
    )
)

# =========================
# 850 hPa UVZ
# =========================
u850 = to_latlon_2d(
    sel_level_mean(u_ano_full, 850, analysis_start, analysis_end)
)

v850 = to_latlon_2d(
    sel_level_mean(v_ano_full, 850, analysis_start, analysis_end)
)

z850 = to_latlon_2d(
    maybe_z_to_gpm(
        sel_level_mean(z_ano_full, 850, analysis_start, analysis_end)
    )
)

# =========================
# 填色变量
# =========================
t2m_plot = to_latlon_2d(
    mean_anom(t2m_ano_full, analysis_start, analysis_end)
)

tcc_plot = to_latlon_2d(
    maybe_tcc_to_percent(
        mean_anom(tcc_ano_full, analysis_start, analysis_end)
    )
)

pre_plot = to_latlon_2d(
    maybe_pre_to_mmday(
        mean_anom(pre_ano_full, analysis_start, analysis_end)
    )
)

# =========================================================
# 色阶设置
# =========================================================
# 图a：T2M
lev_t2m = np.array([
    -3, -2, -1, -0.5, -0.2,
     0.2, 0.5, 1, 2, 3
])

# 图b：TCC (%)
lev_tcc = np.array([
    -20, -15, -10, -5, -2,
      2,   5,  10, 15, 20
])

# 图c：PRE
lev_pre = np.array([
    -8, -6, -4, -2, -1,
     1,  2,  4,  6,  8
])


# =========================================================
# 配色
# =========================================================
# 图a 保持原来的冷暖型
cmap_a = cmaps.GMT_polar[:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:]

# 图b 改为棕紫色系
cmap_b = plt.get_cmap("PuOr")

# 图c 改为褐绿色系
cmap_c = plt.get_cmap("BrBG")

# =========================================================
# 正式绘图
# =========================================================
fig = plt.figure(figsize=(4.3, 8.4))
plt.subplots_adjust(wspace=0.15, hspace=0.28)

title_head = "2015"

# -------------------------
# (a) 200UVZ & T2M
# 不要青藏高原填色
# 风场 scale = 4
# -------------------------
contf1, ax1 = pic_anom(
    fig=fig,
    pic_loc=311,
    lon=lon,
    lat=lat,
    u_plot=u200,
    v_plot=v200,
    z_plot=z200,
    scalar_plot=t2m_plot,
    scalar_levels=lev_t2m,
    z_levels=np.array([-160,  -40, 140,  280]),
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
# 不要青藏高原填色
# 风场 scale = 4
# 棕紫色系
# -------------------------
contf2, ax2 = pic_anom(
    fig=fig,
    pic_loc=312,
    lon=lon,
    lat=lat,
    u_plot=u500,
    v_plot=v500,
    z_plot=z500,
    scalar_plot=tcc_plot,
    scalar_levels=lev_tcc,
    z_levels=np.array([-80,  -20, 100,  200]),
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
# 保留青藏高原填色
# 风场 scale = 8
# 褐绿色系
# -------------------------
contf3, ax3 = pic_anom(
    fig=fig,
    pic_loc=313,
    lon=lon,
    lat=lat,
    u_plot=u850,
    v_plot=v850,
    z_plot=z850,
    scalar_plot=pre_plot,
    scalar_levels=lev_pre,
    z_levels=np.array([-100, -60,  -20, 80,  120]),
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
plt.savefig(
    fr"{PYFILE}/p4/pic/2015异常场_200UVZT2M_500UVZTCC_850UVZPRE.pdf",
    bbox_inches="tight"
)

plt.savefig(
    fr"{PYFILE}/p4/pic/2015异常场_200UVZT2M_500UVZTCC_850UVZPRE.png",
    bbox_inches="tight",
    dpi=600
)

plt.show()
