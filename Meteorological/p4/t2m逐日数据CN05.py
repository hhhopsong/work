import os
import numpy as np
import pandas as pd
import xarray as xr
from climkit.masked import masked
from climkit.filter import *
import matplotlib.pyplot as plt

# =========================================================
# ====================== User settings =====================
# =========================================================

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 合并后的夏季 tem 数据
NC_FILE = "/Volumes/TiPlus7100/data/CHN_GRID_DATA/tem/Tmean_19510101-20250714_0.5.nc"

# 长江流域 shp
SHP_FILE = fr"{PYFILE}/map/self/长江_TP/长江_tp.shp"

# 输出目录
OUT_DIR = fr"{PYFILE}/p4/pic/"
os.makedirs(OUT_DIR, exist_ok=True)

# 输出图片
OUT_FIG = os.path.join(OUT_DIR, r"图3_t2m逐日实际场_三线_CN05")

# 参考气候态年份
CLIM_START = 1961
CLIM_END = 2022

# 合成年份
COMPOSITE_YEARS = [1965, 1974, 1980, 1982, 1987, 1989, 1993, 1999, 2004, 2014]

# 单独分析年份
TARGET_YEAR = 2015

# 滤波参数
FILTER_MONTHS = [5, 6, 7, 8, 9]   # 滤波用 5-9 月
PLOT_MONTHS = [6, 7, 8]            # 展示用 6-8 月
NWTS = 61


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


def build_filter_day_index(time_index: pd.DatetimeIndex) -> np.ndarray:
    """
    为 5/6/7/8/9 月构造统一 day index:
    5月1日 -> 1
    ...
    9月30日 -> 153
    """
    month = time_index.month
    day = time_index.day

    offsets = np.select(
        [
            month == 5,
            month == 6,
            month == 7,
            month == 8,
            month == 9
        ],
        [
            0,    # May
            31,   # Jun
            61,   # Jul
            92,   # Aug
            123   # Sep
        ],
        default=np.nan
    )
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


def month_day_to_summer_day(month: int, day: int) -> int:
    """把月日转换为夏季日序: 6/1 -> 1, 8/31 -> 92"""
    if month == 6:
        return day
    elif month == 7:
        return 30 + day
    elif month == 8:
        return 61 + day
    else:
        raise ValueError("仅支持 6/7/8 月")


# =========================================================
# =========================== Main =========================
# =========================================================

def main():
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

    # 先保留 5-9 月，用于滤波
    da = da.sel(time=slice(f"{CLIM_START}-05-01", f"{CLIM_END}-09-30"))

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

    # 只保留 5-9 月
    df = df[df["month"].isin(FILTER_MONTHS)].copy()

    # 构造 5-9 月滤波日序
    df["filter_day"] = build_filter_day_index(pd.DatetimeIndex(df["time"]))

    # 构造 6-8 月展示日序
    df["summer_day"] = np.nan
    summer_mask = df["month"].isin(PLOT_MONTHS)
    df.loc[summer_mask, "summer_day"] = build_summer_day_index(
        pd.DatetimeIndex(df.loc[summer_mask, "time"])
    )

    # ================= 1) 1961-2022 5-9月气候态实际场 =================
    print("4) 计算 1961-2022 5-9 月逐日气候态实际场...")
    clim_df = (
        df[(df["year"] >= CLIM_START) & (df["year"] <= CLIM_END)]
        .groupby("filter_day", as_index=False)["tem"]
        .mean()
        .rename(columns={"tem": "climatology_actual"})
    )

    # ================= 2) 合成年份逐日实际场及标准差 =================
    print("5) 计算合成年份 5-9 月逐日实际场及逐日标准差...")
    comp_stat_df = (
        df[df["year"].isin(COMPOSITE_YEARS)]
        .groupby("filter_day")["tem"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "composite_actual",
            "std": "composite_std"
        })
    )
    # 如果某些 day 恰好只有一个值，std 会是 NaN，这里补成 0
    comp_stat_df["composite_std"] = comp_stat_df["composite_std"].fillna(0.0)

    # ================= 3) 2015 逐日实际场 =================
    print(f"6) 计算 {TARGET_YEAR} 年 5-9 月逐日实际场...")
    y2015_df = (
        df[df["year"] == TARGET_YEAR][["filter_day", "tem"]]
        .groupby("filter_day", as_index=False)
        .mean()
        .rename(columns={"tem": f"{TARGET_YEAR}_actual"})
    )

    # ================= 4) 先在 5-9 月滤波 =================
    print("7) 在 5-9 月范围做滤波...")
    plot_df_full = clim_df.merge(comp_stat_df, on="filter_day", how="inner")
    plot_df_full = plot_df_full.merge(y2015_df, on="filter_day", how="inner")

    clim_y_full = plot_df_full["climatology_actual"].values
    comp_y_full = plot_df_full["composite_actual"].values
    comp_std_full = plot_df_full["composite_std"].values
    y2015_y_full = plot_df_full[f"{TARGET_YEAR}_actual"].values

    # 滤波：在 5-9 月范围上做
    y2015_anom_filt_full = LanczosFilter(
        y2015_y_full - clim_y_full, 'bandpass', [10, 30], nwts=NWTS
    ).filted()

    comp_anom_filt_full = LanczosFilter(
        comp_y_full - clim_y_full, 'bandpass', [10, 30], nwts=NWTS
    ).filted()

    # 回填到完整 5-9 月长度，两端补 NaN
    y2015_filt_full_nan = np.full(len(y2015_y_full), np.nan)
    comp_filt_full_nan = np.full(len(comp_y_full), np.nan)

    m = len(y2015_anom_filt_full)
    pad_left = (len(y2015_y_full) - m) // 2
    pad_right = len(y2015_y_full) - m - pad_left

    y2015_filt_full_nan[pad_left:len(y2015_y_full) - pad_right] = y2015_anom_filt_full
    comp_filt_full_nan[pad_left:len(comp_y_full) - pad_right] = comp_anom_filt_full

    # 映射到右轴显示位置
    y2015_filt_full_plot = y2015_filt_full_nan + 24
    comp_filt_full_plot = comp_filt_full_nan + 24

    # ================= 5) 再截取 6-8 月用于绘图 =================
    print("8) 截取 6-8 月绘图...")
    # 5月1日=1, 6月1日=32, 8月31日=123
    plot_mask = (plot_df_full["filter_day"] >= 32) & (plot_df_full["filter_day"] <= 123)
    plot_df = plot_df_full.loc[plot_mask].copy()

    # 横轴重新映射成 1~92
    x = np.arange(1, len(plot_df) + 1)

    clim_y = plot_df["climatology_actual"].values
    comp_y = plot_df["composite_actual"].values
    comp_std = plot_df["composite_std"].values
    y2015_y = plot_df[f"{TARGET_YEAR}_actual"].values

    y2015_y_filt = y2015_filt_full_plot[plot_mask.values]
    comp_y_filt = comp_filt_full_plot[plot_mask.values]

    # 横坐标
    tick_positions = [1, 16, 31, 46, 62, 77, 92]
    tick_labels = ["Jun-01", "Jun-16", "Jul-01", "Jul-16", "Aug-01", "Aug-16", "Aug-31"]

    # ================= 作图 =================
    print("9) 绘图...")
    fig, ax = plt.subplots(figsize=(6, 3))

    # 研究区间背景色
    ax.axvspan(30 + 1, 30 + 11, color="#959595", alpha=0.3, zorder=0)
    ax.axvspan(30 + 15, 30 + 26, color="#959595", alpha=0.3, zorder=0)

    # ------------- 填色 -------------
    # 1) 2015 > 气候态：浅红
    ax.fill_between(
        x, y2015_y, clim_y,
        where=(y2015_y > clim_y),
        interpolate=True,
        color="lightcoral",
        alpha=0.35,
        zorder=1
    )

    # 2) 2015 < 气候态：浅蓝
    ax.fill_between(
        x, y2015_y, clim_y,
        where=(y2015_y < clim_y),
        interpolate=True,
        color="deepskyblue",
        alpha=0.35,
        zorder=1
    )

    # 3) 2015 < 合成年：深蓝
    ax.fill_between(
        x, y2015_y, comp_y,
        where=(y2015_y < comp_y),
        interpolate=True,
        color="#007bbb",
        alpha=0.6,
        zorder=2
    )

    # ------------- 三条线 -------------
    # 气候态：黑色实线
    ax.plot(
        x, clim_y,
        color="black",
        linestyle="-",
        linewidth=1,
        label="Clim.",
        zorder=4
    )

    # 合成年：蓝色虚线
    ax.plot(
        x, comp_y,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label="Comp.",
        zorder=4
    )

    # 2015 原始线（这里保留但不显著显示）
    ax.plot(
        x, y2015_y,
        color="#959595",
        linestyle="-",
        linewidth=0,
        zorder=5
    )

    # ------------- 合成年逐日标准差绿色柱 -------------
    y_base = 19.5
    ax.bar(
        x, comp_std,
        color="limegreen",
        bottom=y_base,
        alpha=0.25,
        linewidth=0,
        zorder=1.5
    )

    # ------------- 滤波线 -------------
    mask_gt = (y2015_y_filt > comp_y_filt)
    mask_lt = (y2015_y_filt <= comp_y_filt)

    ax.plot(
        x,
        np.ma.masked_where(~mask_gt, y2015_y_filt),
        color="orangered",
        linestyle="-",
        linewidth=1.3,
        zorder=5
    )

    ax.plot(
        x,
        y2015_y_filt,
        color="green",
        linestyle="-",
        linewidth=1.3,
        zorder=4
    )

    # 右轴
    secax = ax.secondary_yaxis('right')
    secax.set_yticks(np.arange(20, 29, 2))
    secax.set_yticklabels(np.arange(-4, 5, 2), fontsize=14)
    secax.tick_params(axis='y', labelsize=14)

    # ------------- 台风标志 -------------
    typhoons = [
        {"start": (6, 30), "end": (7, 13)},
        {"start": (7, 2),  "end": (7, 10)},
        {"start": (7, 30), "end": (8, 10)},
    ]

    _ty_index = 0
    for tp in typhoons:
        tp["start_day"] = month_day_to_summer_day(*tp["start"])
        tp["end_day"] = month_day_to_summer_day(*tp["end"])

        for iday in range(tp["start_day"], tp["end_day"] + 1):
            ax.text(
                iday,
                19.6 if _ty_index == 0 else (19.6 if _ty_index == 2 else 19.8),
                "·",
                ha="center",
                va="center",
                color='red',
                fontsize=25,
                clip_on=False,
                zorder=10
            )
        _ty_index += 1

    ax.set_xlim(1, 92)
    ax.set_ylim(y_base, 28.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Summer daily T2m", loc='left', fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.legend_handler import HandlerBase

    class HandlerBiColorLine(HandlerBase):
        def create_artists(self, legend, orig_handle,
                           x0, y0, width, height, fontsize, trans):
            xm = x0 + width / 2.0

            line1 = Line2D(
                [x0, xm], [y0 + height / 2, y0 + height / 2],
                color="orangered", linewidth=1.5
            )
            line2 = Line2D(
                [xm, x0 + width], [y0 + height / 2, y0 + height / 2],
                color="green", linewidth=1.5
            )

            line1.set_transform(trans)
            line2.set_transform(trans)

            return [line1, line2]

    class HandlerBiColorPatch(HandlerBase):
        """双色填色图例：左红右蓝"""
        def create_artists(self, legend, orig_handle,
                           x0, y0, width, height, fontsize, trans):
            w2 = width / 2.0

            patch1 = Rectangle(
                (x0, y0), w2, height,
                facecolor="lightcoral", edgecolor="none", alpha=0.35,
                transform=trans
            )
            patch2 = Rectangle(
                (x0 + w2, y0), w2, height,
                facecolor="deepskyblue", edgecolor="none", alpha=0.35,
                transform=trans
            )
            return [patch1, patch2]

    # ===== 主图例：线图 =====
    plt.rcParams['legend.fontsize'] = 8

    line_clim = Line2D([0], [0], color="black", lw=1, linestyle="-", label="Clim.")
    line_comp = Line2D([0], [0], color="blue", lw=1.5, linestyle="--", label="Cool sum. comp.")
    bi_line = Line2D([0], [0], color="none", label="2015 filtered anom.")

    legend1 = ax.legend(
        handles=[bi_line, line_comp, line_clim],
        handler_map={bi_line: HandlerBiColorLine()},
        frameon=False,
        loc="upper left",
        borderaxespad=0.0
    )
    ax.add_artist(legend1)

    # ===== 右侧附加图例 =====
    bar_handle = Rectangle(
        (0, 0), 1, 1,
        facecolor="limegreen",
        edgecolor="none",
        alpha=0.25,
        label="Comp. std"
    )

    fill_handle = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none")
    fill_handle.set_label("2015 anom.")

    typhoon_handle = Line2D(
        [0], [0],
        marker="o",
        color="red",
        linestyle="None",
        markersize=3,
        label="Typhoon"
    )

    legend2 = ax.legend(
        handles=[fill_handle, bar_handle, typhoon_handle],
        handler_map={fill_handle: HandlerBiColorPatch()},
        frameon=False,
        loc="upper right",
        borderaxespad=0.0
    )

    # 加粗边框
    for _ax in fig.axes:
        for spine in _ax.spines.values():
            spine.set_linewidth(1.5)

    plt.rcParams['legend.fontsize'] = 8

    plt.tight_layout()
    plt.savefig(OUT_FIG + ".png", dpi=1000, bbox_inches="tight")
    plt.savefig(OUT_FIG + ".pdf", bbox_inches="tight")
    plt.close()

    print(f"绘图完成，已保存：{OUT_FIG}")

    ds.close()


if __name__ == "__main__":
    main()
