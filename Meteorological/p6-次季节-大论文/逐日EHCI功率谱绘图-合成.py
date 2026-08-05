
"""
逐年逐日 EHCI 的合成全局小波功率谱。

EHCI 为长江流域内超过极端高温温度阈值的格点比例。

处理流程：
1. 计算 1961--2022 年逐日 EHCI；
2. 对每年 6--8 月连续 EHCI 序列分别进行 Morlet 小波分析；
3. WaveletAnalysis 使用 normal=False，不对输入序列标准化；
4. analysis.global_power 直接作为原始量纲全局功率谱，
   不乘年度方差；
5. 90% 全局显著性水平直接使用 WaveletAnalysis 返回值，
   不乘年度方差；
6. 将各年度全局功率谱按年份排列，形成“年份--周期”功率场；
7. 左侧绘制逐年功率场，右侧绘制多年平均全局功率谱；
8. 绘图不再进行任何归一化，直接展示原始 EHCI^2 功率。

绘图调整：
1. 左侧 contourf 最低一级功率区间固定显示为白色；
2. 左右主图之间不留空隙；
3. 左图右边框与右图左边界完全重合，使用单条共用框线。

重要说明：
- EHCI_SMOOTHING_WINDOW=5 时，分析的是 5 天滑动平均后的
  逐日 EHCI，严格来说不是未经平滑的原始逐日 EHCI；
- 将 EHCI_SMOOTHING_WINDOW 改为 1，可关闭滑动平均；
- 左侧图不是对跨年度连续序列进行一次小波变换，而是将
  各年度 6--8 月的全局小波功率谱按年份排列；
- “原始小波分析”在本代码中是指 normal=False，即不将
  各年度 EHCI 序列除以标准差后再分析；
- global_power 和 global_signif 均直接使用 WaveletAnalysis
  返回值，不进行方差回乘。
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr

from climkit.masked import masked
from climkit.wavelet import WaveletAnalysis


# =========================================================
# 配置
# =========================================================
PYFILE = Path("/Volumes/TiPlus7100/PyFile")

TMAX_FILE = PYFILE / "p2/data/Tmax_5Day_filt.nc"

THRESHOLD_SHP = (
    PYFILE
    / "map/地图边界数据/长江区1：25万界线数据集（2002年）"
    / "长江区.shp"
)

YANGTZE_SHP = (
    PYFILE
    / "map/self/长江_TP/长江_tp.shp"
)

OUTPUT_DIR = PYFILE / "p6/pic"

START_YEAR = 1961
END_YEAR = 2023

# 空间温度阈值分位数
TEMPERATURE_QUANTILE = 0.95

# 原始 Tmax_5Day_filt.nc 中：
# day=28 对应 7 月 1 日，
# 因而 6 月 1 日--8 月 31 日理论上对应 day=-2--89。
DAY_SLICE = slice(-2, 89)

# 小波绘图所保留的最大周期
MAX_PERIOD = 32.0

# 小波全局显著性水平
SIGNIFICANCE_LEVEL = 0.90

# 逐日 EHCI 滑动平均窗口：
# 1 表示不平滑；
# 5 表示 5 天中心滑动平均。
EHCI_SMOOTHING_WINDOW = 5

# 左侧色标上限所采用的百分位数。
# 使用 99% 分位数可以避免极个别年份的强峰值压缩其他年份的颜色。
# 设置为 100，则使用全部功率的最大值。
COLORBAR_PERCENTILE = 99.0

# 小波参数
WAVELET_NAME = "Morlet"
WAVELET_J = 7
TIME_STEP = 1.0

# 图片参数
FIGURE_SIZE = (11.2, 5.3)
PNG_DPI = 600

# 左侧功率场基础配色
POWER_CMAP = "YlOrRd"


# =========================================================
# 通用工具
# =========================================================
def normalize_integer_coordinate(
    da: xr.DataArray,
    coord: str,
) -> xr.DataArray:
    """将可能以字符串保存的 year/day 坐标转换为整数。"""
    if coord not in da.coords:
        raise ValueError(f"数据中不存在坐标：{coord!r}")

    values = np.asarray(da[coord].values)

    try:
        integer_values = values.astype(int)
    except (TypeError, ValueError) as exc:
        preview = values[: min(5, values.size)]

        raise ValueError(
            f"无法将坐标 {coord!r} 转换为整数，"
            f"示例值为：{preview}"
        ) from exc

    return da.assign_coords({coord: integer_values})


def check_required_files() -> None:
    """检查输入文件是否存在。"""
    required_paths = (
        TMAX_FILE,
        THRESHOLD_SHP,
        YANGTZE_SHP,
    )

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing_text = "\n".join(
            f"  - {path}"
            for path in missing_paths
        )

        raise FileNotFoundError(
            f"以下必要文件不存在：\n{missing_text}"
        )


def get_output_stem() -> Path:
    """根据平滑窗口生成输出文件名。"""
    if EHCI_SMOOTHING_WINDOW == 1:
        ehci_label = "原始逐日EHCI"
    else:
        ehci_label = (
            f"{EHCI_SMOOTHING_WINDOW}天平滑逐日EHCI"
        )

    filename = (
        f"{ehci_label}_未标准化原始量纲_"
        f"逐年小波功率年份周期场_"
        f"6-8月_{int(SIGNIFICANCE_LEVEL * 100)}显著性"
    )

    return OUTPUT_DIR / filename


def calculate_nanmean(
    data: np.ndarray,
    axis: int,
) -> np.ndarray:
    """
    在避免 RuntimeWarning 的情况下计算 NaN 平均值。

    对全部为 NaN 的位置返回 NaN。
    """
    data = np.asarray(data, dtype=float)

    valid_count = np.sum(
        np.isfinite(data),
        axis=axis,
    )

    data_sum = np.nansum(
        data,
        axis=axis,
    )

    output_shape = np.asarray(data_sum).shape

    mean_value = np.divide(
        data_sum,
        valid_count,
        out=np.full(
            output_shape,
            np.nan,
            dtype=float,
        ),
        where=valid_count > 0,
    )

    return mean_value


def build_year_ticks(
    years: np.ndarray,
) -> np.ndarray:
    """生成年份横坐标刻度。"""
    start_year = int(np.min(years))
    end_year = int(np.max(years))

    first_decade = (
        (start_year + 9) // 10
    ) * 10

    decade_ticks = list(
        range(
            first_decade,
            end_year + 1,
            10,
        )
    )

    ticks = [
        start_year,
        *decade_ticks,
    ]

    ticks = sorted(set(ticks))

    return np.asarray(ticks, dtype=int)


def build_power_colormap(
    color_levels: np.ndarray,
) -> ListedColormap:
    """
    创建最低一级为纯白色的离散功率配色。

    contourf 的颜色对应 levels 之间的区间。
    若共有 N 个 levels，则共有 N-1 个普通颜色区间。

    第一个区间：
        color_levels[0] <= power < color_levels[1]
        固定显示为纯白色。

    其余区间：
        使用 POWER_CMAP 的黄色至红色部分。
    """
    color_levels = np.asarray(
        color_levels,
        dtype=float,
    )

    if color_levels.ndim != 1:
        raise ValueError(
            "color_levels 必须是一维数组。"
        )

    interval_count = color_levels.size - 1

    if interval_count < 1:
        raise ValueError(
            "color_levels 至少需要两个数值。"
        )

    base_cmap = plt.get_cmap(
        POWER_CMAP
    )

    # 只有一个颜色区间时，直接使用白色。
    if interval_count == 1:
        colors = np.asarray(
            [[1.0, 1.0, 1.0, 1.0]]
        )

    else:
        # 跳过 YlOrRd 最浅的接近白色部分，
        # 避免第二级颜色与第一级纯白色不易区分。
        remaining_colors = base_cmap(
            np.linspace(
                0.14,
                1.0,
                interval_count - 1,
            )
        )

        white_color = np.asarray(
            [[1.0, 1.0, 1.0, 1.0]]
        )

        colors = np.vstack(
            [
                white_color,
                remaining_colors,
            ]
        )

    discrete_cmap = ListedColormap(
        colors,
        name=f"{POWER_CMAP}_first_level_white",
    )

    # 超过最高 levels 的值使用基础色表最深颜色。
    discrete_cmap.set_over(
        base_cmap(1.0)
    )

    # NaN 区域设置为透明。
    discrete_cmap.set_bad(
        color=(1.0, 1.0, 1.0, 0.0)
    )

    return discrete_cmap


# =========================================================
# EHCI 数据处理
# =========================================================
def build_daily_ehci() -> xr.DataArray:
    """按照现有事件脚本定义计算逐年逐日 EHCI。"""
    check_required_files()

    print("1) 读取 Tmax 数据……")

    tmax = xr.open_dataarray(
        TMAX_FILE,
        decode_times=False,
    )

    required_dims = {
        "year",
        "day",
        "lat",
        "lon",
    }

    missing_dims = required_dims - set(tmax.dims)

    if missing_dims:
        raise ValueError(
            f"Tmax 数据缺少必要维度："
            f"{sorted(missing_dims)}；"
            f"当前维度为：{list(tmax.dims)}"
        )

    tmax = normalize_integer_coordinate(
        tmax,
        "year",
    )

    tmax = normalize_integer_coordinate(
        tmax,
        "day",
    )

    tmax = (
        tmax
        .sel(year=slice(START_YEAR, END_YEAR))
        .sortby("year")
        .sortby("day")
    )

    if tmax.sizes.get("year", 0) == 0:
        raise ValueError(
            f"Tmax 数据中没有 "
            f"{START_YEAR}--{END_YEAR} 年资料。"
        )

    print(
        "2) 计算极端高温温度阈值……"
    )

    # -----------------------------------------------------
    # 与原事件脚本保持一致：
    #
    # 1. 对阈值区域内的资料求多年、逐日平均；
    # 2. 得到一张空间平均温度场；
    # 3. 对该空间场取 TEMPERATURE_QUANTILE 分位数；
    # 4. 最终得到一个标量温度阈值。
    # -----------------------------------------------------
    threshold_region_tmax = masked(
        tmax,
        str(THRESHOLD_SHP),
    )

    threshold_field = threshold_region_tmax.mean(
        dim=["year", "day"],
        skipna=True,
    )

    threshold_da = threshold_field.quantile(
        TEMPERATURE_QUANTILE,
        dim=["lat", "lon"],
        skipna=True,
    ).squeeze(drop=True)

    temperature_threshold = float(
        np.asarray(
            threshold_da.values
        ).squeeze()
    )

    if not np.isfinite(temperature_threshold):
        raise ValueError(
            "计算得到的极端高温温度阈值为 NaN 或无穷值，"
            "请检查 Tmax 数据和阈值区域 shp 掩膜。"
        )

    print(
        f"   极端高温温度阈值："
        f"{temperature_threshold:.4f}"
    )

    # -----------------------------------------------------
    # 极端高温格点：
    # 1   = 超过温度阈值
    # 0   = 未超过温度阈值
    # NaN = 原资料缺测
    # -----------------------------------------------------
    ehd = xr.where(
        tmax.isnull(),
        np.nan,
        xr.where(
            tmax > temperature_threshold,
            1.0,
            0.0,
        ),
    )

    print(
        "3) 裁剪长江流域并计算 EHCI……"
    )

    ehd = masked(
        ehd,
        str(YANGTZE_SHP),
    )

    # 使用固定流域网格计算分母。
    grid_template = xr.ones_like(
        tmax.isel(
            year=0,
            day=0,
            drop=True,
        ),
        dtype=float,
    )

    basin_grid_template = masked(
        grid_template,
        str(YANGTZE_SHP),
    )

    grid_count = float(
        np.asarray(
            basin_grid_template.sum(
                skipna=True
            ).values
        ).squeeze()
    )

    if (
        not np.isfinite(grid_count)
        or grid_count <= 0
    ):
        raise ValueError(
            f"长江流域有效格点数异常："
            f"{grid_count}"
        )

    print(
        f"   长江流域固定有效格点数："
        f"{grid_count:.0f}"
    )

    # EHCI = 流域内极端高温格点数 / 流域固定格点数
    #
    # min_count=1 可避免某一时次全部为 NaN 时，
    # xarray 将其错误地求和为 0。
    extreme_grid_count = ehd.sum(
        dim=["lat", "lon"],
        skipna=True,
        min_count=1,
    )

    ehci = extreme_grid_count / grid_count

    ehci = (
        ehci
        .sel(day=DAY_SLICE)
        .transpose("year", "day")
    )

    if ehci.sizes.get("day", 0) == 0:
        raise ValueError(
            f"DAY_SLICE={DAY_SLICE} "
            "没有选择到任何日期。"
        )

    day_min = int(ehci.day.min())
    day_max = int(ehci.day.max())
    day_count = ehci.sizes["day"]

    if day_count != 92:
        print(
            "   提示：6--8 月理论上应有 92 天；"
            f"当前资料提供 {day_count} 天，"
            f"day={day_min}--{day_max}。"
        )

    # -----------------------------------------------------
    # EHCI 平滑
    # -----------------------------------------------------
    if EHCI_SMOOTHING_WINDOW < 1:
        raise ValueError(
            "EHCI_SMOOTHING_WINDOW "
            "必须大于或等于 1。"
        )

    if EHCI_SMOOTHING_WINDOW > 1:
        ehci = ehci.rolling(
            day=EHCI_SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        ).mean()

        print(
            f"   已对逐日 EHCI 进行 "
            f"{EHCI_SMOOTHING_WINDOW} 天"
            "中心滑动平均。"
        )
    else:
        print(
            "   未对逐日 EHCI 进行滑动平均。"
        )

    ehci.name = "EHCI"

    ehci.attrs.update(
        {
            "description": (
                "Fraction of Yangtze River basin grid "
                "cells exceeding the extreme-temperature "
                "threshold"
            ),
            "temperature_threshold": (
                temperature_threshold
            ),
            "temperature_quantile": (
                TEMPERATURE_QUANTILE
            ),
            "smoothing_window_days": (
                EHCI_SMOOTHING_WINDOW
            ),
        }
    )

    return ehci


# =========================================================
# 单年度小波分析
# =========================================================
def calculate_global_spectrum(
    ehci_series: np.ndarray,
) -> Optional[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ]
]:
    """
    计算单年度 EHCI 的原始量纲全局小波功率谱和显著性水平。

    本版本不进行标准化和方差回乘：

    1. WaveletAnalysis 使用 normal=False；
    2. analysis.global_power 直接作为全局功率谱；
    3. analysis.global_signif 直接作为全局显著性水平；
    4. global_power 和 global_signif 都不再乘年度方差；
    5. 年度方差仅作为诊断变量保存到 NetCDF。
    """
    series = np.asarray(
        ehci_series,
        dtype=float,
    ).squeeze()

    if series.ndim != 1:
        raise ValueError(
            "EHCI 序列必须是一维，"
            f"当前形状为：{series.shape}"
        )

    if series.size < 4:
        raise ValueError(
            f"EHCI 序列长度过短：{series.size}"
        )

    if not np.isfinite(series).all():
        bad_count = int(
            np.count_nonzero(
                ~np.isfinite(series)
            )
        )

        raise ValueError(
            f"EHCI 序列包含 {bad_count} 个 "
            "NaN 或无穷值，无法进行小波分析。"
        )

    series_variance = float(
        np.var(series)
    )

    # 常数或数值上近似常数的序列不进行小波分析。
    if (
        not np.isfinite(series_variance)
        or np.isclose(
            series_variance,
            0.0,
            rtol=0.0,
            atol=np.finfo(float).eps,
        )
    ):
        return None

    # -----------------------------------------------------
    # 不对输入序列进行标准化。
    # -----------------------------------------------------
    analysis = WaveletAnalysis(
        series,
        wave=WAVELET_NAME,
        dt=TIME_STEP,
        detrend=False,
        normal=False,
        signal=SIGNIFICANCE_LEVEL,
        J=WAVELET_J,
    )

    period = np.asarray(
        analysis.period,
        dtype=float,
    ).squeeze()

    # global_power 直接使用，不乘方差。
    global_power = np.asarray(
        analysis.global_power,
        dtype=float,
    ).squeeze()

    # global_signif 直接使用，不乘方差。
    global_significance = np.asarray(
        analysis.global_signif,
        dtype=float,
    ).squeeze()

    if not (
        period.shape
        == global_power.shape
        == global_significance.shape
    ):
        raise ValueError(
            "周期、全局功率和显著性数组形状不一致："
            f"period={period.shape}, "
            f"power={global_power.shape}, "
            f"significance={global_significance.shape}"
        )

    valid = (
        np.isfinite(period)
        & np.isfinite(global_power)
        & np.isfinite(global_significance)
        & (period > 0)
        & (period <= MAX_PERIOD)
    )

    period = period[valid]
    global_power = global_power[valid]
    global_significance = global_significance[valid]

    if period.size == 0:
        return None

    order = np.argsort(period)

    return (
        period[order],
        global_power[order],
        global_significance[order],
        series_variance,
    )


# =========================================================
# 逐年功率谱场
# =========================================================
def calculate_spectrum_field(
    ehci: xr.DataArray,
) -> Optional[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
]:
    """
    计算逐年原始量纲全局功率谱场和逐年显著性水平。

    返回数组形状：
    power.shape        = (period, year)
    significance.shape = (period, year)
    """
    years = np.asarray(
        ehci.year.values,
        dtype=int,
    )

    period_reference = None

    yearly_results = []
    yearly_variances = []

    print(
        "\n4) 逐年进行未标准化 Morlet 小波分析……"
    )

    for year in years:
        series = np.asarray(
            ehci.sel(year=year).values,
            dtype=float,
        )

        result = calculate_global_spectrum(
            series
        )

        if result is None:
            print(
                f"   {year}: EHCI 序列为常数，"
                "或未得到有效周期，填充为 NaN。"
            )

            yearly_results.append(None)
            yearly_variances.append(np.nan)
            continue

        (
            period,
            power,
            significance,
            variance,
        ) = result

        if period_reference is None:
            period_reference = period.copy()

        elif not np.allclose(
            period,
            period_reference,
            rtol=1e-7,
            atol=1e-10,
        ):
            raise ValueError(
                f"{year} 年的小波周期坐标"
                "与其他年份不一致，"
                "无法合成为功率谱场。"
            )

        yearly_results.append(
            (power, significance)
        )

        yearly_variances.append(
            variance
        )

        power_significance_ratio = np.divide(
            power,
            significance,
            out=np.full_like(
                power,
                np.nan,
            ),
            where=significance > 0,
        )

        if np.isfinite(
            power_significance_ratio
        ).any():
            max_ratio = float(
                np.nanmax(
                    power_significance_ratio
                )
            )
        else:
            max_ratio = np.nan

        print(
            f"   {year}: "
            f"raw_variance={variance:.6e}, "
            "max(power/significance)="
            f"{max_ratio:.4f}"
        )

    if period_reference is None:
        return None

    period_count = period_reference.size

    power_rows = []
    significance_rows = []

    for result in yearly_results:
        if result is None:
            power_rows.append(
                np.full(
                    period_count,
                    np.nan,
                    dtype=float,
                )
            )

            significance_rows.append(
                np.full(
                    period_count,
                    np.nan,
                    dtype=float,
                )
            )

        else:
            power, significance = result

            power_rows.append(power)
            significance_rows.append(
                significance
            )

    # 年份为行，周期为列，再转置为 period × year。
    power_field = np.asarray(
        power_rows,
        dtype=float,
    ).T

    significance_field = np.asarray(
        significance_rows,
        dtype=float,
    ).T

    variance_array = np.asarray(
        yearly_variances,
        dtype=float,
    )

    return (
        years,
        period_reference,
        power_field,
        significance_field,
        variance_array,
    )


# =========================================================
# 保存 NetCDF
# =========================================================
def save_spectrum_data(
    years: np.ndarray,
    period: np.ndarray,
    power: np.ndarray,
    significance: np.ndarray,
    variances: np.ndarray,
) -> Path:
    """保存逐年原始量纲功率谱场及多年平均结果为 NetCDF。"""
    mean_power = calculate_nanmean(
        power,
        axis=1,
    )

    mean_significance = calculate_nanmean(
        significance,
        axis=1,
    )

    power_to_significance = np.divide(
        mean_power,
        mean_significance,
        out=np.full(
            period.size,
            np.nan,
            dtype=float,
        ),
        where=mean_significance > 0,
    )

    valid_year = (
        np.isfinite(power)
        & np.isfinite(significance)
    )

    significant_year = (
        valid_year
        & (power > significance)
    )

    significant_year_count = np.sum(
        significant_year,
        axis=1,
    )

    valid_test_count = np.sum(
        valid_year,
        axis=1,
    )

    significant_year_fraction = np.divide(
        significant_year_count,
        valid_test_count,
        out=np.full(
            period.size,
            np.nan,
            dtype=float,
        ),
        where=valid_test_count > 0,
    )

    # 1 = 显著；0 = 不显著；-1 = 缺测或未检验。
    annual_significant_mask = np.full(
        power.shape,
        -1,
        dtype=np.int8,
    )

    annual_significant_mask[valid_year] = (
        power[valid_year]
        > significance[valid_year]
    ).astype(np.int8)

    dataset = xr.Dataset(
        data_vars={
            "global_power": (
                ("period", "year"),
                power,
                {
                    "long_name": (
                        "Annual non-normalized "
                        "global wavelet power"
                    ),
                    "units": "EHCI^2",
                },
            ),
            "global_significance": (
                ("period", "year"),
                significance,
                {
                    "long_name": (
                        "Annual non-normalized global wavelet "
                        f"{int(SIGNIFICANCE_LEVEL * 100)} "
                        "percent significance level"
                    ),
                    "units": "EHCI^2",
                },
            ),
            "annual_significant_mask": (
                ("period", "year"),
                annual_significant_mask,
                {
                    "long_name": (
                        "Annual significance mask"
                    ),
                    "flag_values": [-1, 0, 1],
                    "flag_meanings": (
                        "invalid not_significant significant"
                    ),
                    "units": "1",
                },
            ),
            "annual_variance": (
                ("year",),
                variances,
                {
                    "long_name": (
                        "Annual raw EHCI variance"
                    ),
                    "units": "EHCI^2",
                },
            ),
            "mean_global_power": (
                ("period",),
                mean_power,
                {
                    "long_name": (
                        "Mean annual non-normalized "
                        "global wavelet power"
                    ),
                    "units": "EHCI^2",
                },
            ),
            "mean_global_significance": (
                ("period",),
                mean_significance,
                {
                    "long_name": (
                        "Mean annual non-normalized global "
                        f"{int(SIGNIFICANCE_LEVEL * 100)} "
                        "percent significance level"
                    ),
                    "units": "EHCI^2",
                },
            ),
            "power_to_significance_ratio": (
                ("period",),
                power_to_significance,
                {
                    "long_name": (
                        "Ratio of mean global power "
                        "to mean global significance"
                    ),
                    "units": "1",
                },
            ),
            "significant_year_fraction": (
                ("period",),
                significant_year_fraction,
                {
                    "long_name": (
                        "Fraction of valid years "
                        "with annual power above "
                        "annual significance"
                    ),
                    "units": "1",
                },
            ),
        },
        coords={
            "year": (
                "year",
                years,
                {
                    "long_name": "Year",
                    "units": "year",
                },
            ),
            "period": (
                "period",
                period,
                {
                    "long_name": (
                        "Fourier-equivalent period"
                    ),
                    "units": "days",
                },
            ),
        },
        attrs={
            "title": (
                "Daily EHCI annual non-normalized "
                "global wavelet spectra"
            ),
            "start_year": START_YEAR,
            "end_year": END_YEAR,
            "summer_months": "June-August",
            "wavelet": WAVELET_NAME,
            "wavelet_J": WAVELET_J,
            "time_step_days": TIME_STEP,
            "significance_level": (
                SIGNIFICANCE_LEVEL
            ),
            "maximum_period_days": (
                MAX_PERIOD
            ),
            "temperature_quantile": (
                TEMPERATURE_QUANTILE
            ),
            "ehci_smoothing_window_days": (
                EHCI_SMOOTHING_WINDOW
            ),
            "wavelet_input_normalized": "false",
            "power_scaling": (
                "WaveletAnalysis called with normal=False; "
                "global_power used directly; "
                "global_significance used directly; "
                "no variance multiplication and "
                "no plotting normalization"
            ),
            "field_definition": (
                "Each column is the non-normalized global "
                "wavelet spectrum of one year's "
                "June-August EHCI sequence"
            ),
        },
    )

    output_file = get_output_stem().with_suffix(
        ".nc"
    )

    dataset.to_netcdf(
        output_file
    )

    print(
        f"\n功率谱数据已保存：\n"
        f"   {output_file}"
    )

    return output_file


# =========================================================
# 绘图
# =========================================================
def plot_spectrum_fields(
    ehci: xr.DataArray,
) -> None:
    """
    绘制双面板图。

    左侧：
        年份--周期逐年原始量纲全局小波功率场。

    右侧：
        多年平均原始量纲全局功率谱与平均显著性线。

    布局：
        左右两幅主图完全相接，不保留水平空隙。
        左图右边框作为两幅图之间的共用边界线。
    """
    field = calculate_spectrum_field(
        ehci
    )

    if field is None:
        raise ValueError(
            "所有年份的 EHCI 序列均为常数，"
            "没有得到有效功率谱场。"
        )

    (
        years,
        period,
        power_raw,
        significance_raw,
        variances,
    ) = field

    save_spectrum_data(
        years=years,
        period=period,
        power=power_raw,
        significance=significance_raw,
        variances=variances,
    )

    # -----------------------------------------------------
    # 多年平均功率谱
    # -----------------------------------------------------
    mean_power_raw = calculate_nanmean(
        power_raw,
        axis=1,
    )

    mean_significance_raw = calculate_nanmean(
        significance_raw,
        axis=1,
    )

    valid_period = (
        np.isfinite(period)
        & np.isfinite(mean_power_raw)
        & np.isfinite(mean_significance_raw)
        & (period > 0)
        & (period <= MAX_PERIOD)
    )

    period = period[valid_period]

    power_raw = power_raw[
        valid_period,
        :
    ]

    significance_raw = significance_raw[
        valid_period,
        :
    ]

    mean_power_raw = mean_power_raw[
        valid_period
    ]

    mean_significance_raw = (
        mean_significance_raw[
            valid_period
        ]
    )

    if period.size == 0:
        raise ValueError(
            "多年平均功率谱或显著性水平"
            "全部为 NaN。"
        )

    # 显著性判断在 WaveletAnalysis 返回的相同原始量纲下完成。
    annual_significant = (
        np.isfinite(power_raw)
        & np.isfinite(significance_raw)
        & (power_raw > significance_raw)
    )

    mean_significant = (
        mean_power_raw
        > mean_significance_raw
    )

    # -----------------------------------------------------
    # 不进行绘图归一化：
    # 左右面板都直接使用原始量纲数据。
    # -----------------------------------------------------
    power_plot = power_raw.copy()

    mean_power_plot = (
        mean_power_raw.copy()
    )

    mean_significance_plot = (
        mean_significance_raw.copy()
    )

    power_unit_label = (
        r""
    )

    mean_power_xlabel = (
        r"Global power"
    )

    # -----------------------------------------------------
    # 诊断信息
    # -----------------------------------------------------
    ratio = np.divide(
        mean_power_raw,
        mean_significance_raw,
        out=np.full_like(
            mean_power_raw,
            np.nan,
        ),
        where=mean_significance_raw > 0,
    )

    print(
        "\n5) 多年平均原始量纲功率谱诊断："
    )

    if np.isfinite(ratio).any():
        best_index = int(
            np.nanargmax(ratio)
        )

        print(
            "   最大 power/significance 比值："
            f"{ratio[best_index]:.4f}"
        )

        print(
            "   对应周期："
            f"{period[best_index]:.3f} 天"
        )

        print(
            "   对应平均功率："
            f"{mean_power_raw[best_index]:.6e}"
        )

        print(
            "   对应平均显著性："
            f"{mean_significance_raw[best_index]:.6e}"
        )

    significant_periods = period[
        mean_significant
    ]

    if significant_periods.size > 0:
        print(
            "   多年平均功率超过平均显著性线的"
            "周期范围："
            f"{significant_periods.min():.3f}--"
            f"{significant_periods.max():.3f} 天"
        )
    else:
        print(
            "   多年平均功率未超过"
            f"平均 {int(SIGNIFICANCE_LEVEL * 100)}% "
            "显著性线。"
        )

    # -----------------------------------------------------
    # 色标范围
    # -----------------------------------------------------
    finite_power = power_plot[
        np.isfinite(power_plot)
        & (power_plot >= 0)
    ]

    if finite_power.size == 0:
        raise ValueError(
            "左侧功率场全部为 NaN。"
        )

    if not (
        0 < COLORBAR_PERCENTILE <= 100
    ):
        raise ValueError(
            "COLORBAR_PERCENTILE 必须位于 "
            "(0, 100] 范围内。"
        )

    color_vmax = float(
        np.nanpercentile(
            finite_power,
            COLORBAR_PERCENTILE,
        )
    )

    if (
        not np.isfinite(color_vmax)
        or color_vmax <= 0
    ):
        color_vmax = float(
            np.nanmax(
                finite_power
            )
        )

    if (
        not np.isfinite(color_vmax)
        or color_vmax <= 0
    ):
        color_vmax = 1.0

    color_levels = np.linspace(
        0.0,
        color_vmax,
        16,
    )

    # 创建最低一级为纯白色的离散配色。
    power_cmap = build_power_colormap(
        color_levels
    )

    # -----------------------------------------------------
    # 坐标
    # -----------------------------------------------------
    log2_period = np.log2(period)

    period_ticks = np.asarray(
        [2, 4, 8, 16, 32],
        dtype=float,
    )

    period_ticks = period_ticks[
        period_ticks <= MAX_PERIOD
    ]

    year_ticks = build_year_ticks(
        years
    )

    # -----------------------------------------------------
    # 创建双面板
    #
    # 使用两行 GridSpec：
    # 第一行为左右两个主图；
    # 第二行为左侧水平色标和右侧空白占位。
    #
    # 这样色标不会单独压缩左侧主图，能够保证左右主图
    # 高度完全一致，并且在 wspace=0 时边框准确相接。
    # -----------------------------------------------------
    print(
        "\n6) 绘制未标准化年份--周期功率场"
        "及多年平均全局功率谱……"
    )

    fig = plt.figure(
        figsize=FIGURE_SIZE,
        constrained_layout=False,
    )

    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[4.8, 1.55],
        height_ratios=[1.0, 0.075],
        left=0.085,
        right=0.985,
        bottom=0.125,
        top=0.865,
        wspace=0.0,
        hspace=0.40,
    )

    ax_field = fig.add_subplot(
        grid[0, 0]
    )

    ax_global = fig.add_subplot(
        grid[0, 1],
        sharey=ax_field,
    )

    colorbar_axis = fig.add_subplot(
        grid[1, 0]
    )

    blank_axis = fig.add_subplot(
        grid[1, 1]
    )

    blank_axis.axis("off")

    # -----------------------------------------------------
    # 左侧：年份--周期功率场
    # -----------------------------------------------------
    masked_power = np.ma.masked_invalid(
        power_plot
    )

    contour = ax_field.contourf(
        years,
        log2_period,
        masked_power,
        levels=color_levels,
        cmap=power_cmap,
        extend="max",
    )

    # 显著性点：
    # 每个黑点表示该年度该周期的原始量纲功率
    # 超过该年度对应的 90% 全局显著性水平。
    year_grid, period_grid = np.meshgrid(
        years,
        log2_period,
    )

    ax_field.scatter(
        year_grid[annual_significant],
        period_grid[annual_significant],
        s=7,
        marker=".",
        color="black",
        linewidths=0,
        alpha=0.85,
        zorder=4,
    )

    ax_field.set_xlim(
        float(years.min()) - 0.5,
        float(years.max()) + 0.5,
    )

    # 周期向下增大，类似常见小波功率谱图。
    ax_field.set_ylim(
        np.log2(MAX_PERIOD),
        np.log2(2.0),
    )

    ax_field.set_xticks(
        year_ticks
    )

    ax_field.set_xticklabels(
        [
            str(year)
            for year in year_ticks
        ],
        rotation=0,
    )

    ax_field.set_yticks(
        np.log2(period_ticks)
    )

    ax_field.set_yticklabels(
        [
            f"{int(value)}"
            for value in period_ticks
        ]
    )

    ax_field.set_xlabel(
        "Year",
        fontsize=12,
    )

    ax_field.set_ylabel(
        "Period (days)",
        fontsize=12,
    )

    ax_field.set_title(
        "(a) Annual wavelet power",
        loc="left",
        fontsize=13,
    )

    ax_field.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    significance_handle = Line2D(
        [],
        [],
        linestyle="none",
        marker=".",
        markersize=7,
        color="black",
        label=(
            f"{int(SIGNIFICANCE_LEVEL * 100)}% "
            "significance"
        ),
    )

    ax_field.legend(
        handles=[significance_handle],
        loc="upper right",
        frameon=False,
        fontsize=8.5,
        handletextpad=0.3,
    )

    # 使用专门的 colorbar_axis，防止色标改变左侧主图尺寸。
    colorbar = fig.colorbar(
        contour,
        cax=colorbar_axis,
        orientation="horizontal",
    )

    colorbar.set_label(
        power_unit_label,
        fontsize=11,
    )

    colorbar.ax.tick_params(
        labelsize=9,
        direction="out",
        width=1.0,
    )

    colorbar.outline.set_linewidth(
        1.0
    )

    # -----------------------------------------------------
    # 右侧：多年平均全局功率谱
    # -----------------------------------------------------
    ax_global.plot(
        mean_power_plot,
        log2_period,
        color="black",
        linewidth=2.0,
        label="Comp. power",
        zorder=3,
    )

    ax_global.plot(
        mean_significance_plot,
        log2_period,
        color="red",
        linewidth=1.2,
        linestyle="--",
        label=(
            f"Comp. {int(SIGNIFICANCE_LEVEL * 100)}% "
            "significance"
        ),
        zorder=2,
    )

    ax_global.fill_betweenx(
        log2_period,
        mean_significance_plot,
        mean_power_plot,
        where=mean_significant,
        color="tab:red",
        alpha=0.25,
        interpolate=True,
        zorder=1,
    )

    combined_global_values = np.concatenate(
        [
            mean_power_plot[
                np.isfinite(
                    mean_power_plot
                )
            ],
            mean_significance_plot[
                np.isfinite(
                    mean_significance_plot
                )
            ],
        ]
    )

    if combined_global_values.size > 0:
        global_xmax = float(
            np.nanmax(
                combined_global_values
            )
        )
    else:
        global_xmax = 1.0

    if (
        not np.isfinite(global_xmax)
        or global_xmax <= 0
    ):
        global_xmax = 1.0

    ax_global.set_xlim(
        0.0,
        global_xmax * 1.08,
    )

    ax_global.set_xlabel(
        mean_power_xlabel,
        fontsize=11,
    )

    ax_global.set_title(
        "(b) Comp. spectrum",
        loc="left",
        fontsize=13,
    )

    ax_global.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
    )

    ax_global.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.3,
    )

    ax_global.legend(
        loc="upper right",
        frameon=False,
        fontsize=8.2,
    )

    # -----------------------------------------------------
    # 左右图无缝连接
    #
    # wspace=0 已使两幅主图位置完全相接。
    #
    # 为避免左图右边框和右图左边框重叠后形成双倍粗线，
    # 隐藏右图左边框，仅保留左图右边框作为共用边界。
    # -----------------------------------------------------
    ax_global.spines["left"].set_visible(
        False
    )

    ax_field.spines["right"].set_zorder(
        10
    )

    # -----------------------------------------------------
    # 全图样式
    # -----------------------------------------------------
    for axis in (
        ax_field,
        ax_global,
    ):
        axis.tick_params(
            direction="out",
            width=1.0,
            labelsize=10,
        )

        for spine in axis.spines.values():
            if spine.get_visible():
                spine.set_linewidth(1.2)

    output_stem = get_output_stem()

    pdf_file = output_stem.with_suffix(
        ".pdf"
    )

    png_file = output_stem.with_suffix(
        ".png"
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    fig.savefig(
        png_file,
        bbox_inches="tight",
        dpi=PNG_DPI,
    )

    plt.close(fig)

    print(
        f"PDF 图片已保存：\n"
        f"   {pdf_file}"
    )

    print(
        f"PNG 图片已保存：\n"
        f"   {png_file}"
    )


# =========================================================
# 主程序
# =========================================================
def main() -> None:
    plt.rcParams["font.family"] = (
        "Times New Roman"
    )

    plt.rcParams["axes.unicode_minus"] = (
        False
    )

    plt.rcParams["mathtext.fontset"] = (
        "stix"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ehci = build_daily_ehci()

    print(
        "\n已计算逐日 EHCI："
        f"{int(ehci.year.min())}--"
        f"{int(ehci.year.max())}，"
        f"共 {ehci.sizes['year']} 年，"
        f"每年 {ehci.sizes['day']} 天。"
    )

    ehci_min = float(
        ehci.min(
            skipna=True
        ).values
    )

    ehci_max = float(
        ehci.max(
            skipna=True
        ).values
    )

    print(
        f"EHCI 总体范围："
        f"{ehci_min:.6f}--"
        f"{ehci_max:.6f}"
    )

    plot_spectrum_fields(
        ehci
    )

    print(
        f"\n全部结果已保存至：\n"
        f"   {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
