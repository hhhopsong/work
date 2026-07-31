# -*- coding: utf-8 -*-
"""
统计1961—2022年逐年第一次大范围极端高温事件爆发时间，
对爆发时间进行Z-score标准化，并导出为NetCDF。

事件定义：
1. 研究时段：每年6月1日—8月31日；
2. EHCI > EHCI_THRESHOLD；
3. 连续满足条件MIN_DURATION天及以上，
   定义为一次大范围极端高温事件；
4. 每年只取第一次符合条件事件的开始日期；
5. 根据日期对应关系：
       day = 28  <=>  7月1日
   自动推算所有day对应的真实日期。

爆发时间标准化：
1. 将6月1日定义为第1天；
2. 将事件开始日期转换为距6月1日的日序；
3. 使用有效事件年份计算Z-score：
       Z = (X - mean) / std
4. 无符合条件事件的年份保留为NaN。

注意：
如果原始数据day最小值为1，则：
    day=1 <=> 6月4日

因此6月1—3日原始数据不存在，代码自动补为NaN，
不会错误参与连续事件判断。
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

from climkit.masked import masked


# ============================================================
# 1. 路径设置
# ============================================================

PYFILE = Path(
    "/Volumes/TiPlus7100/PyFile"
)


# Tmax数据
TMAX_FILE = (
    PYFILE
    / "p2/data/Tmax_5Day_filt.nc"
)


# 计算温度阈值所使用的区域
THRESHOLD_SHP = (
    PYFILE
    / "map/地图边界数据/"
      "长江区1：25万界线数据集（2002年）/"
      "长江区.shp"
)


# 计算长江流域EHCI的区域
YANGTZE_SHP = (
    PYFILE
    / "map/self/长江_TP/长江_tp.shp"
)


# 图片和CSV输出目录
PIC_OUTPUT_DIR = (
    PYFILE
    / "p6/pic"
)


# NetCDF输出目录
NC_OUTPUT_DIR = (
    PYFILE
    / "p6/data"
)


PIC_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


NC_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 统计结果CSV
RESULT_CSV = (
    PIC_OUTPUT_DIR
    / "first_large_scale_extreme_heat_event_JJA_1961_2022.csv"
)


# 爆发时间图片
FIG_FILE = (
    PIC_OUTPUT_DIR
    / "first_large_scale_extreme_heat_onset_JJA_1961_2022.png"
)


# 标准化爆发时间NetCDF
STANDARDIZED_NC = (
    NC_OUTPUT_DIR
    / "standardized_first_large_scale_extreme_heat_onset_JJA_1961_2022.nc"
)


# ============================================================
# 2. 研究参数
# ============================================================

START_YEAR = 1961

END_YEAR = 2022


# 极端高温温度阈值分位数
T_QUANTILE = 0.90


# 大范围极端高温面积比例阈值
#
# 当前设置：
# EHCI > 0.30
EHCI_THRESHOLD = 0.30


# 连续持续时间阈值
#
# 当前设置：
# 连续10天及以上
MIN_DURATION = 10


# ============================================================
# 3. 研究时段
# ============================================================

STUDY_START_MONTH = 6

STUDY_START_DAY = 1

STUDY_END_MONTH = 8

STUDY_END_DAY = 31


# ============================================================
# 4. day编号与真实日期的锚点
# ============================================================
#
# 原始day编号关系：
#
# day=28 <=> 7月1日
#
# 所有day对应日期均根据该锚点自动推算。
#
# ============================================================

ANCHOR_DAY_INDEX = 28

ANCHOR_MONTH = 7

ANCHOR_DAY_OF_MONTH = 1


# ============================================================
# 5. 标准化参数
# ============================================================
#
# ddof=1：
# 使用样本标准差进行Z-score标准化。
#
# ============================================================

STANDARDIZATION_DDOF = 1


# ============================================================
# 6. 工具函数：统一转换year/day坐标
# ============================================================

def normalize_integer_coordinate(
        data_array,
        coordinate_name
):
    """
    将year、day等坐标统一转换为整数。

    用于解决：
        year = "1961"
    与：
        year = 1961

    类型不同导致的xarray.sel切片问题。
    """

    if coordinate_name not in data_array.coords:

        raise KeyError(
            f"数据中不存在坐标：{coordinate_name}\n"
            f"当前坐标：{list(data_array.coords)}"
        )

    original_values = np.asarray(
        data_array[
            coordinate_name
        ].values
    )

    # bytes字符串转换为普通字符串
    if original_values.dtype.kind == "S":

        original_values = np.char.decode(
            original_values,
            "utf-8"
        )

    # 转换为数值
    numeric_values = pd.to_numeric(
        original_values.astype(str),
        errors="raise"
    )

    # 检查是否均为整数
    if not np.allclose(
            numeric_values,
            np.round(numeric_values)
    ):

        raise ValueError(
            f"{coordinate_name}坐标中存在非整数值：\n"
            f"{original_values[:20]}"
        )

    integer_values = np.asarray(
        np.round(
            numeric_values
        ),
        dtype=np.int64
    )

    # 检查转换后是否存在重复坐标
    if (
        len(
            np.unique(
                integer_values
            )
        )
        !=
        len(
            integer_values
        )
    ):

        raise ValueError(
            f"{coordinate_name}"
            f"转换为整数后存在重复值。"
        )

    data_array = (
        data_array
        .assign_coords(
            {
                coordinate_name: (
                    coordinate_name,
                    integer_values
                )
            }
        )
        .sortby(
            coordinate_name
        )
    )

    return data_array


# ============================================================
# 7. 工具函数：day编号转换为真实日期
# ============================================================

def day_index_to_date(
        year,
        day_index
):
    """
    根据锚点：

        day=28 <=> 7月1日

    将任意day编号转换为真实日期。
    """

    anchor_date = pd.Timestamp(
        year=int(year),
        month=ANCHOR_MONTH,
        day=ANCHOR_DAY_OF_MONTH
    )

    offset_days = (
        int(day_index)
        -
        ANCHOR_DAY_INDEX
    )

    actual_date = (
        anchor_date
        +
        pd.Timedelta(
            days=offset_days
        )
    )

    return actual_date


# ============================================================
# 8. 工具函数：识别某年第一次符合条件的事件
# ============================================================

def find_first_large_scale_event(
        yearly_series,
        threshold=0.40,
        min_duration=4
):
    """
    识别某一年第一次大范围极端高温事件。

    条件：
    1. EHCI严格大于threshold；
    2. 连续至少min_duration天；
    3. NaN视为未达到阈值，并中断连续事件。

    Parameters
    ----------
    yearly_series : pd.Series
        某年6—8月逐日EHCI。
        index必须为DatetimeIndex。

    threshold : float
        EHCI面积比例阈值。

    min_duration : int
        最短连续持续天数。

    Returns
    -------
    dict or None
        有事件时返回事件信息；
        无事件时返回None。
    """

    yearly_series = (
        yearly_series
        .sort_index()
    )

    if yearly_series.empty:

        return None

    if not isinstance(
            yearly_series.index,
            pd.DatetimeIndex
    ):

        raise TypeError(
            "yearly_series索引必须为"
            "DatetimeIndex。"
        )

    # EHCI严格大于阈值
    #
    # EHCI等于阈值时不计入事件。
    # NaN转换为False，并中断连续事件。
    event_flag = (
        yearly_series
        .gt(
            threshold
        )
        .fillna(
            False
        )
        .to_numpy(
            dtype=bool
        )
    )

    # 在首尾补False，以识别连续True片段
    padded_flag = np.concatenate(
        [
            np.array(
                [False]
            ),
            event_flag,
            np.array(
                [False]
            )
        ]
    ).astype(
        np.int8
    )

    flag_change = np.diff(
        padded_flag
    )

    # False -> True：事件开始
    run_starts = np.flatnonzero(
        flag_change == 1
    )

    # True -> False：事件结束位置的后一位
    run_ends_exclusive = np.flatnonzero(
        flag_change == -1
    )

    # 每个连续片段的持续时间
    run_lengths = (
        run_ends_exclusive
        -
        run_starts
    )

    # 满足最短持续时间的事件
    valid_run_indices = np.flatnonzero(
        run_lengths
        >=
        min_duration
    )

    # 该年无符合条件事件
    if valid_run_indices.size == 0:

        return None

    # 只取第一次符合条件的事件
    first_event_index = int(
        valid_run_indices[0]
    )

    start_position = int(
        run_starts[
            first_event_index
        ]
    )

    end_position_exclusive = int(
        run_ends_exclusive[
            first_event_index
        ]
    )

    end_position = (
        end_position_exclusive
        -
        1
    )

    duration_days = int(
        run_lengths[
            first_event_index
        ]
    )

    event_values = yearly_series.iloc[
        start_position:
        end_position_exclusive
    ]

    start_date = yearly_series.index[
        start_position
    ]

    end_date = yearly_series.index[
        end_position
    ]

    return {
        "start_date":
            start_date,

        "end_date":
            end_date,

        "duration_days":
            duration_days,

        "mean_ehci":
            float(
                event_values.mean()
            ),

        "max_ehci":
            float(
                event_values.max()
            ),

        "min_ehci":
            float(
                event_values.min()
            )
    }


# ============================================================
# 9. 检查输入文件
# ============================================================

if not TMAX_FILE.exists():

    raise FileNotFoundError(
        f"找不到Tmax数据：\n"
        f"{TMAX_FILE}"
    )


if not THRESHOLD_SHP.exists():

    raise FileNotFoundError(
        f"找不到阈值区域shp：\n"
        f"{THRESHOLD_SHP}"
    )


if not YANGTZE_SHP.exists():

    raise FileNotFoundError(
        f"找不到长江流域shp：\n"
        f"{YANGTZE_SHP}"
    )


# ============================================================
# 10. 读取Tmax数据
# ============================================================
#
# year/day只是编号，
# 不让xarray将其解析为时间。
#
# ============================================================

tmax_5day = xr.open_dataarray(
    TMAX_FILE,
    decode_times=False
)


print(
    "=" * 70
)

print(
    "Original Tmax data"
)

print(
    "=" * 70
)

print(
    tmax_5day
)


# ============================================================
# 11. 检查必要维度
# ============================================================

required_dimensions = {
    "year",
    "day",
    "lat",
    "lon"
}


missing_dimensions = (
    required_dimensions
    -
    set(
        tmax_5day.dims
    )
)


if missing_dimensions:

    tmax_5day.close()

    raise ValueError(
        "数据缺少必要维度："
        f"{sorted(missing_dimensions)}\n"
        "当前维度："
        f"{list(tmax_5day.dims)}"
    )


# ============================================================
# 12. year/day坐标转换为整数
# ============================================================

print(
    "\nBefore coordinate conversion:"
)


print(
    "year dtype =",
    tmax_5day[
        "year"
    ].dtype
)


print(
    "year first values =",
    tmax_5day[
        "year"
    ].values[:5]
)


print(
    "day dtype =",
    tmax_5day[
        "day"
    ].dtype
)


print(
    "day first values =",
    tmax_5day[
        "day"
    ].values[:10]
)


tmax_5day = normalize_integer_coordinate(
    tmax_5day,
    "year"
)


tmax_5day = normalize_integer_coordinate(
    tmax_5day,
    "day"
)


print(
    "\nAfter coordinate conversion:"
)


print(
    "Year range:",
    int(
        tmax_5day[
            "year"
        ].min().item()
    ),
    "to",
    int(
        tmax_5day[
            "year"
        ].max().item()
    )
)


print(
    "Day range:",
    int(
        tmax_5day[
            "day"
        ].min().item()
    ),
    "to",
    int(
        tmax_5day[
            "day"
        ].max().item()
    )
)


# ============================================================
# 13. 输出day与真实日期对应关系
# ============================================================

minimum_day = int(
    tmax_5day[
        "day"
    ].min().item()
)


maximum_day = int(
    tmax_5day[
        "day"
    ].max().item()
)


example_year = START_YEAR


minimum_date = day_index_to_date(
    example_year,
    minimum_day
)


maximum_date = day_index_to_date(
    example_year,
    maximum_day
)


print(
    "\n" + "=" * 70
)

print(
    "Day-to-date mapping"
)

print(
    "=" * 70
)


print(
    f"day={ANCHOR_DAY_INDEX}"
    f" -> "
    f"{ANCHOR_MONTH:02d}-"
    f"{ANCHOR_DAY_OF_MONTH:02d}"
)


print(
    f"Minimum day={minimum_day}"
    f" -> "
    f"{minimum_date.strftime('%m-%d')}"
)


print(
    f"Maximum day={maximum_day}"
    f" -> "
    f"{maximum_date.strftime('%m-%d')}"
)


# ============================================================
# 14. 选择1961—2022年
# ============================================================

available_years = set(
    tmax_5day[
        "year"
    ].values.astype(
        int
    )
)


required_years = set(
    range(
        START_YEAR,
        END_YEAR + 1
    )
)


missing_years = sorted(
    required_years
    -
    available_years
)


if missing_years:

    print(
        "\nWarning: missing years:"
    )

    print(
        missing_years
    )


tmax_period = tmax_5day.sel(
    year=slice(
        START_YEAR,
        END_YEAR
    )
)


if (
    tmax_period.sizes.get(
        "year",
        0
    )
    ==
    0
):

    tmax_5day.close()

    raise ValueError(
        f"没有选到"
        f"{START_YEAR}—{END_YEAR}"
        f"年的数据。"
    )


# ============================================================
# 15. 计算极端高温温度阈值
# ============================================================
#
# 保持原始计算逻辑：
#
# 1. 对阈值区域进行掩膜；
# 2. 对year和day求平均；
# 3. 对多年平均空间场计算空间90%分位数。
#
# ============================================================

threshold_data = masked(
    tmax_period,
    str(
        THRESHOLD_SHP
    )
)


mean_temperature_field = (
    threshold_data
    .mean(
        dim=[
            "year",
            "day"
        ],
        skipna=True
    )
)


t90 = (
    mean_temperature_field
    .quantile(
        T_QUANTILE,
        dim=[
            "lat",
            "lon"
        ],
        skipna=True
    )
    .squeeze(
        drop=True
    )
)


t90_value = float(
    t90.values
)


if not np.isfinite(
        t90_value
):

    tmax_5day.close()

    raise ValueError(
        "计算得到的极端高温阈值为NaN。"
        "请检查数据或shp掩膜。"
    )


print(
    "\n" + "=" * 70
)

print(
    "Temperature threshold"
)

print(
    "=" * 70
)


print(
    f"{T_QUANTILE:.0%} threshold = "
    f"{t90_value:.4f}"
)


# ============================================================
# 16. Tmax转换为逐格点极端高温二值数据
# ============================================================
#
# Tmax > t90  -> 1
# Tmax <= t90 -> 0
# 原始NaN仍保持NaN。
#
# ============================================================

EHD = xr.where(
    tmax_period.isnull(),
    np.nan,
    xr.where(
        tmax_period
        >
        t90_value,
        1.0,
        0.0
    )
)


# 长江流域掩膜
EHD = masked(
    EHD,
    str(
        YANGTZE_SHP
    )
)


EHD.name = (
    "EHD_binary"
)


# ============================================================
# 17. 计算长江流域有效格点数量
# ============================================================

grid_template = xr.ones_like(
    tmax_period.isel(
        year=0,
        day=0,
        drop=True
    ),
    dtype=np.float32
)


yangtze_grid = masked(
    grid_template,
    str(
        YANGTZE_SHP
    )
)


zone_grid_number = float(
    yangtze_grid.sum(
        dim=[
            "lat",
            "lon"
        ],
        skipna=True
    ).values
)


if not np.isfinite(
        zone_grid_number
):

    tmax_5day.close()

    raise ValueError(
        "长江流域有效格点数量为NaN。"
    )


if zone_grid_number <= 0:

    tmax_5day.close()

    raise ValueError(
        "长江流域有效格点数<=0，"
        "请检查shp和经纬度。"
    )


print(
    "\n" + "=" * 70
)

print(
    "Yangtze basin grid"
)

print(
    "=" * 70
)


print(
    f"Valid grid number = "
    f"{zone_grid_number:.0f}"
)


# ============================================================
# 18. 计算逐年逐日EHCI
# ============================================================
#
# EHCI =
#
# 当日发生极端高温的长江流域格点数
# --------------------------------
# 长江流域总有效格点数
#
# ============================================================

extreme_grid_number = EHD.sum(
    dim=[
        "lat",
        "lon"
    ],
    skipna=True
)


EHCI_year_day = (
    extreme_grid_number
    /
    zone_grid_number
)


EHCI_year_day.name = (
    "EHCI"
)


EHCI_year_day = normalize_integer_coordinate(
    EHCI_year_day,
    "year"
)


EHCI_year_day = normalize_integer_coordinate(
    EHCI_year_day,
    "day"
)


print(
    "\nEHCI year-day data:"
)

print(
    EHCI_year_day
)


# ============================================================
# 19. 根据day编号建立真实日期时间序列
# ============================================================

ehci_series_list = []


for year_value in EHCI_year_day[
    "year"
].values:

    year_value = int(
        year_value
    )

    yearly_data = EHCI_year_day.sel(
        year=year_value
    )

    day_indices = (
        yearly_data[
            "day"
        ].values.astype(
            int
        )
    )

    yearly_values = (
        yearly_data
        .to_numpy()
        .astype(
            float
        )
        .ravel()
    )

    # 将day编号转换为实际日期
    actual_dates = pd.DatetimeIndex(
        [
            day_index_to_date(
                year_value,
                day_index
            )
            for day_index in day_indices
        ]
    )

    # 检查日期是否重复
    if actual_dates.duplicated().any():

        tmax_5day.close()

        duplicated_dates = actual_dates[
            actual_dates.duplicated(
                keep=False
            )
        ]

        raise ValueError(
            f"{year_value}年转换后的日期存在重复：\n"
            f"{duplicated_dates}"
        )

    yearly_series_original = pd.Series(
        data=yearly_values,
        index=actual_dates,
        name="EHCI"
    )

    # 建立完整6月1日—8月31日日期序列
    full_study_dates = pd.date_range(
        start=(
            f"{year_value}-"
            f"{STUDY_START_MONTH:02d}-"
            f"{STUDY_START_DAY:02d}"
        ),
        end=(
            f"{year_value}-"
            f"{STUDY_END_MONTH:02d}-"
            f"{STUDY_END_DAY:02d}"
        ),
        freq="D"
    )

    # 原始不存在的日期自动补为NaN
    yearly_series = (
        yearly_series_original
        .reindex(
            full_study_dates
        )
    )

    ehci_series_list.append(
        yearly_series
    )


if not ehci_series_list:

    tmax_5day.close()

    raise ValueError(
        "没有生成EHCI时间序列。"
    )


ehci_series = (
    pd.concat(
        ehci_series_list
    )
    .sort_index()
)


# ============================================================
# 20. 检查EHCI数据
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "JJA EHCI data"
)

print(
    "=" * 70
)


print(
    "Time range:"
)


print(
    ehci_series.index.min()
)


print(
    "to"
)


print(
    ehci_series.index.max()
)


print(
    "\nEHCI minimum =",
    ehci_series.min()
)


print(
    "EHCI maximum =",
    ehci_series.max()
)


if (
    ehci_series.max()
    >
    1.0 + 1e-6
):

    print(
        "\nWARNING: EHCI > 1."
        "\nPlease check numerator/denominator grids."
    )


# ============================================================
# 21. 检查每年6—8月缺测日期
# ============================================================

print(
    "\nMissing dates in JJA:"
)


for year in range(
    START_YEAR,
    END_YEAR + 1
):

    yearly_series = ehci_series[
        ehci_series.index.year
        ==
        year
    ]

    missing_count = int(
        yearly_series
        .isna()
        .sum()
    )

    if missing_count > 0:

        missing_dates = (
            yearly_series[
                yearly_series.isna()
            ]
            .index
            .strftime(
                "%m-%d"
            )
            .tolist()
        )

        print(
            f"{year}: "
            f"{missing_count} missing days, "
            f"examples="
            f"{missing_dates[:10]}"
        )


# ============================================================
# 22. 逐年识别第一次大范围极端高温事件
# ============================================================

event_records = []


for year in range(
    START_YEAR,
    END_YEAR + 1
):

    yearly_series = ehci_series[
        ehci_series.index.year
        ==
        year
    ]

    first_event = find_first_large_scale_event(
        yearly_series=yearly_series,
        threshold=EHCI_THRESHOLD,
        min_duration=MIN_DURATION
    )

    # 当年没有符合条件事件
    if first_event is None:

        event_records.append(
            {
                "year":
                    year,

                "event_occurred":
                    False,

                "start_date":
                    pd.NaT,

                "end_date":
                    pd.NaT,

                "start_month_day":
                    "",

                "end_month_day":
                    "",

                "onset_day_from_june_1":
                    np.nan,

                "duration_days":
                    np.nan,

                "mean_EHCI":
                    np.nan,

                "max_EHCI":
                    np.nan,

                "min_EHCI":
                    np.nan
            }
        )

    # 找到第一次符合条件事件
    else:

        start_date = pd.Timestamp(
            first_event[
                "start_date"
            ]
        )

        end_date = pd.Timestamp(
            first_event[
                "end_date"
            ]
        )

        june_first = pd.Timestamp(
            year=year,
            month=6,
            day=1
        )

        # 6月1日定义为第1天
        onset_day_from_june_1 = (
            start_date
            -
            june_first
        ).days + 1

        event_records.append(
            {
                "year":
                    year,

                "event_occurred":
                    True,

                "start_date":
                    start_date,

                "end_date":
                    end_date,

                "start_month_day":
                    start_date.strftime(
                        "%m-%d"
                    ),

                "end_month_day":
                    end_date.strftime(
                        "%m-%d"
                    ),

                "onset_day_from_june_1":
                    onset_day_from_june_1,

                "duration_days":
                    first_event[
                        "duration_days"
                    ],

                "mean_EHCI":
                    first_event[
                        "mean_ehci"
                    ],

                "max_EHCI":
                    first_event[
                        "max_ehci"
                    ],

                "min_EHCI":
                    first_event[
                        "min_ehci"
                    ]
            }
        )


event_df = pd.DataFrame(
    event_records
)


# 确保按年份排序
event_df = (
    event_df
    .sort_values(
        "year"
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n" + "=" * 70
)

print(
    "First large-scale extreme heat event"
)

print(
    "=" * 70
)


print(
    event_df.to_string(
        index=False
    )
)


# ============================================================
# 23. 爆发时间Z-score标准化
# ============================================================
#
# 使用距6月1日的日序：
#
# standardized_onset =
#     (onset_day_from_june_1 - mean)
#     / standard_deviation
#
# 无事件年份保持为NaN。
#
# standardized_onset < 0：
#     爆发时间早于多年平均；
#
# standardized_onset > 0：
#     爆发时间晚于多年平均。
#
# ============================================================

onset_day_series = pd.to_numeric(
    event_df[
        "onset_day_from_june_1"
    ],
    errors="coerce"
).astype(
    float
)


valid_onset_day_series = (
    onset_day_series
    .dropna()
)


if (
    len(
        valid_onset_day_series
    )
    <=
    STANDARDIZATION_DDOF
):

    tmax_5day.close()

    raise ValueError(
        "有效爆发年份数量不足，"
        "无法计算标准化序列。"
    )


# 多年平均爆发日序
onset_day_mean = float(
    valid_onset_day_series.mean()
)


# 爆发日序标准差
onset_day_std = float(
    valid_onset_day_series.std(
        ddof=STANDARDIZATION_DDOF
    )
)


if (
    not np.isfinite(
        onset_day_std
    )
    or
    onset_day_std <= 0
):

    tmax_5day.close()

    raise ValueError(
        "爆发时间标准差无效，"
        "无法进行标准化。"
    )


event_df[
    "standardized_onset"
] = (
    onset_day_series
    -
    onset_day_mean
) / onset_day_std


print(
    "\n" + "=" * 70
)

print(
    "Standardized onset series"
)

print(
    "=" * 70
)


print(
    f"Valid event years = "
    f"{len(valid_onset_day_series)}"
)


print(
    f"Mean onset day from June 1 = "
    f"{onset_day_mean:.6f}"
)


print(
    f"Standard deviation = "
    f"{onset_day_std:.6f}"
)


print(
    f"Standard deviation ddof = "
    f"{STANDARDIZATION_DDOF}"
)


print(
    event_df[
        [
            "year",
            "event_occurred",
            "onset_day_from_june_1",
            "standardized_onset"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 24. 保存CSV统计结果
# ============================================================
#
# CSV同时包含：
# 1. 原始爆发日期；
# 2. 距6月1日的爆发日序；
# 3. 标准化爆发时间。
#
# ============================================================

event_df.to_csv(
    RESULT_CSV,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d",
    float_format="%.6f"
)


print(
    "\nCSV saved to:"
)

print(
    RESULT_CSV
)


# ============================================================
# 25. 构建标准化爆发时间NetCDF
# ============================================================
#
# 维度：
#     year
#
# 坐标：
#     year = 1961, 1962, ..., 2022
#
# 数据变量：
#     standardized_onset
#     onset_day_from_june_1
#     event_occurred
#
# ============================================================

year_coordinate = (
    event_df[
        "year"
    ]
    .to_numpy(
        dtype=np.int32
    )
)


onset_day_values = (
    event_df[
        "onset_day_from_june_1"
    ]
    .to_numpy(
        dtype=np.float32
    )
)


standardized_onset_values = (
    event_df[
        "standardized_onset"
    ]
    .to_numpy(
        dtype=np.float32
    )
)


event_occurred_values = (
    event_df[
        "event_occurred"
    ]
    .astype(
        np.int8
    )
    .to_numpy()
)


onset_dataset = xr.Dataset(
    data_vars={
        "standardized_onset": (
            ["year"],
            standardized_onset_values,
            {
                "long_name":
                    (
                        "Standardized onset date of the first "
                        "large-scale extreme heat event"
                    ),

                "units":
                    "1",

                "description":
                    (
                        "Z-score of onset_day_from_june_1. "
                        "Negative values indicate earlier onset "
                        "than the climatological mean, and "
                        "positive values indicate later onset. "
                        "NaN indicates no qualifying event."
                    )
            }
        ),

        "onset_day_from_june_1": (
            ["year"],
            onset_day_values,
            {
                "long_name":
                    "Onset day counted from June 1",

                "units":
                    "day",

                "description":
                    (
                        "June 1 is defined as day 1. "
                        "NaN indicates no qualifying event."
                    )
            }
        ),

        "event_occurred": (
            ["year"],
            event_occurred_values,
            {
                "long_name":
                    (
                        "Occurrence flag of the first qualifying "
                        "large-scale extreme heat event"
                    ),

                "units":
                    "1",

                "flag_values":
                    np.array(
                        [0, 1],
                        dtype=np.int8
                    ),

                "flag_meanings":
                    "no_event event_occurred"
            }
        )
    },

    coords={
        "year": (
            ["year"],
            year_coordinate,
            {
                "long_name":
                    "Year"
            }
        )
    },

    attrs={
        "title":
            (
                "Standardized annual onset time of the first "
                "large-scale extreme heat event"
            ),

        "study_period":
            f"{START_YEAR}-{END_YEAR}",

        "study_season":
            "June 1 to August 31",

        "temperature_threshold_quantile":
            float(
                T_QUANTILE
            ),

        "temperature_threshold_value":
            float(
                t90_value
            ),

        "ehci_threshold":
            float(
                EHCI_THRESHOLD
            ),

        "minimum_duration_days":
            int(
                MIN_DURATION
            ),

        "event_definition":
            (
                f"EHCI > {EHCI_THRESHOLD:.4f} for at least "
                f"{MIN_DURATION} consecutive days"
            ),

        "onset_definition":
            (
                "Start date of the first qualifying event "
                "in each year"
            ),

        "standardization_method":
            "Z-score",

        "standardization_formula":
            "(x - mean) / standard_deviation",

        "standard_deviation_ddof":
            int(
                STANDARDIZATION_DDOF
            ),

        "standardization_mean":
            float(
                onset_day_mean
            ),

        "standardization_standard_deviation":
            float(
                onset_day_std
            ),

        "standardization_valid_year_count":
            int(
                len(
                    valid_onset_day_series
                )
            ),

        "day_reference":
            "June 1 is day 1",

        "missing_value_definition":
            "NaN means no qualifying event occurred",

        "day_index_anchor":
            (
                f"day={ANCHOR_DAY_INDEX} corresponds to "
                f"{ANCHOR_MONTH:02d}-{ANCHOR_DAY_OF_MONTH:02d}"
            ),

        "created_by":
            "Python, pandas and xarray"
    }
)


# ============================================================
# 26. 检查NetCDF年份坐标
# ============================================================

expected_year_coordinate = np.arange(
    START_YEAR,
    END_YEAR + 1,
    dtype=np.int32
)


if not np.array_equal(
        year_coordinate,
        expected_year_coordinate
):

    onset_dataset.close()
    tmax_5day.close()

    raise ValueError(
        "event_df中的年份坐标不是完整连续的"
        f"{START_YEAR}—{END_YEAR}。"
    )


if not np.array_equal(
        onset_dataset[
            "year"
        ].values,
        expected_year_coordinate
):

    onset_dataset.close()
    tmax_5day.close()

    raise ValueError(
        "NetCDF年份坐标与预期年份不一致。"
    )


if (
    onset_dataset[
        "standardized_onset"
    ].sizes[
        "year"
    ]
    !=
    len(
        expected_year_coordinate
    )
):

    onset_dataset.close()
    tmax_5day.close()

    raise ValueError(
        "标准化序列长度与年份坐标长度不一致。"
    )


# ============================================================
# 27. 导出NetCDF到p6/data
# ============================================================

compressed_encoding = {
    "standardized_onset": {
        "dtype":
            "float32",

        "_FillValue":
            np.float32(
                np.nan
            ),

        "zlib":
            True,

        "complevel":
            4
    },

    "onset_day_from_june_1": {
        "dtype":
            "float32",

        "_FillValue":
            np.float32(
                np.nan
            ),

        "zlib":
            True,

        "complevel":
            4
    },

    "event_occurred": {
        "dtype":
            "int8",

        "_FillValue":
            None,

        "zlib":
            True,

        "complevel":
            4
    },

    "year": {
        "dtype":
            "int32",

        "_FillValue":
            None
    }
}


try:

    onset_dataset.to_netcdf(
        STANDARDIZED_NC,
        mode="w",
        format="NETCDF4",
        engine="netcdf4",
        encoding=compressed_encoding
    )

    netcdf_engine_used = (
        "netcdf4"
    )


except (
    ImportError,
    ModuleNotFoundError,
    ValueError
) as error:

    print(
        "\nWarning: compressed NetCDF writing failed:"
    )

    print(
        error
    )

    print(
        "Trying default backend without compression."
    )

    fallback_encoding = {
        "standardized_onset": {
            "dtype":
                "float32",

            "_FillValue":
                np.float32(
                    np.nan
                )
        },

        "onset_day_from_june_1": {
            "dtype":
                "float32",

            "_FillValue":
                np.float32(
                    np.nan
                )
        },

        "event_occurred": {
            "dtype":
                "int8",

            "_FillValue":
                None
        },

        "year": {
            "dtype":
                "int32",

            "_FillValue":
                None
        }
    }

    onset_dataset.to_netcdf(
        STANDARDIZED_NC,
        mode="w",
        encoding=fallback_encoding
    )

    netcdf_engine_used = (
        "default xarray backend"
    )


onset_dataset.close()


# ============================================================
# 28. 重新读取并验证NetCDF
# ============================================================

if not STANDARDIZED_NC.exists():

    tmax_5day.close()

    raise FileNotFoundError(
        "NetCDF写出后未找到输出文件：\n"
        f"{STANDARDIZED_NC}"
    )


check_dataset = xr.open_dataset(
    STANDARDIZED_NC,
    decode_times=False
)


try:

    if "year" not in check_dataset.coords:

        raise ValueError(
            "输出NetCDF中不存在year坐标。"
        )

    if "standardized_onset" not in check_dataset.data_vars:

        raise ValueError(
            "输出NetCDF中不存在"
            "standardized_onset变量。"
        )

    if not np.array_equal(
            check_dataset[
                "year"
            ].values.astype(
                np.int32
            ),
            expected_year_coordinate
    ):

        raise ValueError(
            "重新读取后，NetCDF年份坐标不正确。"
        )

    check_standardized_values = (
        check_dataset[
            "standardized_onset"
        ]
        .values
        .astype(
            float
        )
    )

    source_standardized_values = (
        event_df[
            "standardized_onset"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.allclose(
            check_standardized_values,
            source_standardized_values,
            equal_nan=True,
            rtol=1e-6,
            atol=1e-6
    ):

        raise ValueError(
            "重新读取后，标准化序列与原始计算结果不一致。"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "NetCDF validation"
    )

    print(
        "=" * 70
    )

    print(
        check_dataset
    )

    print(
        "\nNetCDF engine used:"
    )

    print(
        netcdf_engine_used
    )

    print(
        "\nStandardized onset NetCDF saved to:"
    )

    print(
        STANDARDIZED_NC
    )


finally:

    check_dataset.close()


# ============================================================
# 29. 将爆发日期映射到统一参考年份
# ============================================================

REFERENCE_YEAR = 2000


plot_dates = []


for start_date in event_df[
    "start_date"
]:

    if pd.isna(
        start_date
    ):

        plot_dates.append(
            np.nan
        )

    else:

        start_date = pd.Timestamp(
            start_date
        )

        reference_date = datetime(
            REFERENCE_YEAR,
            start_date.month,
            start_date.day
        )

        plot_dates.append(
            mdates.date2num(
                reference_date
            )
        )


event_df[
    "plot_date"
] = plot_dates


# ============================================================
# 30. 绘制逐年第一次爆发时间折线图
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        15,
        7
    ),
    dpi=150
)


valid_event = (
    event_df[
        "event_occurred"
    ]
    .astype(
        bool
    )
)


# 无事件年份plot_date为NaN，折线自动断开
ax.plot(
    event_df[
        "year"
    ],
    event_df[
        "plot_date"
    ],
    marker="o",
    markersize=4.5,
    linewidth=1.4,
    label=(
        "First large-scale "
        "extreme heat event"
    )
)


ax.scatter(
    event_df.loc[
        valid_event,
        "year"
    ],
    event_df.loc[
        valid_event,
        "plot_date"
    ],
    s=28,
    zorder=3
)


# ============================================================
# 31. 横轴设置
# ============================================================

ax.set_xlim(
    START_YEAR - 1,
    END_YEAR + 1
)


ax.xaxis.set_major_locator(
    MultipleLocator(
        5
    )
)


ax.xaxis.set_minor_locator(
    MultipleLocator(
        1
    )
)


# ============================================================
# 32. 纵轴设置
# ============================================================

y_min = mdates.date2num(
    datetime(
        REFERENCE_YEAR,
        6,
        1
    )
)


y_max = mdates.date2num(
    datetime(
        REFERENCE_YEAR,
        9,
        1
    )
)


ax.set_ylim(
    y_min,
    y_max
)


ax.yaxis.set_major_locator(
    mdates.DayLocator(
        interval=10
    )
)


ax.yaxis.set_minor_locator(
    mdates.DayLocator(
        interval=1
    )
)


ax.yaxis.set_major_formatter(
    mdates.DateFormatter(
        "%m-%d"
    )
)


# ============================================================
# 33. 图形文字
# ============================================================

ax.set_xlabel(
    "Year",
    fontsize=12
)


ax.set_ylabel(
    "Onset date (month-day)",
    fontsize=12
)


ax.set_title(
    "Annual Onset Date of the First "
    "Large-Scale Extreme Heat Event",
    fontsize=14,
    pad=12
)


ax.grid(
    True,
    which="major",
    linestyle="--",
    linewidth=0.6,
    alpha=0.6
)


ax.tick_params(
    axis="x",
    which="major",
    rotation=45
)


ax.legend(
    loc="best",
    frameon=False
)


fig.tight_layout()


# ============================================================
# 34. 保存图片
# ============================================================

fig.savefig(
    FIG_FILE,
    dpi=1000,
    bbox_inches="tight"
)


plt.show()


plt.close(
    fig
)


print(
    "\nFigure saved to:"
)

print(
    FIG_FILE
)


# ============================================================
# 35. 统计摘要
# ============================================================

number_of_event_years = int(
    valid_event.sum()
)


number_of_no_event_years = int(
    (
        ~valid_event
    ).sum()
)


print(
    "\n" + "=" * 70
)

print(
    "Summary"
)

print(
    "=" * 70
)


print(
    f"Study period: "
    f"{START_YEAR}-{END_YEAR}"
)


print(
    "Study season: "
    "June-August"
)


print(
    f"EHCI threshold: "
    f"> {EHCI_THRESHOLD:.0%}"
)


print(
    f"Minimum duration: "
    f"{MIN_DURATION} consecutive days"
)


print(
    "Years with qualifying events: "
    f"{number_of_event_years}"
)


print(
    "Years without qualifying events: "
    f"{number_of_no_event_years}"
)


print(
    "Standardization method: "
    "Z-score"
)


print(
    "Standardization mean onset day: "
    f"{onset_day_mean:.6f}"
)


print(
    "Standardization standard deviation: "
    f"{onset_day_std:.6f}"
)


print(
    "Standardization ddof: "
    f"{STANDARDIZATION_DDOF}"
)


# ============================================================
# 36. 计算平均、最早、最晚爆发日期
# ============================================================

if number_of_event_years > 0:

    valid_events_df = (
        event_df.loc[
            valid_event
        ]
        .copy()
    )

    average_onset_day = int(
        np.round(
            valid_events_df[
                "onset_day_from_june_1"
            ].mean()
        )
    )

    average_start_date = (
        pd.Timestamp(
            f"{REFERENCE_YEAR}-06-01"
        )
        +
        pd.Timedelta(
            days=
            average_onset_day
            -
            1
        )
    )

    earliest_index = (
        valid_events_df[
            "onset_day_from_june_1"
        ]
        .idxmin()
    )

    earliest_event = event_df.loc[
        earliest_index
    ]

    latest_index = (
        valid_events_df[
            "onset_day_from_june_1"
        ]
        .idxmax()
    )

    latest_event = event_df.loc[
        latest_index
    ]

    print(
        "\nMean onset date: "
        f"{average_start_date.strftime('%m-%d')}"
    )

    print(
        "Earliest onset: "
        f"{int(earliest_event['year'])} "
        f"{earliest_event['start_month_day']}"
    )

    print(
        "Latest onset: "
        f"{int(latest_event['year'])} "
        f"{latest_event['start_month_day']}"
    )

    print(
        "Mean event duration: "
        f"{valid_events_df['duration_days'].mean():.2f} days"
    )


# ============================================================
# 37. 单独打印每年的首次爆发日期和标准化值
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "Annual onset dates and standardized values"
)

print(
    "=" * 70
)


for _, row in event_df.iterrows():

    year = int(
        row[
            "year"
        ]
    )

    if bool(
        row[
            "event_occurred"
        ]
    ):

        print(
            f"{year}: "
            f"{row['start_month_day']}"
            f"  duration="
            f"{int(row['duration_days'])} days"
            f"  max EHCI="
            f"{row['max_EHCI']:.3f}"
            f"  standardized onset="
            f"{row['standardized_onset']:.6f}"
        )

    else:

        print(
            f"{year}: "
            "No qualifying event"
            "  standardized onset=NaN"
        )


# ============================================================
# 38. 关闭原始数据文件
# ============================================================

tmax_5day.close()


print(
    "\n" + "=" * 70
)

print(
    "Output files"
)

print(
    "=" * 70
)


print(
    "CSV:"
)

print(
    RESULT_CSV
)


print(
    "\nFigure:"
)

print(
    FIG_FILE
)


print(
    "\nNetCDF:"
)

print(
    STANDARDIZED_NC
)


print(
    "\nProgram finished successfully."
)
