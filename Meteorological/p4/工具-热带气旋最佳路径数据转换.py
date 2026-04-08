# -*- coding: utf-8 -*-
"""
将 CMA 热带气旋最佳路径 TXT 数据批量转换为 NetCDF(.nc)

输入文件名格式:
    CH1949BST.txt ~ CH2024BST.txt

使用方法:
    1. 安装依赖:
       pip install netCDF4 numpy

    2. 修改下面 main() 里的 input_dir 和 output_dir

    3. 直接运行脚本

NetCDF 设计:
    - storm 维: 每个热带气旋一条头记录
    - obs 维: 所有路径记录平铺存储
    - storm_obs_start_index / storm_obs_end_index:
      用于将某个 storm 对应到 obs 上的记录区间
"""

import os
import re
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from netCDF4 import Dataset, stringtochar, date2num



# =========================
# 数据结构
# =========================

@dataclass
class StormHeader:
    category_flag: str           # AAAAA
    international_id: str        # BBBB
    record_count: int            # CCC
    tc_sequence_with_td: int     # DDDD
    china_tc_id: str             # EEEE
    end_flag: int                # F
    time_interval_hours: int     # G
    storm_name: str              # H...H
    dataset_date: str            # I...I


@dataclass
class TrackRecord:
    yyyymmddhh: str
    intensity_flag: int
    lat_0p1deg_n: int
    lon_0p1deg_e: int
    pres_hpa: int
    wnd_ms: int
    owd_ms: int


# =========================
# 解析函数
# =========================

HEADER_RE = re.compile(r"^66666\b")
FILE_RE = re.compile(r"^CH\d{4}BST\.txt$")


def parse_header(line: str) -> StormHeader:
    """
    头记录格式:
    "AAAAA BBBB  CCC DDDD EEEE F G HHHHHHHHHHHHHHHHHHHH               IIIIIIII       "
    按空白截取

    兼容历史头记录格式:
    1. 国际编号可能是单值，如 7101
    2. 国际编号可能是多值，如 7127,7128
    3. storm_name 可能为空
    4. 日期始终取最后8位数字
    """
    raw = line.rstrip("\n")

    if not raw.strip().startswith("66666"):
        raise ValueError(f"不是合法头记录: {line!r}")

    m_date = re.search(r"(\d{8})\s*$", raw)
    if not m_date:
        raise ValueError(f"头记录缺少8位日期: {line!r}")
    dataset_date = m_date.group(1)

    body = raw[:m_date.start()].rstrip()

    if not body.startswith("66666"):
        raise ValueError(f"头记录缺少分类标志66666: {line!r}")

    rest = body[5:].strip()
    parts = rest.split()

    if len(parts) < 6:
        raise ValueError(f"头记录字段不足，无法解析: {line!r}")

    international_id = parts[0].strip()  # 保持字符串
    record_count = int(parts[1])
    tc_sequence_with_td = int(parts[2])
    china_tc_id = parts[3].strip()  # 改成字符串
    end_flag = int(parts[4])
    time_interval_hours = int(parts[5])

    storm_name = " ".join(parts[6:]).strip() if len(parts) > 6 else ""
    if "nameless" in storm_name.lower():
        storm_name = ""

    return StormHeader(
        category_flag="66666",
        international_id=international_id,
        record_count=record_count,
        tc_sequence_with_td=tc_sequence_with_td,
        china_tc_id=china_tc_id,
        end_flag=end_flag,
        time_interval_hours=time_interval_hours,
        storm_name=storm_name,
        dataset_date=dataset_date,
    )

def safe_int(x, default=-9999):
    x = str(x).strip()
    if x == "":
        return default
    return int(x)

def parse_track_record(line: str) -> TrackRecord:
    """
    最佳路径记录格式:
    "YYYYMMDDHH I LAT LONG PRES     WND  OWD "
    实际处理时按空格 split 更稳妥
    """
    parts = line.strip().split()

    if len(parts) < 5:
        raise ValueError(f"路径记录字段过少: {line!r}")

    yyyymmddhh = parts[0]
    intensity_flag = safe_int(parts[1], default=-9999)
    lat_0p1deg_n = safe_int(parts[2], default=-9999)
    lon_0p1deg_e = safe_int(parts[3], default=-9999)
    pres_hpa = safe_int(parts[4], default=-9999)

    wnd_ms = safe_int(parts[5], default=-9999) if len(parts) >= 6 else -9999
    owd_ms = safe_int(parts[6], default=-9999) if len(parts) >= 7 else -9999

    return TrackRecord(
        yyyymmddhh=yyyymmddhh,
        intensity_flag=intensity_flag,
        lat_0p1deg_n=lat_0p1deg_n,
        lon_0p1deg_e=lon_0p1deg_e,
        pres_hpa=pres_hpa,
        wnd_ms=wnd_ms,
        owd_ms=owd_ms,
    )

def yyyymmddhh_to_datetime(yyyymmddhh: str) -> dt.datetime:
    return dt.datetime(
        int(yyyymmddhh[0:4]),
        int(yyyymmddhh[4:6]),
        int(yyyymmddhh[6:8]),
        int(yyyymmddhh[8:10]),
    )

def yyyymmddhh_to_isoz(yyyymmddhh: str) -> str:
    t = yyyymmddhh_to_datetime(yyyymmddhh)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# =========================
# 读取 TXT
# =========================

def read_txt_file(txt_path: str) -> Tuple[List[StormHeader], List[TrackRecord], np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
        storms: 头记录列表
        records: 全部路径记录(平铺)
        obs_start_index: 每个 storm 对应 obs 起始下标
        obs_end_index: 每个 storm 对应 obs 结束下标
        storm_index_for_obs: 每条 obs 属于哪个 storm
    """
    storms: List[StormHeader] = []
    records: List[TrackRecord] = []
    obs_start_index: List[int] = []
    obs_end_index: List[int] = []
    storm_index_for_obs: List[int] = []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        if not HEADER_RE.match(line.strip()):
            raise ValueError(f"{txt_path} 第 {i + 1} 行不是合法头记录: {line!r}")

        header = parse_header(line)
        storms.append(header)

        start_idx = len(records)

        for _ in range(header.record_count):
            i += 1
            if i >= len(lines):
                raise ValueError(f"{txt_path} 在读取路径记录时提前结束")

            record = parse_track_record(lines[i])
            records.append(record)
            storm_index_for_obs.append(len(storms) - 1)

        end_idx = len(records) - 1
        obs_start_index.append(start_idx)
        obs_end_index.append(end_idx)

        i += 1

    return (
        storms,
        records,
        np.array(obs_start_index, dtype=np.int32),
        np.array(obs_end_index, dtype=np.int32),
        np.array(storm_index_for_obs, dtype=np.int32),
    )


# =========================
# 写入 NetCDF
# =========================

def write_nc(txt_path: str, nc_path: str) -> None:
    storms, records, obs_start_idx, obs_end_idx, storm_index_for_obs = read_txt_file(txt_path)

    nstorm = len(storms)
    nobs = len(records)

    ds = Dataset(nc_path, "w", format="NETCDF4")

    # 维度
    ds.createDimension("storm", nstorm)
    ds.createDimension("obs", nobs)
    ds.createDimension("flag_strlen", 5)
    ds.createDimension("id_strlen", 16)
    ds.createDimension("name_strlen", 20)
    ds.createDimension("date_strlen", 8)
    ds.createDimension("time_strlen", 20)

    # 全局属性
    ds.title = "China Best Track Tropical Cyclone Dataset"
    ds.source = os.path.basename(txt_path)
    ds.institution = "Converted from CHYYYYBST.txt best track files"
    ds.featureType = "trajectory"
    ds.Conventions = "CF-1.8"
    ds.history = f"Created on {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ds.summary = "Converted from CMA tropical cyclone best track TXT files."

    # ===== storm 维变量 =====
    var = ds.createVariable("category_flag", "S1", ("storm", "flag_strlen"))
    var.long_name = "分类标志"
    var.description = "AAAAA, 5字符, '66666'表示最佳路径资料"
    var[:] = stringtochar(np.array([s.category_flag.ljust(5) for s in storms], dtype="S5"))

    var = ds.createVariable("international_id", "S1", ("storm", "id_strlen"))
    var.long_name = "国际编号"
    var.description = "BBBB, 4字符, 年份的最后两位数+两位数的编号"
    var[:] = stringtochar(np.array([s.international_id.ljust(16)[:16] for s in storms], dtype="S16"))

    var = ds.createVariable("record_count", "i4", ("storm",))
    var.long_name = "路径数据记录行数"
    var.description = "CCC, 3字符, 路径数据记录的行数"
    var[:] = np.array([s.record_count for s in storms], dtype=np.int32)

    var = ds.createVariable("tc_sequence_with_td", "i4", ("storm",))
    var.long_name = "包括热带低压在内的热带气旋序号"
    var.description = "DDDD, 4字符, 包括热带低压在内的热带气旋序号"
    var[:] = np.array([s.tc_sequence_with_td for s in storms], dtype=np.int32)

    var = ds.createVariable("china_tc_id", "S1", ("storm", "id_strlen"))
    var.long_name = "我国对热带气旋的编号"
    var.description = "EEEE, 可能为单值，也可能为多值如7127,7128"
    var[:] = stringtochar(np.array([s.china_tc_id.ljust(16)[:16] for s in storms], dtype="S16"))

    var = ds.createVariable("end_flag", "i4", ("storm",))
    var.long_name = "热带气旋终结记录"
    var.description = "F, 0表示消散, 1表示移出西太台风委员会责任海区, 2表示合并, 3表示准静止"
    var.flag_values = np.array([0, 1, 2, 3], dtype=np.int32)
    var.flag_meanings = "dissipated moved_out_of_responsibility_area merged quasi_stationary"
    var[:] = np.array([s.end_flag for s in storms], dtype=np.int32)

    var = ds.createVariable("time_interval_hours", "i4", ("storm",))
    var.long_name = "每行路径间隔小时数"
    var.units = "hour"
    var.description = "G, 2017年以前均为6小时, 2017年开始部分登陆个例有3小时加密记录"
    var[:] = np.array([s.time_interval_hours for s in storms], dtype=np.int32)

    var = ds.createVariable("storm_name", "S1", ("storm", "name_strlen"))
    var.long_name = "热带气旋英文名称"
    var.description = "H...H, 20字符, 名称后加'(-1)n'表示副中心及其序号"
    var[:] = stringtochar(np.array([s.storm_name.ljust(20)[:20] for s in storms], dtype="S20"))

    var = ds.createVariable("dataset_date", "S1", ("storm", "date_strlen"))
    var.long_name = "数据集形成日期"
    var.description = "I...I, 8字符, 数据集形成的日期"
    var[:] = stringtochar(np.array([s.dataset_date.ljust(8)[:8] for s in storms], dtype="S8"))

    var = ds.createVariable("storm_obs_start_index", "i4", ("storm",))
    var.long_name = "该热带气旋路径记录起始下标"
    var.description = "该热带气旋对应 obs 维上的起始位置(从0开始)"
    var[:] = obs_start_idx

    var = ds.createVariable("storm_obs_end_index", "i4", ("storm",))
    var.long_name = "该热带气旋路径记录结束下标"
    var.description = "该热带气旋对应 obs 维上的结束位置(从0开始, 含结束位置)"
    var[:] = obs_end_idx

    # ===== obs 维变量 =====
    var = ds.createVariable("storm_index", "i4", ("obs",))
    var.long_name = "所属热带气旋索引"
    var.description = "该路径记录属于哪个 storm(从0开始)"
    var[:] = storm_index_for_obs

    # 时间先转 datetime，再转为 CF 规范数值时间
    time_datetimes = [yyyymmddhh_to_datetime(r.yyyymmddhh) for r in records]
    time_units = "hours since 1900-01-01 00:00:00"
    time_calendar = "gregorian"
    time_values = date2num(time_datetimes, units=time_units, calendar=time_calendar)

    var = ds.createVariable("time", "f8", ("obs",))
    var.long_name = "时间"
    var.standard_name = "time"
    var.units = time_units
    var.calendar = time_calendar
    var.axis = "T"
    var.description = "UTC time encoded as CF-compliant numeric datetime"
    var[:] = np.array(time_values, dtype=np.float64)

    # 可选：额外保存可直接阅读的 UTC 时间字符串
    var = ds.createVariable("time_str", "S1", ("obs", "time_strlen"))
    var.long_name = "UTC时间字符串"
    var.description = "ISO8601 format: YYYY-MM-DDTHH:MM:SSZ"
    var[:] = stringtochar(
        np.array([yyyymmddhh_to_isoz(r.yyyymmddhh).ljust(20)[:20] for r in records], dtype="S20")
    )

    var = ds.createVariable("intensity_flag", "i4", ("obs",))
    var.long_name = "强度标记"
    var.description = (
        "I, 以正点前2分钟至正点内的平均风速为准; "
        "0=弱于热带低压或等级未知, 1=热带低压, 2=热带风暴, 3=强热带风暴, "
        "4=台风, 5=强台风, 6=超强台风, 9=变性完成"
    )
    var.flag_values = np.array([0, 1, 2, 3, 4, 5, 6, 9], dtype=np.int32)
    var.flag_meanings = "below_td_or_unknown TD TS STS TY STY SuperTY extratropical_completed"
    var[:] = np.array([r.intensity_flag for r in records], dtype=np.int32)

    var = ds.createVariable("latitude", "f4", ("obs",))
    var.long_name = "纬度"
    var.standard_name = "latitude"
    var.units = "degrees_north"
    var.description = "LAT, 纬度(0.1°N), 已转换为十进制度"
    var[:] = np.array([r.lat_0p1deg_n / 10.0 for r in records], dtype=np.float32)

    var = ds.createVariable("longitude", "f4", ("obs",))
    var.long_name = "经度"
    var.standard_name = "longitude"
    var.units = "degrees_east"
    var.description = "LONG, 经度(0.1°E), 已转换为十进制度"
    var[:] = np.array([r.lon_0p1deg_e / 10.0 for r in records], dtype=np.float32)

    var = ds.createVariable("central_pressure", "i4", ("obs",), zlib=True, complevel=4)
    var.long_name = "中心最低气压"
    var.units = "hPa"
    var.description = "PRES, 中心最低气压(hPa)"
    var[:] = np.array([r.pres_hpa for r in records], dtype=np.int32)

    var = ds.createVariable("max_sustained_wind_2min", "i4", ("obs",), zlib=True, complevel=4)
    var.long_name = "2分钟平均近中心最大风速"
    var.units = "m s-1"
    var.description = "WND, 2分钟平均近中心最大风速(MSW, m/s); WND=9表示MSW<10m/s, WND=0为缺测"
    var[:] = np.array([r.wnd_ms for r in records], dtype=np.int32)

    var = ds.createVariable("owd_2min", "i4", ("obs",), zlib=True, complevel=4)
    var.long_name = "2分钟平均风速"
    var.units = "m s-1"
    var.description = (
        "OWD, 2分钟平均风速(m/s): "
        "(a) 对登陆我国的热带气旋, 表示沿海大风风速; "
        "(b) 热带气旋位于南海时, 表示距中心300-500km范围的最大风速"
    )
    var[:] = np.array([r.owd_ms for r in records], dtype=np.int32)

    ds.close()


# =========================
# 批量转换
# =========================

def convert_all_txt_to_nc(input_dir: str, output_dir: str) -> None:
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    txt_files = sorted([f for f in os.listdir(input_dir) if FILE_RE.match(f)])
    if not txt_files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到 CHYYYYBST.txt 文件")

    print(f"共找到 {len(txt_files)} 个 TXT 文件，开始转换...")

    for idx, file_name in enumerate(txt_files, start=1):
        txt_path = os.path.join(input_dir, file_name)
        nc_name = os.path.splitext(file_name)[0] + ".nc"
        nc_path = os.path.join(output_dir, nc_name)

        try:
            write_nc(txt_path, nc_path)
            print(f"[{idx}/{len(txt_files)}] 转换成功: {file_name} -> {nc_name}")
        except Exception as e:
            print(f"[{idx}/{len(txt_files)}] 转换失败: {file_name}")
            print(f"错误信息: {e}")

    print("全部处理完成。")


# =========================
# 主程序
# =========================

def main():
    # ===== 这里改成你自己的路径 =====
    input_dir = r"/Volumes/TiPlus7100/data/Typhoon/CMABSTdata/metadata"     # 存放 CH1949BST.txt ~ CH2024BST.txt 的文件夹
    output_dir = r"/Volumes/TiPlus7100/data/Typhoon/CMABSTdata"   # 输出 nc 文件的文件夹

    convert_all_txt_to_nc(input_dir, output_dir)


if __name__ == "__main__":
    main()
