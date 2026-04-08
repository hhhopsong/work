# -*- coding: utf-8 -*-
import os
import zipfile
import tempfile
from pathlib import Path

import xarray as xr


# ========= 你的目录 =========
DATA = r"/volumes/TiPlus7100/data"
DATA_DIR = fr"{DATA}/ERA5/daily/uvwztSh"
# ==========================


def try_open_nc(nc_path: str) -> xr.Dataset:
    """
    优先按 netcdf4 打开，失败再试 scipy / h5netcdf
    """
    engines = ["netcdf4", "h5netcdf", "scipy", None]
    last_err = None

    for eng in engines:
        try:
            if eng is None:
                ds = xr.open_dataset(nc_path)
            else:
                ds = xr.open_dataset(nc_path, engine=eng)
            return ds
        except Exception as e:
            last_err = e

    raise RuntimeError(f"无法读取 nc 文件: {nc_path}\n最后错误: {last_err}")


def collect_nc_from_zip(fake_nc_path: str):
    """
    把外层文件当 zip 打开，返回其中所有 nc 的 Dataset 列表
    """
    datasets = []

    with zipfile.ZipFile(fake_nc_path, "r") as zf:
        members = zf.namelist()

        # 只取 zip 内部真正的 .nc 文件
        nc_members = [m for m in members if m.lower().endswith(".nc")]

        if not nc_members:
            raise RuntimeError(f"压缩包内没有找到 .nc 文件: {fake_nc_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            for member in nc_members:
                extracted = zf.extract(member, path=tmpdir)
                ds = try_open_nc(extracted)
                datasets.append(ds)

    return datasets


def merge_datasets(ds_list):
    """
    合并多个 Dataset
    这里用 compat='override' / combine_attrs='override'
    避免属性或坐标轻微差异导致报错
    """
    merged = xr.merge(ds_list, compat="override", combine_attrs="override")
    return merged


def main():
    folder = Path(DATA_DIR)
    files = sorted([p for p in folder.iterdir() if p.is_file()])

    print(f"待处理文件数: {len(files)}")

    for fp in files:
        # 例如：
        # 原文件：ERA5_daily_uvwztSh_500_201507.nc
        # 输出： ERA5_daily_uvwztSh_500_201507_unzip.nc
        out_name = fp.stem + "_unzip.nc"
        out_path = fp.with_name(out_name)

        print(f"\n处理: {fp.name}")

        try:
            # 直接把外层文件按 zip 打开
            ds_list = collect_nc_from_zip(str(fp))

            print(f"  压缩包内 nc 数量: {len(ds_list)}")

            merged = merge_datasets(ds_list)

            # 保存
            encoding = {var: {"zlib": True, "complevel": 4} for var in merged.data_vars}

            merged.to_netcdf(out_path, engine="netcdf4", encoding=encoding)

            print(f"  已保存: {out_path.name}")

            # 关闭文件
            for ds in ds_list:
                ds.close()
            merged.close()

        except zipfile.BadZipFile:
            print(f"  不是合法 zip: {fp.name}")
        except Exception as e:
            print(f"  失败: {fp.name}")
            print(f"  错误: {e}")


if __name__ == "__main__":
    main()
