import os
import xarray as xr

# =========================================
# Input / Output settings
# =========================================
input_dir = "/Volumes/TiPlus7100/data/ERA5/daily/uvwztSh"
output_file = "/Volumes/TiPlus7100/p4/data/ERA5_daily_uvwztq_sum.zarr"

start_year = 1961
end_year = 2022
summer_months = ["05", "06", "07", "08", "09"]
levels = [200, 500, 850, 925]

file_prefix = "ERA5_daily_uvwztSh_"

# =========================================
# Collect files by level
# =========================================
files_by_level = {lev: [] for lev in levels}

for year in range(start_year, end_year + 1):
    for month in summer_months:
        for lev in levels:
            fname = f"{file_prefix}{year}{month}_{lev}_unzip.nc"
            fpath = os.path.join(input_dir, fname)
            if os.path.exists(fpath):
                files_by_level[lev].append(fpath)
            else:
                print(f"缺失文件，跳过: {fpath}")

valid_levels = [lev for lev in levels if files_by_level[lev]]
if not valid_levels:
    raise FileNotFoundError("没有找到任何符合条件的文件，请检查目录和文件名。")

for lev in valid_levels:
    files_by_level[lev] = sorted(files_by_level[lev])
    print(f"层次 {lev} 共找到 {len(files_by_level[lev])} 个文件")
    for f in files_by_level[lev][:3]:
        print("  ", f)

# =========================================
# Preprocess
# =========================================
def preprocess(ds):
    ds = ds.squeeze(drop=True)

    # 把 valid_time 统一成 time
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # 如果有多余坐标，可按需删掉
    drop_candidates = [v for v in ["number", "expver"] if v in ds.variables]
    if drop_candidates:
        ds = ds.drop_vars(drop_candidates)

    # 如果原文件里已经有 level 坐标/维度，去掉长度为1的情况，避免后面重复
    if "level" in ds.dims and ds.sizes.get("level", 0) == 1:
        ds = ds.squeeze("level", drop=True)
    elif "level" in ds.coords and "level" not in ds.dims:
        ds = ds.drop_vars("level")

    return ds

# =========================================
# Merge each level along time
# =========================================
level_datasets = []

for lev in valid_levels:
    print(f"\n开始处理层次 {lev} ...")

    ds_lev = xr.open_mfdataset(
        files_by_level[lev],
        combine="nested",
        concat_dim="time",
        preprocess=preprocess,
        parallel=False,
        chunks={"time": 30},
        coords="minimal",
        data_vars="minimal",
        compat="override",
        join="outer",
        engine="netcdf4"
    )

    if "time" in ds_lev.coords:
        ds_lev = ds_lev.sortby("time")

    # 增加 level 维度
    ds_lev = ds_lev.expand_dims(level=[lev])

    level_datasets.append(ds_lev)

# =========================================
# Concat along level
# =========================================
print("\n开始按层次拼接 ...")
ds = xr.concat(
    level_datasets,
    dim="level",
    coords="minimal",
    compat="override",
    join="outer"
)

if "time" in ds.coords:
    ds = ds.sortby("time")

# =========================================
# Ensure output directory exists
# =========================================
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# =========================================
# Encoding
# =========================================
encoding = {}

for var in ds.data_vars:
    encoding[var] = {
        "zlib": True,
        "complevel": 4,
        "shuffle": True
    }

for coord in ds.coords:
    if coord not in encoding:
        encoding[coord] = {"zlib": False}


# =========================================
# Save merged nc
# =========================================
# 关键：统一 chunk，避免 (12, 11, 1, 11, 1, ...)
ds = ds.unify_chunks()
ds = ds.chunk({
    "time": 12,
    "level": -1,   # 整个 level 一块；内存紧张可改小
    "latitude": 90,
    "longitude": 180
})

print("\n开始写出 NetCDF ...")
ds.to_zarr(
    output_file,
    mode="w",
    consolidated=True
)

print(f"合并完成，输出文件：{output_file}")

# =========================================
# Close
# =========================================
ds.close()
for ds_lev in level_datasets:
    ds_lev.close()
