import os
import glob
import xarray as xr

# =========================================
# Input / Output settings
# =========================================
input_dir = "/Volumes/TiPlus7100/data/ERA5/daily/slp_tpp_rad_tcc_eva_peva"
output_file = "/Volumes/TiPlus7100/p4/data/single_level_sum.zarr"

start_year = 1961
end_year = 2022
summer_months = {"05", "06", "07", "08", "09"}

# 文件名格式：
# ERA5_daily_2m_temperature_YYYYMM.nc
file_prefix = "ERA5_daily_slp_tpp_rad_tcc_eva_peva_"

# =========================================
# Collect target files
# =========================================
selected_files = []

for year in range(start_year, end_year + 1):
    for month in sorted(summer_months):
        fname = f"{file_prefix}{year}{month}_unzip.nc"
        fpath = os.path.join(input_dir, fname)
        if os.path.exists(fpath):
            selected_files.append(fpath)
        else:
            print(f"缺失文件，跳过: {fpath}")

if not selected_files:
    raise FileNotFoundError("没有找到任何符合条件的夏季文件，请检查目录和文件名。")

selected_files = sorted(selected_files)
print(f"共找到 {len(selected_files)} 个文件。")
print("前 5 个文件示例：")
for f in selected_files[:5]:
    print("  ", f)

# =========================================
# Open and merge
# =========================================
# combine='by_coords' 会按坐标自动拼接

ds = xr.open_mfdataset(
    selected_files,
    combine="by_coords",
    parallel=False
)

# 如果有 time 坐标，按时间排序
if "valid_time" in ds.coords:
    ds = ds.rename({"valid_time": "time"})

if "time" in ds.coords:
    ds = ds.sortby("time")

# =========================================
# Ensure output directory exists
# =========================================
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# =========================================
# Save merged nc
# =========================================
# 关键：统一 chunk，避免 (12, 11, 1, 11, 1, ...)
ds = ds.unify_chunks()
ds = ds.chunk({
    "time": 12,
    "latitude": -1,
    "longitude": -1
})

ds.to_zarr(
    output_file,
    mode="w",
    consolidated=True
)
print(f"合并完成，输出文件：{output_file}")

# 关闭数据集
ds.close()



