import os
import pandas as pd
import xarray as xr


PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
# ============================================================
# 输出目录
# ============================================================
OUTPUT_DIR = fr"{PYFILE}/p4/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

UVZ_OUTPUT = os.path.join(OUTPUT_DIR, "uvz_7.nc")
T2M_OUTPUT = os.path.join(OUTPUT_DIR, "t2m_7.nc")


# ============================================================
# 6—8月的200、500、850 hPa风场和位势高度
# ============================================================
try:
    uvz = xr.open_dataset(UVZ_OUTPUT)

except (FileNotFoundError, OSError):
    uvz_raw = xr.open_dataset(
        fr"{DATA}/ERA5/ERA5_pressLev/era5_pressLev.nc"
    ).sel(
        date=slice("1961-01-01", "2023-12-31"),
        pressure_level=[200, 500, 850],
        latitude=[90 - i * 0.5 for i in range(361)],
        longitude=[i * 0.5 for i in range(720)]
    )

    uvz = xr.Dataset(
        data_vars={
            "u": (
                ["time", "p", "lat", "lon"],
                uvz_raw["u"].data
            ),
            "v": (
                ["time", "p", "lat", "lon"],
                uvz_raw["v"].data
            ),
            "z": (
                ["time", "p", "lat", "lon"],
                uvz_raw["z"].data
            )
        },
        coords={
            "time": pd.to_datetime(
                uvz_raw["date"].data,
                format="%Y%m%d"
            ),
            "p": uvz_raw["pressure_level"].data,
            "lat": uvz_raw["latitude"].data,
            "lon": uvz_raw["longitude"].data
        }
    )

    uvz_raw.close()

    # 选取1961—2022年6、7、8月，并计算逐年夏季平均
    uvz = uvz.sel(
        time=slice("1961-01-01", "2022-12-31")
    )

    uvz = uvz.sel(
        time=uvz["time.month"].isin([7])
    ).groupby("time.year").mean(
        dim="time",
        keep_attrs=True
    )

    uvz.to_netcdf(UVZ_OUTPUT)


# 选取500 hPa，并调整维度顺序
uvz = uvz.sel(p=500).transpose("year", "lat", "lon")

# 1961—2022年夏季气候平均
uvz_clim = uvz.mean(dim="year")


# ============================================================
# 6—8月2米气温
# ============================================================

t2m_raw = xr.open_dataset(
    fr"{DATA}/ERA5/ERA5_land/uv_2mTTd_sfp_pre_0.nc"
)["t2m"]

t2m_raw = t2m_raw.rename({"valid_time": "date"})

t2m_raw = t2m_raw.sel(
    date=slice("1961-01-01", "2023-12-31")
)

t2m = xr.Dataset(
    data_vars={
        "t2m": (
            ["time", "lat", "lon"],
            t2m_raw.data
        )
    },
    coords={
        "time": pd.to_datetime(
            t2m_raw["date"].data,
            format="%Y%m%d"
        ),
        "lat": t2m_raw["latitude"].data,
        "lon": t2m_raw["longitude"].data
    }
)

# 关闭原始数据文件
t2m_raw.close()

# 选取1961—2022年6、7、8月，并计算逐年夏季平均
t2m = t2m.sel(
    time=slice("1961-01-01", "2022-12-31")
)

t2m = t2m.sel(
    time=t2m["time.month"].isin([7])
).groupby("time.year").mean(
    dim="time",
    keep_attrs=True
)

t2m.to_netcdf(T2M_OUTPUT)


# 调整维度顺序
t2m = t2m.transpose("year", "lat", "lon")

# 1961—2022年夏季气候平均
t2m_clim = t2m.mean(dim="year")
