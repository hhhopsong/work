import pandas as pd
import xarray as xr
import numpy as np

txt_file = "/Volumes/TiPlus7100/p4/data/BSISO.2015.INDEX.NORM.LY.txt"
nc_file = "/Volumes/TiPlus7100/p4/data/2015BSISO.nc"

df = pd.read_csv(
    txt_file,
    sep=r"\s+",
    na_values=[-999.900, -999.9, -999],
    engine="python"
)

df = df.rename(columns={
    "BSISO1-1": "BSISO1_1",
    "BSISO1-2": "BSISO1_2",
    "BSISO2-1": "BSISO2_1",
    "BSISO2-2": "BSISO2_2",
    "BSISO1": "BSISO1_amp",
    "BSISO2": "BSISO2_amp",
})

df["time"] = (
    pd.to_datetime(df["YEAR"].astype(str), format="%Y")
    + pd.to_timedelta(df["DAY"] - 1, unit="D")
)

df = df.set_index("time")

ds = xr.Dataset(
    data_vars={
        "BSISO1_1": ("time", df["BSISO1_1"].astype("float32")),
        "BSISO1_2": ("time", df["BSISO1_2"].astype("float32")),
        "BSISO2_1": ("time", df["BSISO2_1"].astype("float32")),
        "BSISO2_2": ("time", df["BSISO2_2"].astype("float32")),
        "BSISO1_amp": ("time", df["BSISO1_amp"].astype("float32")),
        "BSISO2_amp": ("time", df["BSISO2_amp"].astype("float32")),
    },
    coords={
        "time": df.index
    },
    attrs={
        "description": "Lee et al. BSISO index converted from TXT",
        "source_missing_value": "-999.900",
    }
)

ds["BSISO1_1"].attrs["long_name"] = "BSISO1 principal component 1"
ds["BSISO1_2"].attrs["long_name"] = "BSISO1 principal component 2"
ds["BSISO2_1"].attrs["long_name"] = "BSISO2 principal component 1"
ds["BSISO2_2"].attrs["long_name"] = "BSISO2 principal component 2"
ds["BSISO1_amp"].attrs["long_name"] = "BSISO1 amplitude"
ds["BSISO2_amp"].attrs["long_name"] = "BSISO2 amplitude"

for v in ds.data_vars:
    ds[v].attrs["units"] = "1"

ds.to_netcdf(nc_file)

print(ds)
print(f"Saved to {nc_file}")
