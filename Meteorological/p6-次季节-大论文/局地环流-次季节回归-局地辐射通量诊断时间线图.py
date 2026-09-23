import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from climkit.filter import LanczosFilter
from climkit.masked import masked


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

PYFILE = r'/volumes/TiPlus7100/PyFile'
DATA = r'/volumes/TiPlus7100/data'
RAW_DIR = '/Volumes/TiPlus7100/p4/data'
YEAR_START, YEAR_END = 1961, 2022
LAG_DAYS = np.arange(-24, 25, 3)
SECONDS_PER_DAY = 86400.0


def build_daily_ehci(pyfile, data_dir, threshold_quantile=0.90):
    """按既有定义构造 1961–2022 年逐日 EHCI 原始序列。"""
    tmax_5day = xr.open_dataarray(fr'{pyfile}/p2/data/Tmax_5Day_filt.nc')
    yangtze_shp = fr'{pyfile}/map/self/长江_TP/长江_tp.shp'
    t95 = masked(
        tmax_5day.sel(day=slice('1', '88')),
        fr'{pyfile}/map/地图边界数据/长江区1：25万界线数据集（2002年）/长江区.shp'
    ).mean(dim=['year', 'day']).quantile(threshold_quantile)
    ehd_binary = masked((tmax_5day - t95 > 0).astype(int), yangtze_shp)
    cn05 = xr.open_dataset(fr'{data_dir}/CN05.1/2022/CN05.1_Tmax_2022_daily_025x025.nc')
    zone_stations = masked(
        (cn05 - cn05 + 1).sel(time='2022-01-01'), yangtze_shp
    ).sum()['tmax'].item()
    if not np.isfinite(zone_stations) or zone_stations <= 0:
        raise ValueError(f'长江流域 CN05.1 格点数异常：{zone_stations}')
    ehci_year_day = (ehd_binary.sum(dim=['lat', 'lon'], skipna=True) / zone_stations)
    ehci_year_day = ehci_year_day.transpose('year', 'day')
    years = pd.to_numeric(ehci_year_day.year.values).astype(int)
    dates = np.concatenate([
        pd.date_range(f'{year}-06-04', periods=ehci_year_day.sizes['day'], freq='D').values
        for year in years
    ])
    return xr.DataArray(
        np.asarray(ehci_year_day.values).reshape(-1),
        coords={'time': dates}, dims='time', name='EHCI'
    )


def filter_ehci_10_90d(ehci):
    """逐年对 EHCI 做 10–60 天 Lanczos 带通滤波，再保留有效的 7–8 月。"""
    filtered = []
    for year in np.unique(pd.DatetimeIndex(ehci.time.values).year):
        yearly = ehci.sel(time=slice(f'{year}-06-04', f'{year}-09-29'))
        filtered.append(
            LanczosFilter(yearly, 'bandpass', period=[10, 60], nwts=61).filted()
        )
    return (
        xr.concat(filtered, dim='time')
        .sel(time=slice(f'{YEAR_START}-07-01', f'{YEAR_END}-08-31'))
        .dropna('time')
        .rename('EHCI_10_90d')
    )


def regress_field_on_ehci(field, ehci, lag, field_name):
    """回归 ``field(t + lag)`` 到 ``EHCI(t)``；lag<0 表示场超前。"""
    ehci_dates = pd.DatetimeIndex(pd.to_datetime(ehci.time.values))
    field_dates = ehci_dates + pd.Timedelta(days=lag)
    available_dates = pd.DatetimeIndex(pd.to_datetime(field.time.values))
    valid_time = available_dates.get_indexer(field_dates) >= 0
    if not valid_time.any():
        raise ValueError(f'{field_name} 与 EHCI 在 lag={lag} d 下没有共同样本。')
    ehci_used = ehci.isel(time=valid_time)
    paired_ehci_dates = ehci_dates[valid_time]
    field_used = field.sel(time=field_dates[valid_time]).assign_coords(
        time=('time', paired_ehci_dates.values)
    )
    valid = field_used.notnull()
    field_valid = field_used.where(valid)
    index_valid = xr.broadcast(ehci_used, field_used)[0].where(valid)
    n_valid = valid.sum('time')
    field_anom = field_valid - field_valid.mean('time', skipna=True)
    index_anom = index_valid - index_valid.mean('time', skipna=True)
    numerator = (field_anom * index_anom).sum('time', skipna=True)
    denominator = (index_anom ** 2).sum('time', skipna=True)
    valid_reg = (n_valid >= 2) & (denominator > 0)
    slope = (numerator / denominator.where(valid_reg, 1)).where(valid_reg)
    print(f'{field_name}: n={ehci_used.sizes["time"]}, lag={lag} d')
    return slope


def get_first_var(ds, candidates, label):
    """从异常资料中按候选名称识别变量。"""
    for name in candidates:
        if name in ds.data_vars:
            print(f'{label}: 使用变量 {name}')
            return ds[name]
    raise KeyError(f'未找到 {label}，候选变量为 {candidates}，实际变量为 {list(ds.data_vars)}')


def materialize(da):
    return da.compute() if hasattr(da.data, 'compute') else da


def to_lon180(da):
    """将经度统一为 -180–180°，并按经纬度排序。"""
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180)
    return da.sortby('lon').sortby('lat')


def region_mean_scalar(da, shp_path):
    """按长江流域 shapefile 对二维回归场做格点平均。"""
    da = to_lon180(da)
    return float(masked(da, shp_path).mean(dim=('lat', 'lon'), skipna=True).values)


# 读取并滤波 EHCI；回归覆盖 lead -24 d 至 lag +24 d。
EHCI = filter_ehci_10_90d(build_daily_ehci(PYFILE, DATA))
RAW = xr.open_dataset(
    f'{RAW_DIR}/ERA5_CPC_daily_ano_all_years_MJJAS_1961_2022.nc', chunks='auto'
)
yangtze_shp = fr'{PYFILE}/map/self/长江_TP/长江_tp.shp'

# 这些变量由 ERA5 日累计量生成，单位为 J m-2 day-1；转换为日平均 W m-2。
fluxes = {
    'ssr_ano': ('地表净短波辐射', '#d95f5f', 'Net solar'),
    'str_ano': ('地表净长波辐射', '#4c78a8', 'Net thermal'),
    'sshf_ano': ('地表感热通量', '#2ca25f', 'Sensible heat'),
    'slhf_ano': ('地表潜热通量', '#e6ab02', 'Latent heat'),
}
region_regressions = {name: [] for name in fluxes}

for name, (title, _, _) in fluxes.items():
    field = get_first_var(RAW, [name], title)
    if 'level' in field.dims:
        field = field.isel(level=0, drop=True)
    field = field.sortby('lat').sortby('lon') / SECONDS_PER_DAY
    for lag in LAG_DAYS:
        regression = materialize(
            to_lon180(regress_field_on_ehci(field, EHCI, int(lag), name))
        )
        region_regressions[name].append(region_mean_scalar(regression, yangtze_shp))

print('长江流域地表辐射通量 EHCI lead/lag 回归（W m$^{-2}$）：')
for name, (title, _, _) in fluxes.items():
    print(f'{title} ({name}): {np.asarray(region_regressions[name])}')

# 长江流域区域平均的四项地表辐射通量诊断时间线。
fig, ax = plt.subplots(figsize=(8.4, 4.8))
for name, (_, color, label) in fluxes.items():
    ax.plot(
        LAG_DAYS, region_regressions[name], marker='o', markersize=3.5,
        linewidth=1.4, color=color, label=label
    )
ax.axhline(0, color='black', linewidth=.7)
ax.axvline(0, color='black', linewidth=.8, linestyle='--')
ax.grid(axis='y', linestyle='--', linewidth=.5, alpha=.7)
ax.set_xlim(-24, 24)
ax.set_xticks(LAG_DAYS)
ax.set_xticklabels([f'{day:+d}' if day != 0 else '0' for day in LAG_DAYS])
ax.set_xlabel('Lead/Lag relative to EHCI (day)')
ax.set_ylabel('Surface flux regression (W m$^{-2}$)')
ax.set_title('Yangtze-basin mean surface radiative fluxes', loc='left', fontsize=10)
ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.02), fontsize=9)
ax.tick_params(labelsize=8)
fig.tight_layout()

output_base = fr'{PYFILE}/p6/pic/JJA_EHCI_Yangtze_surface_flux_timeline'
plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
plt.savefig(f'{output_base}.png', bbox_inches='tight', dpi=600)
plt.show()