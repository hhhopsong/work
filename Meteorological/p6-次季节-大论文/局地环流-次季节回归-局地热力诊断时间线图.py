import cmaps
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy.calc as mpcalc

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator
from scipy.stats import t as student_t
from metpy.constants import dry_air_gas_constant as R
from metpy.constants import dry_air_spec_heat_press as cp
from metpy.units import units

from climkit.Cquiver import *
from climkit.filter import LanczosFilter
from climkit.masked import masked
from climkit.lonlat_transform import transform

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"
RAW_DIR = "/Volumes/TiPlus7100/p4/data"
YEAR_START, YEAR_END = 1961, 2022
LAG_DAYS = np.arange(-24, 25, 3)


def build_daily_ehci(pyfile, data_dir, threshold_quantile=0.90):
    """按既有定义构造 1961–2022 年逐日 EHCI 原始序列。"""
    tmax_5day = xr.open_dataarray(fr"{pyfile}/p2/data/Tmax_5Day_filt.nc")
    yangtze_shp = fr"{pyfile}/map/self/长江_TP/长江_tp.shp"
    t95 = (
        masked(
            tmax_5day.sel(day=slice('1', '88')),
            fr"{pyfile}/map/地图边界数据/长江区1：25万界线数据集（2002年）/长江区.shp"
        ).mean(dim=['year', 'day']).quantile(threshold_quantile)
    )
    ehd_binary = masked((tmax_5day - t95 > 0).astype(int), yangtze_shp)
    cn05 = xr.open_dataset(fr"{data_dir}/CN05.1/2022/CN05.1_Tmax_2022_daily_025x025.nc")
    zone_stations = masked((cn05 - cn05 + 1).sel(time='2022-01-01'), yangtze_shp).sum()['tmax'].item()
    if not np.isfinite(zone_stations) or zone_stations <= 0:
        raise ValueError(f'长江流域 CN05.1 格点数异常：{zone_stations}')
    ehci_year_day = (ehd_binary.sum(dim=['lat', 'lon'], skipna=True) / zone_stations)
    ehci_year_day = ehci_year_day.transpose('year', 'day')
    years = pd.to_numeric(ehci_year_day.year.values).astype(int)
    dates = np.concatenate([
        pd.date_range(f'{year}-06-04', periods=ehci_year_day.sizes['day'], freq='D').values
        for year in years
    ])
    return xr.DataArray(np.asarray(ehci_year_day.values).reshape(-1),
                        coords={'time': dates}, dims='time', name='EHCI')


def filter_ehci_10_90d(ehci):
    """逐年对 EHCI 做 10–90 天 Lanczos 带通滤波，再保留有效的 7–8 月。"""
    filtered = []
    for year in np.unique(pd.DatetimeIndex(ehci.time.values).year):
        yearly = ehci.sel(time=slice(f'{year}-06-04', f'{year}-09-29'))
        yearly_bp = LanczosFilter(
            yearly, 'bandpass', period=[10, 60], nwts=61
        ).filted()
        filtered.append(yearly_bp)
    return (
        xr.concat(filtered, dim='time')
        .sel(time=slice(f'{YEAR_START}-07-01', f'{YEAR_END}-08-31'))
        .dropna('time')
        .rename('EHCI_10_90d')
    )


def regress_field_on_ehci(field, ehci, lag, field_name, return_pvalue=False):
    """回归 ``field(t + lag)`` 到 ``EHCI(t)``；lag<0 表示场超前。"""
    ehci_dates = pd.DatetimeIndex(pd.to_datetime(ehci.time.values))
    field_dates = ehci_dates + pd.Timedelta(days=lag)
    available_dates = pd.DatetimeIndex(pd.to_datetime(field.time.values))
    valid_time = available_dates.get_indexer(field_dates) >= 0
    if not valid_time.any():
        raise ValueError(f'{field_name} 与 EHCI 在 lag={lag} d 下没有共同样本。')
    ehci_used = ehci.isel(time=valid_time)
    paired_ehci_dates, paired_field_dates = ehci_dates[valid_time], field_dates[valid_time]
    field_used = field.sel(time=paired_field_dates).assign_coords(time=('time', paired_ehci_dates.values))
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
    if not return_pvalue:
        return slope
    field_ss = (field_anom ** 2).sum('time', skipna=True)
    corr_denominator = np.sqrt(field_ss * denominator)
    valid_corr = valid_reg & (field_ss > 0) & (corr_denominator > 0)
    r_value = xr.where(valid_corr, numerator / corr_denominator.where(valid_corr, 1), 0).clip(-1, 1)
    valid_test = valid_corr & (n_valid > 2) & (np.abs(r_value) < 1)
    t_value = r_value * np.sqrt((n_valid - 2).where(valid_test, 0) /
                                (1 - r_value ** 2).where(valid_test, 1))
    p_value = xr.apply_ufunc(lambda t_, df_: 2 * student_t.sf(np.abs(t_), df_),
                             t_value, n_valid - 2, dask='parallelized', output_dtypes=[float]).where(valid_test)
    return slope, p_value


def get_first_var(ds, candidates, label):
    """从异常资料中按候选名称识别变量。"""
    for name in candidates:
        if name in ds.data_vars:
            print(f'{label}: 使用变量 {name}')
            return ds[name]
    raise KeyError(f'未找到 {label}，候选变量为 {candidates}，实际变量为 {list(ds.data_vars)}')


def calc_daily_temperature_budget(t_da, u_da, v_da, omega_da):
    """按参考脚本的定义计算逐日温度方程，返回 K/s。

    水平梯度和时间差分采用 NumPy 向量化计算，数值定义与 MetPy
    ``advection`` 及参考脚本的中心差分一致，但避免逐日逐层调用函数。
    """
    t_da, u_da, v_da, omega_da = [
        da.transpose('time', 'level', 'lat', 'lon').load()
        for da in (t_da, u_da, v_da, omega_da)
    ]
    times = t_da.time.values
    levels = t_da.level.values.astype(float)
    lats = t_da.lat.values.astype(float)
    lons = t_da.lon.values.astype(float)
    t_values = np.asarray(t_da.values, dtype=float)
    u_values = np.asarray(u_da.values, dtype=float)
    v_values = np.asarray(v_da.values, dtype=float)
    omega_values = np.asarray(omega_da.values, dtype=float)
    p_pa = levels * 100.0

    earth_radius = 6_371_000.0
    dlon = np.deg2rad(np.gradient(lons))[None, None, None, :]
    dlat = np.deg2rad(np.gradient(lats))[None, None, :, None]
    dx = earth_radius * np.cos(np.deg2rad(lats))[None, None, :, None] * dlon
    dy = earth_radius * dlat
    dtdx = np.gradient(t_values, axis=-1) / dx
    dtdy = np.gradient(t_values, axis=-2) / dy
    adv_arr = -(u_values * dtdx + v_values * dtdy)

    time_seconds = (times - times[0]) / np.timedelta64(1, 's')
    dtdt_arr = np.gradient(t_values, time_seconds, axis=0)
    dtdp = np.gradient(t_values, p_pa, axis=1)
    sigma = (R.magnitude * t_values /
             (cp.magnitude * p_pa[None, :, None, None]) - dtdp)
    ver_arr = omega_values * sigma
    q_arr = dtdt_arr - adv_arr - ver_arr

    coords = {'time': times, 'level': levels, 'lat': lats, 'lon': lons}
    return xr.Dataset({
        'adv_T': (('time', 'level', 'lat', 'lon'), adv_arr),
        'ver': (('time', 'level', 'lat', 'lon'), ver_arr),
        'Q': (('time', 'level', 'lat', 'lon'), q_arr),
    }, coords=coords)


def to_lon180(da):
    """将经度统一为 -180–180° 并按经纬度排序。"""
    if float(da.lon.max()) > 180:
        try:
            da = transform(da, 'lon', '360->180')
        except Exception:
            da = da.assign_coords(lon=((da.lon + 180) % 360) - 180)
    return da.sortby('lon').sortby('lat')


def materialize(da):
    return da.compute() if hasattr(da.data, 'compute') else da


def as_numpy(da):
    return np.asarray(materialize(da).values if isinstance(da, xr.DataArray) else da)


def cyclic_dataarray(da):
    da = to_lon180(da)
    values, lon = add_cyclic_point(da.values, coord=da.lon.values)
    return values, lon, da.lat.values


def as_dataarray(arr, ref, name):
    if isinstance(arr, xr.DataArray):
        return to_lon180(arr)
    return xr.DataArray(arr, coords={'lat': ref.lat, 'lon': ref.lon},
                        dims=('lat', 'lon'), name=name)


def _color_list(obj):
    """将 cmaps 的 Colormap 或颜色对象转换为 RGBA 颜色列表。"""
    if hasattr(obj, 'colors'):
        obj = obj.colors

    try:
        values = np.asarray(obj, dtype=float)
        if values.ndim == 1 and values.size in (3, 4):
            return [tuple(values.tolist())]
        if values.ndim == 2 and values.shape[1] in (3, 4):
            return [tuple(row) for row in values.tolist()]
    except (TypeError, ValueError):
        pass

    if isinstance(obj, (list, tuple, np.ndarray)):
        result = []
        for item in obj:
            result.extend(_color_list(item))
        return result

    if mcolors.is_color_like(obj):
        return [obj]
    raise TypeError(f'无法将对象转换为颜色：{type(obj)!r}')


def make_precip_cmap():
    colors = (
        _color_list(cmaps.MPL_RdYlGn[22 + 0:56 - 10])
        + _color_list(cmaps.CBR_wet[0])
        + _color_list(cmaps.MPL_RdYlGn[72 + 10:106 - 0])
    )
    return ListedColormap([mcolors.to_rgba(color) for color in colors], name='precip')


def setup_global_ax(fig, loc, title):
    ax = fig.add_subplot(loc, projection=ccrs.PlateCarree(central_longitude=110))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    ax.set_aspect('auto')
    ax.set_title(title, loc='left', fontsize=10)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=1., color='#AAAAAA')
    ax.add_geometries(
        Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black',
        linewidth=.5,
        zorder=9
    )
    ax.add_geometries(
        Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='gray',
        edgecolor='black',
        linewidth=.5,
        zorder=10
    )
    ax.set_xticks(np.arange(60, 161, 20), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 61, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(which='major', length=4, width=.5, color='black')
    ax.tick_params(which='minor', length=2, width=.2, color='black')
    ax.tick_params(axis='both', labelsize=8, colors='black')
    return ax


def region_mean_scalar(da, shp_path):
    """按长江流域 shapefile 对二维场做格点平均。"""
    da = to_lon180(da).sortby('lat')
    return float(masked(da, shp_path).mean(dim=('lat', 'lon'), skipna=True).values)


def plot_budget_map(fig, spec, field, title, levels, cmap):
    ax = fig.add_subplot(spec, projection=ccrs.PlateCarree(central_longitude=110))
    ax.set_extent([60, 160, 0, 60], crs=ccrs.PlateCarree())
    ax.set_aspect('auto')
    ax.set_title(title, loc='left', fontsize=10)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.7, color='#AAAAAA')
    ax.add_geometries(Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='none', edgecolor='black',
                      linewidth=.6, zorder=5)
    data, lon, lat = cyclic_dataarray(field)
    cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap, extend='both',
                     transform=ccrs.PlateCarree(), antialiased=False)
    ax.set_xticks(np.arange(60, 161, 20), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 61, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=8)
    return ax, cf


def plot_pressure_panel(fig, loc, z, u, v, level, panel_index,
                        precipitation=None, precipitation_levels=None,
                        precipitation_cmap=None):
    ax = setup_global_ax(fig, loc, f'({chr(97 + panel_index)}) {level} hPa Z&UV')
    data_crs = ccrs.PlateCarree()
    ax.tick_params(axis='x', which='both', bottom=False, top=False,
                   labelbottom=False, labeltop=False)
    if precipitation is not None:
        pre_data, pre_lon, pre_lat = cyclic_dataarray(precipitation)
        cf = ax.contourf(pre_lon, pre_lat, pre_data, levels=precipitation_levels,
                         cmap=precipitation_cmap, extend='both', alpha=.8,
                         transform=data_crs, antialiased=False, zorder=1)
        if hasattr(cf, 'collections'):
            for collection in cf.collections:
                collection.set_edgecolor('face')
                collection.set_linewidth(0)
        else:
            cf.set_edgecolor('face')
            cf.set_linewidth(0)
    else:
        cf = None
    z_data, z_lon, z_lat = cyclic_dataarray(z)
    z_levels = np.arange(-600, 601, 200) if level == 200 else np.arange(-300, 301, 100)
    ax.contour(z_lon, z_lat, z_data, levels=z_levels[z_levels < 0],
               colors='#2574a9', linestyles='--', linewidths=.7,
               transform=data_crs, zorder=20)
    ax.contour(z_lon, z_lat, z_data, levels=z_levels[z_levels > 0],
               colors='#c53b47', linestyles='-', linewidths=.7,
               transform=data_crs, zorder=20)
    q_uv = ax.Curlyquiver(as_numpy(u.lon), as_numpy(u.lat), as_numpy(u), as_numpy(v),
                          arrowsize=1, transform=data_crs, scale=.2 if level==200 else .6, linewidth=.6,
                          regrid=22, color='#656565', thinning=['40%', 'min'], nanmax=2,
                          MinDistance=[.2, .4], zorder=101)
    q_uv.key(U=24 if level==200 else 8, label='24 m/s' if level==200 else '8 m/s', color='k', fontproperties={'size': 8}, linewidth=.8,
             arrowsize=6, facecolor='none', edgecolor='none', bbox_to_anchor=(0, 0.13, 1, 1))
    return ax, cf


# 读取并滤波 EHCI；回归覆盖 lead -24 d 至 lag +24 d。
EHCI = filter_ehci_10_90d(build_daily_ehci(PYFILE, DATA))
RAW = xr.open_dataset(f'{RAW_DIR}/ERA5_CPC_daily_ano_all_years_MJJAS_1961_2022.nc', chunks='auto')

# 文件实际只包含 200、500、850、925 hPa；保留全部可用层次，
# 用 850–925 hPa 的温度梯度计算 925 hPa 的静力稳定度。
AVAILABLE_LEVELS = np.asarray(RAW.level.values)
missing_budget_levels = [level for level in (850, 925)
                         if level not in AVAILABLE_LEVELS]
if missing_budget_levels:
    raise ValueError(f'温度诊断缺少层次 {missing_budget_levels} hPa，'
                     f'数据层次为：{AVAILABLE_LEVELS.tolist()}')
BUDGET_LEVELS = [int(level) for level in AVAILABLE_LEVELS
                 if 200 <= level <= 1000]
print(f'温度方程计算层次：{BUDGET_LEVELS}')

# 温度倾向采用中心差分；读取所有 lead/lag 日期及其前后一天，
# 避免对全时段无关日期重复计算温度方程。
ehci_dates = pd.DatetimeIndex(pd.to_datetime(EHCI.time.values))
diagnostic_dates = pd.DatetimeIndex([])
for lag in LAG_DAYS:
    diagnostic_dates = diagnostic_dates.union(
        ehci_dates + pd.Timedelta(days=int(lag))
    )
budget_dates = diagnostic_dates.union(diagnostic_dates - pd.Timedelta(days=1))
budget_dates = budget_dates.union(diagnostic_dates + pd.Timedelta(days=1)).sort_values()

def select_budget_domain(da):
    """裁剪诊断和绘图所需区域，并确保经纬度升序。"""
    da = da.sortby('lat').sortby('lon')
    return da.sel(level=BUDGET_LEVELS, time=budget_dates,
                  lat=slice(-10, 70), lon=slice(50, 170))


t_raw = select_budget_domain(get_first_var(RAW, ['t_ano', 't'], '温度'))
u_raw = select_budget_domain(get_first_var(RAW, ['u_ano', 'u'], '纬向风'))
v_raw = select_budget_domain(get_first_var(RAW, ['v_ano', 'v'], '经向风'))
w_raw = select_budget_domain(get_first_var(RAW, ['w_ano', 'omega_ano', 'w', 'omega'], '垂直速度'))

budget = calc_daily_temperature_budget(t_raw, u_raw, v_raw, w_raw)
yangtze_shp = r'/Volumes/TiPlus7100/PyFile/map/self/长江_TP/长江_tp.shp'
budget_region_mean = {name: [] for name in ('adv_T', 'ver', 'Q')}
term_titles = {
    'adv_T': '水平平流',
    'ver': '垂直运动',
    'Q': '非绝热加热',
}
for name, title in [('adv_T', 'horizontal advection'),
                    ('ver', 'vertical motion'),
                    ('Q', 'diabatic heating')]:
    # 850 和 925 hPa 两层等权平均后，分别进行各 lead/lag 的回归。
    field = budget[name].sel(level=[850, 925]).mean('level') * 86400.0
    for lag in LAG_DAYS:
        regression = materialize(to_lon180(
            regress_field_on_ehci(field, EHCI, int(lag), title)
        ))
        budget_region_mean[name].append(
            region_mean_scalar(regression, yangtze_shp)
        )

print('850–925 hPa 平均的长江流域 EHCI lead/lag 回归温度收支（K/day）：')
for name in ('adv_T', 'ver', 'Q'):
    print(f'{term_titles[name]}: {np.asarray(budget_region_mean[name])}')

# 长江流域区域平均的三项温度诊断时间线。
fig, ax = plt.subplots(figsize=(8.4, 4.8))
line_styles = {
    'adv_T': {'color': '#d95f5f', 'label': 'Adv.'},
    'ver': {'color': '#4c78a8', 'label': 'w'},
    'Q': {'color': '#2ca25f', 'label': 'Q'},
}
for name in ('adv_T', 'ver', 'Q'):
    ax.plot(LAG_DAYS, budget_region_mean[name], marker='o', markersize=3.5,
            linewidth=1.4, **line_styles[name])
ax.axhline(0, color='black', linewidth=.7)
ax.axvline(0, color='black', linewidth=.8, linestyle='--')
ax.grid(axis='y', linestyle='--', linewidth=.5, alpha=.7)
ax.set_xlim(-24, 24)
ax.set_xticks(LAG_DAYS)
ax.set_xticklabels([f'{day:+d}' if day != 0 else '0' for day in LAG_DAYS])
ax.set_xlabel('Lead/Lag relative to EHCI (day)')
ax.set_ylabel('Temperature tendency regression (K day$^{-1}$)')
ax.set_title('850–925 hPa Yangtze-basin mean temperature budget', loc='left', fontsize=10)
ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.02),
          fontsize=9)
ax.tick_params(labelsize=8)
fig.tight_layout()

output_base = fr'{PYFILE}/p6/pic/JJA_EHCI_850-925mean_temperature_budget_timeline'
plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
plt.savefig(f'{output_base}.png', bbox_inches='tight', dpi=600)
plt.show()