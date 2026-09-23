import cmaps
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator
from scipy.stats import t as student_t

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
LEAD = 0


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


def make_tcc_cmap():
    colors = (
        _color_list(cmaps.MPL_PuOr_r[11 + 15:56])
        + _color_list(cmaps.CBR_wet[0])
        + _color_list(cmaps.CBR_wet[0])
        + _color_list(cmaps.CBR_wet[0])
        + _color_list(cmaps.CBR_wet[0])
        + _color_list(cmaps.CBR_wet[0])
        + _color_list(cmaps.CBR_wet[0])
        + _color_list(cmaps.MPL_PuOr_r[64:106 - 15])
    )
    return ListedColormap([mcolors.to_rgba(color) for color in colors], name='tcc')


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


def plot_pressure_panel(fig, loc, z, u, v, level, panel_index,
                        precipitation=None, precipitation_levels=None,
                        precipitation_cmap=None, cloud_cover=None,
                        cloud_cover_levels=None, cloud_cover_cmap=None):
    title = f'({chr(97 + panel_index)}) {level} hPa Z&UV'
    if cloud_cover is not None:
        title += '&TCC'
    ax = setup_global_ax(fig, loc, title)
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
    if cloud_cover is not None:
        tcc_data, tcc_lon, tcc_lat = cyclic_dataarray(cloud_cover)
        tcc_cf = ax.contourf(
            tcc_lon, tcc_lat, tcc_data, levels=cloud_cover_levels,
            cmap=cloud_cover_cmap, extend='both', alpha=1,
            transform=data_crs, antialiased=False, zorder=1
        )
        if hasattr(tcc_cf, 'collections'):
            for collection in tcc_cf.collections:
                collection.set_edgecolor('face')
                collection.set_linewidth(0)
        else:
            tcc_cf.set_edgecolor('face')
            tcc_cf.set_linewidth(0)
    q_uv = ax.Curlyquiver(as_numpy(u.lon), as_numpy(u.lat), as_numpy(u), as_numpy(v),
                          arrowsize=1, transform=data_crs, scale=.2 if level==200 else .6, linewidth=.6,
                          regrid=22, color='#656565', thinning=['40%', 'min'], nanmax=2,
                          MinDistance=[.2, .4], zorder=101)
    q_uv.key(U=24 if level==200 else 8, label='24 m/s' if level==200 else '8 m/s', color='k', fontproperties={'size': 8}, linewidth=.8,
             arrowsize=6, facecolor='none', edgecolor='none', bbox_to_anchor=(0, 0.13, 1, 1))
    return ax, cf


# 读取并滤波 EHCI；同期回归只使用 lag=0。
EHCI = filter_ehci_10_90d(build_daily_ehci(PYFILE, DATA))
RAW = xr.open_dataset(f'{RAW_DIR}/ERA5_CPC_daily_ano_all_years_MJJAS_1961_2022.nc', chunks='auto')
time_slice = slice(f'{YEAR_START}-05-01', f'{YEAR_END}-08-31')
u_raw = RAW['u_ano'].sel(level=[200, 500, 850], time=time_slice)
v_raw = RAW['v_ano'].sel(level=[200, 500, 850], time=time_slice)
z_raw = RAW['z_ano'].sel(level=[200, 500, 850], time=time_slice)
precip_raw = RAW['tp_ano'].sel(time=time_slice)
tcc_raw = RAW['tcc_ano'].sel(time=time_slice)

u_reg = materialize(to_lon180(regress_field_on_ehci(u_raw, EHCI, LEAD, 'u')))
v_reg = materialize(to_lon180(regress_field_on_ehci(v_raw, EHCI, LEAD, 'v')))
z_reg = materialize(to_lon180(regress_field_on_ehci(z_raw, EHCI, LEAD, 'z')))
pre_reg = materialize(to_lon180(regress_field_on_ehci(precip_raw, EHCI, LEAD, 'precip')))
tcc_reg = materialize(to_lon180(regress_field_on_ehci(tcc_raw, EHCI, LEAD, 'tcc')))

PRECIP_LEVELS = np.linspace(-.001, .001, 8)
precip_cmap = make_precip_cmap()
tcc_cmap = make_tcc_cmap()
# TCC 回归系数单位为云量比例；分级边界数与给定色标的颜色数匹配。
TCC_LEVELS = np.linspace(-.5, .5, 8)
fig = plt.figure(figsize=(5, 9))
grid = fig.add_gridspec(3, 1, left=.10, right=.88, hspace=.24)
axes = []
axes.append(plot_pressure_panel(
    fig, grid[0, 0], z_reg.sel(level=200), u_reg.sel(level=200), v_reg.sel(level=200),
    200, 0
)[0])
axes.append(plot_pressure_panel(
    fig, grid[1, 0], z_reg.sel(level=500), u_reg.sel(level=500), v_reg.sel(level=500),
    500, 1, cloud_cover=tcc_reg, cloud_cover_levels=TCC_LEVELS,
    cloud_cover_cmap=tcc_cmap
)[0])
last_ax, last_cf = plot_pressure_panel(
    fig, grid[2, 0], z_reg.sel(level=850), u_reg.sel(level=850), v_reg.sel(level=850),
    850, 2, pre_reg, PRECIP_LEVELS, precip_cmap
)

cbar_ax = fig.add_axes([.90, .12, .02, .76])
cbar = fig.colorbar(last_cf, cax=cbar_ax, orientation='vertical',
                    ticks=PRECIP_LEVELS[::2], drawedges=True)
cbar.ax.tick_params(length=0, labelsize=8)
cbar.dividers.set_linewidth(1.)

for ax in fig.axes:
    for artist in ax.get_children():
        if hasattr(artist, 'set_clip_on'):
            artist.set_clip_on(True)

output_base = fr'{PYFILE}/p6/pic/JJA_EHCI_200ZUV_500ZUVTCC_850ZUV_tp_reg'
plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
plt.savefig(f'{output_base}.png', bbox_inches='tight', dpi=600)
plt.show()