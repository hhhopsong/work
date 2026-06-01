# -*- coding: utf-8 -*-
"""
完整可替换版：只绘制 region maps 相关图，不再自动筛选预测因子、不再逐步回归、不再框选预测因子区域。

功能说明：
1. 只计算并绘制各气象要素与目标指数的相关场。
2. 按 elem + mode 分别出图：
   - mean: 连续两月平均相关图
   - trend: 单月差值相关图，后月 - 前月
   - two_month_trend: 两月平均差值相关图，后两月平均 - 前两月平均
3. 每张图中每个 month_expr 为一个子图。
4. 不再绘制因子虚线框，不再标注 X_name。
5. 显著性打点改为 quiver 零矩阵，并用 regrid_shape 重采样控制密度。
6. 海冰 SIC 使用北极立体投影 NorthPolarStereo，绘图范围限制为 60N 以北。
7. 其他非海冰要素绘图范围限制为 30S 以北。
8. 默认 SCI 风格，每行 6 张图，并压缩横纵间隙。

注意：
- 该脚本依赖你原来的 climkit、本地数据路径和字体环境。
- 如果 Cartopy 第一次使用 Natural Earth 数据，可能会联网下载海岸线数据。
"""

# =========================================================
# 0. imports
# =========================================================
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.path import Path

from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from scipy import stats

from climkit.significance_test import r_test
from climkit.data_read import *
from climkit.corr_reg import corr


read_sic = sic


# =========================================================
# 1. 数据预处理
# =========================================================
def prepare_sic_dataset(ds):
    """Normalize HadISST sea-ice dataset to a stable {'sic': [time, lat, lon]} schema."""
    rename_map = {}
    if 'latitude' in ds.coords and 'lat' not in ds.coords:
        rename_map['latitude'] = 'lat'
    if 'longitude' in ds.coords and 'lon' not in ds.coords:
        rename_map['longitude'] = 'lon'
    if len(rename_map) > 0:
        ds = ds.rename(rename_map)

    if 'sic' not in ds.data_vars:
        for cand in ('ice', 'siconc', 'ci'):
            if cand in ds.data_vars:
                ds = ds.rename({cand: 'sic'})
                break

    if 'sic' not in ds.data_vars:
        raise ValueError(f"SIC variable not found in dataset. Available vars: {list(ds.data_vars)}")

    sic_da = ds['sic']
    vmax = float(np.nanmax(sic_da.values))
    if vmax <= 1.5:
        sic_da = sic_da.where((sic_da >= 0) & (sic_da <= 1.0))
    else:
        sic_da = sic_da.where((sic_da >= 0) & (sic_da <= 100.0))

    out = xr.Dataset({'sic': sic_da})
    out = out.sortby('lat').sortby('lon')
    return out


def prepare_swvl_dataset(swvl_like):
    """
    Normalize soil moisture field to a stable {'swvl': [time, lat, lon]} schema.
    支持 DataArray / Dataset。
    """
    if isinstance(swvl_like, xr.DataArray):
        da = swvl_like.copy()
        if da.name is None:
            da = da.rename('swvl')
        elif da.name != 'swvl':
            da = da.rename('swvl')
        ds = da.to_dataset()

    elif isinstance(swvl_like, xr.Dataset):
        ds = swvl_like.copy()

        rename_map = {}
        if 'latitude' in ds.coords and 'lat' not in ds.coords:
            rename_map['latitude'] = 'lat'
        if 'longitude' in ds.coords and 'lon' not in ds.coords:
            rename_map['longitude'] = 'lon'
        if len(rename_map) > 0:
            ds = ds.rename(rename_map)

        if 'swvl' not in ds.data_vars:
            for cand in ('swvl1', 'soil_moisture', 'sm'):
                if cand in ds.data_vars:
                    ds = ds.rename({cand: 'swvl'})
                    break
    else:
        raise TypeError("swvl_like must be xarray.DataArray or xarray.Dataset")

    rename_map = {}
    if 'latitude' in ds.coords and 'lat' not in ds.coords:
        rename_map['latitude'] = 'lat'
    if 'longitude' in ds.coords and 'lon' not in ds.coords:
        rename_map['longitude'] = 'lon'
    if len(rename_map) > 0:
        ds = ds.rename(rename_map)

    if 'swvl' not in ds.data_vars:
        raise ValueError(f"SWVL variable not found in dataset. Available vars: {list(ds.data_vars)}")

    out = xr.Dataset({'swvl': ds['swvl']})
    out = out.sortby('lat').sortby('lon')
    return out


# =========================================================
# 2. 月份组合与时间平均
# =========================================================
def month_to_season_order(month, cross_month=9):
    """
    把自然月映射到以 cross_month 为起点的季节年顺序。
    例如 cross_month=9: 9->1, 10->2, 11->3, 12->4, 1->5, ..., 8->12
    """
    return ((month - cross_month) % 12) + 1


def generate_avg_month_combos_excluding_predict_month(predict_month=(7, 8)):
    """
    连续两个月平均：12-1, 1-2, 2-3, ..., 11-12 都视为连续。
    组合中不能包含 predict_month。
    """
    combos = []
    pm = set(predict_month)

    for m in range(1, 13):
        m_next = 1 if m == 12 else (m + 1)
        combo = [m, m_next]
        if len(set(combo) & pm) > 0:
            continue
        combos.append(combo)

    return combos


def generate_trend_month_combos_excluding_predict_month(cross_month=9, predict_month=(7, 8)):
    """
    单月趋势：严格按季节年时间顺序 later - earlier。
    月份顺序由 cross_month 定义。
    """
    pm = set(predict_month)
    season_seq = list(range(cross_month, 13)) + list(range(1, cross_month))
    valid_months = [m for m in season_seq if m not in pm]

    combos = []
    for i in range(len(valid_months)):
        for j in range(i):
            earlier = valid_months[j]
            later = valid_months[i]
            combos.append(([later], [earlier]))

    return combos


def generate_two_month_trend_combos_excluding_predict_month(cross_month=9, predict_month=(7, 8)):
    """
    两月平均趋势：后面连续两月平均 - 前面连续两月平均。
    两段内部不能包含 predict_month，两段之间不能重叠月份。
    """
    pm = set(predict_month)
    season_seq = list(range(cross_month, 13)) + list(range(1, cross_month))
    n = len(season_seq)

    segments = []
    for i in range(n):
        seg = [season_seq[i], season_seq[(i + 1) % n]]
        if len(set(seg) & pm) > 0:
            continue
        segments.append(seg)

    combos = []
    for i_later, later_seg in enumerate(segments):
        later_set = set(later_seg)
        for i_earlier, earlier_seg in enumerate(segments):
            if i_earlier >= i_later:
                continue
            earlier_set = set(earlier_seg)
            if len(later_set & earlier_set) > 0:
                continue
            combos.append((later_seg, earlier_seg))

    return combos


def seasonal_mean_same_year(da, months, start_year, end_year):
    """不跨年的月份平均，例如 [3, 4, 5]。"""
    return (
        da.sel(time=da['time.month'].isin(months))
          .sel(time=slice(f'{start_year}', f'{end_year}'))
          .groupby('time.year')
          .mean('time')
          .transpose('year', 'lat', 'lon')
    )


def seasonal_mean_cross_year(da, months, start_year, end_year, cross_month=9):
    """
    跨年月份平均，例如 [11, 12, 1, 2]。
    规则：
    - month >= cross_month 的月份，season_year = 原年份 + 1
    - month <  cross_month 的月份，season_year = 原年份
    """
    da_sel = da.sel(time=da['time.month'].isin(months)).sel(
        time=slice(f'{start_year - 1}', f'{end_year}')
    )

    season_year = xr.where(
        da_sel['time.month'] >= cross_month,
        da_sel['time.year'] + 1,
        da_sel['time.year']
    )

    da_sel = da_sel.assign_coords(season_year=('time', season_year.data))

    out = (
        da_sel.groupby('season_year')
              .mean('time')
              .rename({'season_year': 'year'})
              .transpose('year', 'lat', 'lon')
    )

    return out.sel(year=slice(start_year, end_year))


def get_seasonal_mean(da, months, start_year, end_year, cross_month=9):
    """根据月份是否跨季节年自动计算季节平均。"""
    months = [int(m) for m in months]
    cross_year = any(m >= cross_month for m in months)
    if cross_year:
        return seasonal_mean_cross_year(da, months, start_year, end_year, cross_month)
    return seasonal_mean_same_year(da, months, start_year, end_year)


def format_months(months):
    """用于子图标题的月份表达。"""
    return '/'.join([f'{int(m):02d}' for m in months])


def build_plot_tasks(mode, predict_month=(7, 8), cross_month=9):
    """构造某个 mode 下所有需要绘制的 month_expr 任务。"""
    tasks = []

    if mode == 'mean':
        combos = generate_avg_month_combos_excluding_predict_month(
            predict_month=predict_month
        )
        for months in combos:
            tasks.append({
                'mode': mode,
                'month1': months,
                'month2': None,
                'month_expr': format_months(months),
                'order_key': tuple(month_to_season_order(m, cross_month) for m in months)
            })

    elif mode == 'trend':
        combos = generate_trend_month_combos_excluding_predict_month(
            cross_month=cross_month,
            predict_month=predict_month
        )
        for later_month, earlier_month in combos:
            tasks.append({
                'mode': mode,
                'month1': later_month,
                'month2': earlier_month,
                'month_expr': f'{format_months(later_month)} - {format_months(earlier_month)}',
                'order_key': tuple(
                    [month_to_season_order(m, cross_month) for m in later_month] +
                    [month_to_season_order(m, cross_month) for m in earlier_month]
                )
            })

    elif mode == 'two_month_trend':
        combos = generate_two_month_trend_combos_excluding_predict_month(
            cross_month=cross_month,
            predict_month=predict_month
        )
        for later_months, earlier_months in combos:
            tasks.append({
                'mode': mode,
                'month1': later_months,
                'month2': earlier_months,
                'month_expr': f'{format_months(later_months)} - {format_months(earlier_months)}',
                'order_key': tuple(
                    [month_to_season_order(m, cross_month) for m in later_months] +
                    [month_to_season_order(m, cross_month) for m in earlier_months]
                )
            })

    else:
        raise ValueError("mode must be one of ['mean', 'trend', 'two_month_trend']")

    tasks = sorted(tasks, key=lambda x: x['order_key'])
    return tasks


# =========================================================
# 3. 相关场计算
# =========================================================
def get_field_for_month_expr(field_ds, var_name, TR_time, month1, month2=None, cross_month=9):
    """
    获取某一 month_expr 对应的场：
    - month2 is None: month1 平均
    - month2 not None: month1 平均 - month2 平均
    """
    field_1 = get_seasonal_mean(
        field_ds,
        month1,
        TR_time[0],
        TR_time[1],
        cross_month=cross_month
    )

    if month2 is None:
        out = field_1
    else:
        field_2 = get_seasonal_mean(
            field_ds,
            month2,
            TR_time[0],
            TR_time[1],
            cross_month=cross_month
        )
        out = field_1 - field_2

    return out[var_name]


def calc_corr_map(timeSerie, field_ds, var_name, TR_time, month1, month2=None, cross_month=9):
    """计算目标指数与某个气象场 month_expr 的相关场。"""
    field = get_field_for_month_expr(
        field_ds=field_ds,
        var_name=var_name,
        TR_time=TR_time,
        month1=month1,
        month2=month2,
        cross_month=cross_month
    )

    target = timeSerie.sel(year=slice(f'{TR_time[0]}', f'{TR_time[1]}'))

    # 保证年份完全对齐。
    common_years = np.intersect1d(field['year'].values, target['year'].values)
    field = field.sel(year=common_years)
    target = target.sel(year=common_years)

    corr_arr = corr(target.data, field.data)

    corr_da = xr.DataArray(
        corr_arr,
        coords=[field['lat'], field['lon']],
        dims=['lat', 'lon'],
        name=f'{var_name}_corr'
    )

    return corr_da.sortby('lat').sortby('lon')


# =========================================================
# 4. 绘图工具
# =========================================================
MODE_LABEL = {
    'mean': 'Two-month mean',
    'trend': 'Single-month difference',
    'two_month_trend': 'Two-month difference'
}

ELEM_LABEL = {
    'sst': 'SST',
    'sic': 'SIC',
    'swvl': 'Soil moisture',
    'slp': 'SLP',
    't2m': '2-m temperature'
}


def set_sci_map_style():
    """SCI 风格绘图参数。"""
    plt.rcParams.update({
        'font.family': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 11,
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 600
    })


def make_circular_boundary(n=100):
    """给北极投影设置圆形边界。"""
    theta = np.linspace(0, 2 * np.pi, n)
    center = np.array([0.5, 0.5])
    radius = 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = center + radius * verts
    return Path(circle)


def get_projection(elem):
    """不同要素对应不同投影。"""
    if elem.lower() == 'sic':
        return ccrs.NorthPolarStereo(central_longitude=0)
    return ccrs.PlateCarree(central_longitude=180)


def setup_map_axis(ax, elem, row_id, col_id, nrows_use):
    """设置地图范围、海岸线、刻度和网格。"""
    elem = elem.lower()

    ax.coastlines(resolution='110m', linewidth=0.42, zorder=15)

    if elem == 'sic':
        # 海冰：北极立体投影，只画 60N 以北。
        ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
        ax.set_boundary(make_circular_boundary(), transform=ax.transAxes)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,
            linewidth=0.30,
            color='0.45',
            alpha=0.55,
            linestyle=':'
        )
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
        gl.ylocator = mticker.FixedLocator(np.arange(60, 91, 10))

    else:
        # 非海冰：只画 30S 以北。
        ax.set_extent([-180, 180, -30, 90], crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-30, 91, 30), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())

        if row_id != nrows_use - 1:
            ax.set_xticklabels([])
        if col_id != 0:
            ax.set_yticklabels([])

        ax.tick_params(length=1.8, width=0.45, pad=1.5)


def _sample_sig_points_by_regrid_shape(lons, lats, sig_mask, regrid_shape=30, extent=(-180, 180, -30, 90)):
    """
    对显著性 mask 先做规则块重采样，再返回每个块内最靠近块中心的一个显著格点。

    为什么不用 Cartopy quiver 自带 regrid_shape 直接重采样 masked 零矩阵：
    - 零向量经过矢量场重采样后，非显著区也可能被插值成零向量；
    - quiver 的 minlength 会把这些零向量画成点，导致全图误打点。

    这里保留 regrid_shape 的控制含义：数值越大，显著性点越密；数值越小，显著性点越疏。
    """
    lon_min, lon_max, lat_min, lat_max = extent

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    sig_mask = np.asarray(sig_mask, dtype=bool)

    lon2d, lat2d = np.meshgrid(lons, lats)

    in_extent = (
        (lon2d >= lon_min) & (lon2d <= lon_max) &
        (lat2d >= lat_min) & (lat2d <= lat_max) &
        sig_mask
    )

    if not np.any(in_extent):
        return np.array([]), np.array([])

    # 对全球经纬度图，regrid_shape 理解为经向分箱数；纬向分箱数按图幅比例设置。
    nlon_bin = max(4, int(regrid_shape))
    aspect = max(0.25, (lat_max - lat_min) / max(1e-6, (lon_max - lon_min)))
    nlat_bin = max(4, int(np.ceil(nlon_bin * aspect)))

    lon_edges = np.linspace(lon_min, lon_max, nlon_bin + 1)
    lat_edges = np.linspace(lat_min, lat_max, nlat_bin + 1)

    sample_lons = []
    sample_lats = []

    for iy in range(nlat_bin):
        y0, y1 = lat_edges[iy], lat_edges[iy + 1]
        if iy == nlat_bin - 1:
            lat_bin_mask = (lat2d >= y0) & (lat2d <= y1)
        else:
            lat_bin_mask = (lat2d >= y0) & (lat2d < y1)

        for ix in range(nlon_bin):
            x0, x1 = lon_edges[ix], lon_edges[ix + 1]
            if ix == nlon_bin - 1:
                lon_bin_mask = (lon2d >= x0) & (lon2d <= x1)
            else:
                lon_bin_mask = (lon2d >= x0) & (lon2d < x1)

            hit = in_extent & lat_bin_mask & lon_bin_mask
            if not np.any(hit):
                continue

            yy, xx = np.where(hit)
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            dist2 = (lon2d[yy, xx] - cx) ** 2 + (lat2d[yy, xx] - cy) ** 2
            k = int(np.argmin(dist2))

            sample_lons.append(lon2d[yy[k], xx[k]])
            sample_lats.append(lat2d[yy[k], xx[k]])

    return np.asarray(sample_lons), np.asarray(sample_lats)


def _resolve_regrid_shape(regrid_shape):
    """
    把 regrid_shape 转成规则采样网格大小。

    - int: 解释为纬向点数， 经向点数自动取 2 倍。
    - (nlon, nlat): 直接指定经向、纬向点数。
    """
    if isinstance(regrid_shape, (list, tuple, np.ndarray)) and len(regrid_shape) == 2:
        nlon = int(regrid_shape[0])
        nlat = int(regrid_shape[1])
    else:
        nlat = int(regrid_shape)
        nlon = int(regrid_shape * 2)

    nlon = max(nlon, 4)
    nlat = max(nlat, 4)
    return nlon, nlat


def _nearest_lat_indices(src_lats, target_lats):
    """为目标纬度找到最近的原始纬度索引。"""
    src_lats = np.asarray(src_lats, dtype=float)
    target_lats = np.asarray(target_lats, dtype=float)
    return np.abs(src_lats[None, :] - target_lats[:, None]).argmin(axis=1)


def _nearest_lon_indices(src_lons, target_lons):
    """为目标经度找到最近的原始经度索引，支持 0-360 与 -180-180，并按周期经度计算距离。"""
    src_lons = np.asarray(src_lons, dtype=float) % 360.0
    target_lons = np.asarray(target_lons, dtype=float) % 360.0
    diff = np.abs(((src_lons[None, :] - target_lons[:, None] + 180.0) % 360.0) - 180.0)
    return diff.argmin(axis=1)


def add_significance_quiver_zero(
    ax,
    corr_da,
    TR_time,
    alpha=0.1,
    color='#757575',
    regrid_shape=28,
    min_lat=None,
    max_lat=90.0,
    width=0.0022,
    alpha_quiver=0.72,
    zorder=12
):
    """
    用 quiver 零矩阵替代 scatter 打点标记显著性。

    修正版逻辑：
    1. 先在原始相关场上计算显著性布尔 mask。
    2. 再用 regrid_shape 生成规则 lon-lat 采样网格。
    3. 对每个采样点取最近的原始格点显著性。
    4. 只在采样后仍显著的格点上画 U=0, V=0 的 quiver。

    这样避免直接把 masked zero-vector 交给 Cartopy quiver 的 regrid_shape，
    否则零向量在重采样时可能被插值成全图都有点，或 mask 被处理成全图没点。
    """
    corr_da = corr_da.sortby('lat').sortby('lon')

    n_year = int(TR_time[1] - TR_time[0] + 1)
    r_crit = r_test(n_year, alpha)

    arr = corr_da.values
    lats = corr_da['lat'].values
    lons = corr_da['lon'].values

    sig_mask = np.isfinite(arr) & (np.abs(arr) > r_crit)

    if min_lat is not None:
        sig_mask = sig_mask & (lats[:, None] >= float(min_lat))

    if max_lat is not None:
        sig_mask = sig_mask & (lats[:, None] <= float(max_lat))

    if not np.any(sig_mask):
        return None

    nlon, nlat = _resolve_regrid_shape(regrid_shape)

    lat_min = float(np.nanmin(lats))
    lat_max = float(np.nanmax(lats))
    if min_lat is not None:
        lat_min = max(lat_min, float(min_lat))
    if max_lat is not None:
        lat_max = min(lat_max, float(max_lat))

    if lat_min >= lat_max:
        return None

    # 用 -180~180 作为显示采样经度，transform=PlateCarree 可自动投影到当前 ax。
    target_lons = np.linspace(-180.0, 180.0, nlon, endpoint=False)
    target_lats = np.linspace(lat_min, lat_max, nlat)

    lat_idx = _nearest_lat_indices(lats, target_lats)
    lon_idx = _nearest_lon_indices(lons, target_lons)

    sampled_mask = sig_mask[np.ix_(lat_idx, lon_idx)]

    if not np.any(sampled_mask):
        return None

    lon2d, lat2d = np.meshgrid(target_lons, target_lats)
    x = lon2d[sampled_mask]
    y = lat2d[sampled_mask]

    u0 = np.zeros_like(x, dtype=float)
    v0 = np.zeros_like(y, dtype=float)

    q = ax.quiver(
        x,
        y,
        u0,
        v0,
        transform=ccrs.PlateCarree(),
        color=color,
        alpha=alpha_quiver,
        angles='xy',
        scale_units='xy',
        scale=1,
        pivot='middle',
        width=width,
        headwidth=1,
        headlength=1,
        headaxislength=1,
        minlength=1.35,
        minshaft=1.0,
        zorder=zorder
    )

    return q


def plot_region_corr_maps(
    timeSerie,
    field_data_map,
    TR_time,
    out_dir,
    predict_month=(7, 8, 9, 10),
    elements=('sst', 'sic', 'swvl', 'slp', 't2m'),
    modes=('mean', 'trend', 'two_month_trend'),
    cross_month=9,
    ncols=6,
    figsize_per_panel=(2.35, 1.65),
    cmap='RdBu_r',
    levels=np.arange(-1.0, 1.01, 0.1),
    dpi=600,
    stipple_sig=True,
    stipple_alpha=0.1,
    stipple_color='#757575',
    stipple_regrid_shape_non_sic=30,
    stipple_regrid_shape_sic=24,
    wspace=0.025,
    hspace=0.055
):
    """
    绘制 region maps 相关图。

    field_data_map 格式：
    {
        'sst':  {'ds': sst,    'var': 'sst'},
        'sic':  {'ds': sic_ds, 'var': 'sic'},
        'swvl': {'ds': swvl,   'var': 'swvl'},
        'slp':  {'ds': slp,    'var': 'msl'},
        't2m':  {'ds': t2m,    'var': 't2m'},
    }
    """
    set_sci_map_style()
    os.makedirs(out_dir, exist_ok=True)

    saved_files = []
    corr_cache = {}

    for elem in elements:
        elem_l = elem.lower()

        if elem_l not in field_data_map:
            print(f'[Skip] {elem_l} is not in field_data_map')
            continue

        field_ds = field_data_map[elem_l]['ds']
        var_name = field_data_map[elem_l]['var']

        for mode in modes:
            tasks = build_plot_tasks(
                mode=mode,
                predict_month=predict_month,
                cross_month=cross_month
            )

            if len(tasks) == 0:
                print(f'[Skip] No tasks for {elem_l} - {mode}')
                continue

            n_panel = len(tasks)
            ncols_use = min(ncols, n_panel)
            nrows_use = int(np.ceil(n_panel / ncols_use))

            fig_w = figsize_per_panel[0] * ncols_use
            fig_h = figsize_per_panel[1] * nrows_use + 0.55

            projection = get_projection(elem_l)

            fig, axes = plt.subplots(
                nrows_use,
                ncols_use,
                figsize=(fig_w, fig_h),
                subplot_kw={'projection': projection},
                constrained_layout=False
            )
            axes = np.atleast_1d(axes).ravel()
            mappable = None

            for ipanel, task in enumerate(tasks):
                ax = axes[ipanel]
                ax.set_aspect('auto')
                row_id = ipanel // ncols_use
                col_id = ipanel % ncols_use

                month1 = task['month1']
                month2 = task['month2']
                month_expr = task['month_expr']

                cache_key = (elem_l, mode, tuple(month1), None if month2 is None else tuple(month2))
                if cache_key in corr_cache:
                    corr_da = corr_cache[cache_key]
                else:
                    corr_da = calc_corr_map(
                        timeSerie=timeSerie,
                        field_ds=field_ds,
                        var_name=var_name,
                        TR_time=TR_time,
                        month1=month1,
                        month2=month2,
                        cross_month=cross_month
                    )
                    corr_cache[cache_key] = corr_da

                corr_plot, lon_plot = add_cyclic_point(
                    corr_da.values,
                    coord=corr_da['lon'].values
                )
                lat_plot = corr_da['lat'].values

                mappable = ax.contourf(
                    lon_plot,
                    lat_plot,
                    corr_plot,
                    levels=levels,
                    cmap=cmap,
                    extend='both',
                    transform=ccrs.PlateCarree(),
                    zorder=1
                )

                if stipple_sig:
                    if elem_l == 'sic':
                        quiver_min_lat = 60.0
                        regrid_shape = stipple_regrid_shape_sic
                    else:
                        quiver_min_lat = -30.0
                        regrid_shape = stipple_regrid_shape_non_sic

                    add_significance_quiver_zero(
                        ax=ax,
                        corr_da=corr_da,
                        TR_time=TR_time,
                        alpha=stipple_alpha,
                        color=stipple_color,
                        regrid_shape=regrid_shape,
                        min_lat=quiver_min_lat,
                        width=0.0022 if elem_l != 'sic' else 0.0026,
                        alpha_quiver=0.72,
                        zorder=12
                    )

                setup_map_axis(
                    ax=ax,
                    elem=elem_l,
                    row_id=row_id,
                    col_id=col_id,
                    nrows_use=nrows_use
                )

                panel_label = chr(ord('a') + ipanel) if ipanel < 26 else f'a{ipanel - 25}'
                ax.set_title(
                    f'({panel_label}) {month_expr}',
                    loc='left',
                    fontsize=7.2,
                    pad=2
                )

            for j in range(n_panel, len(axes)):
                fig.delaxes(axes[j])

            fig.subplots_adjust(
                left=0.035,
                right=0.995,
                top=0.925,
                bottom=0.105,
                wspace=wspace,
                hspace=hspace
            )

            cbar = fig.colorbar(
                mappable,
                ax=axes[:n_panel],
                orientation='horizontal',
                fraction=0.032,
                pad=0.035,
                aspect=45
            )
            cbar.set_label('Correlation coefficient', fontsize=8)
            cbar.ax.tick_params(labelsize=7, length=2.0, width=0.45)

            title = (
                f'{ELEM_LABEL.get(elem_l, elem_l.upper())} | '
                f'{MODE_LABEL.get(mode, mode)} correlation maps'
            )
            fig.suptitle(title, y=0.985, fontsize=10.5)

            png_path = os.path.join(out_dir, f'region_corr_{elem_l}_{mode}.png')
            pdf_path = os.path.join(out_dir, f'region_corr_{elem_l}_{mode}.pdf')

            fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')
            plt.close(fig)

            saved_files.extend([png_path, pdf_path])
            print(f'[Saved] {png_path}')
            print(f'[Saved] {pdf_path}')

    return saved_files


# =========================================================
# 5. 执行：只输出 region maps 相关图
# =========================================================
PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

# ---------------------------------------------------------
# 目标指数：EHCI30
# ---------------------------------------------------------
EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
EHCI30 = EHCI.apply(lambda x: (x > 0.5).sum())
EHCI30 = (EHCI30 - EHCI30.mean()) / EHCI30.std('year')
EHCI30 = EHCI30['EHCI']

# ---------------------------------------------------------
# 气象资料
# ---------------------------------------------------------
# 2mT
t2m = era5_land(fr"{DATA}/ERA5/ERA5_land/uv_2mTTd_sfp_pre_0.nc", 1961, 2022, 't2m')

# SLP
slp = era5_s(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc", 1961, 2022, 'msl')

# SST
sst = ersst(fr"{DATA}/NOAA/ERSSTv5/sst.mnmean.nc", 1961, 2022)

# SIC
sic_ds = prepare_sic_dataset(
    read_sic(fr"{DATA}/NOAA/HadISST/HadISST_ice.nc", 1961, 2022)
)

# SWVL = swvl1 + swvl2
swvl1 = era5_land(fr"{DATA}/ERA5/ERA5_land/sm.nc", 1961, 2022, 'swvl1')
swvl2 = era5_land(fr"{DATA}/ERA5/ERA5_land/sm.nc", 1961, 2022, 'swvl2')

# snow depth
sd = era5_s(fr"{DATA}/ERA5/ERA5_singleLev/snow_depth.nc", 1961, 2022, 'sd')

# snow melt
smlt = era5_s(fr"{DATA}/ERA5/ERA5_singleLev/snow_melt.nc", 1961, 2022, 'smlt')

if isinstance(swvl1, xr.Dataset):
    swvl1_da = swvl1['swvl1']
else:
    swvl1_da = swvl1

if isinstance(swvl2, xr.Dataset):
    swvl2_da = swvl2['swvl2']
else:
    swvl2_da = swvl2

swvl = prepare_swvl_dataset((swvl1_da + swvl2_da).rename('swvl'))


# ---------------------------------------------------------
# 参数设置
# ---------------------------------------------------------
TR_time = [1962, 2006]
timeSerie = EHCI30

# 不参与预测的月份。这里保持你原脚本设置：7, 8, 9, 10。
predict_month = [7, 8, 9, 10]

# 季节年起始月份。
cross_month = 9

# 可选：输出 EHCI 线性趋势显著性，方便检查。
slope, intercept_trend, r_value, p_value, std_err = stats.linregress(
    [i for i in range(len(EHCI30))],
    EHCI30
)
print(f"EHCI30 linear trend p-value: {p_value}")


field_data_map = {
    'sst': {
        'ds': sst,
        'var': 'sst'
    },
    'sic': {
        'ds': sic_ds,
        'var': 'sic'
    },
    'swvl': {
        'ds': swvl,
        'var': 'swvl'
    },
    'slp': {
        'ds': slp,
        'var': 'msl'
    },
    't2m': {
        'ds': t2m,
        'var': 't2m'
    },
    'sd': {
        'ds': sd,
        'var': 'sd'
    },
    'smlt': {
        'ds': smlt,
        'var': 'smlt'
    }
}


# ---------------------------------------------------------
# 只绘制 region maps 相关图
# ---------------------------------------------------------
region_fig_dir = fr'{PYFILE}/p5/pic/region_corr_maps_only'

saved_region_files = plot_region_corr_maps(
    timeSerie=timeSerie,
    field_data_map=field_data_map,
    TR_time=TR_time,
    out_dir=region_fig_dir,
    predict_month=predict_month,
    elements=('sst', 'sic', 'swvl', 'slp', 't2m', 'sd', 'smlt'),
    modes=('mean', 'trend', 'two_month_trend'),
    cross_month=cross_month,
    ncols=6,
    figsize_per_panel=(2.35, 1.65),
    cmap='RdBu_r',
    levels=np.arange(-1.0, 1.01, 0.1),
    dpi=600,
    stipple_sig=True,
    stipple_alpha=0.1,              # 90% 显著性
    stipple_color='black',        # 显著性 quiver 零向量点颜色
    stipple_regrid_shape_non_sic=38, # 非海冰重采样密度；点太密可调小，太疏可调大
    stipple_regrid_shape_sic=24,      # 海冰北极图重采样密度；点太密可调小，太疏可调大
    wspace=0.025,
    hspace=0.1
)

print('Region correlation map files:')
for f in saved_region_files:
    print(f)

print('Done.')
