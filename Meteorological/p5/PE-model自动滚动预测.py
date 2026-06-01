# -*- coding: utf-8 -*-
"""
完整可替换版：自动筛选预测因子 + 逐步回归 + 按气象要素和 mode 绘制因子区域图

新增功能：
1. 在 meta_df 生成后，按照 elem + mode 分别出图：
   - mean: 两月平均一张图
   - trend: 单月相减一张图
   - two_month_trend: 双月平均相减一张图
2. 每张图中每个 month_expr 为一个子图。
3. 显著联通区域用虚线框选，并标注 X_name。
4. 出图风格按 SCI 风格设置。
5. 自动筛选因子时仅保留 30S 以北，即 lat >= -30。
6. Region maps 叠加 90% 显著性打点，颜色 #757575。
7. Region maps 默认每行 6 张图，并压缩横纵间隙。

注意：
- 该脚本依赖你原来的 climkit、cmaps、本地数据路径和字体环境。
- 如果你的 Cartopy 第一次使用 Natural Earth 数据，可能会联网下载海岸线数据。
"""

# =========================================================
# 0. imports
# =========================================================
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.lines import lineStyles
from matplotlib.pyplot import quiverkey
from matplotlib.ticker import MultipleLocator, FixedLocator
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmaps

from scipy.ndimage import filters
from scipy import stats
import xarray as xr
import numpy as np
import multiprocessing
import sys
import tqdm as tq
import time
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import os
import ast

from climkit.significance_test import corr_test, r_test
from climkit.TN_WaveActivityFlux import TN_WAF_3D, TN_WAF
from climkit.Cquiver import *
from climkit.data_read import *
from climkit.masked import masked
from climkit.corr_reg import corr, regress
from climkit.lonlat_transform import transform


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
    支持 DataArray / Dataset
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
# 2. 辅助函数
# =========================================================
def _make_slice(coord, v1, v2):
    """根据坐标升降序自动生成 slice"""
    coord_values = coord.values
    if coord_values[0] < coord_values[-1]:
        return slice(min(v1, v2), max(v1, v2))
    else:
        return slice(max(v1, v2), min(v1, v2))


def _to_lon180(da):
    """统一经度到 [-180, 180]，便于输出区域"""
    lon = da.lon.values
    if np.nanmax(lon) > 180:
        return transform(da, 'lon', '360->180')
    return da.sortby('lon')


def month_to_season_order(month, cross_month=9):
    """
    把自然月映射到以 cross_month 为起点的季节年顺序。
    例如 cross_month=9:
    9->1, 10->2, 11->3, 12->4, 1->5, ..., 8->12
    """
    return ((month - cross_month) % 12) + 1


def generate_avg_month_combos_excluding_predict_month(predict_month=[7, 8]):
    """
    连续两个月平均：
    12-1, 1-2, 2-3, ..., 11-12 都视为连续。
    但组合中不能包含 predict_month。
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


def generate_trend_month_combos_excluding_predict_month(cross_month=9, predict_month=[7, 8]):
    """
    单月趋势：
    严格按时间顺序 later - earlier。
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


def generate_two_month_trend_combos_excluding_predict_month(cross_month=9, predict_month=[7, 8]):
    """
    两月平均趋势：
    后面连续两月平均 - 前面连续两月平均。
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


def _find_with_wrap(mask, parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union_with_wrap(mask, parent, a, b):
    ra = _find_with_wrap(mask, parent, a)
    rb = _find_with_wrap(mask, parent, b)
    if ra != rb:
        parent[rb] = ra


def connected_components_periodic_lon(mask, connectivity=2, periodic_lon=True):
    """
    在二维 mask[lat, lon] 上做联通域标记。
    支持经向周期边界：lon=0 与 lon=-1 可相连。
    返回 labels, num_labels。
    """
    nlat, nlon = mask.shape
    labels = np.zeros((nlat, nlon), dtype=np.int32)

    current_label = 0
    parent = [0]

    for i in range(nlat):
        for j in range(nlon):
            if not mask[i, j]:
                continue

            cand = [
                (i - 1, j),
                (i, j - 1),
            ]
            if connectivity == 2:
                cand.extend([
                    (i - 1, j - 1),
                    (i - 1, j + 1),
                ])

            valid_neighbors = []
            for ii, jj in cand:
                if ii < 0:
                    continue
                if periodic_lon:
                    jj = jj % nlon
                else:
                    if jj < 0 or jj >= nlon:
                        continue

                if mask[ii, jj] and labels[ii, jj] > 0:
                    valid_neighbors.append(labels[ii, jj])

            if len(valid_neighbors) == 0:
                current_label += 1
                parent.append(current_label)
                labels[i, j] = current_label
            else:
                min_lab = min(valid_neighbors)
                labels[i, j] = min_lab
                for lb in valid_neighbors:
                    _union_with_wrap(mask, parent, min_lab, lb)

    label_map = {}
    new_label = 0
    for i in range(nlat):
        for j in range(nlon):
            if labels[i, j] > 0:
                root = _find_with_wrap(mask, parent, labels[i, j])
                if root not in label_map:
                    new_label += 1
                    label_map[root] = new_label
                labels[i, j] = label_map[root]

    if periodic_lon and nlon > 1:
        parent2 = list(range(new_label + 1))

        def find2(x):
            while parent2[x] != x:
                parent2[x] = parent2[parent2[x]]
                x = parent2[x]
            return x

        def union2(a, b):
            ra, rb = find2(a), find2(b)
            if ra != rb:
                parent2[rb] = ra

        for i in range(nlat):
            if mask[i, 0] and mask[i, -1] and labels[i, 0] > 0 and labels[i, -1] > 0:
                union2(labels[i, 0], labels[i, -1])

            if connectivity == 2:
                if i > 0:
                    if mask[i, 0] and mask[i - 1, -1] and labels[i, 0] > 0 and labels[i - 1, -1] > 0:
                        union2(labels[i, 0], labels[i - 1, -1])
                    if mask[i, -1] and mask[i - 1, 0] and labels[i, -1] > 0 and labels[i - 1, 0] > 0:
                        union2(labels[i, -1], labels[i - 1, 0])

        root_map = {}
        final_lab = 0
        for i in range(nlat):
            for j in range(nlon):
                if labels[i, j] > 0:
                    r = find2(labels[i, j])
                    if r not in root_map:
                        final_lab += 1
                        root_map[r] = final_lab
                    labels[i, j] = root_map[r]

        num_labels = final_lab
    else:
        num_labels = new_label

    return labels, num_labels


def get_min_lon_arc(lons, lon_mode='360'):
    """
    给定一个联通区域里的所有经度点，找到“最短覆盖弧段”。
    返回 (lon1, lon2, wraps)。
    """
    lons = np.asarray(lons).astype(float)

    if lon_mode == '180':
        lons = (lons + 180) % 360
    else:
        lons = lons % 360

    lons_sorted = np.sort(np.unique(lons))
    if len(lons_sorted) == 1:
        lon1 = float(lons_sorted[0])
        lon2 = float(lons_sorted[0])
        return lon1, lon2, False

    diffs = np.diff(np.r_[lons_sorted, lons_sorted[0] + 360])
    k = np.argmax(diffs)

    start = lons_sorted[(k + 1) % len(lons_sorted)]
    end = lons_sorted[k]

    wraps = start > end
    return float(start), float(end), wraps


def select_region_no_cut(da, lat1, lat2, lon1, lon2):
    """
    支持普通经度区间和跨经线区间的区域选取。
    - 若 lon1 <= lon2: 普通区间
    - 若 lon1 > lon2 : 认为跨越经线，需要拼接两段
    """
    lat_sl = _make_slice(da.lat, lat1, lat2)

    if lon1 <= lon2:
        lon_sl = _make_slice(da.lon, lon1, lon2)
        return da.sel(lat=lat_sl, lon=lon_sl)

    lon_vals = da.lon.values
    if np.nanmax(lon_vals) > 180:
        part1 = da.sel(lat=lat_sl, lon=_make_slice(da.lon, lon1, np.nanmax(lon_vals)))
        part2 = da.sel(lat=lat_sl, lon=_make_slice(da.lon, np.nanmin(lon_vals), lon2))
    else:
        part1 = da.sel(lat=lat_sl, lon=_make_slice(da.lon, lon1, np.nanmax(lon_vals)))
        part2 = da.sel(lat=lat_sl, lon=_make_slice(da.lon, np.nanmin(lon_vals), lon2))

    return xr.concat([part1, part2], dim='lon')


def rolling_corr_centered_index(x, y, window=11):
    """
    计算滑动相关，并把结果索引从窗口右端年改成窗口中心年。
    """
    rc = x.rolling(window=window).corr(y)

    shift_year = (window - 1) // 2
    if isinstance(rc.index, pd.DatetimeIndex):
        rc.index = rc.index - pd.DateOffset(years=shift_year)
    else:
        rc.index = rc.index - shift_year

    return rc


def find_connected_regions_from_corr(
    corr_da,
    alpha=0.1,
    min_size=40,
    split_sign=True,
    connectivity=2,
    periodic_lon=True,
    region_lat_min=-30.0
):
    """
    从 corr 场中提取通过显著性检验且格点数 > min_size 的联通区域。

    新增：
    - region_lat_min=-30.0：只在 30S 以北筛选因子，即 lat >= -30。
      若设为 None，则不限制纬度。
    """
    corr_da = corr_da.copy().sortby('lat')

    lon_vals = corr_da.lon.values
    lat_vals = corr_da.lat.values

    if np.nanmax(lon_vals) > 180:
        lon_mode = '360'
    else:
        lon_mode = '180'

    r_crit = r_test(TR_time[1] - TR_time[0] + 1, alpha)

    arr = corr_da.values
    valid = np.isfinite(arr)

    # =====================================================
    # 新增：只保留 30S 以北格点参与联通域筛选
    # =====================================================
    if region_lat_min is not None:
        lat_keep = lat_vals[:, None] >= float(region_lat_min)
        valid = valid & lat_keep

    if split_sign:
        masks = {
            'pos': valid & (arr > r_crit),
            'neg': valid & (arr < -r_crit),
        }
    else:
        masks = {
            'all': valid & (np.abs(arr) > r_crit),
        }

    regions = []

    for sign_name, mask in masks.items():
        labels, nlab = connected_components_periodic_lon(
            mask,
            connectivity=connectivity,
            periodic_lon=periodic_lon
        )

        for lab_id in range(1, nlab + 1):
            idx = np.where(labels == lab_id)
            ncell = len(idx[0])

            if ncell <= min_size:
                continue

            lats = corr_da.lat.values[idx[0]]
            lons = corr_da.lon.values[idx[1]]
            vals = arr[idx]

            lon1, lon2, wraps = get_min_lon_arc(lons, lon_mode=lon_mode)

            if lon_mode == '180':
                lon1 = ((lon1 + 180) % 360) - 180
                lon2 = ((lon2 + 180) % 360) - 180

            region_info = {
                'sign': sign_name,
                'ncell': int(ncell),
                'lat1': float(np.nanmax(lats)),
                'lat2': float(np.nanmin(lats)),
                'lon1': float(lon1),
                'lon2': float(lon2),
                'wraps': bool(wraps),
                'mean_corr': float(np.nanmean(vals)),
                'max_corr': float(np.nanmax(vals)),
                'min_corr': float(np.nanmin(vals)),
            }
            regions.append(region_info)

    return regions

def get_corr_map_from_predictor_output(pred_out):
    """predictor 返回值转成 corr 字典。"""
    (
        _, _, _, TS, TS_pre, TS_all,
        t2mReg, t2mCorr,
        slpReg, slpCorr,
        sstReg, sstCorr,
        sicReg, sicCorr,
        swvlReg, swvlCorr
    ) = pred_out

    corr_map = {
        'sst': sstCorr,
        'slp': slpCorr,
        't2m': t2mCorr,
        'sic': sicCorr,
        'swvl': swvlCorr
    }

    return corr_map, TS, TS_pre, TS_all


def calc_period_corr_stats(ts, x):
    """
    计算某一时期内 TS 与 X 的相关统计。
    返回: corr, pval, n。
    """
    sub = pd.concat(
        [ts.rename('TS'), x.rename('X')],
        axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()

    n = len(sub)
    if n < 3:
        return np.nan, np.nan, n

    try:
        model = smf.ols("TS ~ X", data=sub).fit()
        corr_val = sub['TS'].corr(sub['X'])
        pval = model.pvalues.get('X', np.nan)
    except Exception:
        corr_val, pval = np.nan, np.nan

    return corr_val, pval, n


def calc_late_rolling_stats(
    rolling_series,
    PR_time,
    r_sig,
    train_sign=None,
    min_count=3
):
    """统计预测期独立样本期上的 rolling corr 表现。"""
    s = rolling_series.copy()

    if isinstance(s.index, pd.DatetimeIndex):
        s_late = s[(s.index.year >= PR_time[0]) & (s.index.year <= PR_time[1])]
    else:
        s_late = s[(s.index >= PR_time[0]) & (s.index <= PR_time[1])]

    s_late = s_late.replace([np.inf, -np.inf], np.nan).dropna()
    late_n = len(s_late)

    if late_n < min_count:
        return {
            'late_n': late_n,
            'late_sig_ratio': np.nan,
            'late_same_sign_ratio': np.nan,
            'late_mean_corr': np.nan,
            'late_sig_mean_corr': np.nan
        }

    s_sig = s_late[np.abs(s_late) > r_sig]

    late_sig_ratio = len(s_sig) / late_n if late_n > 0 else np.nan
    late_mean_corr = s_late.mean() if late_n > 0 else np.nan
    late_sig_mean_corr = s_sig.mean() if len(s_sig) > 0 else np.nan

    if (train_sign is None) or (len(s_sig) == 0):
        late_same_sign_ratio = np.nan
    else:
        late_same_sign_ratio = (np.sign(s_sig.values) == train_sign).mean()

    return {
        'late_n': late_n,
        'late_sig_ratio': late_sig_ratio,
        'late_same_sign_ratio': late_same_sign_ratio,
        'late_mean_corr': late_mean_corr,
        'late_sig_mean_corr': late_sig_mean_corr
    }


def enrich_meta_df_with_period_stats(
    meta_df,
    X_train_all_dict,
    X_pre_all_dict,
    X_roll_all_dict,
    TS,
    TS_pre,
    PR_time,
    rolling_window=11,
    alpha=0.1,
    late_sig_ratio_thres=0.3,
    same_sign_ratio_thres=0.4
):
    """
    给 meta_df 添加：
    - train_corr, train_pval, train_n
    - pre_corr, pre_pval, pre_n
    - late_n, late_sig_ratio, late_same_sign_ratio,
      late_mean_corr, late_sig_mean_corr
    - late_sig_same_sign
    """
    meta_df = meta_df.copy()
    r_sig = r_test(rolling_window, alpha)

    train_corr_list = []
    train_pval_list = []
    train_n_list = []

    pre_corr_list = []
    pre_pval_list = []
    pre_n_list = []

    late_n_list = []
    late_sig_ratio_list = []
    late_same_sign_ratio_list = []
    late_mean_corr_list = []
    late_sig_mean_corr_list = []

    for _, row in meta_df.iterrows():
        xname = row['X_name']

        x_train = X_train_all_dict.get(xname, None)
        x_pre = X_pre_all_dict.get(xname, None)
        x_roll = X_roll_all_dict.get(xname, None)

        tr_corr, tr_pval, tr_n = calc_period_corr_stats(TS, x_train)
        train_corr_list.append(tr_corr)
        train_pval_list.append(tr_pval)
        train_n_list.append(tr_n)

        pr_corr, pr_pval, pr_n = calc_period_corr_stats(TS_pre, x_pre)
        pre_corr_list.append(pr_corr)
        pre_pval_list.append(pr_pval)
        pre_n_list.append(pr_n)

        train_sign = np.sign(tr_corr) if np.isfinite(tr_corr) and tr_corr != 0 else None
        late_stats = calc_late_rolling_stats(
            rolling_series=x_roll,
            PR_time=PR_time,
            r_sig=r_sig,
            train_sign=train_sign,
            min_count=3
        )

        late_n_list.append(late_stats['late_n'])
        late_sig_ratio_list.append(late_stats['late_sig_ratio'])
        late_same_sign_ratio_list.append(late_stats['late_same_sign_ratio'])
        late_mean_corr_list.append(late_stats['late_mean_corr'])
        late_sig_mean_corr_list.append(late_stats['late_sig_mean_corr'])

    meta_df['train_n'] = train_n_list
    meta_df['train_corr'] = train_corr_list
    meta_df['train_pval'] = train_pval_list

    meta_df['pre_n'] = pre_n_list
    meta_df['pre_corr'] = pre_corr_list
    meta_df['pre_pval'] = pre_pval_list

    meta_df['late_n'] = late_n_list
    meta_df['late_sig_ratio'] = late_sig_ratio_list
    meta_df['late_same_sign_ratio'] = late_same_sign_ratio_list
    meta_df['late_mean_corr'] = late_mean_corr_list
    meta_df['late_sig_mean_corr'] = late_sig_mean_corr_list

    meta_df['late_sig_same_sign'] = (
        (meta_df['late_sig_ratio'] >= late_sig_ratio_thres) &
        (meta_df['late_same_sign_ratio'] >= same_sign_ratio_thres)
    )

    return meta_df


def is_late_significant_same_sign(
    rolling_series,
    train_series,
    ts_train,
    PR_time,
    r_sig,
    min_ratio=0.5,
    min_count=3,
    sign_agree_ratio=0.8,
    min_train_corr=0.05
):
    sub_train = pd.concat(
        [ts_train.rename('TS'), train_series.rename('X')],
        axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(sub_train) < 8:
        return False

    train_corr = sub_train['TS'].corr(sub_train['X'])
    if (not np.isfinite(train_corr)) or (abs(train_corr) < min_train_corr):
        return False

    train_sign = np.sign(train_corr)

    s = rolling_series.copy()
    if isinstance(s.index, pd.DatetimeIndex):
        s_late = s[(s.index.year >= PR_time[0]) & (s.index.year <= PR_time[1])]
    else:
        s_late = s[(s.index >= PR_time[0]) & (s.index <= PR_time[1])]

    s_late = s_late.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s_late) < min_count:
        return False

    s_sig = s_late[np.abs(s_late) > r_sig]
    if len(s_sig) < min_count:
        return False

    sig_ratio = len(s_sig) / len(s_late)
    if sig_ratio < min_ratio:
        return False

    same_sign_ratio = (np.sign(s_sig.values) == train_sign).mean()
    if same_sign_ratio < sign_agree_ratio:
        return False

    late_mean_corr = s_sig.mean()
    if (not np.isfinite(late_mean_corr)) or (np.sign(late_mean_corr) != train_sign):
        return False

    return True


def is_late_significant(rolling_series, PR_time, r_sig, min_ratio=0.5, min_count=3):
    """判断某个因子在后期独立样本期是否比较显著。"""
    s = rolling_series.copy()

    if isinstance(s.index, pd.DatetimeIndex):
        s_late = s[(s.index.year >= PR_time[0]) & (s.index.year <= PR_time[1])]
    else:
        s_late = s[(s.index >= PR_time[0]) & (s.index <= PR_time[1])]

    s_late = s_late.replace([np.inf, -np.inf], np.nan).dropna()

    if len(s_late) < min_count:
        return False

    sig_ratio = (np.abs(s_late) > r_sig).mean()
    return sig_ratio >= min_ratio


# =========================================================
# 3. predictor 主函数
# =========================================================
def predictor(timeSerie, TR_time, PR_time, month1, month2=None, predictor_zone=None, cross_month=9):

    if predictor_zone is None:
        predictor_zone = []

    def seasonal_mean_same_year(da, months, start_year, end_year):
        """不跨年的月份平均，例如 [3,4,5]。"""
        return (
            da.sel(time=da["time.month"].isin(months))
              .sel(time=slice(f"{start_year}", f"{end_year}"))
              .groupby("time.year")
              .mean("time")
              .transpose("year", "lat", "lon")
        )

    def seasonal_mean_cross_year(da, months, start_year, end_year, cross_month=cross_month):
        """
        跨年月份平均，例如 [11,12,1,2]。
        规则：
        - month >= cross_month 的月份，season_year = 原年份 + 1
        - month < cross_month 的月份，season_year = 原年份
        """
        da_sel = da.sel(time=da["time.month"].isin(months)).sel(
            time=slice(f"{start_year-1}", f"{end_year}")
        )

        season_year = xr.where(
            da_sel["time.month"] >= cross_month,
            da_sel["time.year"] + 1,
            da_sel["time.year"]
        )

        da_sel = da_sel.assign_coords(season_year=("time", season_year.data))

        out = (
            da_sel.groupby("season_year")
                  .mean("time")
                  .rename({"season_year": "year"})
                  .transpose("year", "lat", "lon")
        )

        return out.sel(year=slice(start_year, end_year))

    def get_seasonal_mean(da, months, start_year, end_year, cross_month=cross_month):
        cross_year = any(m >= cross_month for m in months)
        if cross_year:
            return seasonal_mean_cross_year(da, months, start_year, end_year, cross_month)
        else:
            return seasonal_mean_same_year(da, months, start_year, end_year)

    if month2 is None:
        t2m_imonth = get_seasonal_mean(t2m, month1, TR_time[0], TR_time[1], cross_month)
        slp_imonth = get_seasonal_mean(slp, month1, TR_time[0], TR_time[1], cross_month)
        sst_imonth = get_seasonal_mean(sst, month1, TR_time[0], TR_time[1], cross_month)
        sic_imonth = get_seasonal_mean(sic_ds, month1, TR_time[0], TR_time[1], cross_month)
        swvl_imonth = get_seasonal_mean(swvl, month1, TR_time[0], TR_time[1], cross_month)

        t2m_imonth_pre = get_seasonal_mean(t2m, month1, PR_time[0], PR_time[1], cross_month)
        slp_imonth_pre = get_seasonal_mean(slp, month1, PR_time[0], PR_time[1], cross_month)
        sst_imonth_pre = get_seasonal_mean(sst, month1, PR_time[0], PR_time[1], cross_month)
        sic_imonth_pre = get_seasonal_mean(sic_ds, month1, PR_time[0], PR_time[1], cross_month)
        swvl_imonth_pre = get_seasonal_mean(swvl, month1, PR_time[0], PR_time[1], cross_month)

        t2m_imonth_all = get_seasonal_mean(t2m, month1, TR_time[0], PR_time[1], cross_month)
        slp_imonth_all = get_seasonal_mean(slp, month1, TR_time[0], PR_time[1], cross_month)
        sst_imonth_all = get_seasonal_mean(sst, month1, TR_time[0], PR_time[1], cross_month)
        sic_imonth_all = get_seasonal_mean(sic_ds, month1, TR_time[0], PR_time[1], cross_month)
        swvl_imonth_all = get_seasonal_mean(swvl, month1, TR_time[0], PR_time[1], cross_month)

    else:
        t2m_imonth_1 = get_seasonal_mean(t2m, month1, TR_time[0], TR_time[1], cross_month)
        slp_imonth_1 = get_seasonal_mean(slp, month1, TR_time[0], TR_time[1], cross_month)
        sst_imonth_1 = get_seasonal_mean(sst, month1, TR_time[0], TR_time[1], cross_month)
        sic_imonth_1 = get_seasonal_mean(sic_ds, month1, TR_time[0], TR_time[1], cross_month)
        swvl_imonth_1 = get_seasonal_mean(swvl, month1, TR_time[0], TR_time[1], cross_month)

        t2m_imonth_pre_1 = get_seasonal_mean(t2m, month1, PR_time[0], PR_time[1], cross_month)
        slp_imonth_pre_1 = get_seasonal_mean(slp, month1, PR_time[0], PR_time[1], cross_month)
        sst_imonth_pre_1 = get_seasonal_mean(sst, month1, PR_time[0], PR_time[1], cross_month)
        sic_imonth_pre_1 = get_seasonal_mean(sic_ds, month1, PR_time[0], PR_time[1], cross_month)
        swvl_imonth_pre_1 = get_seasonal_mean(swvl, month1, PR_time[0], PR_time[1], cross_month)

        t2m_imonth_all_1 = get_seasonal_mean(t2m, month1, TR_time[0], PR_time[1], cross_month)
        slp_imonth_all_1 = get_seasonal_mean(slp, month1, TR_time[0], PR_time[1], cross_month)
        sst_imonth_all_1 = get_seasonal_mean(sst, month1, TR_time[0], PR_time[1], cross_month)
        sic_imonth_all_1 = get_seasonal_mean(sic_ds, month1, TR_time[0], PR_time[1], cross_month)
        swvl_imonth_all_1 = get_seasonal_mean(swvl, month1, TR_time[0], PR_time[1], cross_month)

        t2m_imonth_2 = get_seasonal_mean(t2m, month2, TR_time[0], TR_time[1], cross_month)
        slp_imonth_2 = get_seasonal_mean(slp, month2, TR_time[0], TR_time[1], cross_month)
        sst_imonth_2 = get_seasonal_mean(sst, month2, TR_time[0], TR_time[1], cross_month)
        sic_imonth_2 = get_seasonal_mean(sic_ds, month2, TR_time[0], TR_time[1], cross_month)
        swvl_imonth_2 = get_seasonal_mean(swvl, month2, TR_time[0], TR_time[1], cross_month)

        t2m_imonth_pre_2 = get_seasonal_mean(t2m, month2, PR_time[0], PR_time[1], cross_month)
        slp_imonth_pre_2 = get_seasonal_mean(slp, month2, PR_time[0], PR_time[1], cross_month)
        sst_imonth_pre_2 = get_seasonal_mean(sst, month2, PR_time[0], PR_time[1], cross_month)
        sic_imonth_pre_2 = get_seasonal_mean(sic_ds, month2, PR_time[0], PR_time[1], cross_month)
        swvl_imonth_pre_2 = get_seasonal_mean(swvl, month2, PR_time[0], PR_time[1], cross_month)

        t2m_imonth_all_2 = get_seasonal_mean(t2m, month2, TR_time[0], PR_time[1], cross_month)
        slp_imonth_all_2 = get_seasonal_mean(slp, month2, TR_time[0], PR_time[1], cross_month)
        sst_imonth_all_2 = get_seasonal_mean(sst, month2, TR_time[0], PR_time[1], cross_month)
        sic_imonth_all_2 = get_seasonal_mean(sic_ds, month2, TR_time[0], PR_time[1], cross_month)
        swvl_imonth_all_2 = get_seasonal_mean(swvl, month2, TR_time[0], PR_time[1], cross_month)

        t2m_imonth = t2m_imonth_1 - t2m_imonth_2
        slp_imonth = slp_imonth_1 - slp_imonth_2
        sst_imonth = sst_imonth_1 - sst_imonth_2
        sic_imonth = sic_imonth_1 - sic_imonth_2
        swvl_imonth = swvl_imonth_1 - swvl_imonth_2

        t2m_imonth_pre = t2m_imonth_pre_1 - t2m_imonth_pre_2
        slp_imonth_pre = slp_imonth_pre_1 - slp_imonth_pre_2
        sst_imonth_pre = sst_imonth_pre_1 - sst_imonth_pre_2
        sic_imonth_pre = sic_imonth_pre_1 - sic_imonth_pre_2
        swvl_imonth_pre = swvl_imonth_pre_1 - swvl_imonth_pre_2

        t2m_imonth_all = t2m_imonth_all_1 - t2m_imonth_all_2
        slp_imonth_all = slp_imonth_all_1 - slp_imonth_all_2
        sst_imonth_all = sst_imonth_all_1 - sst_imonth_all_2
        sic_imonth_all = sic_imonth_all_1 - sic_imonth_all_2
        swvl_imonth_all = swvl_imonth_all_1 - swvl_imonth_all_2

    timeSerie_train = timeSerie.sel(year=slice(f'{TR_time[0]}', f'{TR_time[1]}')).data
    train_years = pd.to_datetime(np.arange(TR_time[0], TR_time[1] + 1), format='%Y')
    pre_years = pd.to_datetime(np.arange(PR_time[0], PR_time[1] + 1), format='%Y')

    t2mReg, t2mCorr = regress(timeSerie_train, t2m_imonth['t2m'].data), corr(timeSerie_train, t2m_imonth['t2m'].data)
    slpReg, slpCorr = regress(timeSerie_train, slp_imonth['msl'].data), corr(timeSerie_train, slp_imonth['msl'].data)
    sstReg, sstCorr = regress(timeSerie_train, sst_imonth['sst'].data), corr(timeSerie_train, sst_imonth['sst'].data)
    sicReg, sicCorr = regress(timeSerie_train, sic_imonth['sic'].data), corr(timeSerie_train, sic_imonth['sic'].data)
    swvlReg, swvlCorr = regress(timeSerie_train, swvl_imonth['swvl'].data), corr(timeSerie_train, swvl_imonth['swvl'].data)

    t2mReg = xr.DataArray(t2mReg, coords=[t2m_imonth['lat'], t2m_imonth['lon']], dims=['lat', 'lon'], name='t2m_reg')
    slpReg = xr.DataArray(slpReg, coords=[slp_imonth['lat'], slp_imonth['lon']], dims=['lat', 'lon'], name='slp_reg')
    sstReg = xr.DataArray(sstReg, coords=[sst_imonth['lat'], sst_imonth['lon']], dims=['lat', 'lon'], name='sst_reg')
    sicReg = xr.DataArray(sicReg, coords=[sic_imonth['lat'], sic_imonth['lon']], dims=['lat', 'lon'], name='sic_reg')
    swvlReg = xr.DataArray(swvlReg, coords=[swvl_imonth['lat'], swvl_imonth['lon']], dims=['lat', 'lon'], name='swvl_reg')

    t2mCorr = xr.DataArray(t2mCorr, coords=[t2m_imonth['lat'], t2m_imonth['lon']], dims=['lat', 'lon'], name='t2m_corr')
    slpCorr = xr.DataArray(slpCorr, coords=[slp_imonth['lat'], slp_imonth['lon']], dims=['lat', 'lon'], name='slp_corr')
    sstCorr = xr.DataArray(sstCorr, coords=[sst_imonth['lat'], sst_imonth['lon']], dims=['lat', 'lon'], name='sst_corr')
    sicCorr = xr.DataArray(sicCorr, coords=[sic_imonth['lat'], sic_imonth['lon']], dims=['lat', 'lon'], name='sic_corr')
    swvlCorr = xr.DataArray(swvlCorr, coords=[swvl_imonth['lat'], swvl_imonth['lon']], dims=['lat', 'lon'], name='swvl_corr')

    nor_mean = np.mean(timeSerie_train)
    nor_std = np.std(timeSerie_train)
    timeSerie_train = (timeSerie_train - nor_mean) / nor_std
    TS = pd.Series(timeSerie_train, index=train_years, name='TS')

    timeSerie_pre = timeSerie.sel(year=slice(f'{PR_time[0]}', f'{PR_time[1]}'))
    timeSerie_pre = (timeSerie_pre - nor_mean) / nor_std
    TS_pre = pd.Series(timeSerie_pre, index=pre_years, name='TS_pre')

    timeSerie_all = timeSerie.sel(year=slice(f'{TR_time[0]}', f'{PR_time[1]}'))
    timeSerie_all = (timeSerie_all - nor_mean) / nor_std
    TS_all = pd.Series(timeSerie_all, index=np.arange(TR_time[0], PR_time[1] + 1), name='TS_all')

    field_map = {
        'sst': {
            'corr': sstCorr,
            'train': sst_imonth,
            'pre': sst_imonth_pre,
            'all': sst_imonth_all,
            'var': 'sst',
        },
        'slp': {
            'corr': slpCorr,
            'train': slp_imonth,
            'pre': slp_imonth_pre,
            'all': slp_imonth_all,
            'var': 'msl',
        },
        't2m': {
            'corr': t2mCorr,
            'train': t2m_imonth,
            'pre': t2m_imonth_pre,
            'all': t2m_imonth_all,
            'var': 't2m',
        },
        'sic': {
            'corr': sicCorr,
            'train': sic_imonth,
            'pre': sic_imonth_pre,
            'all': sic_imonth_all,
            'var': 'sic',
        },
        'swvl': {
            'corr': swvlCorr,
            'train': swvl_imonth,
            'pre': swvl_imonth_pre,
            'all': swvl_imonth_all,
            'var': 'swvl',
        }
    }

    X_train_dict = {}
    X_pre_dict = {}
    X_rollingCorr_dict = {}

    index = 0
    for izone in predictor_zone:
        index += 1

        elem = izone[0].lower()
        X_zone = izone[1:]

        if elem not in field_map:
            raise ValueError(f"Unsupported predictor element: {elem}. Choose from ['sst', 'slp', 't2m', 'sic', 'swvl'].")

        corr_da = field_map[elem]['corr']
        train_da = field_map[elem]['train']
        pre_da = field_map[elem]['pre']
        all_da = field_map[elem]['all']
        var_name = field_map[elem]['var']

        corr_da_ = corr_da
        train_da_ = train_da
        pre_da_ = pre_da
        all_da_ = all_da

        lat1, lat2, lon1_raw, lon2_raw = X_zone

        if np.nanmax(corr_da_.lon.values) > 180:
            lon1 = lon1_raw % 360
            lon2 = lon2_raw % 360
        else:
            lon1 = lon1_raw
            lon2 = lon2_raw
            if lon1 > 180:
                lon1 = ((lon1 + 180) % 360) - 180
            if lon2 > 180:
                lon2 = ((lon2 + 180) % 360) - 180

        weight = select_region_no_cut(corr_da_, lat1, lat2, lon1, lon2)

        sig_mask = np.abs(weight) > r_test(TR_time[1] - TR_time[0] + 1, 0.1)

        X_train = select_region_no_cut(train_da_[var_name], lat1, lat2, lon1, lon2) * xr.where(sig_mask, weight, np.nan)
        X_train = X_train.mean(['lat', 'lon'])
        X_mean, X_std = X_train.mean(), X_train.std()
        X_train = (X_train - X_mean) / X_std
        X_train = pd.Series(X_train.data, index=train_years, name=f'X{index}_train')

        X_pre = select_region_no_cut(pre_da_[var_name], lat1, lat2, lon1, lon2) * xr.where(sig_mask, weight, np.nan)
        X_pre = X_pre.mean(['lat', 'lon'])
        X_pre = (X_pre - X_mean) / X_std
        X_pre = pd.Series(X_pre.data, index=pre_years, name=f'X{index}_pre')

        X_all = select_region_no_cut(all_da_[var_name], lat1, lat2, lon1, lon2) * xr.where(sig_mask, weight, np.nan)
        X_all = X_all.mean(['lat', 'lon'])
        X_all = (X_all - X_mean) / X_std
        X_all = pd.Series(X_all.data, index=np.arange(TR_time[0], PR_time[1] + 1), name=f'X{index}_all')

        X_rollingCorr = rolling_corr_centered_index(X_all, TS_all, window=11)
        X_rollingCorr.name = f'X{index}_rollingCorr'

        X_train_dict[f'X{index}'] = X_train
        X_pre_dict[f'X{index}'] = X_pre
        X_rollingCorr_dict[f'X{index}'] = X_rollingCorr

    return (
        X_train_dict, X_pre_dict, X_rollingCorr_dict,
        TS, TS_pre, TS_all,
        t2mReg, t2mCorr,
        slpReg, slpCorr,
        sstReg, sstCorr,
        sicReg, sicCorr,
        swvlReg, swvlCorr
    )


# =========================================================
# 4. 自动搜索 predictor
# =========================================================
def auto_build_predictors(
    timeSerie,
    TR_time,
    PR_time,
    predict_month=[7, 8],
    elements=('sst', 'slp', 't2m', 'sic', 'swvl'),
    alpha=0.1,
    min_size=40,
    cross_month=9,
    split_sign=True,
    connectivity=2,
    periodic_lon=True,
    region_lat_min=-30.0,
    verbose=True
):
    """
    自动构造三类 predictor:
    1. mean: 连续两月平均
    2. trend: 单月趋势，后月 - 前月
    3. two_month_trend: 两月平均趋势，后两月平均 - 前两月平均

    新增：
    - region_lat_min=-30.0：自动筛选因子时只保留 30S 以北区域。
    """
    X_train_all_dict = {}
    X_pre_all_dict = {}
    X_roll_all_dict = {}
    meta_records = []

    global_idx = 0
    TS_ref, TS_pre_ref, TS_all_ref = None, None, None

    def _append_predictors(
        pred_out_zone,
        local_meta,
        X_train_all_dict,
        X_pre_all_dict,
        X_roll_all_dict,
        meta_records,
        global_idx
    ):
        X_train_dict, X_pre_dict, X_roll_dict = pred_out_zone[:3]

        for i, meta in enumerate(local_meta, start=1):
            global_idx += 1
            old_key = f'X{i}'
            new_key = f'X{global_idx}'

            X_train_all_dict[new_key] = X_train_dict[old_key]
            X_pre_all_dict[new_key] = X_pre_dict[old_key]
            X_roll_all_dict[new_key] = X_roll_dict[old_key]

            meta['X_name'] = new_key
            meta_records.append(meta)

        return global_idx

    # -----------------------------------------------------
    # A. 连续两月平均因子
    # -----------------------------------------------------
    avg_combos = generate_avg_month_combos_excluding_predict_month(
        predict_month=predict_month
    )

    for months in avg_combos:
        if verbose:
            print(f'\n[平均月份] month1={months}')

        pred_out = predictor(
            timeSerie=timeSerie,
            TR_time=TR_time,
            PR_time=PR_time,
            month1=months,
            month2=None,
            predictor_zone=[],
            cross_month=cross_month
        )

        corr_map, TS_tmp, TS_pre_tmp, TS_all_tmp = get_corr_map_from_predictor_output(pred_out)

        if TS_ref is None:
            TS_ref, TS_pre_ref, TS_all_ref = TS_tmp, TS_pre_tmp, TS_all_tmp

        local_zones = []
        local_meta = []

        for elem in elements:
            regions = find_connected_regions_from_corr(
                corr_map[elem],
                alpha=alpha,
                min_size=min_size,
                split_sign=split_sign,
                connectivity=connectivity,
                periodic_lon=periodic_lon,
                region_lat_min=region_lat_min
            )

            for rg in regions:
                zone = [elem, rg['lat1'], rg['lat2'], rg['lon1'], rg['lon2']]
                local_zones.append(zone)

                local_meta.append({
                    'mode': 'mean',
                    'elem': elem,
                    'month_expr': f'{months[0]}{months[1]}',
                    'month1': tuple(months),
                    'month2': None,
                    'sign': rg['sign'],
                    'ncell': rg['ncell'],
                    'wraps': rg.get('wraps', False),
                    'lat1': rg['lat1'],
                    'lat2': rg['lat2'],
                    'lon1': rg['lon1'],
                    'lon2': rg['lon2'],
                    'mean_corr': rg['mean_corr'],
                    'max_corr': rg['max_corr'],
                    'min_corr': rg['min_corr'],
                })

        if len(local_zones) == 0:
            continue

        pred_out_zone = predictor(
            timeSerie=timeSerie,
            TR_time=TR_time,
            PR_time=PR_time,
            month1=months,
            month2=None,
            predictor_zone=local_zones,
            cross_month=cross_month
        )

        global_idx = _append_predictors(
            pred_out_zone,
            local_meta,
            X_train_all_dict,
            X_pre_all_dict,
            X_roll_all_dict,
            meta_records,
            global_idx
        )

    # -----------------------------------------------------
    # B. 单月趋势因子：后月 - 前月
    # -----------------------------------------------------
    trend_combos = generate_trend_month_combos_excluding_predict_month(
        cross_month=cross_month,
        predict_month=predict_month
    )

    for later_month, earlier_month in trend_combos:
        if verbose:
            print(f'\n[单月趋势] {later_month[0]} - {earlier_month[0]}')

        pred_out = predictor(
            timeSerie=timeSerie,
            TR_time=TR_time,
            PR_time=PR_time,
            month1=later_month,
            month2=earlier_month,
            predictor_zone=[],
            cross_month=cross_month
        )

        corr_map, TS_tmp, TS_pre_tmp, TS_all_tmp = get_corr_map_from_predictor_output(pred_out)

        if TS_ref is None:
            TS_ref, TS_pre_ref, TS_all_ref = TS_tmp, TS_pre_tmp, TS_all_tmp

        local_zones = []
        local_meta = []

        for elem in elements:
            regions = find_connected_regions_from_corr(
                corr_map[elem],
                alpha=alpha,
                min_size=min_size,
                split_sign=split_sign,
                connectivity=connectivity,
                periodic_lon=periodic_lon,
                region_lat_min=region_lat_min
            )

            for rg in regions:
                zone = [elem, rg['lat1'], rg['lat2'], rg['lon1'], rg['lon2']]
                local_zones.append(zone)

                local_meta.append({
                    'mode': 'trend',
                    'elem': elem,
                    'month_expr': f'{later_month[0]}-{earlier_month[0]}',
                    'month1': tuple(later_month),
                    'month2': tuple(earlier_month),
                    'sign': rg['sign'],
                    'ncell': rg['ncell'],
                    'wraps': rg.get('wraps', False),
                    'lat1': rg['lat1'],
                    'lat2': rg['lat2'],
                    'lon1': rg['lon1'],
                    'lon2': rg['lon2'],
                    'mean_corr': rg['mean_corr'],
                    'max_corr': rg['max_corr'],
                    'min_corr': rg['min_corr'],
                })

        if len(local_zones) == 0:
            continue

        pred_out_zone = predictor(
            timeSerie=timeSerie,
            TR_time=TR_time,
            PR_time=PR_time,
            month1=later_month,
            month2=earlier_month,
            predictor_zone=local_zones,
            cross_month=cross_month
        )

        global_idx = _append_predictors(
            pred_out_zone,
            local_meta,
            X_train_all_dict,
            X_pre_all_dict,
            X_roll_all_dict,
            meta_records,
            global_idx
        )

    # -----------------------------------------------------
    # C. 两月平均趋势因子：后两月平均 - 前两月平均
    # -----------------------------------------------------
    two_month_trend_combos = generate_two_month_trend_combos_excluding_predict_month(
        cross_month=cross_month,
        predict_month=predict_month
    )

    for later_months, earlier_months in two_month_trend_combos:
        if verbose:
            print(f'\n[两月平均趋势] {later_months} - {earlier_months}')

        pred_out = predictor(
            timeSerie=timeSerie,
            TR_time=TR_time,
            PR_time=PR_time,
            month1=later_months,
            month2=earlier_months,
            predictor_zone=[],
            cross_month=cross_month
        )

        corr_map, TS_tmp, TS_pre_tmp, TS_all_tmp = get_corr_map_from_predictor_output(pred_out)

        if TS_ref is None:
            TS_ref, TS_pre_ref, TS_all_ref = TS_tmp, TS_pre_tmp, TS_all_tmp

        local_zones = []
        local_meta = []

        for elem in elements:
            regions = find_connected_regions_from_corr(
                corr_map[elem],
                alpha=alpha,
                min_size=min_size,
                split_sign=split_sign,
                connectivity=connectivity,
                periodic_lon=periodic_lon,
                region_lat_min=region_lat_min
            )

            for rg in regions:
                zone = [elem, rg['lat1'], rg['lat2'], rg['lon1'], rg['lon2']]
                local_zones.append(zone)

                local_meta.append({
                    'mode': 'two_month_trend',
                    'elem': elem,
                    'month_expr': f'{later_months[0]}{later_months[1]}-{earlier_months[0]}{earlier_months[1]}',
                    'month1': tuple(later_months),
                    'month2': tuple(earlier_months),
                    'sign': rg['sign'],
                    'ncell': rg['ncell'],
                    'wraps': rg.get('wraps', False),
                    'lat1': rg['lat1'],
                    'lat2': rg['lat2'],
                    'lon1': rg['lon1'],
                    'lon2': rg['lon2'],
                    'mean_corr': rg['mean_corr'],
                    'max_corr': rg['max_corr'],
                    'min_corr': rg['min_corr'],
                })

        if len(local_zones) == 0:
            continue

        pred_out_zone = predictor(
            timeSerie=timeSerie,
            TR_time=TR_time,
            PR_time=PR_time,
            month1=later_months,
            month2=earlier_months,
            predictor_zone=local_zones,
            cross_month=cross_month
        )

        global_idx = _append_predictors(
            pred_out_zone,
            local_meta,
            X_train_all_dict,
            X_pre_all_dict,
            X_roll_all_dict,
            meta_records,
            global_idx
        )

    meta_df = pd.DataFrame(meta_records)

    if len(meta_df) > 0:
        keep_cols = [
            'X_name', 'mode', 'elem', 'month_expr',
            'month1', 'month2',
            'sign', 'ncell', 'wraps',
            'lat1', 'lat2', 'lon1', 'lon2',
            'mean_corr', 'max_corr', 'min_corr'
        ]
        keep_cols = [c for c in keep_cols if c in meta_df.columns]
        meta_df = meta_df[keep_cols]

    return X_train_all_dict, X_pre_all_dict, X_roll_all_dict, meta_df, TS_ref, TS_pre_ref, TS_all_ref

# ============================================================
# 5. 逐步回归前的工具函数
# ============================================================
def calc_vif(df, features):
    if len(features) <= 1:
        return pd.Series([1.0] * len(features), index=features, dtype=float)

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    if len(X) <= len(features):
        return pd.Series([1.0] * len(features), index=features, dtype=float)

    X_const = sm.add_constant(X, has_constant='add')
    vif_values = []
    for i in range(1, X_const.shape[1]):
        try:
            vif_values.append(variance_inflation_factor(X_const.values, i))
        except Exception:
            vif_values.append(np.nan)

    out = pd.Series(vif_values, index=features, dtype=float)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(999.0)
    return out


def diagnose_candidates(df, response='TS', candidates=None):
    if candidates is None:
        candidates = [c for c in df.columns if c != response]

    print("\n================ Candidate diagnostics ================")
    print("Input rows:", len(df))
    print("Input candidates:", len(candidates))

    non_na_counts = df[candidates].notna().sum().sort_values()
    print("\nSmallest non-NA counts:")
    print(non_na_counts.head(10))

    stds = df[candidates].std(numeric_only=True).sort_values()
    print("\nSmallest std:")
    print(stds.head(10))

    corrs = {}
    for x in candidates:
        sub = df[[response, x]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) >= 10:
            try:
                corrs[x] = sub[response].corr(sub[x])
            except Exception:
                pass

    corrs = pd.Series(corrs).dropna()
    if len(corrs) > 0:
        print("\nTop |corr| predictors:")
        print(corrs.abs().sort_values(ascending=False).head(15))
        print("N(|corr|>=0.20):", int((corrs.abs() >= 0.20).sum()))
        print("N(|corr|>=0.25):", int((corrs.abs() >= 0.25).sum()))
        print("N(|corr|>=0.30):", int((corrs.abs() >= 0.30).sum()))
    else:
        print("\nNo usable correlations found.")

    print("=====================================================\n")


def screen_predictors(
    df,
    response='TS',
    candidates=None,
    min_non_na=30,
    min_std=0.05,
    min_abs_corr=0.4,
    p_thres=0.10,
    top_n=40,
    verbose=True
):
    if candidates is None:
        candidates = [c for c in df.columns if c != response]

    records = []

    for x in candidates:
        sub = df[[response, x]].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(sub)

        if n < min_non_na:
            continue

        x_std = sub[x].std()
        if (not np.isfinite(x_std)) or (x_std < min_std):
            continue

        try:
            corr_val = sub[response].corr(sub[x])
        except Exception:
            continue

        if (not np.isfinite(corr_val)) or (abs(corr_val) < min_abs_corr):
            continue

        try:
            model = smf.ols(formula=f"{response} ~ {x}", data=sub).fit()
            pval = model.pvalues.get(x, np.nan)
            adj_r2 = model.rsquared_adj
        except Exception:
            continue

        if (not np.isfinite(pval)) or (pval > p_thres):
            continue

        records.append({
            'predictor': x,
            'n': n,
            'std': x_std,
            'corr': corr_val,
            'abs_corr': abs(corr_val),
            'pval': pval,
            'adj_r2': adj_r2
        })

    if len(records) == 0:
        return [], pd.DataFrame(columns=['predictor', 'n', 'std', 'corr', 'abs_corr', 'pval', 'adj_r2'])

    screen_df = pd.DataFrame(records).sort_values(
        ['abs_corr', 'adj_r2', 'pval'],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    selected = screen_df['predictor'].head(top_n).tolist()

    if verbose:
        print("\n================ Pre-screen result ================")
        print("Predictors kept after univariate screening:", len(selected))
        print(screen_df.head(20))
        print("===================================================\n")

    return selected, screen_df


def remove_high_pairwise_corr(df, features, corr_thres=0.85, response='TS'):
    if len(features) <= 1:
        return features.copy()

    score_dict = {}

    for x in features:
        sub = df[[response, x]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 10:
            continue
        try:
            score_dict[x] = abs(sub[response].corr(sub[x]))
        except Exception:
            score_dict[x] = -np.inf

    features_sorted = sorted(features, key=lambda z: score_dict.get(z, -np.inf), reverse=True)

    kept = []
    for x in features_sorted:
        drop_x = False
        for y in kept:
            sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 10:
                continue
            try:
                xy_corr = sub[x].corr(sub[y])
            except Exception:
                continue

            if np.isfinite(xy_corr) and abs(xy_corr) >= corr_thres:
                drop_x = True
                break

        if not drop_x:
            kept.append(x)

    return kept


def stepwise_selection(
    df,
    response='TS',
    candidates=None,
    p_enter=0.10,
    p_remove=0.15,
    vif_thres=5.0,
    max_steps=4000,
    max_predictors=3,
    verbose=True
):
    if candidates is None:
        candidates = [c for c in df.columns if c != response]

    selected = []
    remaining = candidates.copy()
    step = 0

    while step < max_steps:
        step += 1
        changed = False

        best_feature = None
        best_pval = None

        if len(selected) < max_predictors:
            for feature in remaining:
                trial_features = selected + [feature]
                formula = response + " ~ " + " + ".join(trial_features)
                try:
                    model = smf.ols(formula=formula, data=df).fit()
                    pval = model.pvalues.get(feature, np.nan)
                    if np.isfinite(pval):
                        if (best_pval is None) or (pval < best_pval):
                            best_pval = pval
                            best_feature = feature
                except Exception:
                    continue

        if (best_feature is not None) and (best_pval < p_enter):
            selected.append(best_feature)
            remaining.remove(best_feature)
            changed = True
            if verbose:
                print(f"[Forward] add {best_feature}, p={best_pval:.4f}")

        while len(selected) > 0:
            formula = response + " ~ " + " + ".join(selected)
            model = smf.ols(formula=formula, data=df).fit()
            pvalues = model.pvalues.drop('Intercept', errors='ignore')

            if len(pvalues) == 0:
                break

            worst_feature = pvalues.idxmax()
            worst_pval = pvalues.max()

            if worst_pval > p_remove:
                selected.remove(worst_feature)
                if worst_feature not in remaining:
                    remaining.append(worst_feature)
                changed = True
                if verbose:
                    print(f"[Backward-p] remove {worst_feature}, p={worst_pval:.4f}")
            else:
                break

        while len(selected) >= 2:
            vif = calc_vif(df, selected)
            max_vif_feature = vif.idxmax()
            max_vif_value = vif.max()

            if max_vif_value > vif_thres:
                selected.remove(max_vif_feature)
                if max_vif_feature not in remaining:
                    remaining.append(max_vif_feature)
                changed = True
                if verbose:
                    print(f"[Backward-VIF] remove {max_vif_feature}, VIF={max_vif_value:.2f}")
            else:
                break

        if not changed:
            break

    cleaned = True
    while cleaned and len(selected) > 0:
        cleaned = False

        formula = response + " ~ " + " + ".join(selected)
        model = smf.ols(formula=formula, data=df).fit()
        pvalues = model.pvalues.drop('Intercept', errors='ignore')

        if len(pvalues) > 0 and pvalues.max() > p_remove:
            worst_feature = pvalues.idxmax()
            worst_pval = pvalues.max()
            selected.remove(worst_feature)
            if verbose:
                print(f"[Final-p-clean] remove {worst_feature}, p={worst_pval:.4f}")
            cleaned = True
            continue

        if len(selected) >= 2:
            vif = calc_vif(df, selected)
            if vif.max() > vif_thres:
                worst_feature = vif.idxmax()
                worst_vif = vif.max()
                selected.remove(worst_feature)
                if verbose:
                    print(f"[Final-VIF-clean] remove {worst_feature}, VIF={worst_vif:.2f}")
                cleaned = True

    if len(selected) > max_predictors:
        formula = response + " ~ " + " + ".join(selected)
        model = smf.ols(formula=formula, data=df).fit()
        tvals = model.tvalues.drop('Intercept', errors='ignore').abs().sort_values(ascending=False)
        selected = tvals.index[:max_predictors].tolist()

        if verbose:
            print(f"[Final-limit] keep top {max_predictors} predictors by |t|: {selected}")

    return selected


def fallback_predictors_from_screen(screen_df, max_n=3):
    if len(screen_df) == 0:
        return []
    return screen_df['predictor'].head(max_n).tolist()


# ============================================================
# 6. 新增：按气象要素 + mode 绘制因子区域图
# ============================================================
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


def _safe_month_list(x):
    """
    支持 tuple/list/None/字符串形式的 month1 month2。
    如果 meta_df 从 csv 读回来，month1/month2 可能是字符串。
    """
    if x is None:
        return None

    if isinstance(x, float) and np.isnan(x):
        return None

    if isinstance(x, str):
        xs = x.strip()
        if xs.lower() in ['none', 'nan', '']:
            return None
        try:
            x = ast.literal_eval(xs)
        except Exception:
            x = xs.replace('[', '').replace(']', '')
            x = x.replace('(', '').replace(')', '')
            return [int(v.strip()) for v in x.split(',') if v.strip() != '']

    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x]

    return [int(x)]


def _month_sort_key(row, cross_month=9):
    """按气候季节顺序给 month_expr 排序。"""
    m1 = _safe_month_list(row['month1'])
    m2 = _safe_month_list(row['month2'])

    key = []
    if m1 is not None:
        key.extend([month_to_season_order(m, cross_month) for m in m1])
    if m2 is not None:
        key.extend([month_to_season_order(m, cross_month) for m in m2])

    return tuple(key)


def _lon_to_data(lon, lon_values):
    """把 lon 转换到当前数据经度体系。"""
    lon = float(lon)

    if np.nanmax(lon_values) > 180:
        return lon % 360
    else:
        return ((lon + 180) % 360) - 180


def draw_factor_box(
    ax,
    row,
    lon_values,
    label=True,
    linewidth=1.1,
    zorder=20
):
    """
    在地图上画因子区域虚线框。
    支持跨 0/360 或 -180/180 经线的区域。
    """
    lat_low = min(float(row['lat1']), float(row['lat2']))
    lat_high = max(float(row['lat1']), float(row['lat2']))

    lon1 = _lon_to_data(row['lon1'], lon_values)
    lon2 = _lon_to_data(row['lon2'], lon_values)

    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))

    wraps = bool(row.get('wraps', False)) or (lon1 > lon2)

    if row.get('sign', 'pos') == 'pos':
        edgecolor = '#B22222'
    else:
        edgecolor = '#1F4E79'

    if not wraps:
        segments = [(lon1, lon2)]
    else:
        segments = [(lon1, lon_max), (lon_min, lon2)]

    for a, b in segments:
        if b < a:
            continue

        rect = Rectangle(
            xy=(a, lat_low),
            width=b - a,
            height=lat_high - lat_low,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=(0, (4, 2)),
            transform=ccrs.PlateCarree(),
            zorder=zorder
        )
        ax.add_patch(rect)

    if label:
        x_text = lon1
        y_text = lat_high

        ax.text(
            x_text,
            y_text,
            str(row['X_name']),
            transform=ccrs.PlateCarree(),
            ha='left',
            va='bottom',
            fontsize=6,
            color=edgecolor,
            bbox=dict(
                boxstyle='round,pad=0.12',
                fc='white',
                ec='none',
                alpha=0.65
            ),
            zorder=zorder + 1
        )


def get_corr_da_for_month_expr(
    timeSerie,
    TR_time,
    PR_time,
    elem,
    month1,
    month2,
    cross_month=9
):
    """针对某个 elem + month_expr 重新计算相关场。"""
    pred_out = predictor(
        timeSerie=timeSerie,
        TR_time=TR_time,
        PR_time=PR_time,
        month1=month1,
        month2=month2,
        predictor_zone=[],
        cross_month=cross_month
    )

    corr_map, _, _, _ = get_corr_map_from_predictor_output(pred_out)
    return corr_map[elem].sortby('lat').sortby('lon')


def add_significance_stippling(
    ax,
    corr_da,
    alpha=0.1,
    color='#757575',
    size=1.0,
    stride=1,
    min_lat=None,
    alpha_scatter=0.65,
    zorder=12
):
    """
    在相关场上添加显著性打点。

    参数：
    - alpha=0.1：90% 显著性。
    - color='#757575'：打点颜色。
    - stride：抽稀步长。1 表示不抽稀；2 表示隔一个点画一个。
    - min_lat：若设为 -30，则只给 30S 以北显著区域打点；None 表示全图打点。
    """
    corr_da = corr_da.sortby('lat').sortby('lon')

    r_crit = r_test(TR_time[1] - TR_time[0] + 1, alpha)

    arr = corr_da.values
    lats = corr_da['lat'].values
    lons = corr_da['lon'].values

    sig_mask = np.isfinite(arr) & (np.abs(arr) > r_crit)

    if min_lat is not None:
        sig_mask = sig_mask & (lats[:, None] >= float(min_lat))

    if stride is not None and stride > 1:
        ii, jj = np.indices(sig_mask.shape)
        sig_mask = sig_mask & (ii % stride == 0) & (jj % stride == 0)

    if not np.any(sig_mask):
        return

    lon2d, lat2d = np.meshgrid(lons, lats)

    ax.scatter(
        lon2d[sig_mask],
        lat2d[sig_mask],
        s=size,
        c=color,
        marker='.',
        linewidths=0,
        alpha=alpha_scatter,
        transform=ccrs.PlateCarree(),
        zorder=zorder,
        rasterized=True
    )


def plot_auto_predictor_region_maps(
    meta_df,
    timeSerie,
    TR_time,
    PR_time,
    out_dir,
    elements=('sst', 'sic', 'swvl', 'slp', 't2m'),
    modes=('mean', 'trend', 'two_month_trend'),
    cross_month=9,
    ncols=6,
    figsize_per_panel=(2.35, 1.65),
    cmap='RdBu_r',
    levels=np.arange(-1.0, 1.01, 0.1),
    label_xname=True,
    dpi=600,
    stipple_sig=True,
    stipple_alpha=0.1,
    stipple_color='#757575',
    stipple_size=1.0,
    stipple_stride=1,
    stipple_min_lat=None,
    wspace=0.025,
    hspace=0.055
):
    """
    按 elem + mode 分别绘图。

    新增：
    - 每行默认 6 张图。
    - 子图横纵间隙更小。
    - 显著性打点，默认 90% 显著性，颜色 #757575。
    """
    set_sci_map_style()
    os.makedirs(out_dir, exist_ok=True)

    meta_plot = meta_df.copy()
    saved_files = []

    if len(meta_plot) == 0:
        print('[Region maps] meta_df is empty. Skip region map plotting.')
        return saved_files

    for elem in elements:
        for mode in modes:

            sub_all = meta_plot[
                (meta_plot['elem'].astype(str).str.lower() == elem.lower()) &
                (meta_plot['mode'].astype(str) == mode)
            ].copy()

            if len(sub_all) == 0:
                print(f'[Skip] No predictors for {elem} - {mode}')
                continue

            sub_all['_order_key'] = sub_all.apply(
                lambda r: _month_sort_key(r, cross_month=cross_month),
                axis=1
            )

            month_expr_order = (
                sub_all
                .sort_values('_order_key')
                .drop_duplicates('month_expr')['month_expr']
                .tolist()
            )

            n_panel = len(month_expr_order)
            ncols_use = min(ncols, n_panel)
            nrows_use = int(np.ceil(n_panel / ncols_use))

            fig_w = figsize_per_panel[0] * ncols_use
            fig_h = figsize_per_panel[1] * nrows_use + 0.55

            fig, axes = plt.subplots(
                nrows_use,
                ncols_use,
                figsize=(fig_w, fig_h),
                subplot_kw={
                    'projection': ccrs.PlateCarree(central_longitude=180)
                },
                constrained_layout=False
            )

            axes = np.atleast_1d(axes).ravel()
            mappable = None

            for ipanel, month_expr in enumerate(month_expr_order):
                ax = axes[ipanel]

                sub = sub_all[sub_all['month_expr'] == month_expr].copy()
                row0 = sub.iloc[0]

                month1 = _safe_month_list(row0['month1'])
                month2 = _safe_month_list(row0['month2'])

                corr_da = get_corr_da_for_month_expr(
                    timeSerie=timeSerie,
                    TR_time=TR_time,
                    PR_time=PR_time,
                    elem=elem,
                    month1=month1,
                    month2=month2,
                    cross_month=cross_month
                )

                corr_da = corr_da.sortby('lat').sortby('lon')

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

                # =================================================
                # 新增：显著性打点
                # =================================================
                if stipple_sig:
                    add_significance_stippling(
                        ax=ax,
                        corr_da=corr_da,
                        alpha=stipple_alpha,
                        color=stipple_color,
                        size=stipple_size,
                        stride=stipple_stride,
                        min_lat=stipple_min_lat,
                        alpha_scatter=0.65,
                        zorder=12
                    )

                ax.coastlines(resolution='110m', linewidth=0.42, zorder=15)
                ax.set_global()

                ax.set_xticks(
                    np.arange(-180, 181, 60),
                    crs=ccrs.PlateCarree()
                )
                ax.set_yticks(
                    np.arange(-60, 91, 30),
                    crs=ccrs.PlateCarree()
                )

                ax.xaxis.set_major_formatter(LongitudeFormatter())
                ax.yaxis.set_major_formatter(LatitudeFormatter())

                row_id = ipanel // ncols_use
                col_id = ipanel % ncols_use

                if row_id != nrows_use - 1:
                    ax.set_xticklabels([])
                if col_id != 0:
                    ax.set_yticklabels([])

                ax.tick_params(length=1.8, width=0.45, pad=1.5)

                for _, rg in sub.iterrows():
                    draw_factor_box(
                        ax=ax,
                        row=rg,
                        lon_values=corr_da['lon'].values,
                        label=label_xname,
                        linewidth=1.0,
                        zorder=20
                    )

                panel_label = chr(ord('a') + ipanel)

                ax.set_title(
                    f'({panel_label}) {month_expr}  n={len(sub)}',
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
                f'{ELEM_LABEL.get(elem, elem.upper())} | '
                f'{MODE_LABEL.get(mode, mode)} predictor regions'
            )

            fig.suptitle(title, y=0.985, fontsize=10.5)

            png_path = os.path.join(out_dir, f'auto_regions_{elem}_{mode}.png')
            pdf_path = os.path.join(out_dir, f'auto_regions_{elem}_{mode}.pdf')

            fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')
            plt.close(fig)

            saved_files.extend([png_path, pdf_path])
            print(f'[Saved] {png_path}')
            print(f'[Saved] {pdf_path}')

    return saved_files

# ============================================================
# 7. 执行自动搜索
# ============================================================
PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
EHCI30 = EHCI.apply(lambda x: (x > 0.5).sum())
EHCI30 = (EHCI30 - EHCI30.mean()) / EHCI30.std('year')
EHCI30 = EHCI30['EHCI']

# 2mT
t2m = era5_land(fr"{DATA}/ERA5/ERA5_land/uv_2mTTd_sfp_pre_0.nc", 1961, 2022, 't2m')

# SLP
slp = era5_s(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc", 1961, 2022, 'msl')

# SST
sst = ersst(fr"{DATA}/NOAA/ERSSTv5/sst.mnmean.nc", 1961, 2022)

# SIC
sic_ds = prepare_sic_dataset(read_sic(fr"{DATA}/NOAA/HadISST/HadISST_ice.nc", 1961, 2022))

# SWVL = swvl1 + swvl2
swvl1 = era5_land(fr"{DATA}/ERA5/ERA5_land/sm.nc", 1961, 2022, 'swvl1')
swvl2 = era5_land(fr"{DATA}/ERA5/ERA5_land/sm.nc", 1961, 2022, 'swvl2')

if isinstance(swvl1, xr.Dataset):
    swvl1_da = swvl1['swvl1']
else:
    swvl1_da = swvl1

if isinstance(swvl2, xr.Dataset):
    swvl2_da = swvl2['swvl2']
else:
    swvl2_da = swvl2

swvl = prepare_swvl_dataset((swvl1_da + swvl2_da).rename('swvl'))


def detrend(data):
    return data - np.polyval(np.polyfit(range(len(data)), data, 1), range(len(data)))


TR_time = [1962, 2004]
PR_time = [2005, 2022]
timeSerie = EHCI30

slope, intercept_trend, r_value, p_value, std_err = stats.linregress(
    [i for i in range(len(EHCI30))], EHCI30
)
print(f"##################################{p_value}#########################################")

predict_month = [7, 8, 9, 10]
cross_month = 9

X_train_all_dict, X_pre_all_dict, X_roll_all_dict, meta_df, TS, TS_pre, TS_all = auto_build_predictors(
    timeSerie=timeSerie,
    TR_time=TR_time,
    PR_time=PR_time,
    predict_month=predict_month,
    elements=('sst', 'sic', 'swvl', 'slp', 't2m'),
    alpha=0.1,
    min_size=40,
    cross_month=cross_month,
    split_sign=True,
    connectivity=2,
    periodic_lon=True,
    region_lat_min=-30.0,   # 只筛选 30S 以北因子
    verbose=True
)

meta_df = enrich_meta_df_with_period_stats(
    meta_df=meta_df,
    X_train_all_dict=X_train_all_dict,
    X_pre_all_dict=X_pre_all_dict,
    X_roll_all_dict=X_roll_all_dict,
    TS=TS,
    TS_pre=TS_pre,
    PR_time=PR_time,
    rolling_window=11,
    alpha=0.1,
    late_sig_ratio_thres=0.4,
    same_sign_ratio_thres=0.8
)

meta_df = meta_df.sort_values(
    by=['late_sig_same_sign', 'pre_corr', 'train_corr', 'ncell'],
    ascending=[False, False, False, False]
).reset_index(drop=True)

print(f'共筛得 {len(X_train_all_dict)} 个候选预测因子')
print(meta_df.head(30))

meta_df.to_csv(
    fr"{PYFILE}/p5/data/auto_predictor_catalog.csv",
    index=False,
    encoding='utf-8-sig'
)


# ============================================================
# 8. 新增调用：输出每个气象要素和每个 mode 的因子空间分布图
# ============================================================
region_fig_dir = fr'{PYFILE}/p5/pic/auto_predictor_region_maps'

saved_region_files = plot_auto_predictor_region_maps(
    meta_df=meta_df,
    timeSerie=timeSerie,
    TR_time=TR_time,
    PR_time=PR_time,
    out_dir=region_fig_dir,
    elements=('sst', 'sic', 'swvl', 'slp', 't2m'),
    modes=('mean', 'trend', 'two_month_trend'),
    cross_month=cross_month,
    ncols=6,                         # 每行 6 张图
    figsize_per_panel=(2.35, 1.65),   # 更紧凑的 SCI 排版
    cmap='RdBu_r',
    levels=np.arange(-1.0, 1.01, 0.1),
    label_xname=True,
    dpi=600,
    stipple_sig=True,                 # 显著性打点
    stipple_alpha=0.1,                # 90% 显著性
    stipple_color='#757575',          # 打点颜色
    stipple_size=1.0,
    stipple_stride=1,                 # 若点太密可改成 2
    stipple_min_lat=None,             # 全图显著性打点；若只想 30S 以北打点，改成 -30.0
    wspace=0.025,
    hspace=0.055
)

print('Region map files:')
for f in saved_region_files:
    print(f)


# ============================================================
# 9. 逐步回归与最终预测图
# ============================================================
plt.rcParams['font.family'] = ['AVHershey Simplex', 'AVHershey Duplex', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(5, 4))
fig.subplots_adjust(hspace=0.35)
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

index_N = 2   # 最终逐步回归最多保留的因子个数

predictor_results = [(X_train_all_dict, X_pre_all_dict, X_roll_all_dict)]

all_X_train = {}
all_X_pre = {}
all_X_rollingCorr = {}

factor_count = 1
for train_dict, pre_dict, rolling_dict in predictor_results:
    for old_key in train_dict.keys():
        new_key = f'X{factor_count}'
        all_X_train[new_key] = train_dict[old_key].rename(new_key)
        all_X_pre[new_key] = pre_dict[old_key].rename(new_key)
        all_X_rollingCorr[new_key] = rolling_dict[old_key].rename(new_key)
        factor_count += 1

candidate_predictors = list(all_X_train.keys())
print("Available predictors:", candidate_predictors[:20], '...', candidate_predictors[-20:])
print("Total candidate predictors:", len(candidate_predictors))

df_train_full = pd.concat(
    [TS.rename('TS')] + [all_X_train[x].rename(x) for x in candidate_predictors],
    axis=1
)

df_pre_full = pd.concat(
    [TS_pre.rename('TS')] + [all_X_pre[x].rename(x) for x in candidate_predictors],
    axis=1
)

df_train_step = df_train_full.copy()

for c in df_train_step.columns:
    df_train_step[c] = pd.to_numeric(df_train_step[c], errors='coerce')
for c in df_pre_full.columns:
    df_pre_full[c] = pd.to_numeric(df_pre_full[c], errors='coerce')

diagnose_candidates(
    df=df_train_step,
    response='TS',
    candidates=candidate_predictors
)

screened_predictors, screen_df = screen_predictors(
    df=df_train_step,
    response='TS',
    candidates=candidate_predictors,
    min_non_na=30,
    min_std=0.05,
    min_abs_corr=0.4,
    p_thres=0.10,
    top_n=40,
    verbose=True
)

if len(screened_predictors) == 0:
    raise ValueError("Pre-screen found no usable predictors. Check X construction, NaN ratio, and variance.")

screened_predictors = remove_high_pairwise_corr(
    df=df_train_step,
    features=screened_predictors,
    corr_thres=0.6,
    response='TS'
)

print("Predictors after pairwise-corr filtering:", screened_predictors)

df_step = df_train_step[['TS'] + screened_predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()

print("Training rows used for stepwise regression:", len(df_step))
print("Candidate predictors used in stepwise:", screened_predictors)

model_predictors = stepwise_selection(
    df=df_step,
    response='TS',
    candidates=screened_predictors,
    p_enter=0.10,
    p_remove=0.15,
    vif_thres=5.0,
    max_steps=4000,
    max_predictors=index_N,
    verbose=True
)

if len(model_predictors) == 0:
    model_predictors = fallback_predictors_from_screen(screen_df, max_n=index_N)
    print("Stepwise returned empty. Fallback predictors:", model_predictors)

if len(model_predictors) == 0:
    raise ValueError("No usable predictors after fallback.")

model_predictors = model_predictors[:index_N]
print("Selected predictors:", model_predictors)

plot_predictors = screened_predictors.copy()
selected_predictors = model_predictors.copy()

df_train = pd.concat(
    [TS.rename('TS')] + [all_X_train[x].rename(x) for x in selected_predictors],
    axis=1
).replace([np.inf, -np.inf], np.nan)

df_pre = pd.concat(
    [TS_pre.rename('TS')] + [all_X_pre[x].rename(x) for x in selected_predictors],
    axis=1
).replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# rollingCorr 绘图
# ------------------------------------------------------------
ax_rollingCorr = fig.add_subplot(gs[0])
ax_rollingCorr.set_ylim(-0.5, 1)

r_sig_11 = r_test(11, 0.1)

late_sig_predictors = meta_df.loc[
    meta_df['late_sig_same_sign'] == True, 'X_name'
].tolist()

late_sig_predictors = [x for x in late_sig_predictors if x in plot_predictors]
print("Late-significant predictors:", late_sig_predictors)

line_handles = []
line_labels = []

for i, xname in enumerate(plot_predictors):
    is_selected = xname in model_predictors
    is_late_sig = xname in late_sig_predictors

    if is_selected:
        color = 'red'
        lw = 1.8
        alpha = 1.0
        zorder = 3
    elif is_late_sig:
        color = 'orange'
        lw = 1.4
        alpha = 0.95
        zorder = 2
    else:
        color = '#959595'
        lw = 0.9
        alpha = 0.75
        zorder = 1

    line, = ax_rollingCorr.plot(
        all_X_rollingCorr[xname].index,
        all_X_rollingCorr[xname].values,
        color=color,
        linewidth=lw,
        linestyle='-',
        alpha=alpha,
        zorder=zorder,
        label=xname
    )

    line_handles.append(line)
    line_labels.append(xname)

h1 = ax_rollingCorr.axhline(
    y=r_sig_11,
    color='black',
    linestyle='--',
    linewidth=1,
    label='90%',
    alpha=0.7
)
h2 = ax_rollingCorr.axhline(
    y=0,
    color='#999999',
    linestyle='-',
    linewidth=0.5,
    alpha=0.7
)

legend_ncol = min(4, max(1, int(np.ceil(len(plot_predictors) / 8))))
leg = ax_rollingCorr.legend(
    handles=line_handles + [h1],
    labels=line_labels + ['90%'],
    loc='lower right',
    fontsize=6,
    ncol=legend_ncol,
    frameon=False
)

for txt in leg.get_texts()[:-1]:
    label = txt.get_text()

    if label in model_predictors:
        txt.set_fontweight('bold')
        txt.set_color('red')
        txt.set_alpha(1.0)
    elif label in late_sig_predictors:
        txt.set_fontweight('bold')
        txt.set_color('orange')
        txt.set_alpha(0.9)
    else:
        txt.set_fontweight('normal')
        txt.set_color('#666666')
        txt.set_alpha(0.75)

leg.get_texts()[-1].set_fontweight('normal')
leg.get_texts()[-1].set_color('black')
leg.get_texts()[-1].set_alpha(0.8)

# ------------------------------------------------------------
# 最终回归建模
# ------------------------------------------------------------
df_model_train = df_train[['TS'] + selected_predictors].dropna().copy()

formula = "TS ~ " + " + ".join(model_predictors)
print("Final formula:", formula)

model = smf.ols(formula=formula, data=df_model_train).fit()
print(model.summary())

intercept = model.params['Intercept']
coef_dict = {x: model.params[x] for x in model_predictors}

if len(model_predictors) >= 2:
    final_vif = calc_vif(df_model_train[['TS'] + model_predictors], model_predictors)
    print("Final VIF:")
    print(final_vif)

# ------------------------------------------------------------
# 预测
# ------------------------------------------------------------
df_train['predicted_TS'] = model.predict(df_train)
df_pre['inDependent_pre'] = model.predict(df_pre)
df_train['residuals'] = df_train['TS'] - df_train['predicted_TS']

TS_all_plot = pd.concat([TS.rename('TS'), TS_pre.rename('TS')])

# ------------------------------------------------------------
# 预测图
# ------------------------------------------------------------
ax_predict = fig.add_subplot(gs[1])
ax_predict.set_ylim(-3, 3)

ax_predict.plot(TS_all_plot.index, TS_all_plot.values, color='black', linestyle='-', linewidth=1.5, label='Obs')
ax_predict.plot(df_train.index, df_train['predicted_TS'], color='blue', linestyle='--', linewidth=1.5, label='Reforecast')
ax_predict.plot(df_pre.index, df_pre['inDependent_pre'], color='red', linestyle=(0, (1, 1)), linewidth=1.5, label='Independent forecast')
ax_predict.axhline(y=0, color='#999999', linestyle='-', linewidth=0.5, alpha=0.5)
ax_predict.legend(loc='lower right', fontsize=6, ncol=3, frameon=False)
ax_predict.axvline(x=pd.to_datetime(f'{TR_time[1]}-6-30'), color='orange', linestyle='-', linewidth=1)

train_eval = df_train[['TS', 'predicted_TS']].dropna()
pre_eval = df_pre[['TS', 'inDependent_pre']].dropna()

tcc_text_train = f"TCC={train_eval['TS'].corr(train_eval['predicted_TS']):.2f}" if len(train_eval) > 1 else "TCC=nan"
rmse_text_train = f"RMSE={np.sqrt(np.mean((train_eval['TS'] - train_eval['predicted_TS'])**2)):.2f}" if len(train_eval) > 0 else "RMSE=nan"

ax_predict.text(
    0.08, 0.80, f'{tcc_text_train}\n{rmse_text_train}',
    transform=ax_predict.transAxes,
    ha='center', va='bottom', fontsize=6,
    bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='blue', alpha=0.6),
    zorder=10
)

tcc_text_pre = f"TCC={pre_eval['TS'].corr(pre_eval['inDependent_pre']):.2f}" if len(pre_eval) > 1 else "TCC=nan"
rmse_text_pre = f"RMSE={np.sqrt(np.mean((pre_eval['TS'] - pre_eval['inDependent_pre'])**2)):.2f}" if len(pre_eval) > 0 else "RMSE=nan"

ax_predict.text(
    0.92, 0.80, f'{tcc_text_pre}\n{rmse_text_pre}',
    transform=ax_predict.transAxes,
    ha='center', va='bottom', fontsize=6,
    bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='red', alpha=0.6),
    zorder=10
)

# ------------------------------------------------------------
# 回归方程文本
# ------------------------------------------------------------
equation_terms = [f'{intercept:.2f}']
for x in model_predictors:
    coef = coef_dict[x]
    sign = '+' if coef >= 0 else '-'
    equation_terms.append(f' {sign} {abs(coef):.2f}*{x}')
func = 'TS = ' + ''.join(equation_terms)

ax_predict.text(
    0.5, 0.88, func,
    transform=ax_predict.transAxes,
    ha='center', va='bottom', fontsize=8,
    bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='none', alpha=0.6),
    zorder=10
)

print("Final selected predictors:", model_predictors)
print("Final equation:", func)

plt.savefig(fr'{PYFILE}/p5/pic/pemodle_auto.pdf', bbox_inches='tight')
plt.savefig(fr'{PYFILE}/p5/pic/pemodle_auto.png', dpi=600, bbox_inches='tight')
plt.close(fig)

print('Done.')