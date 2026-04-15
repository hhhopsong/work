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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmaps

from scipy.ndimage import filters
import xarray as xr
import numpy as np
import multiprocessing
import sys
import tqdm as tq
import time
import pandas as pd

from climkit.significance_test import corr_test, r_test
from climkit.TN_WaveActivityFlux import TN_WAF_3D, TN_WAF
from climkit.Cquiver import *
from climkit.data_read import *
from climkit.masked import masked
from climkit.corr_reg import corr, regress
from climkit.lonlat_transform import transform


read_sic = sic


def prepare_sic_dataset(ds):
    """Normalize HadISST sea-ice dataset to a stable {'sic': [time, lat, lon]} schema."""
    # Normalize coordinate names first.
    rename_map = {}
    if 'latitude' in ds.coords and 'lat' not in ds.coords:
        rename_map['latitude'] = 'lat'
    if 'longitude' in ds.coords and 'lon' not in ds.coords:
        rename_map['longitude'] = 'lon'
    if len(rename_map) > 0:
        ds = ds.rename(rename_map)

    # Normalize variable name to "sic".
    if 'sic' not in ds.data_vars:
        for cand in ('ice', 'siconc', 'ci'):
            if cand in ds.data_vars:
                ds = ds.rename({cand: 'sic'})
                break

    if 'sic' not in ds.data_vars:
        raise ValueError(f"SIC variable not found in dataset. Available vars: {list(ds.data_vars)}")

    # Keep valid physical range only. (0~1 or 0~100 are both supported)
    sic_da = ds['sic']
    vmax = float(np.nanmax(sic_da.values))
    if vmax <= 1.5:
        sic_da = sic_da.where((sic_da >= 0) & (sic_da <= 1.0))
    else:
        sic_da = sic_da.where((sic_da >= 0) & (sic_da <= 100.0))

    out = xr.Dataset({'sic': sic_da})
    out = out.sortby('lat').sortby('lon')
    return out


def predictor(timeSerie, TR_time, PR_time, month1, month2=None, predictor_zone=None, cross_month=9):

    if predictor_zone is None:
        predictor_zone = []

    def seasonal_mean_same_year(da, months, start_year, end_year):
        """不跨年的月份平均，例如 [3,4,5]"""
        return (
            da.sel(time=da["time.month"].isin(months))
              .sel(time=slice(f"{start_year}", f"{end_year}"))
              .groupby("time.year")
              .mean("time")
              .transpose("year", "lat", "lon")
        )

    def seasonal_mean_cross_year(da, months, start_year, end_year, cross_month=cross_month):
        """
        跨年月份平均，例如 [11,12,1,2]
        规则：
        - month >= cross_month 的月份，season_year = 原年份 + 1
        - month < cross_month 的月份，season_year = 原年份
        最终 season_year 作为输出 year
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

        t2m_imonth_pre = get_seasonal_mean(t2m, month1, PR_time[0], PR_time[1], cross_month)
        slp_imonth_pre = get_seasonal_mean(slp, month1, PR_time[0], PR_time[1], cross_month)
        sst_imonth_pre = get_seasonal_mean(sst, month1, PR_time[0], PR_time[1], cross_month)
        sic_imonth_pre = get_seasonal_mean(sic_ds, month1, PR_time[0], PR_time[1], cross_month)

        t2m_imonth_all = get_seasonal_mean(t2m, month1, TR_time[0], PR_time[1], cross_month)
        slp_imonth_all = get_seasonal_mean(slp, month1, TR_time[0], PR_time[1], cross_month)
        sst_imonth_all = get_seasonal_mean(sst, month1, TR_time[0], PR_time[1], cross_month)
        sic_imonth_all = get_seasonal_mean(sic_ds, month1, TR_time[0], PR_time[1], cross_month)
    else:
        t2m_imonth_1 = get_seasonal_mean(t2m, month1, TR_time[0], TR_time[1], cross_month)
        slp_imonth_1 = get_seasonal_mean(slp, month1, TR_time[0], TR_time[1], cross_month)
        sst_imonth_1 = get_seasonal_mean(sst, month1, TR_time[0], TR_time[1], cross_month)
        sic_imonth_1 = get_seasonal_mean(sic_ds, month1, TR_time[0], TR_time[1], cross_month)

        t2m_imonth_pre_1 = get_seasonal_mean(t2m, month1, PR_time[0], PR_time[1], cross_month)
        slp_imonth_pre_1 = get_seasonal_mean(slp, month1, PR_time[0], PR_time[1], cross_month)
        sst_imonth_pre_1 = get_seasonal_mean(sst, month1, PR_time[0], PR_time[1], cross_month)
        sic_imonth_pre_1 = get_seasonal_mean(sic_ds, month1, PR_time[0], PR_time[1], cross_month)

        t2m_imonth_all_1 = get_seasonal_mean(t2m, month1, TR_time[0], PR_time[1], cross_month)
        slp_imonth_all_1 = get_seasonal_mean(slp, month1, TR_time[0], PR_time[1], cross_month)
        sst_imonth_all_1 = get_seasonal_mean(sst, month1, TR_time[0], PR_time[1], cross_month)
        sic_imonth_all_1 = get_seasonal_mean(sic_ds, month1, TR_time[0], PR_time[1], cross_month)

        t2m_imonth_2 = get_seasonal_mean(t2m, month2, TR_time[0], TR_time[1], cross_month)
        slp_imonth_2 = get_seasonal_mean(slp, month2, TR_time[0], TR_time[1], cross_month)
        sst_imonth_2 = get_seasonal_mean(sst, month2, TR_time[0], TR_time[1], cross_month)
        sic_imonth_2 = get_seasonal_mean(sic_ds, month2, TR_time[0], TR_time[1], cross_month)

        t2m_imonth_pre_2 = get_seasonal_mean(t2m, month2, PR_time[0], PR_time[1], cross_month)
        slp_imonth_pre_2 = get_seasonal_mean(slp, month2, PR_time[0], PR_time[1], cross_month)
        sst_imonth_pre_2 = get_seasonal_mean(sst, month2, PR_time[0], PR_time[1], cross_month)
        sic_imonth_pre_2 = get_seasonal_mean(sic_ds, month2, PR_time[0], PR_time[1], cross_month)

        t2m_imonth_all_2 = get_seasonal_mean(t2m, month2, TR_time[0], PR_time[1], cross_month)
        slp_imonth_all_2 = get_seasonal_mean(slp, month2, TR_time[0], PR_time[1], cross_month)
        sst_imonth_all_2 = get_seasonal_mean(sst, month2, TR_time[0], PR_time[1], cross_month)
        sic_imonth_all_2 = get_seasonal_mean(sic_ds, month2, TR_time[0], PR_time[1], cross_month)

        t2m_imonth = t2m_imonth_1 - t2m_imonth_2
        slp_imonth = slp_imonth_1 - slp_imonth_2
        sst_imonth = sst_imonth_1 - sst_imonth_2
        sic_imonth = sic_imonth_1 - sic_imonth_2

        t2m_imonth_pre = t2m_imonth_pre_1 - t2m_imonth_pre_2
        slp_imonth_pre = slp_imonth_pre_1 - slp_imonth_pre_2
        sst_imonth_pre = sst_imonth_pre_1 - sst_imonth_pre_2
        sic_imonth_pre = sic_imonth_pre_1 - sic_imonth_pre_2

        t2m_imonth_all = t2m_imonth_all_1 - t2m_imonth_all_2
        slp_imonth_all = slp_imonth_all_1 - slp_imonth_all_2
        sst_imonth_all = sst_imonth_all_1 - sst_imonth_all_2
        sic_imonth_all = sic_imonth_all_1 - sic_imonth_all_2

    timeSerie_train = timeSerie.sel(year=slice(f'{TR_time[0]}', f'{TR_time[1]}')).data
    train_years = pd.to_datetime(np.arange(TR_time[0], TR_time[1] + 1), format='%Y')
    pre_years = pd.to_datetime(np.arange(PR_time[0], PR_time[1] + 1), format='%Y')

    t2mReg, t2mCorr = regress(timeSerie_train, t2m_imonth['t2m'].data), corr(timeSerie_train, t2m_imonth['t2m'].data)
    slpReg, slpCorr = regress(timeSerie_train, slp_imonth['msl'].data), corr(timeSerie_train, slp_imonth['msl'].data)
    sstReg, sstCorr = regress(timeSerie_train, sst_imonth['sst'].data), corr(timeSerie_train, sst_imonth['sst'].data)
    sicReg, sicCorr = regress(timeSerie_train, sic_imonth['sic'].data), corr(timeSerie_train, sic_imonth['sic'].data)
    t2mReg = xr.DataArray(t2mReg, coords=[t2m_imonth['lat'], t2m_imonth['lon']],
                          dims=['lat', 'lon'], name='t2m_reg')
    slpReg = xr.DataArray(slpReg, coords=[slp_imonth['lat'], slp_imonth['lon']],
                          dims=['lat', 'lon'], name='slp_reg')
    sstReg = xr.DataArray(sstReg, coords=[sst_imonth['lat'], sst_imonth['lon']],
                          dims=['lat', 'lon'], name='sst_reg')
    sicReg = xr.DataArray(sicReg, coords=[sic_imonth['lat'], sic_imonth['lon']],
                          dims=['lat', 'lon'], name='sic_reg')
    t2mCorr = xr.DataArray(t2mCorr, coords=[t2m_imonth['lat'], t2m_imonth['lon']],
                           dims=['lat', 'lon'], name='t2m_corr')
    slpCorr = xr.DataArray(slpCorr, coords=[slp_imonth['lat'], slp_imonth['lon']],
                           dims=['lat', 'lon'], name='slp_corr')
    sstCorr = xr.DataArray(sstCorr, coords=[sst_imonth['lat'], sst_imonth['lon']],
                           dims=['lat', 'lon'], name='sst_corr')
    sicCorr = xr.DataArray(sicCorr, coords=[sic_imonth['lat'], sic_imonth['lon']],
                           dims=['lat', 'lon'], name='sic_corr')

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

    # 统一管理四类场
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
        }
    }

    X_train_dict = {}
    X_pre_dict = {}
    X_rollingCorr_dict = {}

    index = 0
    for izone in predictor_zone:
        index += 1

        elem = izone[0].lower()
        X_zone = izone[1:]   # [lat1, lat2, lon1, lon2]

        if elem not in field_map:
            raise ValueError(f"Unsupported predictor element: {elem}. Choose from ['sst', 'slp', 't2m', 'sic'].")

        corr_da = field_map[elem]['corr']
        train_da = field_map[elem]['train']
        pre_da = field_map[elem]['pre']
        all_da = field_map[elem]['all']
        var_name = field_map[elem]['var']

        # --------------------------------------------------
        # 经度体系只做“必要统一”，不再人为切断联通域
        # predictor_zone 的 lon 以 corr_da 当前经度体系为准
        # --------------------------------------------------
        corr_da_ = corr_da
        train_da_ = train_da
        pre_da_ = pre_da
        all_da_ = all_da

        # 如果 predictor_zone 给的是负经度，而数据是 0~360，经度转换成 0~360
        if np.nanmax(corr_da_.lon.values) > 180:
            lon1 = X_zone[2] % 360
            lon2 = X_zone[3] % 360
        else:
            lon1 = X_zone[2]
            lon2 = X_zone[3]
            if lon1 > 180:
                lon1 = ((lon1 + 180) % 360) - 180
            if lon2 > 180:
                lon2 = ((lon2 + 180) % 360) - 180


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

        # 权重场
        weight = select_region_no_cut(corr_da_, lat1, lat2, lon1, lon2)

        sig_mask = np.abs(weight) > r_test(TR_time[1] - TR_time[0] + 1, 0.1)

        # ---------- train ----------
        X_train = select_region_no_cut(train_da_[var_name], lat1, lat2, lon1, lon2) * xr.where(sig_mask, weight, np.nan)

        X_train = X_train.mean(['lat', 'lon'])
        X_mean, X_std = X_train.mean(), X_train.std()
        X_train = (X_train - X_mean) / X_std
        X_train = pd.Series(X_train.data, index=train_years, name=f'X{index}_train')

        # ---------- pre ----------
        X_pre = select_region_no_cut(pre_da_[var_name], lat1, lat2, lon1, lon2) * xr.where(sig_mask, weight, np.nan)

        X_pre = X_pre.mean(['lat', 'lon'])
        X_pre = (X_pre - X_mean) / X_std
        X_pre = pd.Series(X_pre.data, index=pre_years, name=f'X{index}_pre')

        # ---------- all / rolling corr ----------
        X_all = select_region_no_cut(all_da_[var_name], lat1, lat2, lon1, lon2) * xr.where(sig_mask, weight, np.nan)

        X_all = X_all.mean(['lat', 'lon'])
        X_all = (X_all - X_mean) / X_std
        X_all = pd.Series(X_all.data, index=np.arange(TR_time[0], PR_time[1] + 1), name=f'X{index}_all')

        # 与目标序列做11年滑动相关（索引改为窗口中心年，避免右端年份偏移）
        X_rollingCorr = rolling_corr_centered_index(X_all, TS_all, window=11)
        X_rollingCorr.name = f'X{index}_rollingCorr'

        X_train_dict[f'X{index}'] = X_train
        X_pre_dict[f'X{index}'] = X_pre
        X_rollingCorr_dict[f'X{index}'] = X_rollingCorr

    return X_train_dict, X_pre_dict, X_rollingCorr_dict, TS, TS_pre, TS_all, t2mReg, t2mCorr, slpReg, slpCorr, sstReg, sstCorr, sicReg, sicCorr


## 请续写程序，我需要自动选出predictor里 所有对应corr矩阵通过90%检验的过检联通区域（联通区域点数量大于40个）当作预测因子Xn。
# 首先遍历sst slp t2m 和 sic，连续组合平均月份，进行12 23 34 45月等等选出因子
# 其次遍历sst slp t2m 和 sic，排列组合趋势月份，进行后月减前月5-4 5-3 5-2 5-1月等等选出因子
from scipy.ndimage import label, generate_binary_structure

# =========================================================
# 1. 一些辅助函数
# =========================================================
def _make_slice(coord, v1, v2):
    """根据坐标升降序自动生成 slice"""
    coord_values = coord.values
    if coord_values[0] < coord_values[-1]:   # 升序
        return slice(min(v1, v2), max(v1, v2))
    else:                                    # 降序
        return slice(max(v1, v2), min(v1, v2))


def _to_lon180(da):
    """统一经度到 [-180, 180]，便于输出区域"""
    lon = da.lon.values
    if np.nanmax(lon) > 180:
        return transform(da, 'lon', '360->180')
    return da.sortby('lon')


def find_connected_regions_from_corr(
    corr_da,
    alpha=0.1,
    min_size=40,
    split_sign=True,
    connectivity=2,
    periodic_lon=True
):
    """
    从 corr 场中提取通过 90% 显著性检验且格点数 > min_size 的联通区域。
    注意：
    - 不做 180° 截断
    - 支持经向周期连接
    """
    corr_da = corr_da.copy().sortby('lat')

    # 经度不转换，保留原始形式
    lon_vals = corr_da.lon.values
    if np.nanmax(lon_vals) > 180:
        lon_mode = '360'
    else:
        lon_mode = '180'

    r_crit = r_test(TR_time[1] - TR_time[0] + 1, alpha)

    arr = corr_da.values
    valid = np.isfinite(arr)

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
                'wraps': bool(wraps),   # 是否跨越 0/360 或 ±180
                'mean_corr': float(np.nanmean(vals)),
                'max_corr': float(np.nanmax(vals)),
                'min_corr': float(np.nanmin(vals)),
            }
            regions.append(region_info)

    return regions


def select_region_no_cut(da, lat1, lat2, lon1, lon2):
    """
    支持普通经度区间和跨经线区间的区域选取。
    约定：
    - 若 lon1 <= lon2: 普通区间
    - 若 lon1 > lon2 : 认为跨越经线，需要拼接两段
    """
    lat_sl = _make_slice(da.lat, lat1, lat2)

    # 普通区间
    if lon1 <= lon2:
        lon_sl = _make_slice(da.lon, lon1, lon2)
        return da.sel(lat=lat_sl, lon=lon_sl)

    # 跨经线区间：拼接两段
    lon_vals = da.lon.values
    if np.nanmax(lon_vals) > 180:
        # 0~360 体系
        part1 = da.sel(lat=lat_sl, lon=_make_slice(da.lon, lon1, np.nanmax(lon_vals)))
        part2 = da.sel(lat=lat_sl, lon=_make_slice(da.lon, np.nanmin(lon_vals), lon2))
    else:
        # -180~180 体系
        part1 = da.sel(lat=lat_sl, lon=_make_slice(da.lon, lon1, np.nanmax(lon_vals)))
        part2 = da.sel(lat=lat_sl, lon=_make_slice(da.lon, np.nanmin(lon_vals), lon2))

    return xr.concat([part1, part2], dim='lon')


def get_corr_map_from_predictor_output(pred_out):
    """
    predictor 返回值转成 corr 字典
    """
    (
        _, _, _, TS, TS_pre, TS_all,
        t2mReg, t2mCorr,
        slpReg, slpCorr,
        sstReg, sstCorr,
        sicReg, sicCorr
    ) = pred_out

    corr_map = {
        'sst': sstCorr,
        'slp': slpCorr,
        't2m': t2mCorr,
        'sic': sicCorr
    }

    return corr_map, TS, TS_pre, TS_all


def rolling_corr_centered_index(x, y, window=11):
    """
    计算滑动相关，并把结果索引从“窗口右端年”改成“窗口中心年”。

    例如 window=11:
    原 rolling 结果对 1972 的值，实际对应 1962-1972 这个窗，
    中心年应为 1967，所以索引整体向前平移 5 年。
    """
    rc = x.rolling(window=window).corr(y)

    shift_year = (window - 1) // 2
    if isinstance(rc.index, pd.DatetimeIndex):
        rc.index = rc.index - pd.DateOffset(years=shift_year)
    else:
        rc.index = rc.index - shift_year

    return rc


def calc_period_corr_stats(ts, x):
    """
    计算某一时期内 TS 与 X 的相关统计
    返回: corr, pval, n
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
    """
    统计预测期（独立样本期）上的 rolling corr 表现。
    注意：rolling_series 的索引应已处理为“窗口中心年”。

    返回：
    late_n                后期有效窗口数
    late_sig_ratio        后期 |corr| > r_sig 的比例
    late_same_sign_ratio  后期显著窗口中，与训练期符号一致的比例
    late_mean_corr        后期平均滑动相关
    late_sig_mean_corr    后期显著窗口平均相关
    """
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

        # ---------- train ----------
        tr_corr, tr_pval, tr_n = calc_period_corr_stats(TS, x_train)
        train_corr_list.append(tr_corr)
        train_pval_list.append(tr_pval)
        train_n_list.append(tr_n)

        # ---------- pre ----------
        pr_corr, pr_pval, pr_n = calc_period_corr_stats(TS_pre, x_pre)
        pre_corr_list.append(pr_corr)
        pre_pval_list.append(pr_pval)
        pre_n_list.append(pr_n)

        # ---------- rolling late ----------
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



def month_to_season_order(month, cross_month=9):
    """
    把自然月映射到以 cross_month 为起点的“季节年顺序”
    例如 cross_month=9:
    9->1, 10->2, 11->3, 12->4, 1->5, 2->6, ..., 8->12
    """
    return ((month - cross_month) % 12) + 1


def generate_avg_month_combos_excluding_predict_month(predict_month=[7, 8]):
    """
    连续2个月平均：
    12-1, 1-2, 2-3, ..., 11-12 都视为连续
    但组合中不能包含 predict_month
    返回格式：
    [[12,1], [1,2], [2,3], ...]
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


def month_to_season_order(month, cross_month=9):
    """
    把自然月映射到以 cross_month 为起点的季节年顺序
    例如 cross_month=9:
    9->1, 10->2, 11->3, 12->4, 1->5, ..., 8->12
    """
    return ((month - cross_month) % 12) + 1


def generate_avg_month_combos_excluding_predict_month(predict_month=[7, 8]):
    """
    连续两个月平均：
    12-1, 1-2, 2-3, ..., 11-12 都视为连续
    但组合中不能包含 predict_month
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
    严格按时间顺序 later - earlier
    月份顺序由 cross_month 定义
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
    后面连续两月平均 - 前面连续两月平均

    约束：
    1. 两段都必须是连续两个月（允许 12-1）
    2. 不能包含 predict_month
    3. 两段不能重叠
    4. 时间上 later_seg 晚于 earlier_seg
    """
    pm = set(predict_month)

    season_seq = list(range(cross_month, 13)) + list(range(1, cross_month))
    n = len(season_seq)

    # 先生成所有合法连续两月段
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

            # 不允许重叠月份
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
    返回 labels, num_labels
    """
    nlat, nlon = mask.shape
    labels = np.zeros((nlat, nlon), dtype=np.int32)

    # 第一遍：普通扫描赋初值
    current_label = 0
    parent = [0]

    for i in range(nlat):
        for j in range(nlon):
            if not mask[i, j]:
                continue

            neighbors = []

            # 已扫描过的邻居
            # 4邻域: 上、左
            # 8邻域: 左上、上、右上、左
            cand = [
                (i-1, j),     # up
                (i, j-1),     # left
            ]
            if connectivity == 2:
                cand.extend([
                    (i-1, j-1),   # up-left
                    (i-1, j+1),   # up-right
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

    # 第二遍：压缩标签
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

    # 第三步：补做左右边界合并（周期经向）
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
                # 斜向跨边界
                if i > 0:
                    if mask[i, 0] and mask[i-1, -1] and labels[i, 0] > 0 and labels[i-1, -1] > 0:
                        union2(labels[i, 0], labels[i-1, -1])
                    if mask[i, -1] and mask[i-1, 0] and labels[i, -1] > 0 and labels[i-1, 0] > 0:
                        union2(labels[i, -1], labels[i-1, 0])

        # 重编码
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
    给定一个联通区域里的所有经度点，找到“最短覆盖弧段”
    返回 (lon1, lon2, wraps)
    - lon1, lon2: 用于 slice 的区间端点
    - wraps=True 表示该区域跨越 0/360，经度区间需要特殊处理
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

    # 找最大缺口，最短覆盖弧就是其余部分
    diffs = np.diff(np.r_[lons_sorted, lons_sorted[0] + 360])
    k = np.argmax(diffs)

    start = lons_sorted[(k + 1) % len(lons_sorted)]
    end = lons_sorted[k]

    wraps = start > end   # 例如 start=350, end=20 表示跨 0 经线
    return float(start), float(end), wraps


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


# =========================================================
# 2. 自动搜索 predictor
# =========================================================
def auto_build_predictors(
    timeSerie,
    TR_time,
    PR_time,
    predict_month=[7, 8],
    elements=('sst', 'slp', 't2m', 'sic'),
    alpha=0.1,
    min_size=40,
    cross_month=9,
    split_sign=True,
    connectivity=2,
    periodic_lon=True,
    verbose=True
):
    """
    自动构造三类 predictor:
    1. mean: 连续两月平均
    2. trend: 单月趋势（后月 - 前月）
    3. two_month_trend: 两月平均趋势（后两月平均 - 前两月平均）

    返回
    ----
    X_train_all_dict, X_pre_all_dict, X_roll_all_dict, meta_df, TS, TS_pre, TS_all
    """
    X_train_all_dict = {}
    X_pre_all_dict = {}
    X_roll_all_dict = {}
    meta_records = []

    global_idx = 0
    TS_ref, TS_pre_ref, TS_all_ref = None, None, None

    def _append_predictors(pred_out_zone, local_meta, X_train_all_dict, X_pre_all_dict, X_roll_all_dict, meta_records, global_idx):
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
                periodic_lon=periodic_lon
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
                    'lat1': rg['lat1'],
                    'lat2': rg['lat2'],
                    'lon1': rg['lon1'],
                    'lon2': rg['lon2'],
                    'wraps': rg.get('wraps', False),
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
            pred_out_zone, local_meta,
            X_train_all_dict, X_pre_all_dict, X_roll_all_dict,
            meta_records, global_idx
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
                periodic_lon=periodic_lon
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
                    'lat1': rg['lat1'],
                    'lat2': rg['lat2'],
                    'lon1': rg['lon1'],
                    'lon2': rg['lon2'],
                    'wraps': rg.get('wraps', False),
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
            pred_out_zone, local_meta,
            X_train_all_dict, X_pre_all_dict, X_roll_all_dict,
            meta_records, global_idx
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
                periodic_lon=periodic_lon
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
                    'lat1': rg['lat1'],
                    'lat2': rg['lat2'],
                    'lon1': rg['lon1'],
                    'lon2': rg['lon2'],
                    'wraps': rg.get('wraps', False),
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
            pred_out_zone, local_meta,
            X_train_all_dict, X_pre_all_dict, X_roll_all_dict,
            meta_records, global_idx
        )

    # -----------------------------------------------------
    # D. 汇总输出
    # -----------------------------------------------------
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




# =========================================================
# 3. 执行自动搜索
# =========================================================
PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

EHCI = xr.open_dataset(f"{PYFILE}/p5/data/EHCI_daily.nc")
EHCI = EHCI.groupby('time.year')
EHCI30 = EHCI.apply(lambda x: (x > 0.6).sum())
EHCI30 = (EHCI30 - EHCI30.mean()) / EHCI30.std('year')
EHCI30 = EHCI30['EHCI']

# 2mT
t2m = era5_land(fr"{DATA}/ERA5/ERA5_land/uv_2mTTd_sfp_pre_0.nc", 1961, 2022, 't2m')
# SLP
slp = era5_s(fr"{DATA}/ERA5/ERA5_singleLev/ERA5_sgLEv.nc", 1961, 2022, 'msl')
# sst
sst = ersst(fr"{DATA}/NOAA/ERSSTv5/sst.mnmean.nc", 1961, 2022)
# sic
sic_ds = prepare_sic_dataset(read_sic(fr"{DATA}/NOAA/HadISST/HadISST_ice.nc", 1961, 2022))

TR_time = [1962, 2004]
PR_time = [2005, 2022]
timeSerie = EHCI30

predict_month = [6, 7, 8, 9, 10, 11, 12]

X_train_all_dict, X_pre_all_dict, X_roll_all_dict, meta_df, TS, TS_pre, TS_all = auto_build_predictors(
    timeSerie=timeSerie,
    TR_time=TR_time,
    PR_time=PR_time,
    predict_month=predict_month,
    elements=('sst', 'sic'),
    alpha=0.1,
    min_size=40,
    cross_month=9,
    split_sign=True,
    connectivity=2,
    periodic_lon=True,
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

# 保存信息表
meta_df.to_csv(
    fr"{PYFILE}/p5/data/auto_predictor_catalog.csv",
    index=False,
    encoding='utf-8-sig'
)

# %%
plt.rcParams['font.family'] = ['AVHershey Simplex', 'AVHershey Duplex', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(5, 4))
fig.subplots_adjust(hspace=0.35)
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================
# 0. 参数设置
# ============================================================
index_N = 2   # 最终逐步回归最多保留的因子个数

# ============================================================
# 1. 收集 predictor 结果
# ============================================================
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


# ============================================================
# 2. 组装训练/预测数据
# ============================================================
df_train_full = pd.concat(
    [TS.rename('TS')] + [all_X_train[x].rename(x) for x in candidate_predictors],
    axis=1
)

df_pre_full = pd.concat(
    [TS_pre.rename('TS')] + [all_X_pre[x].rename(x) for x in candidate_predictors],
    axis=1
)

# 训练期去掉第一年（按你原逻辑）
df_train_step = df_train_full.copy()

# 统一数值化
for c in df_train_step.columns:
    df_train_step[c] = pd.to_numeric(df_train_step[c], errors='coerce')
for c in df_pre_full.columns:
    df_pre_full[c] = pd.to_numeric(df_pre_full[c], errors='coerce')


# ============================================================
# 3. 工具函数
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

    ranked = []
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

        # Forward
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

        # Backward by p
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

        # Backward by VIF
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

    # Final clean
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

    # 最终数量限制
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


def is_late_significant(rolling_series, PR_time, r_sig, min_ratio=0.5, min_count=3):
    """
    判断某个因子在后期（独立样本期）是否比较显著
    """
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
# ============================================================
# 4. 候选因子诊断
# ============================================================
diagnose_candidates(
    df=df_train_step,
    response='TS',
    candidates=candidate_predictors
)


# ============================================================
# 5. 单因子预筛选
# ============================================================
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


# ============================================================
# 6. 去除两两高度相关变量
# ============================================================
screened_predictors = remove_high_pairwise_corr(
    df=df_train_step,
    features=screened_predictors,
    corr_thres=0.6,
    response='TS'
)

print("Predictors after pairwise-corr filtering:", screened_predictors)


# ============================================================
# 7. 逐步回归（最多 index_N 个因子）
# ============================================================
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

# fallback
if len(model_predictors) == 0:
    model_predictors = fallback_predictors_from_screen(screen_df, max_n=index_N)
    print("Stepwise returned empty. Fallback predictors:", model_predictors)

if len(model_predictors) == 0:
    raise ValueError("No usable predictors after fallback.")

model_predictors = model_predictors[:index_N]
print("Selected predictors:", model_predictors)


# ============================================================
# 8. 建模与绘图变量分离
# ============================================================
plot_predictors = screened_predictors.copy()      # 画 rollingCorr 用
selected_predictors = model_predictors.copy()     # 建模用

df_train = pd.concat(
    [TS.rename('TS')] + [all_X_train[x].rename(x) for x in selected_predictors],
    axis=1
).replace([np.inf, -np.inf], np.nan)

df_pre = pd.concat(
    [TS_pre.rename('TS')] + [all_X_pre[x].rename(x) for x in selected_predictors],
    axis=1
).replace([np.inf, -np.inf], np.nan)


# ============================================================
# 9. rollingCorr 绘图
#    红色：最终入模
#    橘色：后期显著
#    灰色：其他
# ============================================================
ax_rollingCorr = fig.add_subplot(gs[0])
ax_rollingCorr.set_ylim(-0.5, 1)

# 11年滑动相关的90%显著性阈值
r_sig_11 = r_test(11, 0.1)

# 判断哪些因子在后期比较显著
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

    # 颜色优先级：最终入模 > 后期显著 > 普通
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

# 显著性参考线
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

# 图例文字样式同步
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

# 90% 图例
leg.get_texts()[-1].set_fontweight('normal')
leg.get_texts()[-1].set_color('black')
leg.get_texts()[-1].set_alpha(0.8)


# ============================================================
# 10. 最终回归建模
# ============================================================
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


# ============================================================
# 11. 预测
# ============================================================
df_train['predicted_TS'] = model.predict(df_train)
df_pre['inDependent_pre'] = model.predict(df_pre)
df_train['residuals'] = df_train['TS'] - df_train['predicted_TS']

TS_all = pd.concat([TS.rename('TS'), TS_pre.rename('TS')])


# ============================================================
# 12. 预测图
# ============================================================
ax_predict = fig.add_subplot(gs[1])
ax_predict.set_ylim(-3, 3)

ax_predict.plot(TS_all.index, TS_all.values, color='black', linestyle='-', linewidth=1.5, label='Obs')
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


# ============================================================
# 13. 回归方程文本
# ============================================================
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

