import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import tqdm as tq

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker, gridspec
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy import ndimage
from scipy.stats import ttest_ind, pearsonr
import matplotlib.path as mpath
import matplotlib.patches as patches

from climkit.Cquiver import *
from climkit.TN_WaveActivityFlux import TN_WAF_3D
from climkit.masked import masked
from climkit.significance_test import r_test
from climkit.lonlat_transform import *
from climkit.filter import *
from climkit.corr_reg import *
import cmaps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import tqdm as tq

from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker, gridspec
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy import ndimage
from scipy.stats import ttest_ind, pearsonr
import matplotlib.path as mpath
import matplotlib.patches as patches

from climkit.Cquiver import *
from climkit.TN_WaveActivityFlux import TN_WAF_3D
from climkit.masked import masked
from climkit.significance_test import r_test
from climkit.lonlat_transform import *
from climkit.filter import *
from climkit.corr_reg import *

from matplotlib import ticker
from metpy.calc import vertical_velocity
from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants


from matplotlib import ticker
from metpy.calc import vertical_velocity
from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants

import xeofs

def regress_map(field, pc):
    """field: time, lat, lon; pc: time"""
    pc = (pc - pc.mean("time")) / pc.std("time")
    return xr.cov(field, pc, dim="time") / pc.var("time")


# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

CLIM = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_clim_sum.nc").sel(mmdd=slice("06-01", "08-31"))
ANO = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_ano_2015_MJJAS.nc").sel(time=slice("2015-06-01", "2015-08-31"))
BP = xr.open_dataset("/Volumes/TiPlus7100/p4/data/ERA5_CPC_daily_bp_2015_JJA_10-30d.nc")


u_bp = BP['u_bp']
v_bp = BP['v_bp']
z_bp = BP['z_bp']
t_bp = BP['t_bp']
w_bp = BP['w_bp']
olr_bp = BP['olr_bp']
t2m_bp = BP['t2m_bp']

u_clim = CLIM['u_clim'].mean(dim='mmdd')
v_clim = CLIM['v_clim'].mean(dim='mmdd')
u_ano = ANO['u_ano'].mean(dim='time')
v_ano = ANO['v_ano'].mean(dim='time')

t2m_bp = masked(t2m_bp, fr"{PYFILE}/map/self/长江_TP/长江_tp.shp") # 取反位向
svd_t2m_v200 = xeofs.cross.MCA(n_modes=4, standardize=False, use_coslat=True).fit(t2m_bp, v_bp.sel(level=200, lon=slice(-80, 160), lat=slice(36, 75)), dim="time")
svd_t2m_u850 = xeofs.cross.MCA(n_modes=4, standardize=False, use_coslat=True).fit(t2m_bp, u_bp.sel(level=850, lon=slice(40, 160), lat=slice(-10, 24)), dim="time")
svd_t2m_olr = xeofs.cross.MCA(n_modes=4, standardize=False, use_coslat=True).fit(t2m_bp, olr_bp.sel(lon=slice(40, 160), lat=slice(-10, 24)), dim="time")


# =========================
# draw SVD: Z200&UV&T2m + scores + lag correlations
# draw SVD: Z850&UV&T2m + scores + lag correlations
# =========================
from scipy.stats import pearsonr
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

proj = ccrs.PlateCarree(central_longitude=0)

fig = plt.figure(figsize=(18, 6.5))

gs = gridspec.GridSpec(
    nrows=2, ncols=3,
    width_ratios=[1.05, 1.05, 0.85],
    height_ratios=[1, 1],
    wspace=0.18, hspace=0.3
)

clevs_t2m = np.array([
    -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4
]) * 5

cmap = cmaps.GMT_polar[2:10] + cmaps.CBR_wet[0] + cmaps.GMT_polar[10:-2]
cf_last = None


def lag_corr(x, y=None, max_lag=30):
    """
    lag > 0: x leads y
    lag < 0: x lags y
    """
    if y is None:
        y = x

    x = xr.DataArray(x).values
    y = xr.DataArray(y).values

    x = (x - np.nanmean(x)) / np.nanstd(x)
    y = (y - np.nanmean(y)) / np.nanstd(y)

    lags = np.arange(-max_lag, max_lag + 1)
    rs = []

    for lag in lags:
        if lag < 0:
            xx = x[-lag:]
            yy = y[:lag]
        elif lag > 0:
            xx = x[:-lag]
            yy = y[lag:]
        else:
            xx = x
            yy = y

        mask = np.isfinite(xx) & np.isfinite(yy)

        if mask.sum() > 3:
            r = pearsonr(xx[mask], yy[mask])[0]
        else:
            r = np.nan

        rs.append(r)

    return lags, np.array(rs)


def draw_one(row, level, svd_obj, left_score_name, right_score_name,
             map_extent, box_blue, box_orange, title_left, title_right,
             vector_label, score_label):

    global cf_last

    if level <= 500:
        t2m_mode = svd_obj.components()[0].sel(mode=1)
        atm_score = svd_obj.scores()[1].sel(mode=1)
        t2m_score = svd_obj.scores()[0].sel(mode=1)

        t2m_mode *= -10
        t2m_score /= -100
        atm_score /= -100

    elif level == 850:
        t2m_mode = svd_obj.components()[0].sel(mode=1)
        atm_score = svd_obj.scores()[1].sel(mode=1)
        t2m_score = svd_obj.scores()[0].sel(mode=1)

        t2m_mode *= -10
        t2m_score /= -100
        atm_score /= -100

    z_reg = transform(z_bp, 'lon', '360->180')
    u_reg = transform(u_bp, 'lon', '360->180')
    v_reg = transform(v_bp, 'lon', '360->180')
    olr_reg = transform(olr_bp, 'lon', '360->180')


    z_reg = regress(atm_score, z_reg.sel(level=level))
    u_reg = regress(atm_score, u_reg.sel(level=level))
    v_reg = regress(atm_score, v_reg.sel(level=level))

    if level <= 500:
        olr_reg = regress(atm_score, olr_reg)
        olr_reg = olr_reg.where((z_reg.lat >= -15) & (z_reg.lat <= 15))

        WAF = TN_WAF_3D(
            u_clim.sel(level=level),
            v_clim.sel(level=level),
            z_reg,
            single_level=200
        )

        waf_x, waf_y = WAF

        waf_x = waf_x.where(
            (u_reg.lon >= map_extent[0]) & (u_reg.lon <= map_extent[1]) &
            (u_reg.lat >= map_extent[2]) & (u_reg.lat <= map_extent[3])
        )

        waf_y = waf_y.where(
            (u_reg.lon >= map_extent[0]) & (u_reg.lon <= map_extent[1]) &
            (u_reg.lat >= map_extent[2]) & (u_reg.lat <= map_extent[3])
        )

    z_reg = z_reg.where(
        (z_reg.lon >= map_extent[0]) & (z_reg.lon <= map_extent[1]) &
        (z_reg.lat >= map_extent[2]) & (z_reg.lat <= map_extent[3])
    )

    u_reg = u_reg.where(
        (u_reg.lon >= map_extent[0]) & (u_reg.lon <= map_extent[1]) &
        (u_reg.lat >= map_extent[2]) & (u_reg.lat <= map_extent[3])
    )

    v_reg = v_reg.where(
        (v_reg.lon >= map_extent[0]) & (v_reg.lon <= map_extent[1]) &
        (v_reg.lat >= map_extent[2]) & (v_reg.lat <= map_extent[3])
    )

    # ================= left map =================
    ax = fig.add_subplot(gs[row, 0], projection=proj)
    ax.set_aspect('auto')

    extent = [-90, 170, 0, 80] if level == 200 else [20, 180, -20, 50]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(
        cfeature.COASTLINE.with_scale('110m'),
        linewidth=1.5,
        color="#BBBBBB"
    )

    ax.add_geometries(
        Reader(fr'{PYFILE}/map/self/长江_TP/长江_tp.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black',
        linewidth=.5
    )

    ax.add_geometries(
        Reader(fr'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary2500m_长江流域/TPBoundary2500m_长江流域.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='gray',
        edgecolor='black',
        linewidth=.5
    )

    if level >= 600:
        ax.add_geometries(
            Reader(f'{PYFILE}/map/地图边界数据/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp').geometries(),
            ccrs.PlateCarree(),
            facecolor='#909090',
            edgecolor='#909090',
            linewidth=0,
            hatch='.',
            zorder=10
        )

    if level <= 500:
        ax.set_xticks(np.arange(extent[0], extent[1], 40), crs=proj)
        ax.set_yticks(np.arange(extent[2], extent[3] + 1, 15), crs=proj)
    elif level == 850:
        ax.set_xticks(np.arange(extent[0], extent[1], 30), crs=proj)
        ax.set_yticks(np.arange(extent[2], extent[3] + 1, 10), crs=proj)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=12)

    cf = ax.contourf(
        t2m_mode.lon,
        t2m_mode.lat,
        t2m_mode,
        levels=clevs_t2m,
        cmap=cmap,
        extend='both',
        transform=proj
    )

    cf_last = cf

    zlev_pos = [60, 180] if level == 200 else [1, 20]
    zlev_neg = [-180, -60] if level == 200 else [-30, -10]

    ax.contour(
        z_reg.lon,
        z_reg.lat,
        z_reg,
        levels=zlev_pos,
        colors='red',
        linewidths=1.2,
        transform=proj
    )

    ax.contour(
        z_reg.lon,
        z_reg.lat,
        z_reg,
        levels=zlev_neg,
        colors='blue',
        linewidths=1.2,
        linestyles='--',
        transform=proj
    )

    z_reg.to_netcdf(fr"/Volumes/TiPlus7100/p4/data/SVD{level}V_{level}Z.nc")

    if level <= 500:
        q = ax.Curlyquiver(
            u_reg.lon,
            u_reg.lat,
            u_reg,
            v_reg,
            arrowsize=1,
            transform=ccrs.PlateCarree(central_longitude=0),
            scale=2,
            linewidth=1,
            regrid=18,
            color="#555555",
            thinning=["20%", "min"],
            nanmax=2,
            MinDistance=[0.2, 0.4]
        )

        q.key(
            U=4,
            label=r"4 $m/s$",
            fontproperties={'size': 8},
            facecolor='none',
            bbox_to_anchor=(0, 0.14, 1, 1),
            edgecolor="none",
            arrowsize=6,
            linewidth=1,
            intetval=0.8
        )

        q_waf = ax.Curlyquiver(
            u_reg.lon,
            u_reg.lat,
            waf_x,
            waf_y,
            integration_direction="stick_both",
            arrowsize=2,
            transform=ccrs.PlateCarree(central_longitude=0),
            scale=20,
            linewidth=2,
            regrid=12,
            color="purple",
            thinning=["60%", "min"],
            nanmax=2,
            MinDistance=[0.2, 0.4]
        )

        q_waf.key(
            U=0.5,
            label=r"0.5 ${m}^{2}/{s}^{2}$",
            fontproperties={'size': 8},
            facecolor='none',
            bbox_to_anchor=(-0.1, 0.14, 1, 1),
            edgecolor="none",
            arrowsize=1,
            linewidth=1,
            intetval=0.8
        )

    elif level == 850:
        q = ax.Curlyquiver(
            u_reg.lon,
            u_reg.lat,
            u_reg,
            v_reg,
            arrowsize=1,
            transform=ccrs.PlateCarree(central_longitude=0),
            scale=10,
            linewidth=1,
            regrid=18,
            color="#555555",
            thinning=["35%", "min"],
            nanmax=2,
            MinDistance=[0.2, 0.4]
        )

        q.key(
            U=0.5,
            label=r"0.5 $m/s$",
            fontproperties={'size': 8},
            facecolor='none',
            bbox_to_anchor=(0, 0.14, 1, 1),
            edgecolor="none",
            arrowsize=1,
            linewidth=1,
            intetval=0.8
        )

    ax.add_patch(
        patches.Rectangle(
            (box_blue[0], box_blue[2]),
            box_blue[1] - box_blue[0],
            box_blue[3] - box_blue[2],
            linewidth=1.8,
            edgecolor='royalblue',
            facecolor='none',
            linestyle='--',
            transform=proj
        )
    )

    ax.set_title(title_left, loc='left', fontsize=16)

    # ================= right score =================
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.set_aspect('auto')

    dates = pd.to_datetime(t2m_score.time.values)

    xscore = (t2m_score - t2m_score.mean()) / t2m_score.std()
    yscore = (atm_score - atm_score.mean()) / atm_score.std()

    ax2.plot(dates, xscore, color='red', lw=1.5, label='T2m')
    ax2.plot(dates, yscore, color='blue', lw=1.5, label=score_label)

    ax2.set_xlim(dates.min(), dates.max())
    ax2.axhline(0, color='0.75', lw=0.7)

    ax2.set_ylim(-4, 4)
    ax2.set_yticks(np.arange(-4, 4.1, 2))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    ax2.tick_params(labelsize=12)
    ax2.legend(frameon=False, fontsize=12, bbox_to_anchor=(-0.01, 1.03), loc='upper left')

    try:
        corvar = float(svd_obj.squared_covariance_fraction().sel(mode=1) * 100)
    except Exception:
        corvar = float(svd_obj.explained_variance_ratio().sel(mode=1) * 100)

    r, p = pearsonr(xscore.values, yscore.values)
    ptxt = 'p<0.01' if p < 0.01 else f'p={p:.2f}'

    ax2.text(
        0.43, 0.92,
        f'CorVar={corvar:.0f}%',
        transform=ax2.transAxes,
        fontsize=12
    )

    ax2.text(
        0.75, 0.92,
        f'r={r:.2f} ({ptxt})',
        transform=ax2.transAxes,
        fontsize=12
    )

    ax2.set_title(title_right, loc='left', fontsize=16)
    import matplotlib.dates as mdates

    # x轴只显示 月-日
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # 控制刻度密度（可选，建议加）
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=20))

    return ax, ax2, xscore, yscore


# ================= first row: 200 hPa =================
ax200, ax200_ts, t2m200_score, v200_score = draw_one(
    row=0,
    level=200,
    svd_obj=svd_t2m_v200,
    left_score_name='T2m',
    right_score_name='V200',
    map_extent=[-90, 170, 20, 80],
    box_blue=[-80, 160, 36, 75],
    box_orange=[-70, 0, -20, 45],
    title_left='(a) SVD (T2m&V200)',
    title_right='(b) Time series',
    vector_label='1.5 m/s',
    score_label='V200'
)


# ================= second row: 850 hPa =================
ax850, ax850_ts, t2m850_score, u850_score = draw_one(
    row=1,
    level=850,
    svd_obj=svd_t2m_olr,
    left_score_name='T2m',
    right_score_name='OLR',
    map_extent=[20, 180, -20, 50],
    box_blue=[40, 160, -10, 24],
    box_orange=[90, 122, 20, 35],
    title_left='(c) SVD (T2m&OLR)',
    title_right='(d) Time series',
    vector_label='1.5 m/s',
    score_label='U850'
)


# ================= third column: lag correlations =================
gs_right = gridspec.GridSpecFromSubplotSpec(
    3, 1,
    subplot_spec=gs[:, 2],
    hspace=0.25
)

max_lag = 30

lag_v200, r_v200 = lag_corr(v200_score, max_lag=max_lag)
lag_cross, r_cross = lag_corr(v200_score, u850_score, max_lag=max_lag)
lag_u850, r_u850 = lag_corr(u850_score, max_lag=max_lag)

lag_panels = [
    (lag_v200, r_v200, '(e) auto-V200'),
    (lag_cross, r_cross, '(f) V200&U850'),
    (lag_u850, r_u850, '(g) auto-U850'),
]

n_eff = len(v200_score.time)
r_sig = 1.96 / np.sqrt(n_eff - 3)

for i, (lags, rs, title) in enumerate(lag_panels):
    ax_lag = fig.add_subplot(gs_right[i, 0])

    ax_lag.plot(lags, rs, color='black', lw=2.2)

    ax_lag.axhline(0, color='black', lw=1.0, ls='--', alpha=0.8)
    ax_lag.axvline(0, color='black', lw=1.0, ls='--', alpha=0.8)

    ax_lag.axhline(r_sig, color='red', lw=1.2, ls=(0, (8, 5)), alpha=0.7)
    ax_lag.axhline(-r_sig, color='red', lw=1.2, ls=(0, (8, 5)), alpha=0.7)

    ax_lag.set_xlim(-max_lag, max_lag)
    ax_lag.set_ylim(-1.0, 1.0)

    ax_lag.set_xticks(np.arange(-30, 31, 10))
    ax_lag.set_yticks(np.arange(-0.8, 0.81, 0.4))

    ax_lag.xaxis.set_minor_locator(MultipleLocator(2))
    ax_lag.yaxis.set_minor_locator(MultipleLocator(0.2))

    ax_lag.tick_params(labelsize=11, direction='out', length=4, width=1.2)
    ax_lag.tick_params(which='minor', length=2, width=1)

    if i < 2:
        ax_lag.set_xticklabels([])

    ax_lag.set_title(title, loc='left', fontsize=16)

    for spine in ax_lag.spines.values():
        spine.set_linewidth(2.0)


# ================= colorbar =================
cax = inset_axes(
    ax850,
    width="100%",
    height="5%",
    loc='lower left',
    bbox_to_anchor=(0, -0.18, 1, 1),
    bbox_transform=ax850.transAxes,
    borderpad=0
)

cb = fig.colorbar(cf_last, cax=cax, orientation='horizontal', drawedges=True)
cb.set_ticks(clevs_t2m)
cb.ax.tick_params(length=0, labelsize=9)
cb.dividers.set_linewidth(2.5)
cb.outline.set_linewidth(2.5)


# ================= borders =================
for ax_ in fig.axes:
    for spine in ax_.spines.values():
        spine.set_linewidth(2.5)


plt.savefig(fr"{PYFILE}/p4/pic/SVD_UVZ.png", bbox_inches='tight', dpi=600)
plt.savefig(fr"{PYFILE}/p4/pic/SVD_UVZ.pdf", bbox_inches='tight')
plt.show()

