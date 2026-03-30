import xarray as xr
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.mpl.clip_path import bbox_to_path
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec
from metpy.calc import vertical_velocity
from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants
import cmaps

from climkit.TN_WaveActivityFlux import TN_WAF_3D
from climkit.Cquiver import *
from climkit.data_read import *
from climkit.lonlat_transform import *


PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"


def regress(time_series, data, var_name=None):
    """
    对 time_series 与 data 的第一个维度做线性回归和相关分析，并进行显著性检验。

    Parameters
    ----------
    time_series : array-like or xr.DataArray
        一维时间序列，长度应等于 data 的时间维长度
    data : xr.Dataset / xr.DataArray / np.ndarray
        待回归数据，要求第一个维度是时间维
    var_name : str, optional
        当 data 是 xr.Dataset 时，需要指定变量名；
        若不指定且 Dataset 只有一个变量，则自动取该变量
    alpha : float, default 0.05
        显著性水平

    Returns
    -------
    regression_map : xr.DataArray or np.ndarray
        回归系数场
    correlation_map : xr.DataArray or np.ndarray
        相关系数场
    pvalue_map : xr.DataArray or np.ndarray
        相关系数对应的 p 值
    significance_mask : xr.DataArray or np.ndarray
        是否通过显著性检验（True/False）
    """
    from scipy import stats
    # -----------------------------
    # 1. 统一 data 类型
    # -----------------------------
    original_is_xarray = False
    coords = None
    dims = None
    name = None

    if isinstance(data, xr.Dataset):
        if var_name is not None:
            da = data[var_name]
        else:
            data_vars = list(data.data_vars)
            if len(data_vars) == 1:
                da = data[data_vars[0]]
            else:
                raise ValueError(
                    "data 是 xr.Dataset，包含多个变量，请通过 var_name 指定变量名。"
                )
        original_is_xarray = True
        coords = {k: v for k, v in da.coords.items() if k in da.dims[1:]}
        dims = da.dims[1:]
        name = da.name if da.name is not None else "data"

    if isinstance(data, xr.DataArray):
        da = data
        original_is_xarray = True
        coords = {k: v for k, v in da.coords.items() if k in da.dims[1:]}
        dims = da.dims[1:]
        name = da.name if da.name is not None else "data"

    elif isinstance(data, np.ndarray):
        da = None
    else:
        raise TypeError("data 必须是 xr.Dataset、xr.DataArray 或 np.ndarray")

    # -----------------------------
    # 2. time_series 转 np.array
    # -----------------------------
    if isinstance(time_series, xr.DataArray):
        ts = time_series.values
    else:
        ts = np.asarray(time_series)

    ts = np.squeeze(ts)
    if ts.ndim != 1:
        raise ValueError("time_series 必须是一维数组")

    # -----------------------------
    # 3. data 转 np.array
    # -----------------------------
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        data_np = np.asarray(da.values)
    else:
        data_np = np.asarray(data)

    if data_np.shape[0] != len(ts):
        raise ValueError(
            f"time_series 长度 ({len(ts)}) 必须等于 data 第一个维度长度 ({data_np.shape[0]})"
        )

    # -----------------------------
    # 4. reshape 为 (time, space)
    # -----------------------------
    reshaped_data = data_np.reshape(len(ts), -1)

    # -----------------------------
    # 5. 去均值
    # -----------------------------
    ts_anom = ts - np.mean(ts)
    data_anom = reshaped_data - np.mean(reshaped_data, axis=0)

    # -----------------------------
    # 6. 回归系数和相关系数
    # -----------------------------
    numerator = np.sum(data_anom * ts_anom[:, np.newaxis], axis=0)
    denominator = np.sum(ts_anom ** 2)

    regression_coef = numerator / denominator

    data_std_term = np.sqrt(np.sum(data_anom ** 2, axis=0))
    ts_std_term = np.sqrt(np.sum(ts_anom ** 2))

    correlation = numerator / (data_std_term * ts_std_term)

    # 避免浮点误差导致 |r| > 1
    correlation = np.clip(correlation, -1.0, 1.0)

    # -----------------------------
    # 7. 显著性检验（对相关系数做 t 检验）
    #    t = r * sqrt((n-2)/(1-r^2))
    # -----------------------------
    n = len(ts)
    dof = n - 2

    with np.errstate(divide="ignore", invalid="ignore"):
        t_value = correlation * np.sqrt(dof / (1.0 - correlation ** 2))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), df=dof))


    # -----------------------------
    # 8. reshape 回原空间维度
    # -----------------------------
    spatial_shape = data_np.shape[1:]
    regression_map = regression_coef.reshape(spatial_shape)
    pvalue_map = p_value.reshape(spatial_shape)

    # -----------------------------
    # 9. 若原数据是 xarray，则还原成 DataArray
    # -----------------------------
    if original_is_xarray:
        regression_map = xr.DataArray(
            regression_map,
            coords=coords,
            dims=dims,
            name=f"{name}_regression"
        )

        pvalue_map = xr.DataArray(
            pvalue_map,
            coords=coords,
            dims=dims,
            name=f"{name}_pvalue"
        )

    return regression_map, pvalue_map



K_type = xr.open_dataset(fr"{PYFILE}/p2/data/Time_type_AverFiltAll0.9%_0.3%_3.nc")
lat_range = [15, -5]
Z = xr.open_dataset(fr"{PYFILE}/p2/data/Z.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(lat_range[0], lat_range[1]), lon=slice(0, 360)) / 9.8
U = xr.open_dataset(fr"{PYFILE}/p2/data/U.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(lat_range[0], lat_range[1]), lon=slice(0, 360))
V = xr.open_dataset(fr"{PYFILE}/p2/data/V.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(lat_range[0], lat_range[1]), lon=slice(0, 360))
T = xr.open_dataset(fr"{PYFILE}/p2/data/T.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(lat_range[0], lat_range[1]), lon=slice(0, 360))
Q = xr.open_dataset(fr"{PYFILE}/p2/data/Q.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(lat_range[0], lat_range[1]), lon=slice(0, 360))
W = xr.open_dataset(fr"{PYFILE}/p2/data/W.nc").sel(level=[100, 150, 200, 300, 400, 500, 600, 700, 850, 900, 1000], lat=slice(lat_range[0], lat_range[1]), lon=slice(0, 360))
# W = vertical_velocity(W['w'] * units('Pa/s') , W['level'] * units.hPa, T['t'] * units.degC)
Pre = xr.open_dataset(fr"{PYFILE}/p2/data/pre.nc").sel(lat=slice(lat_range[0], lat_range[1]), lon=slice(0, 360))
Sst = xr.open_dataset(fr"{PYFILE}/p2/data/sst.nc").sel(lat=slice(lat_range[0], lat_range[1]), lon=slice(0, 360))
Terrain = xr.open_dataset(fr"{DATA}/NOAA/ETOPO/ETOPO_2022_v1_30s_N90W180_bed.nc").sel(lat=slice(lat_range[1], lat_range[0]), lon=slice(-180, 180))['z'].astype(np.float64).mean(dim='lat', skipna=True)
Z = transform(Z['z'], lon_name='lon', type='180->360')
U = transform(U['u'], lon_name='lon', type='180->360')
V = transform(V['v'], lon_name='lon', type='180->360')
Q = transform(Q['q'], lon_name='lon', type='180->360')
W = transform(W['w'], lon_name='lon', type='180->360')
Pre = transform(Pre['pre'], lon_name='lon', type='180->360')
Sst = transform(Sst['sst'], lon_name='lon', type='180->360')
Terrain = transform(Terrain, lon_name='lon', type='180->360')

Qu = Q * units('kg/kg') * U * units('m/s') / constants.g * 1000
Qv = Q * units('kg/kg') * V * units('m/s') / constants.g * 1000
dx, dy = mpcalc.lat_lon_grid_deltas(Qu.lon, Qu.lat)
# 计算水汽通量散度
Q_div = np.array([[mpcalc.divergence(Qu[iYear, iLev, :, :], Qv[iYear, iLev, :, :], dx=dx, dy=dy) for iLev in range(len(Qu['level']))] for iYear in range(len(Qu['year']))])


Terrain_ver = np.array(Terrain)
Terrain_ver = 1013 * (1 - 6.5/2.88e5 * Terrain_ver)**5.255
lon_Terrain = Terrain.lon


#### 拉尼娜型海温
zone = [120, 360-80, 5, -5]
### 合成分析
time_series = (Sst.sel(lon=slice(zone[0], zone[1]), lat=slice(zone[2], zone[3]))
               - Sst.sel(lon=slice(zone[0], zone[1]), lat=slice(zone[2], zone[3])).mean(['year']))
trend = regress(np.array(range(len(time_series))), time_series.data)[0]
time_series = time_series - np.arange(len(time_series))[:, np.newaxis, np.newaxis] * trend

## 中下游型
K_series = K_type.sel(type=1)['K'].data
K_series = (K_series - np.mean(K_series))/np.std(K_series)
corr_mid_lower = regress(K_series, Sst.sel(lon=slice(zone[0], zone[1]), lat=slice(zone[2], zone[3])).data)[0]

def shift_lon_start(lon, *fields, lon0=40):
    lon = np.asarray(lon)
    idx = np.searchsorted(lon, lon0)

    lon_new = np.concatenate([lon[idx:], lon[:idx] + 360])

    out = [lon_new]
    for f in fields:
        f = np.asarray(f)
        # 默认经度是最后一维；如果不是最后一维，再单独改 axis
        f_new = np.concatenate([f[..., idx:], f[..., :idx]], axis=-1)
        out.append(f_new)
    return out

fig = plt.figure(figsize=(8, 4))
spec = gridspec.GridSpec(ncols=1, nrows=2, wspace=0, hspace=0, height_ratios=[2, 1.4])  # 设置子图比例
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

Z_reg = regress(K_series, Z)[0]
U_reg = regress(K_series, U)[0]
W_reg = regress(K_series, W)[0]
Q_div_reg = regress(K_series, Q_div)[0]
Pre_reg = regress(K_series, Pre)[0]
Sst_reg = regress(K_series, Sst)[0]
Z_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], Z_reg.data),},
                  coords={'level': Z['level'], 'lat': Z['lat'], 'lon': Z['lon']})
U_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], U_reg.data)},
                  coords={'level': U['level'], 'lat': U['lat'], 'lon': U['lon']})
W_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], W_reg.data)},
                  coords={'level': W['level'], 'lat': W['lat'], 'lon': W['lon']})
Q_div_nc = xr.Dataset({'reg': (['level', 'lat', 'lon'], Q_div_reg.data)},
                        coords={'level': Qv['level'], 'lat': Qv['lat'], 'lon': Qv['lon']})
Pre_nc = xr.Dataset({'reg': (['lat', 'lon'], Pre_reg.data)},
                    coords={'lat': Pre['lat'], 'lon': Pre['lon']})
Sst_nc = xr.Dataset({'reg': (['lat', 'lon'], Sst_reg.data)},
                    coords={'lat': Sst['lat'], 'lon': Sst['lon']})
W_nc['reg'] *= 1000   # Pa->0.001Pa



# ---- 剖面图数据平移到 40E 开始 ----
TRS_LON = 220
lon_u = U_nc['lon'].to_numpy()
u_mean = U_nc['reg'].mean('lat').to_numpy()
w_mean = W_nc['reg'].mean('lat').to_numpy()

lon_q = Q_div_nc['lon'].to_numpy()
q_mean = Q_div_nc['reg'].mean('lat', skipna=True).to_numpy()

lon_sst = Sst_nc['lon'].to_numpy()
sst_lon_mean = Sst_nc['reg'].mean('lat', skipna=True).to_numpy()

lon_pre = Pre_nc['lon'].to_numpy()
pre_lon_mean = Pre_nc['reg'].mean('lat', skipna=True).to_numpy()

lon_ter = np.asarray(lon_Terrain)
ter_ver = np.asarray(Terrain_ver)

lon_u_s, u_mean_s, w_mean_s = shift_lon_start(lon_u, u_mean, w_mean, lon0=TRS_LON)
lon_q_s, q_mean_s = shift_lon_start(lon_q, q_mean, lon0=TRS_LON)
lon_sst_s, sst_lon_mean_s = shift_lon_start(lon_sst, sst_lon_mean, lon0=TRS_LON)
lon_pre_s, pre_lon_mean_s = shift_lon_start(lon_pre, pre_lon_mean, lon0=TRS_LON)
lon_ter_s, ter_ver_s = shift_lon_start(lon_ter, ter_ver, lon0=TRS_LON)


f_ax = fig.add_subplot(spec[0])
f_ax.set_title(r'(d) MLR-type U$\omega$&moisture_flux_div', loc='left', fontsize=16, pad=12)
f_ax.set_ylabel('Pressure(hPa)', fontsize=12)
f_ax.set_xlim(TRS_LON, 360+TRS_LON)
f_ax.set_ylim(1000, 100)

f_ax.fill_between(
    lon_ter_s, ter_ver_s, 1010,
    where=ter_ver_s < 1010,
    facecolor='#454545', clip_on=False, zorder=10
)

vec = f_ax.Curlyquiver(
    lon_u_s, U_nc['level'], u_mean_s, w_mean_s*2,
    arrowsize=1.5, scale=30, linewidth=.5, regrid=18,
    color='k', zorder=8.5, MinDistance=[0.4, 0.5], regrid_reso=9
)
vec.key(
    U=5, label='5 m/s (10$^{-2}$ Pa/s)', loc='upper right',
    edgewidth=0, edgecolor='none', arrowsize=6, linewidth=.4,
    intetval=1.1, fontproperties={'size': 9},
    bbox_to_anchor=[-0.02, 0.20, 1, 1]
)

lev_z = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -.2, .2, 1.0, 1.5, 2.0, 2.5, 3.0])

q_div = f_ax.contourf(
    lon_q_s, Q_div_nc['level'], q_mean_s,
    levels=lev_z * 10e-9,
    cmap=cmaps.MPL_RdYlGn_r[22:56] + cmaps.CBR_wet[0] + cmaps.MPL_RdYlGn_r[72:106],
    zorder=8, extend='both'
)
q_div_nf = f_ax.contour(
    lon_q_s, Q_div_nc['level'], q_mean_s,
    levels=lev_z * 10e-9,
    colors='w', linewidths=.2, zorder=8, linestyles='solid'
)

f_pre = fig.add_subplot(spec[1])
f_sst = f_pre.twinx()

f_pre.axhline(y=0, color='#757575', linestyle='--', linewidth=1, zorder=5)
f_pre.plot(lon_pre_s, pre_lon_mean_s, color='green', linewidth=1, label='PRE', zorder=5)
f_sst.plot(lon_sst_s, sst_lon_mean_s, color='blue', linewidth=1, label='SST', zorder=5)

f_pre.set_xlim(TRS_LON, 360+TRS_LON)
f_sst.set_xlim(TRS_LON, 360+TRS_LON)

xticks = [20+TRS_LON, 80+TRS_LON, 140+TRS_LON, 200+TRS_LON, 260+TRS_LON, 320+TRS_LON]
xticklabels = ['120°W', '60°W', '0°', '60°E', '120°E', '180°']

for ax in [f_ax, f_pre]:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


# 对齐 x 轴与上方主图
# f_sst.set_xlim(0, 360)
# f_sst.set_xticks([0, 60, 120, 180, 240, 300, 360])
# f_sst.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
f_pre.set_ylim(-.45, .3)
f_pre.set_yticks([-.4, -.2, 0, .2])
f_pre.set_yticklabels(['-0.4', '-0.2', ' 0   ', ' 0.2'])
f_sst.set_ylim(-.45, .3)
f_sst.set_yticks([-.4, -0.2, 0, 0.2])
f_sst.set_yticklabels(['-0.4', '-0.2', ' 0 ', ' 0.2'])
f_sst.tick_params(axis='y', labelsize=10, width=1, length=8, colors='blue')
f_pre.tick_params(axis='x', labelsize=10, width=1, length=8, colors='black')
f_pre.tick_params(axis='y', labelsize=10, width=1, length=8, colors='green')

f_pre.spines['left'].set_color('green')
f_pre.spines['right'].set_visible(False)
f_sst.spines['right'].set_color('blue')
f_sst.spines['left'].set_visible(False)

# 视需要添加 y 轴标签与图例
f_sst.set_ylabel('', fontsize=11)
f_sst.legend(loc='upper right', bbox_to_anchor=(0.90, 1), fontsize=8, frameon=False)
f_pre.legend(loc='upper right', fontsize=8, frameon=False)
f_sst.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.5)

for spine in f_ax.spines.values():
    spine.set_zorder(100)
    spine.set_linewidth(1)  # 设置边框线宽
for spine in f_sst.spines.values():
    spine.set_linewidth(1)  # 设置边框线宽
for spine in f_pre.spines.values():
    spine.set_linewidth(1)  # 设置边框线宽

# color bar位置
position = fig.add_axes([0.196, 0.00, 0.64, 0.03])
cb1 = plt.colorbar(q_div, cax=position, orientation='horizontal', drawedges=True)#orientation为水平或垂直
cb1.outline.set_edgecolor('black')  # 将colorbar边框调为黑色
cb1.outline.set_linewidth(1.0)  # 加粗 colorbar 边框线宽
cb1.dividers.set_color('black') # 将colorbar内间隔线调为黑色
try:
    cb1.dividers.set_linewidth(1.0)  # 加粗隔断线宽
except Exception:
    # 某些 matplotlib 版本下 dividers 可能不是支持 set_linewidth 的对象，退回到设置每个 divider 的 linewidth
    for d in getattr(cb1, 'dividers', []):
        try:
            d.set_linewidth(1.0)
        except Exception:
            pass
cb1.ax.tick_params(length=0, labelsize=12, width=1)#length为刻度线的长度
cb1.locator = ticker.FixedLocator(lev_z) # colorbar上的刻度值个数
cb1.set_ticks(lev_z * 10e-9)
cb1.set_ticklabels([f'{i}' for i in lev_z])
# cb1.set_label('Divergence(10$^{-9}$ g·m$^{-2}$·s$^{-1}$)', fontsize=16)

# f_ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
# f_ax.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
f_ax.set_xticks([])
f_ax.tick_params(labelsize=10, width=1, length=8)  # length为刻度线的长度
#f_ax.set_yscale('log')
f_ax.set_yticks([1000, 850, 700, 600, 500, 400, 300, 200, 100])
f_ax.set_yticklabels(['1000','850', '700', '600', '500', '400', '300', '200', '100'])

plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r2赤道纬向环流合成归因.png", dpi=600, bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r2赤道纬向环流合成归因.pdf", bbox_inches='tight')
