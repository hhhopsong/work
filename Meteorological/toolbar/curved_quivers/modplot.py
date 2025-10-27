"""
2D 向量场的流线型矢量绘图。
作者
hopsong

Github项目地址
https://github.com/hhhopsong/work/blob/master/Meteorological/toolbar/curved_quivers/modplot.py
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
# 数据处理三方库
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import numpy as np


# 绘图三方库
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib import _api, cm, patches
from matplotlib.streamplot import TerminateTrajectory
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# 辅助三方库
import tqdm as tq
import warnings

# 加速计算三方库
from func_timeout import func_set_timeout, FunctionTimedOut
from numba import njit, prange
import math



class VHead(patches.ArrowStyle._Base):
    """
    一个自定义的 VHead 箭头样式。
    通过实现 transmute 方法来保证兼容性。
    """

    def __init__(self, head_length=0.4, head_width=0.4):
        self.head_length = head_length
        self.head_width = head_width
        super().__init__()

    def transmute(self, path, mutation_size, transform):
        """
        接收原始路径，返回带有箭头的完整新路径。
        这是创建箭头样式的“经典”方法。
        """
        # 获取箭身路径的最后一个点（即箭头的目标位置）和倒数第二个点（用来确定方向）
        x_end, y_end = path.vertices[-1]
        if len(path.vertices) > 1:
            x_start, y_start = path.vertices[-2]
        else:
            x_start, y_start = x_end - 1, y_end
        direction_vec = np.array([x_end - x_start, y_end - y_start])
        norm = np.linalg.norm(direction_vec)
        direction_vec = direction_vec / (norm if norm != 0 else 1)
        arrow_angle_rad = np.arctan2(direction_vec[1], direction_vec[0])
        hl = self.head_length * mutation_size
        hw = self.head_width * mutation_size
        rotation_matrix = np.array([
            [np.cos(arrow_angle_rad), -np.sin(arrow_angle_rad)],
            [np.sin(arrow_angle_rad), np.cos(arrow_angle_rad)]
        ])
        end_point = np.array([x_end, y_end])

        # --- ✨ 核心修改在这里 ---

        # 1. 定义 V 形的三个点（在原点坐标系，尖端朝向原点）
        #    不再需要 gap 和 p1_end, p2_end
        prong1_start_local = np.array([-hl, hw / 2.0])
        vertex_local = np.array([0, 0])  # 交汇的顶点就是原点
        prong2_start_local = np.array([-hl, -hw / 2.0])

        # 2. 旋转并平移这三个点
        prong1_start = np.dot(rotation_matrix, prong1_start_local) + end_point
        vertex = np.dot(rotation_matrix, vertex_local) + end_point  # 这其实就是 end_point
        prong2_start = np.dot(rotation_matrix, prong2_start_local) + end_point

        # 3. 构建新的顶点列表和指令列表
        all_verts = [
            prong1_start,
            vertex,
            prong2_start
        ]

        all_codes = [
            Path.MOVETO,  # 提笔，移动到 V 形一侧的起点
            Path.LINETO,  # 画线到顶点
            Path.LINETO  # 从顶点继续画线到另一侧的起点
        ]

        # 返回新的路径和是否可填充的标志
        return Path(all_verts, all_codes), False
patches.ArrowStyle.register("v", VHead)


def lontransform(data, lon_name='lon', type='180->360'):
    """
    将经纬度从180->360或360->180转换
    Parameters
    ----------
    data : xarray.DataArray
        数据
    lon_name : str
        经度名称
    type : str
        转换类型，180->360或360->180
    Returns
    -------
    data : xarray.DataArray
        转换后的数据
    """
    if type == '180->360':
        data.coords[lon_name] = np.mod(data[lon_name], 360.)
        return data.reindex({lon_name: np.sort(data[lon_name])})
    elif type == '360->180':
        data.coords[lon_name] = data[lon_name].where(data[lon_name] <= 180, data[lon_name] - 360)
        return data.reindex({lon_name: np.sort(data[lon_name])})
    else:
        raise ValueError('type must be 180->360 or 360->180')

@njit(fastmath=True)
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    rad = math.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = ((lon2 - lon1 + 180.0) % 360.0 - 180.0) * rad
    a = math.sin(dlat * 0.5) ** 2 + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon * 0.5) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


# 四邻域反距离加权（基于球面距离）
@njit(fastmath=True, parallel=True)
def _geo_idw_four(field2d, lat_src, lon_src, Yq, Xq, power=1.0, eps=1e-9):
    # 计算各点所在的网格单元索引
    nlat = lat_src.size
    nlon = lon_src.size
    out = np.empty(Yq.size, dtype=np.float64)

    # 展平只为简化并行循环；返回时再 reshape
    y = Yq.ravel()
    x = Xq.ravel()

    for k in prange(y.size):
        yy = y[k]
        xx = x[k]

        # 二分查找所在单元
        i = np.searchsorted(lat_src, yy) - 1
        j = np.searchsorted(lon_src, xx) - 1
        if i < 0: i = 0
        if j < 0: j = 0
        if i > nlat - 2: i = nlat - 2
        if j > nlon - 2: j = nlon - 2

        i0 = i
        i1 = i + 1
        j0 = j
        j1 = j + 1

        lat00 = lat_src[i0]; lon00 = lon_src[j0]
        lat10 = lat_src[i1]; lon10 = lon_src[j0]
        lat01 = lat_src[i0]; lon01 = lon_src[j1]
        lat11 = lat_src[i1]; lon11 = lon_src[j1]

        d00 = _haversine_scalar(yy, xx, lat00, lon00)
        d10 = _haversine_scalar(yy, xx, lat10, lon10)
        d01 = _haversine_scalar(yy, xx, lat01, lon01)
        d11 = _haversine_scalar(yy, xx, lat11, lon11)

        # 精确落在格点上（阈值按米计）
        if d00 < 1e-6:
            out[k] = field2d[i0, j0]
            continue
        if d10 < 1e-6:
            out[k] = field2d[i1, j0]
            continue
        if d01 < 1e-6:
            out[k] = field2d[i0, j1]
            continue
        if d11 < 1e-6:
            out[k] = field2d[i1, j1]
            continue

        w00 = 1.0 / ((d00 + eps) ** power)
        w10 = 1.0 / ((d10 + eps) ** power)
        w01 = 1.0 / ((d01 + eps) ** power)
        w11 = 1.0 / ((d11 + eps) ** power)

        num = (w00 * field2d[i0, j0] +
               w10 * field2d[i1, j0] +
               w01 * field2d[i0, j1] +
               w11 * field2d[i1, j1])
        den = w00 + w10 + w01 + w11
        out[k] = num / den

    return out.reshape(Yq.shape)


def adjust_sub_axes(ax_main, ax_sub, shrink, lr=1.0, ud=1.0, width=1.0, height=1.0):
    '''
    将ax_sub调整到ax_main的右下角.shrink指定缩小倍数。
    当ax_sub是GeoAxes时,需要在其设定好范围后再使用此函数
    :param ax_main: 主图
    :param ax_sub: 子图
    :param shrink: 缩小倍数
    :param lr: 左右间距(>1：左偏, <1：右偏)
    :param ud: 上下间距(>1：上偏, <1：下偏)
    :param width: 宽度缩放比例
    :param height: 高度比例
    '''
    bbox_main = ax_main.get_position()
    bbox_sub = ax_sub.get_position()
    wnew = bbox_main.width * shrink * width
    hnew = bbox_main.height * shrink * height
    bbox_new = mtransforms.Bbox.from_extents(
        bbox_main.x1 - lr*wnew, bbox_main.y0 + (ud-1)*hnew,
        bbox_main.x1 - (lr-1)*wnew, bbox_main.y0 + ud*hnew
    )
    ax_sub.set_position(bbox_new)


class Curlyquiver:
    def __init__(self, ax, x, y, U, V, lon_trunc=None, linewidth=.5, color='black', cmap=None, norm=None, arrowsize=.5,
                 arrowstyle='v', transform=None, zorder=None, start_points='interleaved', scale=1., masked=True, regrid=30,
                 regrid_reso=2.5, integration_direction='both', mode='loose', nanmax=None, center_lon=180., alpha=1.,
                 thinning=['0%', 'min'], MinDistance=[0, 1]):
        """绘制矢量曲线.

            *x*, *y* : 1d arrays
                *规则* 网格.
            *u*, *v* : 2d arrays
                ``x`` 和 ``y`` 方向变量。行数应与 ``y`` 的长度匹配，列数应与 ``x`` 匹配.
            *lon_trunc* : float
                经度截断
            *linewidth* : numeric or 2d array
                给定与速度形状相同的二维阵列，改变线宽。
            *color* : matplotlib color code, or 2d array
                矢量颜色。给定一个与 ``u`` , ``v`` 形状相同的数组时，将使用*cmap*将值转换为*颜色*。
            *cmap* : :class:`~matplotlib.colors.Colormap`
                用于绘制矢量的颜色图。仅在使用*cmap*进行颜色绘制时才需要。
            *norm* : :class:`~matplotlib.colors.Normalize`
                用于将数据归一化。
                如果为 ``None`` ，则将（最小，最大）拉伸到（0,1）。只有当*color*为数组时才需要。
            *arrowsize* : float
                箭头大小
            *arrowstyle* : str
                箭头样式规范。
                详情请参见：:class:`~matplotlib.patches.FancyArrowPatch`.
            *start_points*: Nx2 array
                矢量起绘点的坐标。在数据坐标系中，与 ``x`` 和 ``y`` 数组相同。
                当 ``start_points`` 为 'interleaved' 时，会根据 ``x`` 和 ``y`` 数组自动生成蜂窝状起绘点。
            *zorder* : int
                ``zorder`` 属性决定了绘图元素的绘制顺序,数值较大的元素会被绘制在数值较小的元素之上。
            *scale* : float(0-100)
                矢量的最大长度。
            *masked* : bool
                原数据是否为掩码数组
            *regrid* : int(>=2)
                是否重新插值网格
            *regrid_reso* : float
                重新插值网格分辨率
            *integration_direction* : {'forward', 'backward', 'both'}, default: 'both'
                矢量向前、向后或双向绘制。
            *mode* : {'loose', 'strict'}, default: 'loose'
                流线边界绘制模式.
                'loose': 流线绘制时，线性外拓数据边界(Nan值计为0进行插值).
                'strict': 流线绘制时，严格裁切数据边界.
            *nanmax* : float
                风速单位一
            *center_lon* : float
                中心经度
                中心经度，默认180.
            *alpha* : float(0-1)
                矢量透明度，默认1.
            *thinning* : [float , str]
                float为百分位阈值阈值，长度超出此百分位阈值的流线将不予绘制。
                str为采样方式，'max'、'min'或'range'。
                例如：[10, 'max']，将不予绘制超过10的 streamline。
                例如：[10, 'min']，将不予绘制小于10的 streamline。
                例如：[[10, 20], 'range']，将绘制长度在10~20之间的 streamline。
            *MinDistance* : [float1, float2]
                最小距离阈值。
                float1为最小距离阈值，流线之间的最小距离（格点间距为单位一）.
                float2为重叠部分占总线长的百分比.

            Returns:

                *stream_container* : StreamplotSet
                    具有属性的容器对象

                        - lines: `matplotlib.collections.LineCollection` of streamlines

                        - arrows: collection of `matplotlib.patches.FancyArrowPatch`
                          objects representing arrows half-way along stream
                          lines.

                *unit* : float
                    矢量的单位长度
                *nanmax* : float
                    矢量的最大长度
        """

        self.axes = ax
        self.x = x
        self.y = y
        self.U = U
        self.V = V
        self.lon_trunc = lon_trunc if lon_trunc is not None else center_lon - 180.
        self.linewidth = linewidth
        self.color = color
        self.cmap = cmap
        self.norm = norm
        self.arrowsize = arrowsize
        self.arrowstyle = arrowstyle
        self.transform = transform
        self.zorder = zorder
        self.start_points = start_points
        self.scale = scale
        self.masked = masked
        self.regrid = regrid
        self.regrid_reso = regrid_reso
        self.integration_direction = integration_direction
        self.mode = mode
        self.NanMax = nanmax
        self.center_lon = center_lon
        self.thinning = thinning
        self.MinDistance = MinDistance
        self.alpha = alpha

        self.quiver = self.quiver()
        self.nanmax = self.quiver[2]
    def quiver(self):
        return velovect(self.axes, self.x, self.y, self.U, self.V, self.lon_trunc, self.linewidth, self.color,
                        self.cmap, self.norm, self.arrowsize, self.arrowstyle, self.transform, self.zorder,
                        self.start_points, self.scale, self.masked, self.regrid, self.regrid_reso, self.integration_direction,
                        self.mode, self.NanMax, self.center_lon, self.thinning, self.MinDistance, self.alpha)

    def key(self, fig, U=1., shrink=0.15, angle=0., label='1', lr=1., ud=1., fontproperties={'size': 5},
            width_shrink=1., height_shrink=1., edgecolor='k', arrowsize=None, linewidth=None, color=None):
        '''
        曲线矢量图例
        :param fig: 画布总底图
        :param axes: 目标图层
        :param quiver: 曲线矢量图层
        :param U: 风速
        :param angle: 角度
        :param label: 标签
        :param fontproperties: 字体属性
        :param lr: 左右偏移(>1：左偏, <1：右偏)
        :param ud: 上下偏移(>1：上偏, <1：下偏)
        :param width_shrink: 宽度缩放比例
        :param height_shrink: 高度缩放比例
        :param edgecolor: 边框颜色

        :return: None
        '''
        arrowsize = arrowsize if arrowsize is not None else self.arrowsize
        linewidth = linewidth if linewidth is not None else self.linewidth
        color = color if color is not None else self.color
        velovect_key(fig, self.axes, self.quiver, shrink, U, angle, label, color=color, arrowstyle=self.arrowstyle,
                     linewidth=linewidth, fontproperties=fontproperties, lr=lr, ud=ud, width_shrink=width_shrink,
                     height_shrink=height_shrink, arrowsize=arrowsize, edgecolor=edgecolor)


def velovect(axes, x, y, u, v, lon_trunc=0., linewidth=.5,    color='black',
               cmap=None,      norm=None,    arrowsize=.5,    arrowstyle='v',
               transform=None, zorder=None,  start_points= 'interleaved',
               scale=100.,     masked=True,  regrid=30,       regrid_reso=2.5, integration_direction='both',
               mode='loose',   nanmax=None,  center_lon=180., thinning=[1, 'random'],   MinDistance=[0.1, 0.5],
               alpha=1.,       latlon_zoom='True'):
    """绘制矢量曲线"""

    # 填充nan值为0
    if masked:
        u = np.where(np.isnan(u), 0, u)
        v = np.where(np.isnan(v), 0, v)

    # 检查y是否升序
    if y[0] > y[-1]:
        warnings.warn('已将Y轴反转，因为Y轴坐标轴为非增长序列。', UserWarning)
        y = y[::-1]
        u = u[::-1]
        v = v[::-1]

    # 数据类型转化
    try:
        if isinstance(x, xr.DataArray):
            x = x.data
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError('x 的数据类型必须是 xarray.DataArray 或者 numpy.ndarray')
    except:
        pass
    try:
        if isinstance(y, xr.DataArray):
            y = y.data
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise ValueError('y 的数据类型必须是 xarray.DataArray 或者 numpy.ndarray')
    except:
        pass
    try:
        if isinstance(u, xr.DataArray):
            u = u.data
        elif isinstance(u, np.ndarray):
            pass
        else:
            raise ValueError('u 的数据类型必须是 xarray.DataArray 或者 numpy.ndarray')
    except:
        pass
    try:
        if isinstance(v, xr.DataArray):
            v = v.data
        elif isinstance(v, np.ndarray):
            pass
        else:
            raise ValueError('v 的数据类型必须是 xarray.DataArray 或者 numpy.ndarray')
    except:
        pass

    #检验center_lon范围
    if center_lon < -180 or center_lon > 360:
        raise ValueError('center_lon 的范围必须在-180~360之间。')
    center_lon = center_lon + 360 if center_lon < 0 else center_lon


    # 获取axes范围
    try:
        extent = axes.get_extent()
        extent = np.array(extent)
        extent[0] = extent[0] + center_lon
        extent[1] = extent[1] + center_lon
        MAP = True
    except AttributeError:
        extent = axes.get_xlim() + axes.get_ylim()
        MAP = False

    # 检测坐标系是否为非线性（如对数坐标）
    is_x_log = False
    is_y_log = False
    try:
        # 检查坐标轴的scale属性
        x_scale = axes.xaxis.get_scale()
        y_scale = axes.yaxis.get_scale()
        is_x_log = (x_scale == 'log')
        is_y_log = (y_scale == 'log')
    except AttributeError:
        pass

    if MAP:
        # 经纬度重排列为-180~0~180
        u = xr.DataArray(u, coords={'lat': y, 'lon': x}, dims=['lat', 'lon'])
        v = xr.DataArray(v, coords={'lat': y, 'lon': x}, dims=['lat', 'lon'])
        if x[-1] > 180:
            u = lontransform(u, lon_name='lon', type='360->180')
            v = lontransform(v, lon_name='lon', type='360->180')
        x = u.lon
        y = u.lat
        u = u.data
        v = v.data
        # 环球插值
        if (-90 < y[0]) or (90 > y[-1]):
            warnings.warn('高纬地区数据缺测，已进行延拓(fill np.nan)', UserWarning)
            bound_err = False
        else:
            bound_err = True
        if x[0] + 360 == x[-1] or np.abs(x[0] - x[-1] < 1e-4):
            # 同时存在-180和180则除去180
            u = u[:, :-1]
            v = v[:, :-1]
            x = x[:-1]
            u = np.concatenate([u, u, u], axis=1)
            v = np.concatenate([v, v, v], axis=1)
            x = np.concatenate([x - 360, x, x + 360])
            u_global_interp = RegularGridInterpolator((y, x), u, method='linear', bounds_error=bound_err)
            v_global_interp = RegularGridInterpolator((y, x), v, method='linear', bounds_error=bound_err)
        else:
            u = np.concatenate([u, u, u], axis=1)
            v = np.concatenate([v, v, v], axis=1)
            x = np.concatenate([x - 360, x, x + 360])
            u_global_interp = RegularGridInterpolator((y, x), u, method='linear', bounds_error=bound_err)
            v_global_interp = RegularGridInterpolator((y, x), v, method='linear', bounds_error=bound_err)

        x_1degree = np.arange(-181, 181.5, 1)
        y_1degree = np.arange(-90, 90.5, 1)
        cent_int = center_lon//1
        cent_flt = center_lon%1
        X_1degree_cent, Y_1degree = np.meshgrid((x_1degree + cent_int + cent_flt)[::int(regrid_reso//1)], y_1degree[::int(regrid_reso//1)])
        u_1degree = u_global_interp((Y_1degree, X_1degree_cent))
        v_1degree = v_global_interp((Y_1degree, X_1degree_cent))
    else:
        x_1degree = x
        y_1degree = y
        cent_flt = center_lon%regrid_reso
        u_1degree = u
        v_1degree = v

    REGRID_LEN = 1 if isinstance(regrid, int) else len(regrid)
    if regrid:
        # 将网格插值为正方形等间隔网格
        if MAP:
            x = np.arange(-180, 180 + regrid_reso/2, regrid_reso)
            y = np.arange(-89, 89 + regrid_reso/2, regrid_reso)
            U = RegularGridInterpolator(
                (y_1degree[::int(regrid_reso // 1)], (x_1degree + cent_flt)[::int(regrid_reso // 1)]), u_1degree,
                method='linear', bounds_error=True)
            V = RegularGridInterpolator(
                (y_1degree[::int(regrid_reso // 1)], (x_1degree + cent_flt)[::int(regrid_reso // 1)]), v_1degree,
                method='linear', bounds_error=True)
        else:
            x = np.arange(x[0], x[-1] + 1e-5, regrid_reso)
            y = np.arange(y[0], y[-1] + 1e-5, regrid_reso*5.5556)
            U = RegularGridInterpolator(
                (y_1degree, (x_1degree + cent_flt)), u_1degree,
                method='linear', bounds_error=True)
            V = RegularGridInterpolator(
                (y_1degree, (x_1degree + cent_flt)), v_1degree,
                method='linear', bounds_error=True)
        ## 裁剪绘制区域的数据->得到正确的regird
        if REGRID_LEN == 2:
            regrid_x = regrid[0]
            regrid_y = regrid[1]
        else:
            regrid_x = regrid
            regrid_y = regrid

        x_delta = np.linspace(x[0], x[-1], regrid_x, retstep=True)[1]
        y_delta = np.linspace(y[0], y[-1], regrid_y, retstep=True)[1]

        # 重新插值
        if is_x_log or is_y_log:
            X, Y = np.meshgrid(x, y)
            # 对数坐标下使用更安全的插值方法
            u = U((Y, X))
            v = V((Y, X))
        elif REGRID_LEN == 2:
            X, Y = np.meshgrid(x, y)
            u = U((Y, X))
            v = V((Y, X))
        elif x_delta < y_delta:
            X, Y = np.meshgrid(x, y)
            u = U((Y, X))
            v = V((Y, X))
            if MAP: zone_scale = np.abs(extent[0] - extent[1]) / np.abs(x[0] - x[-1]) # 区域裁剪对风矢的缩放比例
        else:
            X, Y = np.meshgrid(x, y)
            u = U((Y, X))
            v = V((Y, X))
            if MAP: zone_scale = np.abs(extent[2] - extent[3]) / np.abs(y[0] - y[-1]) # 区域裁剪对风矢的缩放比例
    else:
        raise ValueError('regrid 必须为非零整数')

    # 风速归一化
    wind = np.ma.sqrt(u ** 2 + v ** 2)     # scale缩放
    nanmax = np.nanmax(wind) if nanmax == None else nanmax
    if MAP: wind_shrink = 1 / nanmax / scale * zone_scale
    else: wind_shrink = 1 / nanmax / scale
    u = u * wind_shrink
    v = v * wind_shrink

    if regrid_x * regrid_y >= 2000: warnings.warn('流线绘制格点过多，可能导致计算速度过慢!', RuntimeWarning)
    _api.check_in_list(['both', 'forward', 'backward'], integration_direction=integration_direction)
    grains = 1
    # 由于对数坐标，在此对对应对数坐标进行处理
    if is_x_log: x = np.log10(x)
    if is_y_log: y = np.log10(y)
    grid = Grid(x, y)
    mask = StreamMask(10)
    dmap = DomainMap(grid, mask)

    if zorder is None:
        zorder = mlines.Line2D.zorder

    # default to data coordinates
    if transform is None:
        transform = axes.transData

    if color is None:
        color = axes._get_lines.get_next_color()

    if linewidth is None:
        linewidth = matplotlib.rcParams['lines.linewidth']

    line_kw = {}
    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        if color.shape != grid.shape:
            raise ValueError(
                "如果 'color' 参数被设定, 则其数据维度必须和 'Grid(x,y)' 相同")
        line_colors = []
        color = np.ma.masked_invalid(color)
    else:
        line_kw['color'] = color
        arrow_kw['color'] = color

    if isinstance(linewidth, np.ndarray):
        if linewidth.shape != grid.shape:
            raise ValueError(
                "如果 'linewidth' 参数被设定, 则其数据维度必须和 'Grid(x,y)' 相同")
        line_kw['linewidth'] = []
    else:
        line_kw['linewidth'] = linewidth
        arrow_kw['linewidth'] = linewidth

    line_kw['zorder'] = zorder
    arrow_kw['zorder'] = zorder

    ## Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape:
        raise ValueError("'u' 和 'v' 的维度必须和 'Grid(x,y)' 相同")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)
    magnitude = np.ma.sqrt(u**2 + v**2)
    #magnitude /= np.nanmax(magnitude)
	
    resolution = 47666666e-8 # 分辨率(最小可分辨度为47666666e-8)
    minlength = .9*resolution
    integrate = get_integrator(u, v, dmap, minlength, resolution, magnitude, integration_direction=integration_direction, mode=mode, axes_scale=[is_x_log, is_y_log])
    trajectories = []
    edges = []

    ## 生成绘制网格
    if is_x_log:
        # 对数坐标下使用对数等间距点
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        if x_min <= 0:  # 对数坐标不能有负值或零
            x_min = np.min(x[x > 0])
        x_draw = np.logspace(np.log10(x_min), np.log10(x_max), regrid_x)

    if is_y_log:
        # 对数坐标下使用对数等间距点
        y_min, y_max = np.nanmin(y), np.nanmax(y)
        if y_min <= 0:  # 对数坐标不能有负值或零
            y_min = np.min(y[y > 0])
        y_draw = np.logspace(np.log10(y_min), np.log10(y_max), regrid_y)

    if not is_x_log and not is_y_log:
        # 只有在两个轴都是线性时才应用原来的逻辑
        if MAP:
            x_draw_delta = np.linspace(extent[0], extent[1], regrid_x, retstep=True)[1]
            y_draw_delta = np.linspace(extent[2], extent[3], regrid_y, retstep=True)[1]
            if REGRID_LEN == 2:
                x_draw = np.arange(extent[0] - center_lon + x_draw_delta / 2, extent[1] - center_lon, x_draw_delta)
                y_draw = np.arange(extent[2] + y_draw_delta / 2, extent[3], y_draw_delta)
            elif x_draw_delta < y_draw_delta:
                x_draw = np.arange(extent[0] - center_lon + x_draw_delta / 2, extent[1] - center_lon, x_draw_delta)
                y_draw = np.arange(extent[2] + x_draw_delta / 2, extent[3], x_draw_delta)
            else:
                x_draw = np.arange(extent[0] - center_lon + y_draw_delta / 2, extent[1] - center_lon, y_draw_delta)
                y_draw = np.arange(extent[2] + y_draw_delta / 2, extent[3], y_draw_delta)
        else:
            x_draw = x
            y_draw = y

    # 处理超出-180~180范围的经度
    if MAP:
        x_draw = np.where(x_draw > 180, x_draw - 360, x_draw)
        x_draw = np.where(x_draw < -180, x_draw + 360, x_draw)

    if start_points is None:
        if regrid:
            X_re, Y_re = np.meshgrid(x_draw, y_draw)
            start_points = np.array([X_re.flatten(), Y_re.flatten()]).T
        else:
            start_points=_gen_starting_points(x,y,grains)
    elif start_points == 'interleaved':
        if regrid:
            X_re, Y_re = np.meshgrid(x_draw, y_draw)
            if len(x_draw) > 1 and len(y_draw) > 1:
                horizontal_shift = (x_draw[1] - x_draw[0]) / 2.0
                X_re[1::2] += horizontal_shift
                mask_ = np.ones_like(X_re, dtype=bool)
                mask_[1::2, -1] = False
                X_re = X_re[mask_]
                Y_re = Y_re[mask_]
            start_points = np.array([X_re.flatten(), Y_re.flatten()]).T
        else:
            warnings.warn('绘制点未成功插值为六边形: start_points 为 "interleaved" 时, regrid 的值必须非 False', UserWarning)
            start_points=_gen_starting_points(x,y,grains)


    sp2 = np.asanyarray(start_points, dtype=float).copy()
    # 检查start_points是否在数据边界之外
    for xs, ys in sp2:
        if not (grid.x_origin <= xs <= grid.x_origin + grid.width
                and grid.y_origin <= ys <= grid.y_origin + grid.height):
            if (np.abs(xs - grid.x_origin) < 1e-8 or np.abs(xs - grid.x_origin - grid.width) < 1e-8
                    or np.abs(ys - grid.y_origin) < 1e-8 or np.abs(ys - grid.y_origin - grid.height) < 1e-8):
                warnings.warn(f"起绘点 ({xs}, {ys}) 位于数据边界上，可能会导致路径积分失败。", UserWarning)
            else:
                raise ValueError("起绘点 ({}, {}) 超出数据边界".format(xs, ys))

    # Convert start_points from data to array coords
    # Shift the seed points from the bottom left of the data so that
    # data2grid works properly.
    sp2[:, 0] -= grid.x_origin
    sp2[:, 1] -= grid.y_origin

    @func_set_timeout(1)
    def integrate_timelimit(xg, yg):
        return integrate(xg, yg)

    traj_length = []
    for xs, ys in tq.tqdm(sp2, desc='路径积分', colour='green', unit='points', total=len(sp2)):
        xg, yg = dmap.data2grid(xs, ys)
        xg = np.clip(xg, 0, grid.nx - 1)
        yg = np.clip(yg, 0, grid.ny - 1)
        try:
            integrate_ = integrate_timelimit(xg, yg)
        except FunctionTimedOut:
            print(f"({xg}, {yg})流线绘制超时，已自动跳过该流线.")
            continue
        t = integrate_[0:2] if integrate_[0][0] is not None else None
        if t is not None:
            trajectories.append(t[0])
            edges.append(t[1])
            D = distance(t[0][0], t[0][1]) if ~np.isnan(distance(t[0][0], t[0][1])) else 0
            traj_length.append(D)

    # 稀疏化
    from operator import itemgetter
    combined = list(zip(traj_length, trajectories, edges))
    combined.sort(key=itemgetter(0), reverse=True)  # 按第 0 个元素（traj_length）降序
    traj_length, trajectories, edges = map(list, zip(*combined))

    # 稀疏化
    if thinning[0] != 1:
        len_index = len(traj_length)
        index0 = 0
        index1 = len_index
        wind_to_traj_length = traj_length[0] / np.nanmax(np.ma.sqrt(u ** 2 + v ** 2))
        if thinning[1] == 'range':  #################### 取整 ####################
            if isinstance(thinning[0][0], str):
                if thinning[0][0][-1] == "%":
                    index1 = int((1 - eval((thinning[0][0][:-1])) / 100) * len_index)
                    index0 = int((1 - eval((thinning[0][1][:-1])) / 100) * len_index)
                else:
                    raise ValueError('thinning 的两个参数必须为 0 到 1 间的值, 或 0% 到 100% 间的百分比')
            else:
                thres1 = thinning[0][0] * wind_shrink * wind_to_traj_length
                index1 = np.where(np.array(traj_length) >= thres1)[0][0]
                thres0 = thinning[0][1] * wind_shrink * wind_to_traj_length
                index0 = np.where(np.array(traj_length) <= thres0)[0][0]
        elif thinning[1] == 'max':
            if isinstance(thinning[0], str):
                if thinning[0][-1] == "%":
                    index0 = int((1 - eval(thinning[0][:-1]) / 100) * len_index)
                else:
                    raise ValueError('thinning 的第一个参数必须为 0 到 1 间的值, 或 0% 到 100% 间的百分比')
            else:
                thres1 = thinning[0] * wind_shrink * wind_to_traj_length
                index1 = np.where(np.array(traj_length) <= thres1)[0][0]
        elif thinning[1] == 'min':
            if isinstance(thinning[0], str):
                if thinning[0][-1] == "%":
                    index1 = int((1 - eval(thinning[0][:-1]) / 100) * len_index)
                else:
                    raise ValueError('thinning 的第一个参数必须为 0 到 1 间的值, 或 0% 到 100% 间的百分比')
            else:
                thres0 = thinning[0] * wind_shrink * wind_to_traj_length
                index0 = np.where(np.array(traj_length) >= thres0)[0][0]
        # 得到白化后的轨迹
        trajectories = trajectories[index0:index1]
        edges = edges[index0:index1]
        traj_length = traj_length[index0:index1]

    if MinDistance[0] > 0 and MinDistance[1] < 1:
        distance_limit_tlen = []
        distance_limit_traj = []
        distance_limit_edges = []
        for i in range(len(trajectories)):
            if np.isnan(traj_length[i]) or np.isinf(traj_length[i]):
                continue
            if i == 0:
                distance_limit_tlen.append(traj_length[i])
                distance_limit_traj.append(trajectories[i])
                distance_limit_edges.append(edges[i])
            else:
                add_signl = True
                for i_in in range(len(distance_limit_traj)):
                    too_close_percent = traj_overlap(trajectories[i], distance_limit_traj[i_in], MinDistance[0])[0]
                    if too_close_percent >= MinDistance[1]:
                        add_signl = False
                        break
                if add_signl:
                    distance_limit_tlen.append(traj_length[i])
                    distance_limit_traj.append(trajectories[i])
                    distance_limit_edges.append(edges[i])
        traj_length, trajectories, edges = distance_limit_tlen, distance_limit_traj, distance_limit_edges


    # 单位
    try:
        unit = 1 / nanmax / scale
    except:
        unit = np.nan
        warnings.warn('格点与投影转换有误,矢量单位将不会绘制!', UserWarning)

    if use_multicolor_lines:
        if norm is None:
            norm = mcolors.Normalize(color.min(), color.max())
        if cmap is None:
            cmap = cm.get_cmap(matplotlib.rcParams['image.cmap'])
        else:
            cmap = cm.get_cmap(cmap)

    streamlines = []
    arrows = []
    t_len_max = np.nanmax(traj_length)
    for t_len, t, edge in zip(traj_length, trajectories, edges):
        tgx = np.array(t[0])
        tgy = np.array(t[1])
		
        # 从网格坐标重新缩放为数据坐标
        tx, ty = dmap.grid2data(*np.array(t))
        tx += grid.x_origin
        ty += grid.y_origin

        # 对对数坐标进行解码处理
        if is_x_log: tx = 10 ** tx
        if is_y_log: ty = 10 ** ty

        points = np.transpose([tx, ty]).reshape(-1, 1, 2)
        streamlines.extend(np.hstack([points[:-1], points[1:]]))

        # Add arrows half way along each trajectory.
        s = np.cumsum(np.sqrt(np.diff(tx) ** 2 + np.diff(ty) ** 2))
        # 箭头方向平滑
        # flit_index = len(tx) // 15 + 1
        flit_index = 5
        if len(tx) <= 10:
            flit_index = 5
        for i in range(flit_index):
            try:
                n = np.searchsorted(s, s[-(flit_index - i)])
                break
            except:
                continue
        arrow_tail = (tx[-1], ty[-1])
        arrow_head = (tx[-2], ty[-2])

        arrow_sizes = (0.35 + 0.65 * np.log(((np.e-1) * t_len / t_len_max) + 1)) * arrowsize
        arrow_kw['mutation_scale'] = 10 * arrow_sizes

        if isinstance(linewidth, np.ndarray):
            line_widths = interpgrid(linewidth, tgx, tgy, masked=masked, mode=mode)[:-1]
            line_kw['linewidth'].extend(line_widths)
            arrow_kw['linewidth'] = line_widths[n]

        if use_multicolor_lines:
            color_values = interpgrid(color, tgx, tgy, masked=masked, mode=mode)[:-1]
            line_colors.append(color_values)
            arrow_kw['color'] = cmap(norm(color_values[n]))
        
        if not edge:
            if MAP:
                p = patches.FancyArrowPatch(
                    arrow_head, arrow_tail, transform=transform, **arrow_kw)
            else:
                # 将数据坐标转换为显示坐标
                display_coords_head = axes.transData.transform(np.array([arrow_head]))
                display_coords_tail = axes.transData.transform(np.array([arrow_tail]))

                # 计算方向向量
                direction = display_coords_head[0] - display_coords_tail[0]
                if np.sqrt(np.sum(direction**2)) == 0:
                    continue  # 避免零长度向量

                # 标准化方向向量
                direction = direction / np.sqrt(np.sum(direction**2))

                # 设置箭头长度为arrowsize的倍数（这里使用10作为基础倍数，可以根据需要调整）
                arrow_length = 1 * arrowsize

                # 计算新的箭头尾部坐标（头部保持不变）
                new_tail_display = display_coords_head[0] - direction * arrow_length

                # 将显示坐标转回数据坐标
                new_coords_data = axes.transData.inverted().transform(
                    np.vstack([display_coords_head[0], new_tail_display]))

                arrow_head_visual = new_coords_data[0].tolist()
                arrow_tail_visual = new_coords_data[1].tolist()

                # 使用视觉一致的坐标创建箭头
                p = patches.FancyArrowPatch(
                    arrow_head_visual, arrow_tail_visual, transform=transform, **arrow_kw)
        else:
            continue
        
        # ds = np.sqrt((arrow_tail[0]-arrow_head[0])**2+(arrow_tail[1]-arrow_head[1])**2)
        # if ds<1e-15: continue  # 移除极小的箭头
        p.set_alpha(alpha)
        axes.add_patch(p)
        arrows.append(p)

    if alpha>=.999:
        lc = mcollections.LineCollection(
            streamlines, transform=transform, capstyle='round', **line_kw)
        lc.sticky_edges.x[:] = [grid.x_origin, grid.x_origin + grid.width]
        lc.sticky_edges.y[:] = [grid.y_origin, grid.y_origin + grid.height]
        if use_multicolor_lines:
            lc.set_array(np.ma.hstack(line_colors))
            lc.set_cmap(cmap)
            lc.set_norm(norm)
        axes.add_collection(lc)
    else:
        # this part is powered by GPT5
        # streamlines: list of arrays, 每个 array 是 (N_i, 2) 的坐标点
        verts, codes = [], []
        for sl in streamlines:
            sl = np.asarray(sl)
            if sl.size == 0:
                continue
            verts.append(sl[0])
            codes.append(Path.MOVETO)
            verts.extend(sl[1:])
            codes.extend([Path.LINETO] * (len(sl) - 1))

        path = Path(np.asarray(verts, float), codes)
        patch = PathPatch(
            path,
            facecolor=line_kw.get("color", "C0"),
            edgecolor=line_kw.get("color", "C0"),
            lw=line_kw.get("linewidth", 1.0),
            capstyle='round',
            joinstyle='round',
            transform=transform,
            alpha=alpha
        )

        patch.sticky_edges.x[:] = [grid.x_origin, grid.x_origin + grid.width]
        patch.sticky_edges.y[:] = [grid.y_origin, grid.y_origin + grid.height]
        axes.add_patch(patch)

    axes.autoscale_view()

    ac = mcollections.PatchCollection(arrows)
    stream_container = StreamplotSet(lc, ac) if alpha>=.999 else StreamplotSet(patch, ac)
    return stream_container, unit, nanmax

	
class StreamplotSet(object):

    def __init__(self, lines, arrows, **kwargs):
        self.lines = lines
        self.arrows = arrows


# Coordinate definitions
# ========================
class DomainMap(object):
    """Map representing different coordinate systems.

    Coordinate definitions:

    * axes-coordinates goes from 0 to 1 in the domain.
    * data-coordinates are specified by the input x-y coordinates.
    * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,
      where N and M match the shape of the input data.
    * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,
      where N and M are user-specified to control the density of streamlines.

    This class also has methods for adding trajectories to the StreamMask.
    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g., if the trajectory is very short) just call `undo_trajectory`.
    """

    def __init__(self, grid, mask):
        self.grid = grid
        self.mask = mask
        # Constants for conversion between grid- and mask-coordinates
        self.x_grid2mask = (mask.nx - 1) / grid.nx
        self.y_grid2mask = (mask.ny - 1) / grid.ny

        self.x_mask2grid = 1. / self.x_grid2mask
        self.y_mask2grid = 1. / self.y_grid2mask

        self.x_data2grid = 1. / grid.dx
        self.y_data2grid = 1. / grid.dy

    def grid2mask(self, xi, yi):
        """Return nearest space in mask-coords from given grid-coords."""
        return (int((xi * self.x_grid2mask) + 0.5),
                int((yi * self.y_grid2mask) + 0.5))

    def mask2grid(self, xm, ym):
        return xm * self.x_mask2grid, ym * self.y_mask2grid

    def data2grid(self, xd, yd):
        return xd * self.x_data2grid, yd * self.y_data2grid

    def grid2data(self, xg, yg):
        return xg / self.x_data2grid, yg / self.y_data2grid

    def start_trajectory(self, xg, yg):
        xm, ym = self.grid2mask(xg, yg)
        self.mask._start_trajectory(xm, ym)

    def reset_start_point(self, xg, yg):
        xm, ym = self.grid2mask(xg, yg)
        self.mask._current_xy = (xm, ym)

    def update_trajectory(self, xg, yg):
        
        xm, ym = self.grid2mask(xg, yg)
        #self.mask._update_trajectory(xm, ym)

    def undo_trajectory(self):
        self.mask._undo_trajectory()
        

class Grid(object):
    """Grid of data."""
    def __init__(self, x, y):

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            x_row = x[0, :]
            if not np.allclose(x_row, x):
                raise ValueError("The rows of 'x' must be equal")
            x = x_row
        else:
            raise ValueError("'x' can have at maximum 2 dimensions")

        if y.ndim == 1:
            pass
        elif y.ndim == 2:
            y_col = y[:, 0]
            if not np.allclose(y_col, y.T):
                raise ValueError("The columns of 'y' must be equal")
            y = y_col
        else:
            raise ValueError("'y' can have at maximum 2 dimensions")

        self.nx = len(x)
        self.ny = len(y)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.x_origin = np.nanmin(x)
        self.y_origin = np.nanmin(y)

        self.width = np.nanmax(x) - np.nanmin(x)
        self.height = np.nanmax(y) - np.nanmin(y)

    @property
    def shape(self):
        return self.ny, self.nx

    def within_grid(self, xi, yi):
        """Return True if point is a valid index of grid."""
        # Note that xi/yi can be floats; so, for example, we can't simply check
        # `xi < self.nx` since `xi` can be `self.nx - 1 < xi < self.nx`
        return xi >= 0 and xi <= self.nx - 1 and yi >= 0 and yi <= self.ny - 1

class StreamMask(object):
    """Mask to keep track of discrete regions crossed by streamlines.

    The resolution of this grid determines the approximate spacing between
    trajectories. Streamlines are only allowed to pass through zeroed cells:
    When a streamline enters a cell, that cell is set to 1, and no new
    streamlines are allowed to enter.
    """

    def __init__(self, density):
        if np.isscalar(density):
            if density <= 0:
                raise ValueError("If a scalar, 'density' must be positive")
            self.nx = self.ny = int(30 * density)
        else:
            if len(density) != 2:
                raise ValueError("'density' can have at maximum 2 dimensions")
            self.nx = int(30 * density[0])
            self.ny = int(30 * density[1])
        self._mask = np.zeros((self.ny, self.nx))
        self.shape = self._mask.shape

        self._current_xy = None

    def __getitem__(self, *args):
        return self._mask.__getitem__(*args)

    def _start_trajectory(self, xm, ym):
        """Start recording streamline trajectory"""
        self._traj = []
        self._update_trajectory(xm, ym)

    def _undo_trajectory(self):
        """Remove current trajectory from mask"""
        for t in self._traj:
            self._mask.__setitem__(t, 0)

    def _update_trajectory(self, xm, ym):
        """Update current trajectory position in mask.

        If the new position has already been filled, raise `InvalidIndexError`.
        """
        #if self._current_xy != (xm, ym):
        #    if self[ym, xm] == 0:
        self._traj.append((ym, xm))
        self._mask[ym, xm] = 1
        self._current_xy = (xm, ym)
        #    else:
        #        raise InvalidIndexError


# Integrator definitions
#========================
def get_integrator(u, v, dmap, minlength, resolution, magnitude, integration_direction='both', masked=True, mode='loose', axes_scale=[False, False]):
    axes_scale = axes_scale

    # rescale velocity onto grid-coordinates for integrations.
    u, v = dmap.data2grid(u, v)

    # speed (path length) will be in axes-coordinates
    u_ax = u / dmap.grid.nx
    v_ax = v / dmap.grid.ny
    speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2)

    if integration_direction == 'both':
        speed = speed / 2.

    def forward_time(xi, yi):
        ds_dt = interpgrid(speed, xi, yi, masked=masked, mode=mode, axes_scale=axes_scale)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        ui = interpgrid(u, xi, yi, masked=masked, mode=mode, axes_scale=axes_scale)
        vi = interpgrid(v, xi, yi, masked=masked, mode=mode, axes_scale=axes_scale)
        return ui * dt_ds, vi * dt_ds

    def backward_time(xi, yi):
        dxi, dyi = forward_time(xi, yi)
        return -dxi, -dyi

    def integrate(x0, y0):
        """Return x, y grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary
        or when it crosses into an already occupied cell in the StreamMask. The
        resulting trajectory is None if it is shorter than `minlength`.
        """

        stotal, x_traj, y_traj, m_total = 0., [], [], []

        
        dmap.start_trajectory(x0, y0)

        if integration_direction in ['both', 'backward']:
            stotal_, x_traj_, y_traj_, m_total_, hit_edge = _integrate_rk12(x0, y0, dmap, backward_time, resolution, magnitude, axes_scale=[False, False])
            stotal += stotal_
            x_traj += x_traj_[::-1]
            y_traj += y_traj_[::-1]
            m_total += m_total_[::-1]

        if integration_direction in ['both', 'forward']:
            dmap.reset_start_point(x0, y0)
            stotal_, x_traj_, y_traj_, m_total_, hit_edge = _integrate_rk12(x0, y0, dmap, forward_time, resolution, magnitude, axes_scale=[False, False])
            stotal += stotal_
            x_traj += x_traj_[1:]
            y_traj += y_traj_[1:]
            m_total += m_total_[1:]

        if len(x_traj)>1:
            return (x_traj, y_traj), hit_edge, m_total, stotal
        else:  # reject short trajectories
            dmap.undo_trajectory()
            return (None, None), hit_edge, m_total, stotal

    return integrate

def _integrate_rk12(x0, y0, dmap, f, resolution, magnitude, masked=True, mode='loose', axes_scale=[False, False]):
    """2nd-order Runge-Kutta algorithm with adaptive step size.

    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:

    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.

    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.

    This integrator is about 1.5 - 2x as fast as both the RK4 and RK45
    solvers in most setups on my machine. I would recommend removing the
    other two to keep things simple.
    """
    # This error is below that needed to match the RK4 integrator. It
    # is set for visual reasons -- too low and corners start
    # appearing ugly and jagged. Can be tuned.
    maxerror = 2e-4

    # This limit is important (for all integrators) to avoid the
    # trajectory skipping some mask cells. We could relax this
    # condition if we use the code which is commented out below to
    # increment the location gradually. However, due to the efficient
    # nature of the interpolation, this doesn't boost speed by much
    # for quite a bit of complexity.
    maxds = min(1. / dmap.mask.nx, 1. / dmap.mask.ny, 6e-4)

    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    xf_traj = []
    yf_traj = []
    m_total = []
    hit_edge = False
    axes_scale = axes_scale
    
    while dmap.grid.within_grid(xi, yi):
        xf_traj.append(xi)
        yf_traj.append(yi)
        m_total.append(interpgrid(magnitude, xi, yi, masked=masked, mode=mode, axes_scale=axes_scale))
        try:
            k1x, k1y = f(xi, yi)
            k2x, k2y = f(xi + ds * k1x,
                         yi + ds * k1y)
        except IndexError:
            # Out of the domain on one of the intermediate integration steps.
            # Take an Euler step to the boundary to improve neatness.
            # 在其中一个中间集成步骤中脱离域。向边界迈出欧拉步以提高整洁度。
            ds, xf_traj, yf_traj = _euler_step(xf_traj, yf_traj, dmap, f)
            stotal += ds
            hit_edge = True
            break
        except TerminateTrajectory:
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)

        nx, ny = dmap.grid.shape
        # Error is normalized to the axes coordinates
        error = np.sqrt(((dx2 - dx1) / nx) ** 2 + ((dy2 - dy1) / ny) ** 2)

        # Only save step if within error tolerance
        if error < maxerror:
            xi += dx2
            yi += dy2
            
            if not dmap.grid.within_grid(xi, yi):
                hit_edge=False
            
            if (stotal + ds) > resolution*np.mean(m_total):
                s_remaining = resolution*np.mean(m_total) - stotal
                fraction = s_remaining / ds
                if (fraction-1) < -1:
                    break  # 防止出现负值导致负步长
                # 按比例缩放最后一步的位移
                xi += dx2 * (fraction-1)
                yi += dy2 * (fraction-1)
                dmap.update_trajectory(xi, yi)
                if not dmap.grid.within_grid(xi, yi):
                    hit_edge = False
                # 将总长度精确地更新到目标值
                stotal += s_remaining
                # 将这个精确的终点加入轨迹
                xf_traj.append(xi)
                yf_traj.append(yi)
                m_total.append(interpgrid(magnitude, xi, yi, masked=masked, mode=mode, axes_scale=axes_scale))
                break

            dmap.update_trajectory(xi, yi)
            stotal += ds

        # recalculate stepsize based on step error
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)

    return stotal, xf_traj, yf_traj, m_total, hit_edge


def _euler_step(xf_traj, yf_traj, dmap, f):
    """Simple Euler integration step that extends streamline to boundary."""
    ny, nx = dmap.grid.shape
    xi = xf_traj[-1]
    yi = yf_traj[-1]
    cx, cy = f(xi, yi)
    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = xi / -cx
    else:
        dsx = (nx - 1 - xi) / cx
    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = yi / -cy
    else:
        dsy = (ny - 1 - yi) / cy
    ds = min(dsx, dsy)
    xf_traj.append(xi + cx * ds)
    yf_traj.append(yi + cy * ds)
    return ds, xf_traj, yf_traj


# 实用功能
# ========================
#####################插值过于宽松，增加一般模式，引入nan值，不将nan值看作0#####################
def interpgrid(a, xi, yi, masked=True, mode='loose', axes_scale=[False, False]):
    """Fast 2D, linear interpolation on an integer grid/整数网格上的快速二维线性插值"""

    Ny, Nx = np.shape(a)
    if isinstance(xi, np.ndarray):
        x = xi.astype(int)
        y = yi.astype(int)
        # Check that xn, yn don't exceed max index
        xn = np.clip(x + 1, 0, Nx - 1)
        yn = np.clip(y + 1, 0, Ny - 1)
    else:
        x = int(xi)
        y = int(yi)
        # conditional is faster than clipping for integers
        if x == (Nx - 1):
            xn = x
        else:
            xn = x + 1
        if y == (Ny - 1):
            yn = y
        else:
            yn = y + 1

    a00 = a[y, x]
    a01 = a[y, xn]
    a10 = a[yn, x]
    a11 = a[yn, xn]
    xt = xi - x if not axes_scale[0] else 10 ** xi - 10 ** x
    yt = yi - y if not axes_scale[1] else 10 ** yi - 10 ** y
    if mode == 'loose':
        a0 = a00 * (1 - xt) + a01 * xt
        a1 = a10 * (1 - xt) + a11 * xt
        ai = a0 * (1 - yt) + a1 * yt
    elif mode == 'strict':
        zeros = np.where(np.array([a00, a01, a10, a11]) == 0.0)[0]
        if len(zeros) >= 2:
            ai = 0.0
        elif len(zeros) == 1:
            distance1 = np.sqrt((x - xi) ** 2 + (y - yi) ** 2)
            distance2 = np.sqrt((xn - xi) ** 2 + (y - yi) ** 2)
            distance3 = np.sqrt((x - xi) ** 2 + (yn - yi) ** 2)
            distance4 = np.sqrt((xn - xi) ** 2 + (yn - yi) ** 2)
            distances = np.array([distance1, distance2, distance3, distance4])
            if np.argmin(distances) == zeros[0]:
                ai = 0.0
            else:
                a0 = a00 * (1 - xt) + a01 * xt
                a1 = a10 * (1 - xt) + a11 * xt
                ai = a0 * (1 - yt) + a1 * yt
        else:
            a0 = a00 * (1 - xt) + a01 * xt
            a1 = a10 * (1 - xt) + a11 * xt
            ai = a0 * (1 - yt) + a1 * yt

    if not isinstance(xi, np.ndarray):
        if np.ma.is_masked(ai) and (not masked):
            raise TerminateTrajectory

    return ai


def _gen_starting_points(x,y,grains):
    
    eps = np.finfo(np.float32).eps
    
    tmp_x =  np.linspace(x.min()+eps, x.max()-eps, grains)
    tmp_y =  np.linspace(y.min()+eps, y.max()-eps, grains)
    
    xs = np.tile(tmp_x, grains)
    ys = np.repeat(tmp_y, grains)

    seed_points = np.array([list(xs), list(ys)])
    
    return seed_points.T


def distance(x, y):
    """Calculate the sum_distance between some points."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) < 2:
        return None

    x = np.asarray(x)
    y = np.asarray(y)
    dx = np.diff(x)
    dy = np.diff(y)

    return np.sqrt(dx**2 + dy**2).sum()

def traj_overlap(traj1, traj2, threshold=0.01):
    from scipy.spatial import cKDTree
    """
    检查两条轨迹是否重叠
    :param traj1: 第一条轨迹，格式为 (x, y)
    :param traj2: 第二条轨迹，格式为 (x, y)
    :param threshold: 重叠的距离阈值
    :return:  返回两条轨迹重叠部分占两条轨迹长度的各自百分比，如 (p1, p2)
    """

    # 将轨迹转换为Numpy数组
    points1 = np.column_stack(traj1)
    points2 = np.column_stack(traj2)

    if np.isnan(points1).any() or np.isnan(points2).any():
        warnings.warn("Trajectory contains NaN values.")
        return 1, 1
    elif np.isinf(points1).any() or np.isinf(points2).any():
        warnings.warn("Trajectory contains Inf values.")
        return 1, 1

    # 构建KD树用于快速邻近搜索
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    # 找出轨迹1中与轨迹2的点距离小于threshold的点
    overlap1 = tree1.query_ball_tree(tree2, threshold)
    count1 = sum(map(bool, overlap1))

    # 找出轨迹2中与轨迹1的点距离小于threshold的点
    overlap2 = tree2.query_ball_tree(tree1, threshold)
    count2 = sum(map(bool, overlap2))

    # 计算百分比
    percent1 = count1 / len(points1) if len(points1) > 0 else 0
    percent2 = count2 / len(points2) if len(points2) > 0 else 0

    return percent1, percent2


def velovect_key(fig, axes, quiver, shrink=0.15, U=1., angle=0., label='1', color='k', arrowstyle='->', linewidth=.5,
                 fontproperties={'size': 5}, lr=1., ud=1., width_shrink=1., height_shrink=1., arrowsize=1., edgecolor='k'):
    '''
    曲线矢量图例
    :param fig: 画布总底图
    :param axes: 目标图层
    :param quiver: 曲线矢量图层
    :param X: 图例横坐标
    :param Y: 图例纵坐标
    :param U: 风速
    :param angle: 角度
    :param label: 标签
    :param color: 颜色
    :param arrowstyle: 箭头样式
    :param linewidth: 线宽
    :param fontproperties: 字体属性
    :param lr: 左右偏移(>1：左偏, <1：右偏)
    :param ud: 上下偏移(>1：上偏, <1：下偏)
    :param width_shrink: 宽度缩放比例
    :param height_shrink: 高度缩放比例

    :return: None
    '''
    axes_sub = fig.add_axes([0, 0, 1, 1])
    adjust_sub_axes(axes, axes_sub, shrink=shrink, lr=lr, ud=ud, width=width_shrink, height=height_shrink)
    # 不显示刻度和刻度标签
    axes_sub.set_xticks([])
    axes_sub.set_yticks([])
    axes_sub.set_xlim(-1, 1)
    axes_sub.set_ylim(-2, 1)
    for spine in axes_sub.spines.values():
        spine.set_edgecolor(edgecolor)
    dt_ds = quiver[1]
    if np.isnan(dt_ds):
        return
    U_trans = U * dt_ds / shrink / 2
    # 绘制图例
    x, y = U_trans * np.cos(angle) * 2. / width_shrink, U_trans * np.sin(angle) * 3. / height_shrink
    arrow = patches.FancyArrowPatch(
    (x-(1e-1)*np.cos(angle), y-(1e-1)*np.sin(angle)), (x, y)
              , arrowstyle=arrowstyle, mutation_scale=10 * arrowsize, linewidth=linewidth, color=color)
    axes_sub.add_patch(arrow)
    lines = [[[-x, y], [x, -y]]]
    lc = mcollections.LineCollection(lines, capstyle='round', linewidth=linewidth, color=color)
    axes_sub.add_collection(lc)
    axes_sub.text(0, -1.5, label, ha='center', va='center', color=color, fontproperties=fontproperties)

if __name__ == '__main__':
    "test"
    x = np.linspace(-180, 180, 361*5)
    y = np.linspace(-90, 90, 180*5)
    Y, X = np.meshgrid(y, x)

    U = np.linspace(-1, 1, X.shape[0])[np.newaxis, :] * np.ones(X.shape).T
    V = np.linspace(1, -1, X.shape[1])[:, np.newaxis] * np.ones(X.shape).T
    #####
    fig = matplotlib.pyplot.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree(100.5))
    ax1.set_extent([-50, 130, -80, 80], crs=ccrs.PlateCarree())
    a1 = Curlyquiver(ax1, x, y, U, V, regrid=20, scale=10, color='k', linewidth=0.8, arrowsize=1, center_lon=100.5, MinDistance=[0.1, 0.1], arrowstyle='v', thinning=['0%', 'min'], alpha=0.6, zorder=100)
    ax1.contourf(x, y, U, levels=[-1, 0, 1], cmap=plt.cm.PuOr_r, transform=ccrs.PlateCarree(0), extend='both',alpha=0.5, zorder=10)
    ax1.contourf(x, y, V, levels=[-1, 0, 1], cmap=plt.cm.RdBu, transform=ccrs.PlateCarree(0), extend='both',alpha=0.5, zorder=10)
    # ax1.quiver(x, y, U, V, transform=ccrs.PlateCarree(0), regrid_shape=20, scale=25)
    a1.key(fig, shrink=0.15)
    ax1.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.2)

    for artist in ax1.get_children():
        # 强制开启裁剪
        artist.set_clip_on(True)
    plt.savefig('test.pdf', bbox_inches='tight')
    plt.show()
