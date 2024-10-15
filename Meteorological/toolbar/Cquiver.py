import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numba
import tqdm as tq
import warnings
import cartopy.crs as ccs
import scipy.ndimage as scind
from cartopy.util import add_cyclic_point
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

import sys
sys.path.append('d:/CODES/Python/Meteorological')
from toolbar.sub_adjust import adjust_sub_axes

def curly_vector(axes, x, y, U, V, lon_trunc, transform=None, color='k', regrid=20, streamgrid=15, linewidth=1, direction='both', density=1, scale=10, arrowstyle='simple', arrowsize=7, head_length=0.4, head_width=0.2, head_dist=1, scaling=False):
    """
    Warning:务必在调用函数前设置经纬度范围(set_exten)!网格间距需要各自等差!
    绘制曲线矢量
    :param axes: 目标图层
    :param x: 经度序列
    :param y: 纬度序列
    :param U: x方向风速
    :param V: y方向风速
    :param lon_trunc: 经度截断
    :param transform: 投影
    :param color: 颜色
    :param regrid: 插值网格密度
    :param streamgrid: 流线网格密度(过大将影响绘制速度!)
    :param linewidth: 线宽
    :param direction: 矢量方向
    :param density: 流线绘制密度
    :param scale: 缩放
    :param arrowstyle: 箭头样式
    :param arrowsize: 箭头大小
    :param head_length: 箭头长度
    :param head_width: 箭头宽度
    :param scaling: 是否缩放线宽
    :return: 曲线矢量图层, 风速最大值
    """
    # 数据检查
    if len(x)  != U.shape[0] or len(y) != U.shape[1] or len(x) != V.shape[0] or len(y) != V.shape[1]:
        raise ValueError('风速场维度与格点维度不匹配!')
    if x[0] > x[-1] or y[0] > y[-1]:
        warnings.warn('经纬度序列非严格增长,将进行重排列!')
        U = U[::-1, :] if x[0] > x[-1] else U
        U = U[:, ::-1] if y[0] > y[-1] else U
        V = V[::-1, :] if x[0] > x[-1] else V
        V = V[:, ::-1] if y[0] > y[-1] else V
        x = x[::-1] if x[0] > x[-1] else x
        y = y[::-1] if y[0] > y[-1] else y

    x = x.data if isinstance(x, xr.DataArray) else x
    y = y.data if isinstance(y, xr.DataArray) else y
    # 经纬度重排列为-180~0~180
    if x[-1] > 180:
        U = np.concatenate([U[np.argmax(x > 180):, :], U[np.argmax(x >= 0):np.argmax(x > 180), :]], axis=0)
        V = np.concatenate([V[np.argmax(x > 180):, :], V[np.argmax(x >= 0):np.argmax(x > 180), :]], axis=0)
        x = np.concatenate([x[x > 180] - 360, x[np.argmax(x >= 0):np.argmax(x > 180)]])
    
    # 获取axes经纬度范围
    extent = axes.get_extent()
    # 将网格插值为正方形等间隔网格
    warnings.warn('非正方形格点，将进行插值!')
    U = RegularGridInterpolator((x, y), U, method='linear')
    V = RegularGridInterpolator((x, y), V, method='linear')
    # 截取范围内的经纬度
    x_extent = np.where((x >= extent[0]) & (x < extent[1]))[0]
    y_extent = np.where((y >= extent[2]) & (y < extent[3]))[0]
    x = x[x_extent]
    y = y[y_extent]

    x_stream, y_stream, U_stream, V_stream = x, y, U, V
    x = np.linspace(x[0], x[-1], regrid)
    y = np.linspace(y[0], y[-1], regrid)
    x_stream = np.linspace(x_stream[0], x_stream[-1], streamgrid)
    y_stream = np.linspace(y_stream[0], y_stream[-1], streamgrid)
    if np.abs(x[0] - x[1]) < np.abs(y[0] - y[1]):
        x = np.arange(x[0], x[-1], np.abs(x[0] - x[1]))
        y = np.arange(y[0], y[-1], np.abs(x[0] - x[1]))
        X, Y = np.meshgrid(x, y)
        U = U((X, Y))
        V = V((X, Y))
        x_stream = np.arange(x_stream[0], x_stream[-1], np.abs(x_stream[0] - x_stream[1]))
        y_stream = np.arange(y_stream[0], y_stream[-1], np.abs(x_stream[0] - x_stream[1]))
        X_stream, Y_stream = np.meshgrid(x_stream, y_stream)
        U_stream = U_stream((X_stream, Y_stream))
        V_stream = V_stream((X_stream, Y_stream))
    else:
        x = np.arange(x[0], x[-1], np.abs(y[0] - y[1]))
        y = np.arange(y[0], y[-1], np.abs(y[0] - y[1]))
        X, Y = np.meshgrid(x, y)
        U = U((X, Y))
        V = V((X, Y))
        x_stream = np.arange(x_stream[0], x_stream[-1], np.abs(y_stream[0] - y_stream[1]))
        y_stream = np.arange(y_stream[0], y_stream[-1], np.abs(y_stream[0] - y_stream[1]))
        X_stream, Y_stream = np.meshgrid(x_stream, y_stream)
        U_stream = U_stream((X_stream, Y_stream))
        V_stream = V_stream((X_stream, Y_stream))
    
    if len(x_stream) * len(y_stream) >= 1000:
        warnings.warn('流线绘制精度高，可能导致计算速度过慢!', RuntimeWarning)

    # 初始化
    wind_speed = np.sqrt(U_stream**2 + V_stream**2) # 风速
    start_points = np.array([X.flatten(), Y.flatten()]).T # 起始点
    lon_trunc = lon_trunc - 360 if lon_trunc > 180 else lon_trunc
    # 横纵画图单位同化
    y2x = (x[-1] - x[0]) / (y[-1] - y[0])
    V_trans = V * y2x
    wind_speed = np.sqrt(U**2 + V_trans**2) # 风速
    norm_flat = wind_speed.flatten()/np.nanmax(wind_speed) # 归一化展平
    # 参数配置
    if transform is None:
        transform = axes.projection
    if linewidth is None:
        linewidth = matplotlib.rcParams['lines.linewidth']
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    TrueIndex = np.where(~np.isnan(norm_flat))[0]
    start_points = start_points[TrueIndex]
    norm_flat = norm_flat[TrueIndex] # 剔除无效点
    axes.quiver(X_stream, Y_stream, U_stream, V_stream, color='blue', scale=scale/200, scale_units='xy', transform=transform)
    for i in tq.trange(start_points.shape[0], desc='绘制曲线矢量', leave=True):
        # 轨迹绘制
        if scaling:
            linewidth = .5 * norm_flat[i]  # 缩放线宽
        strm = axes.streamplot(X_stream, Y_stream, U_stream, V_stream, color=color, start_points=np.array([start_points[i,:]]), minlength=.05*norm_flat[i]/scale, maxlength=.5*norm_flat[i]/scale,
                integration_direction=direction, density=density, arrowsize=0, transform=transform, linewidth=linewidth)
        # 箭头绘制
        arrow_start = [line.vertices[-1] for line in strm.lines.get_paths()]
        if arrow_start == []:
            continue  # 无效箭头
        try:
            arrow_end = [line.vertices[-3] for line in strm.lines.get_paths()]
        except:
            arrow_end = [line.vertices[-2] for line in strm.lines.get_paths()]
        arrow_start = arrow_start[0]
        arrow_end = arrow_end[0]
        arrow_end =  arrow_start + (arrow_start - arrow_end) * 10**(-5)
        a_start = arrow_start[0] - 360 - lon_trunc if arrow_start[0] - lon_trunc > 180 else arrow_start[0] - lon_trunc
        a_end = arrow_end[0] - 360 - lon_trunc if arrow_end[0] - lon_trunc > 180 else arrow_end[0] - lon_trunc
        a_start = a_start + 360 if a_start < -180 else a_start
        a_end = a_end + 360 if a_end < -180 else a_end
        # 网格偏移避免异常箭头
        if np.abs(a_start - a_end) < 90:
            if np.min([a_start, a_end]) <= 0 <= np.max([a_start, a_end]) and extent[0] + 360 == extent[1]:
                error = 10 ** (-3)
                arrow_start = [arrow_start[0] - error, arrow_start[1]]
                arrow_end =  [arrow_end[0] - error, arrow_end[1]]
        arrow_start = [arrow_start[0] + 180 + lon_trunc, arrow_start[1]]
        arrow_end = [arrow_end[0] + 180 + lon_trunc, arrow_end[1]]
        arrows = patches.FancyArrowPatch(arrow_start, arrow_end, color=color, mutation_scale=arrowsize, transform=transform)
        arrows.set_arrowstyle(arrowstyle+f', head_length={head_length}, head_width={head_width}')
        axes.add_patch(arrows)
    return axes.quiver(X_stream, Y_stream, np.full(U_stream.shape, np.nan), np.full(V_stream.shape, np.nan), scale=scale/315, scale_units='xy', color='blue', transform=transform), np.max(wind_speed)


def curly_vector_key(fig, axes, quiver, X=.93, Y=.105, U=None, angle=0, label='', labelpos='S', color='k', linewidth=.08, fontproperties={'size': 5}):
    '''
    曲线矢量图例
    :param fig: 画布
    :param axes: 目标图层
    :param quiver: 曲线矢量图层
    :param X: 图例横坐标
    :param Y: 图例纵坐标
    :param U: 风速
    :param angle: 角度
    :param label: 标签
    :param labelpos: 标签位置
    :param color: 颜色
    :param fontproperties: 字体属性
    :return: None
    '''
    if U is None:
        U = 1
    else:
        U = U / quiver[1] 
    axes_sub = fig.add_axes([0, 0, 1, 1])
    adjust_sub_axes(axes, axes_sub, shrink=0.05)
    # 不显示刻度和刻度标签
    axes_sub.set_xticks([])
    axes_sub.set_yticks([])
    # 让图例在子图层中居中
    axes_sub.quiverkey(quiver[0], X=X, Y=Y, U=U, angle=angle, label=label, linewidth=linewidth,
                                  labelpos=labelpos, color=color, fontproperties=fontproperties)
    


if __name__ == '__main__':
    "test"
    x = np.linspace(-180, 180, 1440)
    y = np.linspace(-90, 90, 720)
    X, Y = np.meshgrid(x, y)
    U = np.ones(X.shape).T
    V = np.zeros(X.shape).T
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection=ccs.PlateCarree())
    ax1.set_extent([-180, 180, -90, 90])
    a1 = curly_vector(ax1, x, y, U, V, lon_trunc=180, scale=25)
    curly_vector_key(fig, ax1, a1, U=4, label='4 m/s')
    plt.savefig('D:/PyFile/pic/test.png', dpi=500)
    plt.show()
