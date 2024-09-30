import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tqdm as tq
import warnings
import cartopy.crs as ccs
import scipy.ndimage as scind
from scipy.interpolate import RegularGridInterpolator
from sub_adjust import adjust_sub_axes

def curly_vector(axes, x, y, U, V, lon_trunc, transform=None, color='k', linewidth=1, direction='backward', density=10, scale=10, arrowstyle='simple', arrowsize=7, head_length=0.4, head_width=0.2, head_dist=1, scaling=False):
    """
    Warning:务必在调用函数前设置经纬度范围(set_exten)!网格间距需要各自等差!
    """
    # 数据检查
    if len(x) * len(y) != U.shape[0] * U.shape[1] or len(x) * len(y) != V.shape[0] * V.shape[1]:
        raise ValueError('风速场维度与格点维度不匹配!')
    if x[0] > x[-1] or y[0] > y[-1]:
        warnings.warn('RuntimeWarning: 经纬度序列非严格增长,将进行重排列!')
        x = x[::-1] if x[0] > x[-1] else x
        y = y[::-1] if y[0] > y[-1] else y
        U = U[::-1, :] if x[0] > x[-1] else U
        U = U[:, ::-1] if y[0] > y[-1] else U
        V = V[::-1, :] if x[0] > x[-1] else V
        V = V[:, ::-1] if y[0] > y[-1] else V

    # 将网格插值为正方形等间隔网格
    if np.abs(x[0] - x[1]) != np.abs(y[0] - y[1]):
        warnings.warn('RuntimeWarning: 非正方形格点，将进行插值!')
        if np.abs(x[0] - x[1]) < np.abs(y[0] - y[1]):
            U = RegularGridInterpolator((x, y), U, method='linear')
            V = RegularGridInterpolator((x, y), V, method='linear')
            y = np.arange(y[0], y[-1] + np.abs(x[0] - x[1]), np.abs(x[0] - x[1]))
            X, Y = np.meshgrid(x, y)
            U = U((X, Y))
            V = V((X, Y))
        else:
            U = RegularGridInterpolator((x, y), U, method='linear')
            V = RegularGridInterpolator((x, y), V, method='linear')
            x = np.arange(x[0], x[-1] + np.abs(y[0] - y[1]), np.abs(y[0] - y[1]))
            X, Y = np.meshgrid(x, y)
            U = U((X, Y))
            V = V((X, Y))
    else:
        X, Y = np.meshgrid(x, y)
    
    if len(x) * len(y) >= 500:
        warnings.warn('RuntimeWarning: 格点过多，可能导致计算速度过慢!')

    # 初始化
    wind_speed = np.sqrt(U**2 + V**2) # 风速
    norm_flat = wind_speed.flatten()/np.max(wind_speed) # 归一化展平
    start_points = np.array([X.flatten(), Y.flatten()]).T # 起始点
    lon_trunc = lon_trunc - 360 if lon_trunc > 180 else lon_trunc
    # 横纵画图单位同化
    y2x = (x[-1] - x[0]) / (y[-1] - y[0])
    V_trans = V * y2x
    wind_speed = np.sqrt(U**2 + V_trans**2) # 风速
    norm_flat = wind_speed.flatten()/np.max(wind_speed) # 归一化展平
    # 参数配置
    if transform is None:
        transform = axes.projection
    if linewidth is None:
        linewidth = matplotlib.rcParams['lines.linewidth']
    if scaling:
        linewidth=.5*norm_flat[i]  # 缩放线宽
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    for i in tq.trange(start_points.shape[0], desc='绘制曲线矢量', leave=True):
        arrow_start = start_points[i, :]
        arrow_end = arrow_start + np.array([U.flatten()[i], V.flatten()[i]]) / np.max(wind_speed)*10**(-5)
        a_start = arrow_start[0] - 360 + lon_trunc if arrow_start[0] + lon_trunc > 180 else arrow_start[0] + lon_trunc
        a_end = arrow_end[0] - 360 + lon_trunc if arrow_end[0] + lon_trunc > 180 else arrow_end[0] + lon_trunc
        axes.streamplot(X,Y,U,V, color=color, start_points=np.array([start_points[i,:]]), minlength=.05*norm_flat[i]/scale, maxlength=1*norm_flat[i]/scale, 
                integration_direction=direction, density=density, arrowsize=0, transform=transform, linewidth=linewidth)
        arrows = patches.FancyArrowPatch(arrow_start, arrow_end, color=color,mutation_scale=arrowsize, transform=transform)
        # 只显示箭头头部
        if np.abs(a_start - a_end) < 90:
            if np.min([a_start, a_end]) <= 0 <= np.max([a_start, a_end]):
                continue  # 跨越截断精度不绘制箭头
        arrows.set_arrowstyle(arrowstyle+f', head_length={head_length}, head_width={head_width}')
        axes.add_patch(arrows)
    return axes.quiver(X, Y, np.full(U.shape, np.nan), np.full(V.shape, np.nan), scale=scale/315, scale_units='xy', color='blue', transform=transform), np.max(wind_speed)


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
    x = np.linspace(-180, 180, 10)
    y = np.linspace(-90, 90, 20)
    X, Y = np.meshgrid(x, y)
    U = np.random.randn(*X.shape).T
    V = np.random.randn(*X.shape).T
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection=ccs.PlateCarree())
    ax1.set_extent([-180, 180, -90, 90])
    a1 = curly_vector(ax1, x, y, U, V, lon_trunc=180)
    curly_vector_key(fig, ax1, a1,U=4, label='4 m/s')
    plt.show()
