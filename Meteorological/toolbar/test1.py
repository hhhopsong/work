import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tqdm as tq
import warnings
import cartopy.crs as ccs
import scipy.ndimage as scind
from scipy.interpolate import interpolate

def curly_vector(axes, x, y, U, V, transform=None, color='k', linewidth=1, direction='backward', density=10, scale=10, scaling=False):
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
            U = interpolate.RegularGridInterpolator((x, y), U, method='linear')
            V = interpolate.RegularGridInterpolator((x, y), V, method='linear')
            y = np.arange(y[0], y[-1] + np.abs(x[0] - x[1]), np.abs(x[0] - x[1]))
            X, Y = np.meshgrid(x, y)
            U = U((X, Y))
            V = V((X, Y))
        else:
            U = interpolate.RegularGridInterpolator((x, y), U, method='linear')
            V = interpolate.RegularGridInterpolator((x, y), V, method='linear')
            x = np.arange(x[0], x[-1] + np.abs(y[0] - y[1]), np.abs(y[0] - y[1]))
            X, Y = np.meshgrid(x, y)
            U = U((X, Y))
            V = V((X, Y))
    else:
        X, Y = np.meshgrid(x, y)
    
    if len(x) * len(y) >= 3000:
        warnings.warn('RuntimeWarning: 格点过多，可能导致计算速度过慢!')

    # 初始化
    wind_speed = np.sqrt(U**2 + V**2) # 风速
    norm_flat = wind_speed.flatten()/np.max(wind_speed) # 归一化展平
    start_points = np.array([X.flatten(), Y.flatten()]).T # 起始点
    ###################################################箭头防异常!!!!
    Q1 = axes.quiver(X, Y, np.full(U.shape, np.nan), np.full(V.shape, np.nan), scale=scale/315, scale_units='xy', color='blue', transform=transform, headaxislength=0, headlength=0, headwidth=0)
    axes.quiverkey(Q1, X=0.9, Y=0.9, U=1, angle=0, label=f'{np.max(wind_speed)} m/s',
                                  labelpos='E', color='green', fontproperties={'size': 5})  # linewidth=1为箭头的大小
    ##################################################
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
    for i in tq.trange(start_points.shape[0], desc='绘制曲线矢量', leave=False):
        arrow_start = start_points[i, :]
        arrow_end = arrow_start + np.array([U.flatten()[i], V.flatten()[i]]) / np.max(wind_speed)*10**(-5)
        arrow_delta = np.sqrt((arrow_end[0] - arrow_start[0])**2 + (arrow_end[1] - arrow_start[1])**2)
        axes.streamplot(X,Y,U,V, color=color, start_points=np.array([start_points[i,:]]), minlength=.1*norm_flat[i]/scale, maxlength=1*norm_flat[i]/scale,
                integration_direction=direction, density=density, arrowsize=0.0, transform=transform, linewidth=linewidth)
        # 将地理坐标转换为显示坐标
        arrow_start_display = axes.projection.transform_point(arrow_start[0], arrow_start[1], transform)
        arrow_end_display = axes.projection.transform_point(arrow_end[0], arrow_end[1], transform)
        arrows = patches.FancyArrowPatch(arrow_start_display, arrow_end_display, color=color, arrowstyle='->', mutation_scale=scale, transform=transform)
        axes.add_patch(arrows)
    #axes.quiver(X,Y,U/np.max(wind_speed), V/np.max(wind_speed), scale=scale*1.2, color=color, transform=transform, headaxislength=0, headlength=0, headwidth=0)

if __name__ == '__main__':
    w = 3
    x = np.linspace(-180, 180, 10)
    y = np.linspace(-90, 90, 10)

    X, Y = np.meshgrid(x, y)
    U = -Y*10
    #U = np.zeros_like(X)
    V = X
    #V = np.zeros_like(Y)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection=ccs.PlateCarree())
    ax1.set_extent([-180, 180, -90, 90])
    curly_vector(ax1, x, y, U, V)
    plt.savefig("C:/Users/10574/Desktop/curly_vector.png", dpi=1000)
