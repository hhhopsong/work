import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tqdm as tq
import warnings
import cartopy.crs as ccs


def curly_vector(axes, x, y, U, V, transform=None, color='k', linewidth=1, direction='backward', density=10, scale=5, scaling=False):
    """
    Warning:务必在调用函数前设置经纬度范围(set_exten)!
    """
    # 数据检查
    if len(x) * len(y) != U.shape[0] * U.shape[1] or len(x) * len(y) != V.shape[0] * V.shape[1]:
        raise ValueError('风速场维度与格点维度不匹配!')
    if len(x) * len(y) >= 3000:
        warnings.warn('RuntimeWarning: 格点过多，可能导致计算速度过慢!')
    X, Y = np.meshgrid(x, y)
    wind_speed = np.sqrt(U**2 + V**2) # 风速
    norm_flat = wind_speed.flatten()/np.max(wind_speed) # 归一化展平
    start_points = np.array([X.flatten(), Y.flatten()]).T # 起始点
    # 参数配置
    if transform is None:
        transform = axes.projection
    if linewidth is None:
        linewidth = matplotlib.rcParams['lines.linewidth']
    if scaling:
        linewidth=.5*norm_flat[i]  # 缩放线宽
    for i in tq.trange(start_points.shape[0], desc='绘制曲线矢量', leave=False):
        axes.streamplot(X,Y,U/np.max(wind_speed),V/np.max(wind_speed), color=color, start_points=np.array([start_points[i,:]]), minlength=.1*norm_flat[i]/scale, maxlength=1*norm_flat[i]/scale,
                integration_direction=direction, density=density, arrowsize=0.0, transform=transform, linewidth=linewidth)
        arrow_start = start_points[i, :]
        arrow_end = arrow_start + np.array([U.flatten()[i], V.flatten()[i]]) / np.max(wind_speed)
        # 将地理坐标转换为显示坐标
        arrow_start_display = axes.projection.transform_point(arrow_start[0], arrow_start[1], transform)
        arrow_end_display = axes.projection.transform_point(arrow_end[0], arrow_end[1], transform)
        arrows = patches.FancyArrowPatch(arrow_start_display, arrow_end_display, color=color, arrowstyle='->', mutation_scale=scale, transform=transform)
        axes.add_patch(arrows)
    axes.quiver(X,Y,U/np.max(wind_speed), V/np.max(wind_speed), scale=scale, color=color, transform=axes.transData)

if __name__ == '__main__':
    w = 3
    x = np.linspace(-180, 180, 20)
    y = np.linspace(-90, 90, 20)

    X, Y = np.meshgrid(x, y)
    U = -Y
    V = X
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection=ccs.PlateCarree())
    ax1.set_extent([-180, 180, -90, 90])
    curly_vector(ax1, x, y, U, V)
    plt.savefig("C:/Users/86136/Desktop/curly_vector.png", dpi=1000)


'''plt.subplot(121)
plt.title('scaling only the length')
for i in tq.trange(start_points.shape[0]):
    plt.streamplot(X,Y,U,V, color='k', start_points=np.array([start_points[i,:]]),minlength=.95*norm_flat[i]*scale, maxlength=1.0*norm_flat[i]*scale,
                integration_direction='backward', density=10, arrowsize=0.0)
arrows = patches.FancyArrowPatch((X,Y), (U,V), color='k', arrowstyle='->', mutation_scale=10)
plt.add_patch(arrows)'''
