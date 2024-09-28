import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tqdm as tq
import warnings
import cartopy.crs as ccs


def curly_vector(axes, x, y, U, V, transform=ccs.PlateCarree(), color='k', 
                 linewidth=1, direction='backward', density=10, arrowsize=0, 
                 arrowstyle='-|>', scale=5, scale_all=False):
    """
    WARNING:请务必设置axes的投影类型及经纬度范围(set_extent)!
    生成曲线矢量场
    Parameters
    ----------
    axes : matplotlib.axes.Axes
        绘图对象
    x : 1D array-like
        经度
    y : 1D array-like
        纬度
    U/V : 2D array-like
        水平风速
    transform : cartopy.crs
        投影类型
    color : str
        曲线颜色
    linewidth : float
        曲线宽度
    direction : str
        曲线方向
    density : int
        曲线密度
    arrowsize : float
        箭头大小
    scale : float
        缩放比例
    scale_all : bool
        是否缩放曲线整体
    """
    if scale_all:
        # 曲线缩放
        linewidth=.5*norm_flat[i]
    else: 
        if linewidth is None:
            linewidth = matplotlib.rcParams['lines.linewidth']
    if transform is None:
        transform = axes.projection
    if len(x) * len(y) != U.shape[0] * U.shape[1] or len(x) * len(y) != V.shape[0] * V.shape[1]:
        raise ValueError('风速场维度与格点维度不匹配!')
    if len(x) * len(y) >= 500:
        warnings.warn('RuntimeWarning: 格点过多，可能导致计算速度过慢!')
    X, Y = np.meshgrid(x, y)
    wind_speed = np.sqrt(U**2 + V**2) # 风速
    norm_flat = wind_speed.flatten() # 展平
    start_points = np.array([X.flatten(), Y.flatten()]).T # 起始点
    start_geopoint = transform.transform_points(ccs.Geodetic(), start_points[:, 0], start_points[:, 1])[:, :2]
    for i in tq.trange(start_points.shape[0], desc='绘制曲线矢量场', leave=False):
        axes.streamplot(X,Y,U,V, color=color, start_points=start_geopoint[i:i+1, :], minlength=0.1*norm_flat[i]*scale, maxlength=1.01*norm_flat[i]*scale,
                integration_direction=direction, density=density, arrowsize=0.0, linewidth=linewidth, transform=transform)
    axes.quiver(X, Y, U, V, scale=scale, transform=transform, pivot='tip', angles='xy', scale_units='xy')

if __name__ == '__main__':
    w = 3
    #生成全球2.5度经纬度格点
    y = np.linspace(-90, 90, 15)
    x = np.linspace(-180, 180, 15)

    X, Y = np.meshgrid(x, y)
    #生成一个全球的风速场
    U = -Y
    V = X
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection=ccs.PlateCarree())
    ax1.set_extent([-180, 180, -90, 90], ccs.PlateCarree())
    curly_vector(ax1, x, y, U, V)
    plt.savefig(r'C:/Users/10574/Desktop/test.png', dpi=2000)
    plt.show()

