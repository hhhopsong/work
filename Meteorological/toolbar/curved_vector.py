import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tqdm as tq
import warnings
import cartopy.crs as ccs


def curly_vector(ax, x, y, u, v, transform=ccs.PlateCarree(),color='k', direction='backward', density=10, arrowsize=0.0, scale=0.2):
    if len(x) * len(y) != U.shape[0] * U.shape[1] or len(x) * len(y) != V.shape[0] * V.shape[1]:
        raise ValueError('风速场维度与格点维度不匹配!')
    if len(x) * len(y) >= 3000:
        warnings.warn('RuntimeWarning: 格点过多，可能导致计算速度过慢!')
    Y, X = np.meshgrid(y, x)
    wind_speed = np.sqrt(U**2 + V**2) # 风速
    norm_flat = wind_speed.flatten() # 展平
    start_points = np.array([X.flatten(), Y.flatten()]).T # 起始点
    scale = scale / np.max(norm_flat) # 缩放比例
    for i in tq.trange(start_points.shape[0]):
        ax.streamplot(x,y,U,V, color=color, start_points=np.array([start_points[i,:]]), minlength=.95*norm_flat[i]*scale, maxlength=1.0*norm_flat[i]*scale,
                integration_direction=direction, density=density, arrowsize=arrowsize, transform=transform)
    arrows = patches.FancyArrowPatch((X,Y), (U,V), color=color, arrowstyle='->', mutation_scale=10)
    ax.add_patch(arrows)

w = 3
x = np.linspace(-w, w, 20)
y = np.linspace(-w, w, 20)

Y, X = np.meshgrid(y, x)

U = -Y
V = X
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection=ccs.PlateCarree())
curly_vector(ax1, x, y, U, V)


norm = np.sqrt(U**2 + V**2)
norm_flat = norm.flatten()

start_points = np.array([X.flatten(),Y.flatten()]).T

plt.clf()
scale = .2/np.max(norm)

plt.subplot(121)
plt.title('scaling only the length')
for i in tq.trange(start_points.shape[0]):
    plt.streamplot(X,Y,U,V, color='k', start_points=np.array([start_points[i,:]]),minlength=.95*norm_flat[i]*scale, maxlength=1.0*norm_flat[i]*scale,
                integration_direction='backward', density=10, arrowsize=0.0)
arrows = patches.FancyArrowPatch((X,Y), (U,V), color='k', arrowstyle='->', mutation_scale=10)
plt.add_patch(arrows)
plt.axis('square')



plt.subplot(122)
plt.title('scaling length, arrowhead and linewidth')
for i in tq.trange(start_points.shape[0]):
    plt.streamplot(X,Y,U,V, color='k', start_points=np.array([start_points[i,:]]),minlength=.95*norm_flat[i]*scale, maxlength=1.0*norm_flat[i]*scale,
                integration_direction='backward', density=10, arrowsize=0.0, linewidth=.5*norm_flat[i])
plt.quiver(X,Y,U/np.max(norm), V/np.max(norm),scale=30)

plt.axis('square')
plt.show() 