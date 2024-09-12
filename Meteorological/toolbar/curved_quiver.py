import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs


def curved_quiver(ax, X, Y, U, V, color='k',scaling=False,length_scale=0, scale=30, minlength=.95, maxlength=1.0, integration_direction='backward', density=10, arrowsize=0.0, transform=ccrs.PlateCarree()):
    norm = np.sqrt(U**2 + V**2)
    norm_flat = norm.flatten()
    start_points = np.array([X.flatten(),Y.flatten()]).T
    if length_scale == 0:
        scale_ = .2/np.max(norm)
    else:
        scale_ = length_scale/np.max(norm)
    if not scaling:
        for i in range(start_points.shape[0]):
            ax.streamplot(X,Y,U,V, color=color, start_points=np.array([start_points[i,:]]),minlength=minlength*norm_flat[i]*scale_, maxlength=maxlength*norm_flat[i]*scale_,
                        integration_direction=integration_direction, density=density, arrowsize=arrowsize, transform=transform)
        ax.quiver(X,Y,U/norm, V/norm,scale=scale)
    else:
        for i in range(start_points.shape[0]):
            ax.streamplot(X,Y,U,V, color=color, start_points=np.array([start_points[i,:]]),minlength=.95*norm_flat[i]*scale_, maxlength=1.0*norm_flat[i]*scale_,
                        integration_direction=integration_direction, density=density, arrowsize=arrowsize, linewidth=.5*norm_flat[i], transform=transform)
        ax.quiver(X,Y,U/np.max(norm), V/np.max(norm),scale=scale)


if __name__ == '__main__':
    plt.show()
