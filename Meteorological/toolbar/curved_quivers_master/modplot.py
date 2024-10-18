"""
Streamline plotting for 2D vector fields.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.streamplot import TerminateTrajectory

import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import cartopy.crs as ccrs

import matplotlib
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import tqdm as tq
from toolbar.sub_adjust import adjust_sub_axes


__all__ = ['velovect']


def velovect(axes, x, y, u, v, lon_trunc=180, linewidth=None, color='black',
               cmap=None, norm=None, arrowsize=1, arrowstyle='->',
               transform=None, zorder=None, start_points=None,
               scale=1., grains=1, masked=True, regrid=0, integration_direction='both'):
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
    *zorder* : int
        ``zorder`` 属性决定了绘图元素的绘制顺序,数值较大的元素会被绘制在数值较小的元素之上。
    *scale* : float
        矢量的最大长度。
    *grains* : int
        绘图网格点数。
    *masked* : bool
        原数据是否为掩码数组
    *regrid* : int
        是否重新插值网格
    *integration_direction* : {'forward', 'backward', 'both'}, default: 'both'
        矢量向前、向后或双向绘制。

    Returns:

        *stream_container* : StreamplotSet
            具有属性的容器对象

                - lines: `matplotlib.collections.LineCollection` of streamlines

                - arrows: collection of `matplotlib.patches.FancyArrowPatch`
                  objects representing arrows half-way along stream
                  lines.

            此容器将来可能会更改，以允许更改线条和箭头的颜色图、alpha等，但这些更改应该会向下兼容。
        *scale* : float
            矢量的最大长度。
    """

    # 填充nan值为0
    if masked:
        u = np.where(np.isnan(u), 0, u)
        v = np.where(np.isnan(v), 0, v)

    # 检查y是否升序
    if y[0] > y[-1]:
        print('Velovect Waring: Y reversed, because y is descending.')
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
            raise ValueError('x must be a xarray.DataArray or numpy.ndarray')
    except:
        pass
    try:
        if isinstance(y, xr.DataArray):
            y = y.data
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise ValueError('y must be a xarray.DataArray or numpy.ndarray')
    except:
        pass
    try:
        if isinstance(u, xr.DataArray):
            u = u.data
        elif isinstance(u, np.ndarray):
            pass
        else:
            raise ValueError('u must be a xarray.DataArray or numpy.ndarray')
    except:
        pass
    try:
        if isinstance(v, xr.DataArray):
            v = v.data
        elif isinstance(v, np.ndarray):
            pass
        else:
            raise ValueError('v must be a xarray.DataArray or numpy.ndarray')
    except:
        pass

    # 经纬度重排列为-180~0~180
    if x[-1] > 180:
        u = np.concatenate([u[:, np.argmax(x > 180):], u[:, np.argmax(x >= 0):np.argmax(x > 180)]], axis=1)
        v = np.concatenate([v[:, np.argmax(x > 180):], v[:, np.argmax(x >= 0):np.argmax(x > 180)]], axis=1)
        x = np.concatenate([x[x > 180] - 360, x[np.argmax(x >= 0):np.argmax(x > 180)]])

    if regrid:
        # 将网格插值为正方形等间隔网格
        U = RegularGridInterpolator((y, x), u, method='linear')
        V = RegularGridInterpolator((y, x), v, method='linear')
        x = np.linspace(x[0], x[-1], regrid)
        y = np.linspace(y[0], y[-1], regrid)
        if np.abs(x[0] - x[1]) < np.abs(y[0] - y[1]):
            x = np.arange(x[0], x[-1], np.abs(x[0] - x[1]))
            y = np.arange(y[0], y[-1], np.abs(x[0] - x[1]))
            X, Y = np.meshgrid(x, y)
            u = U((Y, X))
            v = V((Y, X))
        else:
            x = np.arange(x[0], x[-1], np.abs(y[0] - y[1]))
            y = np.arange(y[0], y[-1], np.abs(y[0] - y[1]))
            X, Y = np.meshgrid(x, y)
            u = U((Y, X))
            v = V((Y, X))

    _api.check_in_list(['both', 'forward', 'backward'], integration_direction=integration_direction)
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
                "If 'color' is given, must have the shape of 'Grid(x,y)'")
        line_colors = []
        color = np.ma.masked_invalid(color)
    else:
        line_kw['color'] = color
        arrow_kw['color'] = color

    if isinstance(linewidth, np.ndarray):
        if linewidth.shape != grid.shape:
            raise ValueError(
                "If 'linewidth' is given, must have the shape of 'Grid(x,y)'")
        line_kw['linewidth'] = []
    else:
        line_kw['linewidth'] = linewidth
        arrow_kw['linewidth'] = linewidth

    line_kw['zorder'] = zorder
    arrow_kw['zorder'] = zorder

    ## Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape:
        raise ValueError("'u' and 'v' must be of shape 'Grid(x,y)'")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)
    magnitude = np.sqrt(u**2 + v**2)
    magnitude/=np.max(magnitude)
	
    resolution = scale/grains
    minlength = .9*resolution
    integrate = get_integrator(u, v, dmap, minlength, resolution, magnitude)

    trajectories = []
    edges = []
    
    if start_points is None:
        if regrid:
            start_points = np.array([X.flatten(), Y.flatten()]).T
        else:
            start_points=_gen_starting_points(x,y,grains)
    
    sp2 = np.asanyarray(start_points, dtype=float).copy()

    # Check if start_points are outside the data boundaries
    for xs, ys in sp2:
        if not (grid.x_origin <= xs <= grid.x_origin + grid.width
                and grid.y_origin <= ys <= grid.y_origin + grid.height):
            raise ValueError("Starting point ({}, {}) outside of data "
                             "boundaries".format(xs, ys))

    # Convert start_points from data to array coords
    # Shift the seed points from the bottom left of the data so that
    # data2grid works properly.
    sp2[:, 0] -= grid.x_origin
    sp2[:, 1] -= grid.y_origin

    # 获取axes范围
    try:
        extent = axes.get_extent()
    except AttributeError:
        extent = axes.get_xlim() + axes.get_ylim()

    for xs, ys in sp2:
        xg, yg = dmap.data2grid(xs, ys)
        xg = np.clip(xg, 0, grid.nx - 1)
        yg = np.clip(yg, 0, grid.ny - 1)
        t = integrate(xg, yg)
        if t is not None:
            trajectories.append(t[0])
            edges.append(t[1])

    if use_multicolor_lines:
        if norm is None:
            norm = mcolors.Normalize(color.min(), color.max())
        if cmap is None:
            cmap = cm.get_cmap(matplotlib.rcParams['image.cmap'])
        else:
            cmap = cm.get_cmap(cmap)

    streamlines = []
    arrows = []
    for t, edge in tq.tqdm(zip(trajectories,edges), desc='Drawing streamlines'):
        tgx = np.array(t[0])
        tgy = np.array(t[1])
        
		
        # Rescale from grid-coordinates to data-coordinates.
        tx, ty = dmap.grid2data(*np.array(t))
        tx += grid.x_origin
        ty += grid.y_origin

        
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
        arrow_tail = (tx[n], ty[n])
        arrow_head = (tx[-1], ty[-1])

        # 网格偏移避免异常箭头
        arrow_start = np.array([arrow_head[0], arrow_head[1]])
        arrow_end = np.array([arrow_tail[0], arrow_tail[1]])
        delta = arrow_start - arrow_end
        if np.sqrt(delta[0] ** 2 + delta[1] ** 2) == 0.: continue  # 长度为0的箭头
        delta = delta / np.sqrt(delta[0] ** 2 + delta[1] ** 2)
        delta = delta * 10**0
        arrow_end =  arrow_start + delta
        a_start = arrow_start[0] - 360 - lon_trunc if arrow_start[0] - lon_trunc > 180 else arrow_start[0]  - lon_trunc
        a_end = arrow_end[0] - 360 - lon_trunc if arrow_end[0] - lon_trunc > 180 else arrow_end[0]  - lon_trunc
        a_start = a_start + 360 if a_start < -180 else a_start
        a_end = a_end + 360 if a_end < -180 else a_end
        # 网格偏移避免异常箭头
        if np.abs(a_start - a_end) < 90:
            if np.min([a_start, a_end]) <= 0 <= np.max([a_start, a_end]) and extent[0] + 360 == extent[1]:
                arrow_start = [arrow_start[0] - delta[0] * 1.001, arrow_start[1] - delta[0] * 1.001]
                arrow_end =  [arrow_end[0] - delta[0] * 1.001, arrow_end[1] - delta[0] * 1.001]
        arrow_head = [arrow_start[0], arrow_start[1]]
        arrow_tail = [arrow_end[0], arrow_end[1]]

        # 防止出现纬度超过90度
        if np.abs(arrow_head[1]) >= 90 or np.abs(arrow_tail[1]) >= 90:
            error = np.argmax([np.abs(arrow_head[1]), np.abs(arrow_tail[1])])
            error = [arrow_head[1], arrow_tail[1]][error]
            if error > 0:
                error -= (90 - 1e-5)
                arrow_head = np.array([arrow_head[0], arrow_head[1] - error])
                arrow_tail = np.array([arrow_tail[0], arrow_tail[1] - error])
            else:
                error -= (-90 + 1e-5)
                arrow_head = np.array([arrow_head[0], arrow_head[1] - error])
                arrow_tail = np.array([arrow_tail[0], arrow_tail[1] - error])

        if isinstance(linewidth, np.ndarray):
            line_widths = interpgrid(linewidth, tgx, tgy, masked=masked)[:-1]
            line_kw['linewidth'].extend(line_widths)
            arrow_kw['linewidth'] = line_widths[n]

        if use_multicolor_lines:
            color_values = interpgrid(color, tgx, tgy, masked=masked)[:-1]
            line_colors.append(color_values)
            arrow_kw['color'] = cmap(norm(color_values[n]))
        
        if not edge:
            p = patches.FancyArrowPatch(
                arrow_head, arrow_tail, transform=transform, **arrow_kw)
        else:
            continue
        
        ds = np.sqrt((arrow_tail[0]-arrow_head[0])**2+(arrow_tail[1]-arrow_head[1])**2)

        if ds<1e-15: continue  #remove vanishingly short arrows that cause Patch to fail

        axes.add_patch(p)
        arrows.append(p)

    lc = mcollections.LineCollection(
        streamlines, transform=transform, capstyle='round', **line_kw)
    lc.sticky_edges.x[:] = [grid.x_origin, grid.x_origin + grid.width]
    lc.sticky_edges.y[:] = [grid.y_origin, grid.y_origin + grid.height]
    if use_multicolor_lines:
        lc.set_array(np.ma.hstack(line_colors))
        lc.set_cmap(cmap)
        lc.set_norm(norm)
    axes.add_collection(lc)
    axes.autoscale_view()

    ac = mcollections.PatchCollection(arrows)
    stream_container = StreamplotSet(lc, ac)
    return stream_container, scale

	

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

        self.x_origin = x[0]
        self.y_origin = y[0]

        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]

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

def get_integrator(u, v, dmap, minlength, resolution, magnitude, integration_direction='both', masked=True):

    # rescale velocity onto grid-coordinates for integrations.
    u, v = dmap.data2grid(u, v)

    # speed (path length) will be in axes-coordinates
    u_ax = u / dmap.grid.nx
    v_ax = v / dmap.grid.ny
    speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2)

    if integration_direction == 'both':
        speed = speed / 2.

    def forward_time(xi, yi):
        ds_dt = interpgrid(speed, xi, yi, masked=masked)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        ui = interpgrid(u, xi, yi, masked=masked)
        vi = interpgrid(v, xi, yi, masked=masked)
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

        stotal, x_traj, y_traj = 0., [], []

        
        dmap.start_trajectory(x0, y0)

        if integration_direction in ['both', 'backward']:
            stotal_, x_traj_, y_traj_, m_total, hit_edge = _integrate_rk12(x0, y0, dmap, backward_time, resolution, magnitude)
            stotal += stotal_
            x_traj += x_traj_[::-1]
            y_traj += y_traj_[::-1]

        if integration_direction in ['both', 'forward']:
            dmap.reset_start_point(x0, y0)
            stotal_, x_traj_, y_traj_, m_total, hit_edge = _integrate_rk12(x0, y0, dmap, forward_time, resolution, magnitude)
            stotal += stotal_
            x_traj += x_traj_[1:]
            y_traj += y_traj_[1:]


        if len(x_traj)>1:
            return (x_traj, y_traj), hit_edge
        else:  # reject short trajectories
            dmap.undo_trajectory()
            return None

    return integrate


def _integrate_rk12(x0, y0, dmap, f, resolution, magnitude, masked=True):
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
    maxerror = 0.0003

    # This limit is important (for all integrators) to avoid the
    # trajectory skipping some mask cells. We could relax this
    # condition if we use the code which is commented out below to
    # increment the location gradually. However, due to the efficient
    # nature of the interpolation, this doesn't boost speed by much
    # for quite a bit of complexity.
    maxds = min(1. / dmap.mask.nx, 1. / dmap.mask.ny, 0.1)

    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    xf_traj = []
    yf_traj = []
    m_total = []
    hit_edge = False
    
    while dmap.grid.within_grid(xi, yi):
        xf_traj.append(xi)
        yf_traj.append(yi)
        m_total.append(interpgrid(magnitude, xi, yi, masked=masked))
        try:
            k1x, k1y = f(xi, yi)
            k2x, k2y = f(xi + ds * k1x,
                         yi + ds * k1y)
        except IndexError:
            # Out of the domain on one of the intermediate integration steps.
            # Take an Euler step to the boundary to improve neatness.
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
            
            dmap.update_trajectory(xi, yi)
            
            if not dmap.grid.within_grid(xi, yi):
                hit_edge=True
            
            if (stotal + ds) > resolution*np.mean(m_total):
                break
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


# Utility functions
# ========================

def interpgrid(a, xi, yi, masked=True):
    """Fast 2D, linear interpolation on an integer grid"""

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
    xt = xi - x
    yt = yi - y
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




def velovect_key(fig, axes, quiver, shrink=0.15, U=1., angle=0., label='1', color='k', arrowstyle='->', linewidth=.5, fontproperties={'size': 5}):
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
    :param labelpos: 标签位置
    :param color: 颜色
    :param fontproperties: 字体属性
    :return: None
    '''
    axes_sub = fig.add_axes([0, 0, 1, 1])
    adjust_sub_axes(axes, axes_sub, shrink=shrink)
    # 不显示刻度和刻度标签
    axes_sub.set_xticks([])
    axes_sub.set_yticks([])
    axes_sub.set_xlim(-1, 1)
    axes_sub.set_ylim(-2, 1)
    U = U * quiver[1] * 1e2 / 2.6
    # 绘制图例
    arrow = patches.FancyArrowPatch(
        (-U*np.cos(angle), -U*np.sin(angle)), (U*np.cos(angle), U*np.sin(angle)), arrowstyle=arrowstyle,
               mutation_scale=10, linewidth=linewidth)
    axes_sub.add_patch(arrow)
    axes_sub.text(0, -1.5, label, ha='center', va='center', color=color, fontproperties=fontproperties)


if __name__ == '__main__':
    "test"
    x = np.linspace(-180, 180, 1440)
    y = np.linspace(-90, 90, 720)
    Y, X = np.meshgrid(y, x)
    U = np.ones(X.shape).T
    V = np.zeros(X.shape).T
    fig = matplotlib.pyplot.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    a1 = velovect(ax1, x, y, U, V, regrid=5, lon_trunc=180, scale=0.02,color='black')
    velovect_key(fig, ax1, a1)
    plt.savefig('D:/PyFile/pic/test.png', dpi=1000)
    plt.show()
