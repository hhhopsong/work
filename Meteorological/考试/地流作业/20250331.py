import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

x = np.arange(-10, 10.25, .25)
y = np.arange(-10, 10.25, .25)
X, Y = np.meshgrid(x, y)

Re = .3
n0 = 1.
k = .5
l = .5
f = 7.2921 * 10**-5
sigma = f * 1.25
θ = k * X + l * Y
phi =  2

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


def animate(frame):
    ax.clear()
    t = frame * 10e2  # 时间参数

    n = Re * n0 * np.cos(θ - sigma * t)
    u = - (9.8 * np.abs(n0)) / (f**2 - sigma**2) * (k * sigma * np.cos(θ - sigma * t + phi) - l * f * np.sin(θ - sigma * t + phi))
    v = - (9.8 * np.abs(n0)) / (f**2 - sigma**2) * (l * sigma * np.cos(θ - sigma * t + phi) + k * f * np.sin(θ - sigma * t + phi))

    surf = ax.plot_surface(X, Y, n, cmap='coolwarm', lw=0.5, rstride=8, cstride=8,alpha=0.4)

    ax.contourf(X, Y, n, zdir='z', offset=-6, cmap='coolwarm')
    ax.quiver(X[::4, ::4], Y[::4, ::4], np.full_like(n, -6)[::4, ::4], u[::4, ::4], v[::4, ::4], np.full_like(n, 0)[::4, ::4], length=0.000004, color='k')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-6, 2)
    ax.view_init(elev=30, azim=225)
    return surf


ani = animation.FuncAnimation(fig, animate, frames=200, interval=30, blit=False)
ani.save('wave.gif', writer='pillow', fps=30, dpi=300)