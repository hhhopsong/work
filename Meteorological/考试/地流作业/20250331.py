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
sigma = f * 2
θ = k * X + l * Y
phi = 0
L = 2
N = 0.15
Cx = f * 2
C0 = f * 4
H0 = 0.01

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


def animate(frame):
    ax.clear()
    t = frame * 10e2  # 时间参数

    # # g
    # n = Re * n0 * np.cos(θ - sigma * t)
    # u = - (9.8 * np.abs(n0)) / (f**2 - sigma**2) * (k * sigma * np.cos(θ - sigma * t + phi) - l * f * np.sin(θ - sigma * t + phi))
    # v = - (9.8 * np.abs(n0)) / (f**2 - sigma**2) * (l * sigma * np.cos(θ - sigma * t + phi) + k * f * np.sin(θ - sigma * t + phi))

    n = n0 * (np.cos(N * np.pi * Y / L) - L * f / N / np.pi / Cx * np.sin(N * np.pi * Y / L)) * np.cos(k * X - sigma * t + phi)
    u = n0 / H0 * (C0**2 / Cx * np.cos(N * np.pi / L * Y) - L * f / N / np.pi * np.sin(N * np.pi / L * Y)) * np.cos(k * X - sigma * t + phi)
    v = -n0 * L / H0 / sigma / N / np.pi * (f**2 + C0**2 * N**2 * np.pi**2 / L**2) * np.sin(N * np.pi / L * Y) * np.sin(k * X - sigma * t + phi)

    surf = ax.plot_surface(X, Y, n, cmap='coolwarm', lw=0.5, rstride=8, cstride=8,alpha=0.4)

    ax.contourf(X, Y, n, zdir='z', offset=-6, cmap='coolwarm')
    ax.quiver(X[::4, ::4], Y[::4, ::4], np.full_like(n, -6)[::4, ::4], u[::4, ::4], v[::4, ::4], np.full_like(n, 0)[::4, ::4], length=0.4, color='k')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-6, 2)
    ax.view_init(elev=30, azim=225)
    return surf


ani = animation.FuncAnimation(fig, animate, frames=200, interval=30, blit=False)
ani.save('wave.gif', writer='pillow', fps=30, dpi=300)