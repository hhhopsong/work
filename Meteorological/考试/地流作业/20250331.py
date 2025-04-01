import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

nx, ny = 200, 200  # 网格大小
x = np.linspace(-10, 10, nx)
y = np.linspace(-10, 10, ny)
X, Y = np.meshgrid(x, y)

Re = 1.
n0 = 1.
k = 2
l = 2
sigma = 2
θ = k * X + l * Y

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-3, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(elev=30, azim=225)


def animate(frame):
    ax.clear()
    t = frame / 20.0  # 时间参数

    n = Re * n0 * np.cos(θ - sigma * t)

    surf = ax.plot_surface(X, Y, n, cmap='Blues', linewidth=0, antialiased=True)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=30, azim=225)
    return surf

ani = animation.FuncAnimation(fig, animate, frames=200, interval=300, blit=False)
ani.save('wave.gif', writer='pillow', fps=30)