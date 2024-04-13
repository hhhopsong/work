import cmaps
import numpy as np
from scipy.ndimage import filters


# 计算兰伯托地图投影放大系数
def lambert_map_scale(i, j):
    # km, s
    d = 300000
    a = 6371000
    k = 0.7156
    le = 11423370
    Ω = 7.292e-5
    l = np.sqrt((10*d+i*d) ** 2 + (-4*d+j*d) ** 2)
    L = (le ** (2/k) - l ** (2/k)) / (le ** (2/k) + l ** (2/k))
    f = 2 * Ω * L
    m = k * l / (a * np.sqrt(1 - L ** 2))
    return m, f

# 计算绝对涡度和绝对涡度平流
def abs_vorticity(z):
    shape = z.shape
    d = 300000
    f_ = 1e-4
    n = np.zeros(shape)
    n.fill(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            m, f = lambert_map_scale(i, j)
            if i == 0 or i == shape[0] - 1 or j == 0 or j == shape[1] - 1:
                # 边界涡度计算
                n[i, j] = 0
            elif i == 1 or i == shape[0] - 2 or j == 1 or j == shape[1] - 2:
                # γ边界条件
                n[i, j] = 9.8 * m**2 / (d**2 * f_) * (z[i+1, j] - 2*z[i, j] + z[i-1, j] + z[i, j+1] - 2*z[i, j] + z[i, j-1]) + f
            else:
                n[i, j] = 9.8 * m**2 / (d**2 * f_) * (z[i+1, j] - 2*z[i, j] + z[i-1, j] + z[i, j+1] - 2*z[i, j] + z[i, j-1]) + f
    # 计算绝对涡度平流F
    ## F = -1/4 * ((z[i-1, j] - z[i+1, j]) * (n[i, j+1] - n[i, j-1]) - (z[i, j+1] - z[i, j-1]) * (n[i-1, j] - n[i+1, j]))
    F = np.zeros(shape)
    F.fill(0)
    F[1:-1, 1:-1] = -1/4 * ((z[0:-2, :] - z[2:, :])[:, 1:-1] * (n[:, 2:] - n[:, 0:-2])[1:-1, :] - (z[:, 0:-2] - z[:, 2:])[1:-1, :] * (n[2:, :] - n[0:-2, :])[:, 1:-1])
    return F

# 超张驰迭代法求解正压涡度方程
def hyper_viscosity(z, F, a=1.5):
    E = 1e-5
    α = a  # 超张驰系数 （1.2到1.8之间）
    shape = z.shape
    x = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == 0 or i == shape[0] - 1 or j == 0 or j == shape[1] - 1:
                x[i, j] = 0
            elif i == 1 or i == shape[0] - 2 or j == 1 or j == shape[1] - 2:
                x[i, j] = 0
    while True:
        delta = 0.0
        for i in range(2, shape[0]-2):
            for j in range(2, shape[1]-2):
                before = x[i, j]
                x[i, j] = x[i, j] + α / 4 * (x[i+1, j] + x[i-1, j] + x[i, j+1] + x[i, j-1] - 4*x[i, j] + F[i, j])
                delta = max(delta, abs(x[i, j] - before))
        if delta < E:
            break
    return x

# 时间积分
def time_integration(z, x, dt, N=1):
    global n
    shape = z.shape
    if N == 0:
        for i in range(shape[0]):
            for j in range(shape[1]):
                z[i, j] = z[i, j] + x[i, j] * dt
        n += 1
        return z
    else:
        for i in range(shape[0]):
            for j in range(shape[1]):
                z[i, j] = z[i, j] + 2 * x[i, j] * dt
        n += 1
        return z

# 进行积分
def integral(z, N=6, dt=360, fliter_index=.1):
    global n
    n = 0
    z0 = z
    z1 = z
    N = N * 3600 // dt
    for i in range(N):
        if n == 0:
            x = hyper_viscosity(z, abs_vorticity(z0), a=1.5)
            z1 = filters.gaussian_filter(time_integration(z0, x, dt, n), fliter_index)
            #z1 = time_integration(z0, x, dt, n)
        elif n % 2 == 1:
            x = hyper_viscosity(z, abs_vorticity(z1), a=1.5)
            #z0 = filters.gaussian_filter(time_integration(z0, x, dt, n), fliter_index)
            z0 = time_integration(z0, x, dt, n)
        else:
            x = hyper_viscosity(z, abs_vorticity(z0), a=1.5)
            #z1 = filters.gaussian_filter(time_integration(z1, x, dt, n), fliter_index)
            z1 = time_integration(z1, x, dt, n)
    if n % 2 == 1:
        return z0
    else:
        return z1


# 500hPa初始位势高度场
with open("h730429.dat", "r") as f:
    data = f.readlines()
    z = []
    for line in data:
        z.append([eval(i) for i in line.split()])
z0 = np.array(z)*10
z1 = np.array(z)*10
z_t = integral(z1, N=24, dt=1200, fliter_index=0)
# 画图
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)
a_ = ax1.contour(z0[2:-2, 2:-2], colors='black', linewidths=1, level=[i for i in range(5080, 5880, 40)])
a = ax1.contourf(z_t[2:-2, 2:-2], cmap=cmaps.MPL_gist_rainbow_r, levels=[i for i in range(5080, 5880, 40)])
ax1.invert_yaxis()
plt.clabel(a_, inline=True, fontsize=10, fmt='%d', inline_spacing=10)
cb1 = plt.colorbar(a)
plt.show()
'''for i in range(16):
    if i <= 8:
        print(f' {i + 1}\t', end=" ")
    else:
        print(f'{i + 1}\t', end=" ")
    for j in range(20):
        print(f'{lambert_map_scale(i, j)[0]:.3f}', end=" ")
    print()
print('   ', end="")
for j in range(20):
    print(f'{j + 1:> 5}', end=" ")'''