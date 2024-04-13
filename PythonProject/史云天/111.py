# 模拟参数为0.9的泊松过程。绘制出其样本路径图
# 生成泊松过程的样本路径
'''import numpy as np
lam = 0.9
T = 1
N = 1000
dt = T/N
t = np.linspace(0,T,N+1)
X = np.zeros(N+1)
X[0] = 0
for i in range(N):
    X[i+1] = X[i] + np.random.poisson(lam*dt)
# 绘制泊松过程的样本路径
import matplotlib.pyplot as plt
plt.plot(t,X)
plt.xlabel('t')
plt.ylabel('X(t)')
plt.title('Poisson Process')
plt.show()
'''
#模拟到达率为1，服务速率为2的M/M/1排队系统。绘制出其样本路径图
# 生成M/M/1排队系统的样本路径
'''import numpy as np
lam = 1
mu = 2
T = 10
N = 1000
dt = T/N
t = np.linspace(0,T,N+1)
X = np.zeros(N+1)
X[0] = 0
for i in range(N):
    X[i+1] = X[i] + np.random.poisson(lam*dt) - np.random.poisson(mu*dt)
# 绘制M/M/1排队系统的样本路径
import matplotlib.pyplot as plt
plt.plot(t,X)
plt.xlabel('t')
plt.ylabel('X(t)')
plt.title('M/M/1 Queue')
plt.show()
'''
#模拟突发过程，其马尔可夫到达过程矩阵为D0=(-100,0;-1,1),D1=(99,1;0,0)。绘制出其样本路径图
# 生成突发过程的样本路径
import numpy as np
T = 10
N = 1000
dt = T/N
t = np.linspace(0,T,N+1)
X = np.zeros(N+1)
X[0] = 0
for i in range(N):
    if X[i] == 0:
        X[i+1] = X[i] + np.random.poisson(100*dt)
    else:
        X[i+1] = X[i] + np.random.poisson(dt)
# 绘制突发过程的样本路径
import matplotlib.pyplot as plt
plt.plot(t,X)
plt.xlabel('t')
plt.ylabel('X(t)')
plt.title('Burst Process')
plt.show()



