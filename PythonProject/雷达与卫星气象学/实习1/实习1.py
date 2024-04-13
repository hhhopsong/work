import numpy as np
import matplotlib.pyplot as plt

# 数据处理
A = np.zeros((256, 256))
for i in range(0, 256):
    for j in range(0, 256):
        A[i, j] = i + j
B = np.uint8(A)
C = A / 2
D = np.uint8(C)
# 绘制图像
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.imshow(B, cmap='gray')
ax2 = fig.add_subplot(212)
ax2.imshow(D, cmap='gray')
ax1.axis('off')
ax2.axis('off')
plt.show()
