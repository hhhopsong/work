import numpy as np


A = np.mat([[4, -2, -4], [-2, 17, 10], [-4, 10, 9]])
r = max(np.linalg.eig(A)[0])
print(f'A的谱半径为{r}')
