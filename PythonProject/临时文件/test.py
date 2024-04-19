import numpy as np
from scipy.stats import t
N=36
t_critical = t.ppf(0.975, N - 2)
r = [0.25, 0.35, 0.45, 0.50, 0.66, 0.52]
for i in range(len(r)):
    pas = 0
    t_ = r[i] * np.sqrt((N - 2) / (1 - r[i] ** 2))
    if t_ > t_critical:
        pas = 1
    print(f'{r[i]:.2f}\t{pas}')