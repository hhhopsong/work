from toolbar.significance_test import r_test

# print(r_test(62, 0.05))
# print(r_test(62, 0.01))
# print(r_test(62, 0.005))

from toolbar.corr_reg import cort, corr
import numpy as np

a = np.array([np.sin(i) for i in range(60)])
b = np.array([np.sin(i) + i for i in range(60)])

print(corr(a, b))
print(cort(a, b))

