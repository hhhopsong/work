import numpy as np
import matplotlib.pyplot as plt
import h5py

# 数据读取
file = "D:\雷达气象实习资料\实习1\FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_20180518080000_20180518081459_4000M_V0001.HDF"
info = h5py.File(file, 'r')
r=list(info.keys())

# 图像显示
count = 1
for i in r[r.index('NOMChannel01'):r.index('NOMChannel14')+1]:
    try:
        x = info[i]
        x = np.array(x)
        x[x > 4000] = 0
        plt.imshow(x, cmap='gray')
        plt.axis('off')
        plt.savefig(i + '-08.png', dpi=1500)
        plt.show()
        print(count)
        count += 1
    except:
        pass
