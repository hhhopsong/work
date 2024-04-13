# 浅层神经网络的不稳定性试验设计

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# 试验设计思路：Xobs为生成-1~1之间的随机数，Yobs根据设置的神经网络结构生成，如下图。
# 不卡死设置初始权重的种子，将训练集上训练所得参数应用于测试集，计算得出测试集Yobs与通过该神经网络计算得到的YML的差值，将运行10次得到的结果进行集合分析
# 可发现这10个集合成员的标准差较大，模型每次训练所的结果存在较大波动，即表现出了小样本时浅层神经网络的不稳定性。



