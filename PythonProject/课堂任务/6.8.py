import jieba
import matplotlib.pyplot as mat
import random

'''mat.rcParams['font.sans-serif'] = ['SimHei']
days = range(1, 11)

high_temp = [random.randint(30, 40) for i in range(10)]
low_temp = [random.randint(19, 23) for ii in range(10)]

mat.title('天气预报', fontsize=28)
mat.xlabel('日期')
mat.ylabel('温度')
mat.subplot(2, 1, 1)
mat.bar(days, high_temp, color='orange', width=0.4)
mat.xticks(range(1, 11))
mat.subplot(212)
mat.plot(days, low_temp, color='b')
mat.figure()
#  mat.plot(days, low_temp, color='blue', marker='o')
mat.plot(days, low_temp, color='blue', marker='o')
mat.legend(['最高温', '最低温'])  # 图例
mat.xticks(range(1, 11))

mat.show()'''

print(jieba.lcut_for_search('申富明的npy是母零,李薪阳真的不是猛一'))
