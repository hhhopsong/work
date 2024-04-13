# 统计文件data_1.txt行数  （不包括空白行）
# data1_1.txt中的空白行都是由\n构成的空白行
# task1_1.py

with open('data1_1.txt', 'r', encoding = 'utf-8') as f:
    data = f.readlines()  #一次读入所有数据，按行读取，保存到列表中

n = 0
for line in data:
    if line.strip('\n') == '':
        continue
    n += 1
print(f'共{n}行')
