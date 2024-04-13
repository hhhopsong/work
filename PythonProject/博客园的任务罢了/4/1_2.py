# 统计文件data_1.txt行数  （不包括空白行）
# data1_1.txt中的空白行都是由\n构成的空白行
# task1_2.py

with open('data1_1.txt', 'r', encoding = 'utf-8') as f:
    n = 0
    for line in f:  #逐行遍历处理
        if line.strip('\n') == '':
            continue
        n += 1

print(f'共{n}行')
