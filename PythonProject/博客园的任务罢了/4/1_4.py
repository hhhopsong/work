# 统计文件data1_2.txt行数  （不包括空白行）
# data1_2.txt中的空白行包括由空格、Tab键（\t）、换行（\n）构成的空白行
# task1_4.py

with open('data1_2.txt', 'r', encoding = 'uft-8') as f:
    n = 0
    for line in f:
        if line.isspace():
            continue
        n += 1

print(f'共{n}行')
