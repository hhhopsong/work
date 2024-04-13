# 统计其中只出现过一次的独特行行数，在屏幕上打印输出结果。
# task2.py

with open('dat2.txt', 'r', encoding = 'utf-8') as f:
    data = f.read().strip('\n')

unique_line = []
for line in data:
    if data.count(line) == 1:
        unique_line.append(line)

print(f'共{len(unique_line)}独特行')
for i in unique_line:
    print(i)
