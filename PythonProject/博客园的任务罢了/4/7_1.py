import random

with open('data7.txt', 'r', encoding='gbk')as f:
    data = f.read().split('\n')

info = eval(input('输入随机抽点人数：'))
# 随机抽取模块
output = set()
while len(output) < info:
    rand = random.randint(0, len(data) - 1)
    output.add(str(data[rand]))
# 写入模块
with open('lucky.txt', 'w+', encoding='utf-8')as f:
    for line in output:
        f.write(line + '\n')
        print(line)
