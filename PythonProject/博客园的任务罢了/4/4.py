# 使用csv模块进行csv格式的读写
# 把两维列表中的数据写入CSV格式的文件data4.csv中
# task4.py

import csv

ls = [ ['城市', '大致人口'],
       ['南京', '850万'],
       ['纽约', '2300万'],
       ['东京', '3800万'],
       ['巴黎', '1000万'] ]

with open('data4.csv', 'r', encoding = 'utf-8')as f:
    writer = csv.writer(f)  # 使用csv模块的writer类创建对象
    writer.writerows(ls)

# 从data4.csv中读出数据，把逗号换成\t,分行打印到屏幕上
with open('data4.txt', 'r', encoding = 'utf-8')as f:
    reader = csv.reader(f) # 使用csv模块的reader类创建对象
    for line in reader:
        print('\t'.join(line))
