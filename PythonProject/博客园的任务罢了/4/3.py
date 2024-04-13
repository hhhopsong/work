# 不使用csv模块，使用python内置的读写操作进行csv格式的读写
# 把两维列表中的数据写入CSV格式的文件data3.csv中
# task3.py

ls = [ ['城市', '大致人口'],
       ['南京', '850万'],
       ['纽约', '2300万'],
       ['东京', '3800万'],
       ['巴黎', '1000万'] ]

with open('data3.csv', 'w', encoding = 'utf-8') as f:
    for line in ls:
        data = ','.join(line) + '\n'
        f.write(data)


# 从data3.csv中读出数据，把逗号换成\t,分行打印输出到屏幕上
with open('data3.csv', 'r', encoding = 'utf-8')as f:
    data = f.read()

print(data.replace(',', '\t'), end = '')
