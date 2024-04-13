'''
x = [1,2,3,2]

y = ['aa','bb','cc','dd','cc']

d = {}

for index in range(len(x)):

    d[x[index]] = y[index]

print(d)
'''


# !/usr/bin/python

# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]


# 列表
random = [(2, 2), (3, 4), (4, 1), (1, 3)]

# 指定第二个元素排序
random.sort(key=takeSecond)

# 输出类别
print('排序列表：', random)
