
x = 0
y = 0
list1 = []
word = '我的愿望清单'
while x < 3:
    info = input('输入想要加入愿望清单的事情:')
    list1.append(info)
    x += 1
print('{:-^50}'.format(word))
for i in list1:
    y += 1
    print(f'  {y}.{i}')

