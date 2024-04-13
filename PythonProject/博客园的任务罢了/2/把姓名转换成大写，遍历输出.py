
# 把姓名转换成大写，遍历输出
name_list = ['david bowie', 'louis armstrong', 'leonard cohen', 'bob dylan', 'cocteau twins']

n = 1
for name in name_list:
    print(f'{n}:{name.title()}')
    n += 1
