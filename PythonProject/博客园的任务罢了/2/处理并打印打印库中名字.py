
name_list = ['david bowie', 'louis armstrong', 'leonard cohen', 'bob dylan', 'cocteau twins']
n = 0
name_list.sort()
for name in name_list:
    x = name.title()
    n += 1
    print(f'{n}.{x}')
