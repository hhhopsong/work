
def search(data, info):
    count = -1
    if info in data:
        for i in data:
            count += 1
            if info == i:
                return count
    else:
        return None


members = ['Joe', 'Linda', 'May', 'George']
name = input('输入要查找的姓名: ')


ls = [('在', ',索引是', search(members, name)) if search(members, name) is not None else ('不在', '', '')]
print(f'{name}{ls[0][0]}成员列表中{ls[0][1]}{ls[0][2]}')
