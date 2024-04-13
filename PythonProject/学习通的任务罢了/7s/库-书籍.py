with open('book_info.txt', encoding='utf-8') as data_lines:
    datas = data_lines.read().split('\n')
data = [i.strip('\n') for i in datas if i.strip('\n') != '']
with open('book_info2.txt', 'w+', encoding='utf-8') as data_lines:
    for i in data:
        data_lines.write(i + '\n')
y=        print(i)
