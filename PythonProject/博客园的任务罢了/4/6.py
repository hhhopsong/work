
with open('data6_1.txt', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')

data1 = [str(i).split('\t') for i in data]

data1.sort(key=lambda x: x[2], reverse=True)
with open('data6_2.txt', 'w+',encoding='utf-8')as f:
    for line in data1:
        for i in line:
            f.write(i + '\t')
            print(i + '\t', end='')
        f.write('\n')
        print()
