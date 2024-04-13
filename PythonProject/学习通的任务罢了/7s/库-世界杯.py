
with open('world_cup_champion_info.txt') as line:
    datas = line.readlines()
data1 = [i.split(':')[0] for i in datas]
data2 = [i.split(':')[1].strip(' \n') for i in datas]
data = dict(zip(data1, data2))

info1 = input('输入年份：')
if info1 in data:
    print(f'{info1}年冠军球队是{data[info1]}')
else:
    print(f'{info1}年没有举办世界杯')

info2 = input('输入队伍：')
if info2 in data.values():
    print(f'{info2}夺冠年份：', end='')
    for i in data.keys():
        if info2 == data[i]:
            print(i, end=' ')
else:
    print(f'{info2}没有获得过冠军')
