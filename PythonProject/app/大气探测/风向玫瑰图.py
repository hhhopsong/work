'''
北（N）、北北东（NNE）、东北（NE）、东东北（ENE）
东（E）、东东南（ESE）、东南（SE）南东南（SSE）
南（S）、南西南（SSW）、西南（SW）、西西南（WSW）
西（W）、西西北（WNW）、西北（NW）、北西北（NNW）
根据风向的定义,从方位角在337.5°±11.25°范围内吹来的风的风向都记为NNW.
'''

with open('2004-2006期间午后14点的风速-合肥站.csv', 'r', encoding='utf-8') as line:
    dates = line.readlines()

exl = []

for i in range(1, len(dates)):
    exl.append(dates[i].split(',')[4] + ',' + dates[i].split(',')[5] + ',' + dates[i].split(',')[2])

direction = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
data = dict()
for i in direction:
    data[i] = []
    for j in exl:
        if i == j.split(',')[0] and eval(j.split(',')[1]) != 0:
            data[i].append(j.split(',')[1].strip('\n'))

for i in direction:
    average = 0.0
    percent = 0.0
    for ii in data[i]:
        average += eval(ii) / len(data[i])
    percent = len(data[i])/1096.0
    data[i] = f'{percent},{average}'

lines = open('全年处理结果.csv', 'w+', encoding='gbk')

lines.write('全年风向,风向频率,平均风速\n')

for i in direction:
    lines.write(i)
    lines.write(f',{eval(data[i].split(",")[0])*100:.2f}%,{eval(data[i].split(",")[1]):.2f}\n')


Spr = dict()
Sum = dict()
Aut = dict()
Win = dict()
seasons = [Spr, Sum, Aut, Win]
for season in seasons:
    for i in direction:
        season[i] = []
        if season == Spr:
            for j in exl:
                if i == j.split(',')[0] and eval(j.split(',')[1]) != 0 and 3 <= eval(j.split(',')[2]) <= 5:
                    season[i].append(j.split(',')[1].strip('\n'))
        if season == Sum:
            for j in exl:
                if i == j.split(',')[0] and eval(j.split(',')[1]) != 0 and 6 <= eval(j.split(',')[2]) <= 8:
                    season[i].append(j.split(',')[1].strip('\n'))
        if season == Aut:
            for j in exl:
                if i == j.split(',')[0] and eval(j.split(',')[1]) != 0 and 9 <= eval(j.split(',')[2]) <= 11:
                    season[i].append(j.split(',')[1].strip('\n'))
        else:
            for j in exl:
                if i == j.split(',')[0] and eval(j.split(',')[1]) != 0 and (12 <= eval(j.split(',')[2]) or eval(j.split(',')[2]) <= 2):
                    season[i].append(j.split(',')[1].strip('\n'))
    he = 0
    for i in direction:
        he += len(season[i])
    for i in direction:
        average = 0.0
        percent = 0.0
        for ii in season[i]:
            average += eval(ii) / len(season[i])
        percent = len(season[i])/he
        season[i] = f'{percent},{average}'

lines.write('春季风向,春季风向频率,春季平均风速\n')
for i in direction:
    lines.write(i)
    lines.write(f',{eval(Spr[i].split(",")[0])*100:.2f}%,{eval(Spr[i].split(",")[1]):.2f}\n')
lines.write('夏季风向,夏季风向频率,夏季平均风速\n')
for i in direction:
    lines.write(i)
    lines.write(f',{eval(Sum[i].split(",")[0])*100:.2f}%,{eval(Sum[i].split(",")[1]):.2f}\n')
lines.write('秋季风向,秋季风向频率,秋季平均风速\n')
for i in direction:
    lines.write(i)
    lines.write(f',{eval(Aut[i].split(",")[0])*100:.2f}%,{eval(Aut[i].split(",")[1]):.2f}\n')
lines.write('冬季风向,冬季风向频率,冬季平均风速\n')
for i in direction:
    lines.write(i)
    lines.write(f',{eval(Win[i].split(",")[0])*100:.2f}%,{eval(Win[i].split(",")[1]):.2f}\n')
lines.close()

