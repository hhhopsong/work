from math import *


data0 = open('全年处理结果.csv', 'r+', encoding='utf-8')
data0 = data0.readlines()

year = data0[1:17]
Spr = data0[18:34]
Sum = data0[35:51]
Aut = data0[52:68]
Win = data0[69:85]
data = [year, Spr, Sum, Aut, Win]


for i in data:
    for ii in range(len(i)):
        i[ii] = i[ii].strip('\n')
        i[ii] = i[ii].split(',')[0] + ',' + str((eval(i[ii].split(',')[1].strip('%'))/100)*eval(i[ii].split(',')[2]))


Average_wind = [0, 0, 0, 0, 0]
A_num = 0
for i in data:
    x = 0
    y = 0
    rad = 90
    for ii in range(len(i)):
        rad = rad * pi / 180
        x += eval(i[ii].split(',')[1]) * cos(rad)
        y += eval(i[ii].split(',')[1]) * sin(rad)
        rad -= 22.5
    Average_wind[A_num] = 90 -(atan2(y, x) * 180 / pi)
    A_num += 1

print(Average_wind)