
data = {1930:'乌拉圭',
        1934:'意大利',
        1938:'意大利',
        1950:'乌拉圭',
        1954:'西德',
        1958:'巴西',
        1962:'巴西',
        1966:'英格兰',
        1970:'巴西',
        1974:'西德',
        1978:'阿根廷',
        1982:'意大利',
        1986:'阿根廷',
        1990:'西德',
        1994:'巴西',
        1998:'法国',
        2002:'巴西',
        2006:'意大利',
        2010:'西班牙',
        2014:'德国',
        2018:'法国'}
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


