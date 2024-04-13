import pandas as pd

def str_insert(str0, pos, str1, special='no_end'):
    if special == 'no_end':
        str_list = list(str0)
        str_list.insert(pos, str1)
        str_out = ''.join(str_list)
        return str_out
    elif special == 'end':
        str_list = list(str0)
        str_list.append(str1)
        str_out = ''.join(str_list)
        return str_out


name = input('请输入源文件名:')
time0 = input('请输入开学第一周周一日期(格式:2021.02.21):')
name0 = input('请命名处理后文件:')

data = pd.read_excel(f'{name}.xls','Sheet1',index_col=0)
data.to_csv(f'{name}.csv',encoding='gbk')
with open(f'{name}.csv', 'r', encoding='gbk') as line:
    dates = line.readlines()

days = []
for j in range(2, 7):
    days.append([])
    for i in range(2, 8):
        if dates[i].split(',')[j] != ' ':
            days[j-2].append(dates[i].split(',')[j])

for i in range(5):
    days[i] = [ii.replace('◆', '◇').split('◇') for ii in days[i]]


gets = []
for j in range(5):
    gets.append([])
    dx = -1
    for i in range(len(days[j])):
        dx += 1
        gets[j].append([])
        subject = ''
        subject2 = ''

        all_time = ''
        all_time2 = ''
        week = '1-18'
        week2 = '1-18'
        time = ''
        time2 = ''
        double = ''
        double2 = ''

        room = room2 = ''
        teacher = teacher2 = ''
        for ii in range(len(days[j][i])):
            if ii == 0:
                subject = days[j][i][ii]
            elif ii == 1:
                if len(days[j][i]) < 5:
                    teacher = days[j][i][ii].strip('()')
                    room = 'N/A'
                else:
                    for iii in range(len(days[j][i][ii])):
                        teacher = days[j][i][ii].split('(')[0]
                        week0 = days[j][i][ii].split('(')[1].strip('()')
                        week = str_insert('', 0, week0)
            elif ii == 2:
                if len(days[j][i]) < 5:
                    time = str_insert(days[j][i][ii].strip('{}'), 1, '-', special='end')
                else:
                    room = days[j][i][ii]
            elif ii == 4:
                if days[j][i][ii].split("{")[0] != '':
                    double = str_insert('()', 1, days[j][i][ii].split("周")[0])
                time = str_insert(str_insert(days[j][i][ii].split('{')[1].strip('节}'), 1, '-'), 3, '节', special='end')
            elif ii == 5:
                subject2 = days[j][i][ii]
            elif ii == 6:
                for iii in range(len(days[j][i][ii])):
                    teacher2 = days[j][i][ii].split('(')[0]
                    week0 = days[j][i][ii].split('(')[1].strip('()')
                    week2 = str_insert('', 0, week0)
            elif ii == 7:
                room2 = days[j][i][ii]
            elif ii == 9:
                double2 = str_insert('()', 1, days[j][i][ii].split('周')[0])
                time2 = str_insert(str_insert(days[j][i][ii].split('{')[1].strip('节}'), 1, '-'), 3, '节', special='end')
        subject = subject
        all_time = week + double + '周' + time
        room = room
        teacher = teacher
        gets[j][dx].append(subject)
        gets[j][dx].append(all_time)
        gets[j][dx].append(room)
        gets[j][dx].append(teacher)
        gets[j][dx] = '\n'.join(gets[j][dx]).strip('\n')
        if len(days[j][i]) > 5:
            dx += 1
            gets[j].append([])
            subject2 = subject2
            all_time2 = week2 + double2 + '周' + time2
            room2 = room2
            teacher2 = teacher2
            gets[j][dx].append(subject2)
            gets[j][dx].append(all_time2)
            gets[j][dx].append(room2)
            gets[j][dx].append(teacher2)
            gets[j][dx] = '\n'.join(gets[j][dx]).strip('\n')

biggest = max([len(i) for i in gets])
results = []
for i in range(5):
    for ii in range(len(gets[i]), biggest):
        gets[i].insert(ii, '\n\n\n')

for i in range(biggest):
    results.append([])
    for ii in range(5):
        bridge = gets[ii][i]
        results[i].append(bridge)

for i in range(biggest, 11):
    results.append([])
    for ii in range(5):
        results[i].insert(ii, '\n\n\n')
for i in range(11):
    for ii in range(5):
        results[i][ii] = ''.join(results[i][ii])


a = 0
b = '\n'
shabi = "课程时长\n, （分钟）"
nice0 = '周一,周二,周三,周四,周五,周六,周日,,开学第一周周一日期,,' + time0 + '\n'
nice1 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,,上课时间,"{shabi}"\n'
a += 1
nice2 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,第1节,8:00,45\n'
a += 1
nice3 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,第2节,8:55,45\n'
a += 1
nice4 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,第3节,10:10,45\n'
a += 1
nice5 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,第4节,11:05,45\n'
a += 1
nice6 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,第5节,13:45,45\n'
a += 1
nice7 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,第6节,14:40,45\n'
a += 1
nice8 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,第7节,15:55,45\n'
a += 1
nice9 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,第8节,16:50,45\n'
a += 1
nice10 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,第9节,18:45,45\n'
a += 1
nice11 = f'"{results[a][0]}", "{results[a][1]}", "{results[a][2]}", "{results[a][3]}", "{results[a][4]}",,,,第10节,19:45,45\n'


with open(f'{name0}.csv', 'w+', encoding='gbk') as line0:
    line0.writelines(str(nice0).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice1).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice2).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice3).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice4).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice5).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice6).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice7).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice8).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice9).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice10).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))
    line0.writelines(str(nice11).replace('"(', '"').replace(')"', '"').replace("\'", "'").replace("'", "").replace(", ", ",").replace('"\n,\n,\n,"', ""))

col1 = '周一,周二,周三,周四,周五,周六,周日,,开学第一周周一日期,,2022.02.21\n'
col2 = '"课程名称（必填）\n', '周数节数（必填）\n', '教室（必填）\n', '教师（非必填）",,,,,,,,,上课时间,"课程时长\n', '（分钟）"\n'
col3 = ',"举例课程A\n', '1-17周1-2节\n', '教室\n', '教师","举例课程B\n', '1-17(单)周3-6节\n', '教室\n', '教师","举例课程C\n', '1-6,8-9周7-8节\n', '教室\n', '教师","举例课程D\n', '1-17周3,5节\n', '教室\n', '教师",,,,第1节,8:00,45\n'
col4 = ',"举例课程E\n', '1-6(双),8-9周3,5,8节\n', '教室\n', '教师","举例课程F\n', '1-6(单),11,13周3,5,8节\n', '教室\n', '教师",,,,,,第2节,8:55,45\n'
col5 = ',,,,,,,,第3节,10:10,45\n'
col6 = ',,,,,,,,第4节,11:05,45\n'
col7 = ',,,,,,,,第5节,13:45,45\n'
col8 = ',,,,,,,,第6节,14:40,45\n'
col9 = ',,,,,,,,第7节,15:55,45\n'
col10 = ',,,,,,,,第8节,16:50,45\n'
col11 = ',,,,,,,,第9节,18:45,45\n'
col12 = ',,,,,,,,第10节,19:45,45\n'
