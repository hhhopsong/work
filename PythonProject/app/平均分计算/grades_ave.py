grade = []
with open('grades.txt', 'r') as f:
    grade = f.readlines()
for i in range(len(grade)):
    grade[i] = grade[i].strip().split('\t')
ave_important1 = [0, 0]
ave_important = [0, 0]
ave_unimportant = [0, 0]
ave_all = [0, 0]
for i in range(len(grade)):
    if grade[i][4] == '优秀':
        grade[i][4] = '92'
    elif grade[i][4] == '良好':
        grade[i][4] = '85'
    elif grade[i][4] == '中等':
        grade[i][4] = '75'
    elif grade[i][4] == '及格':
        grade[i][4] = '65'
    elif grade[i][4] == '不及格':
        grade[i][4] = '50'
    if '必' in grade[i][3]:
        ave_important[0] += eval(grade[i][4]) * eval(grade[i][7])
        ave_important[1] += eval(grade[i][7])
        if '体育' not in grade[i][2]:
            ave_important1[0] += eval(grade[i][4]) * eval(grade[i][7])
            ave_important1[1] += eval(grade[i][7])
    elif '选' in grade[i][3]:
        ave_unimportant[0] += eval(grade[i][4]) * eval(grade[i][7])
        ave_unimportant[1] += eval(grade[i][7])
    ave_all[0] += eval(grade[i][4]) * eval(grade[i][7])
    ave_all[1] += eval(grade[i][7])
print(f'*' * 30)
ave_important1_ = [0, 0]
ave_important_ = [0, 0]
ave_unimportant_ = [0, 0]
ave_all_ = [0, 0]
item0 = '1'
xueqi = 0
for i in range(len(grade)):
    item1 = grade[i][1]
    if item1 == item0 and i != len(grade) - 1:
        if grade[i][4] == '优秀':
            grade[i][4] = '92'
        elif grade[i][4] == '良好':
            grade[i][4] = '85'
        elif grade[i][4] == '中等':
            grade[i][4] = '75'
        elif grade[i][4] == '及格':
            grade[i][4] = '65'
        elif grade[i][4] == '不及格':
            grade[i][4] = '50'
        if '必' in grade[i][3]:
            ave_important_[0] += eval(grade[i][4]) * eval(grade[i][7])
            ave_important_[1] += eval(grade[i][7])
            if '体育' not in grade[i][2]:
                ave_important1_[0] += eval(grade[i][4]) * eval(grade[i][7])
                ave_important1_[1] += eval(grade[i][7])
        elif '选' in grade[i][3]:
            ave_unimportant_[0] += eval(grade[i][4]) * eval(grade[i][7])
            ave_unimportant_[1] += eval(grade[i][7])
        ave_all_[0] += eval(grade[i][4]) * eval(grade[i][7])
        ave_all_[1] += eval(grade[i][7])
    else:
        print(f'*第{xueqi+1}学期必修均分(-体育):{ave_important1_[0] / ave_important1_[1]:.4f}{"*": >3}')
        print(f'*第{xueqi+1}学期必修均分(+体育):{ave_important_[0] / ave_important_[1]:.4f}{"*": >3}')
        print(f'*' * 30)
        item0 = item1
        xueqi += 1
        ave_important1_ = [0, 0]
        ave_important_ = [0, 0]
        ave_unimportant_ = [0, 0]
        ave_all_ = [0, 0]
        if item1 == item0:
            if grade[i][4] == '优秀':
                grade[i][4] = '92'
            elif grade[i][4] == '良好':
                grade[i][4] = '85'
            elif grade[i][4] == '中等':
                grade[i][4] = '75'
            elif grade[i][4] == '及格':
                grade[i][4] = '65'
            elif grade[i][4] == '不及格':
                grade[i][4] = '50'
            if '必' in grade[i][3]:
                ave_important_[0] += eval(grade[i][4]) * eval(grade[i][7])
                ave_important_[1] += eval(grade[i][7])
                if '体育' not in grade[i][2]:
                    ave_important1_[0] += eval(grade[i][4]) * eval(grade[i][7])
                    ave_important1_[1] += eval(grade[i][7])
            elif '选' in grade[i][3]:
                ave_unimportant_[0] += eval(grade[i][4]) * eval(grade[i][7])
                ave_unimportant_[1] += eval(grade[i][7])
            ave_all_[0] += eval(grade[i][4]) * eval(grade[i][7])
            ave_all_[1] += eval(grade[i][7])
print(f'*必修课平均分(不含体育):{ave_important1[0] / ave_important1[1]:.4f}{"*": >3}')
print(f'*必修课平均分(含体育):{ave_important[0] / ave_important[1]:.4f}{"*": >4}')
print(f'*选修课平均分:{ave_unimportant[0] / ave_unimportant[1]:.4f}{"*": >11}')
print(f'*所有课平均分:{ave_all[0] / ave_all[1]:.4f}{"*":>11}')
print(f'*' * 30)