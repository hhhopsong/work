import pprint
import 搜题入库


with open('习概题库.txt', 'r', encoding='utf-8') as f:
    data0 = f.readlines()
questions = dict()
ques = ''

for i in range(len(data0)):
    if '【' in data0[i]:
        ques = ''
        ques = data0[i][data0[i].index('】')+1:].strip()
        questions[ques] = ''
    else:
        questions[ques] += data0[i]
pprint.pprint(len(questions.items()))
with open('习概.txt', 'r', encoding='gbk') as f:
    data1 = f.readlines()
ans = 搜题入库.join_split(data1, '正确答案')
for i in range(len(ans)):
    ans[i] = ans[i].strip()
    if i < 10:
        ans[i] = ans[i][1:]
    elif i < 100:
        ans[i] = ans[i][2:]
    else:
        ans[i] = ans[i][3:]
for i in range(len(ans)):
    bridge = ans[i].split('\n')
    bridge[0] = bridge[0][:-11]
    ans[i] = '\n'.join(bridge)

result = []
num = 0
for i in range(len(questions)):
    info = 0
    for ii in range(len(ans)):
        if questions[i] == ans[ii].split('\n')[0].strip():
            result.append(f'{i+1}.' + questions[i] + '\n' + '\n'.join(ans[ii].split('\n')[1:]) + '\n')
            info = 1
            num += 1
            break
    if info == 0:
        result.append(f'{i+1}.' + questions[i] + '\n' + 'WARNING: 未找到答案\n')
#pprint.pprint(len(result))