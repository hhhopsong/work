import random  # 随机数
import pyttsx3  # 语音输出
import datetime  # 时间模块


# 打印游戏信息
poke_data = {'1': '平民', '2': '\033[1;33m预言家\033[0m', '3': '\033[1;33m女巫\033[0m',
             '4': '\033[1;33m猎人\033[0m', '5': '\033[1;33m守卫\033[0m', 'a': '\033[1;31m普通狼人\033[0m',
             'b': '\033[1;31m白狼王\033[0m'}
print('目前支持的职业目前有(数字为代码)：1-平民，2-预言家，3-女巫，4-猎人，5-守卫，a-普通狼人,b-白狼王')

# 人物信息导入
info = input('输入人物信息(输入编号,用"-"隔开):').split('-')
num_info = len(info)
goodpeople = [poke_data[i] for i in info if '1' <= i <= '5']
badpeople = [poke_data[i] for i in info if 'a' <= i <= 'b']
num_commompeople = num_info - len(goodpeople) - len(badpeople)
major = goodpeople + badpeople + ['平民'] * num_commompeople

# 号码牌生成
number = [i + 1 for i in range(num_info)]

#  随机分配职业
random.shuffle(major)
data = dict(zip(number, major))

#  上帝分发身份信息
for i in range(num_info):
    engine = pyttsx3.init()
    engine.setProperty('200', '100')
    No = i + 1
    engine.say(f'请{No}号上前获取身份！')
    engine.runAndWait()
    input(f'按回车显示{No}号身份！')  # 身份显示----接口
    engine.say('你的身份是')
    engine.runAndWait()
    for a in range(5):
        print('*' * 200)
    print(f'{data[No]:*^200}')
    for a in range(5):
        print('*' * 200)
    engine.say('请您确认身份后按下回车键！')
    engine.runAndWait()
    input('身份确认')  # 身份确认----接口
    for a in range(10):
        print('*' * 200)

# Day 1
engine = pyttsx3.init()
engine.setProperty('200', '100')
engine.say('天黑请闭眼！')
engine.runAndWait()

# 1狼人
engine.say('狼人请睁眼！功能狼举手示意！请在互相确认身份后给出击杀的目标！')
engine.runAndWait()
for a in range(10):
    print('*' * 200)
print(f'仍存活的玩家是{list(data.keys())}')
info_1 = input('击杀目标(不选择杀人请直接按回车):')  # 狼人击杀目标----接口(务必以功能狼最大)
for a in range(20):
    print('*' * 200)
engine.say('狼人，请闭眼！')
engine.runAndWait()

# 2预言家
engine.say('预言家请睁眼！请给出你要查验的对象！')
engine.runAndWait()
for a in range(10):
    print('*' * 200)
print(f'仍存活的玩家是{list(data.keys())}')
info_2 = input(f'查验对象:')  # 预言家查验----接口
engine.say('他的身份是')
engine.runAndWait()
if data[eval(info_2)] != '\033[1;31m普通狼人\033[0m' and data[eval(info_2)] != '\033[1;31m白狼王\033[0m':
    print('\033[1;32m好人\033[0m')
else:
    print('\033[1;31m坏人\033[0m')
for a in range(5):
    print('*' * 200)
engine.say('预言家，请按回车确认后闭眼！')
engine.runAndWait()
input('预言家确认查验对象')  # 预言家确认查验对象---接口
for a in range(20):
    print('*' * 200)

# 3女巫
drug_good = 1
drug_bad = 1
engine.say('女巫请睁眼！今晚该号玩家死亡！你是否要使用解药！')
engine.runAndWait()
for a in range(10):
    print('*' * 200)
if drug_good == 1:
    if info_1 == '':
        print('\033[1;32m无人死亡！\033[0m')
    else:
        print(f'\033[1;31m{info_1}号死亡！\033[0m')
else:
    print('解药已用尽，不能得知死讯！')
info_3_1 = input('你是否要使用解药！(Y/N):')  # 女巫解药----接口
if info_3_1 == 'Y':
    drug_good -= 1
for a in range(10):
    print('*' * 200)
engine.say('你是否要使用毒药！')
engine.runAndWait()
info_3_2 = input('你是否要使用毒药！(Y/N)以及要使用的对象，并用“-”分开（例如：N;Y-3）:').split('-')  # 女巫毒药----接口
if info_3_2[0] == 'Y':
    drug_bad -= 1
for a in range(20):
    print('*' * 200)
engine.say('女巫，请闭眼！')
engine.runAndWait()

# 4猎人
info_4 = ''
if '4' in info:
    engine.say('猎人请睁眼！今晚你的开枪状态是')
    engine.runAndWait()
    for a in range(10):
        print('*' * 200)
    if drug_good < 0:
        print('\033[1;32m可以开枪！\033[0m')
    elif len(info_3_2) == 1:
        print('\033[1;32m可以开枪！\033[0m')
    elif data[eval(info_3_2[1])] != '\033[1;33m猎人\033[0m':
        print('\033[1;32m可以开枪！\033[0m')
    else:
        print('\033[1;31m不可以开枪！\033[0m')
    for a in range(5):
        print('*' * 200)
    engine.say('猎人，请闭眼！')
    engine.runAndWait()
    for a in range(20):
        print('*' * 200)

# 5守卫
info_5 = ''
if '5' in info:
    engine.say('守卫请睁眼！今晚你要守卫的人是')
    engine.runAndWait()
    for a in range(10):
        print('*' * 200)
    print(f'仍存活的玩家是{list(data.keys())}！')
    info_5 = input('今晚你要守卫的人是(不选择守人请直接按回车):')
    for a in range(20):
        print('*' * 200)
    engine.say('守卫，请闭眼！')
    engine.runAndWait()

# 死亡运算
death = []
if info_1 != '':
    if drug_good == 0:
        if info_3_1 == 'Y':
            if info_5 == info_1:
                del data[eval(info_1)]
                death.append(info_1)
    if info_3_1 == 'N':
        if info_5 != info_1:
            del data[eval(info_1)]
            death.append(info_1)
if info_3_2[0] == 'Y':
    if drug_bad == 0:
        del data[eval(info_3_2[1])]
        death.append(info_3_2[1])
random.shuffle(death)
if '平民' not in list(data.values()):
    engine.say('狼人获胜，游戏结束！')
    engine.runAndWait()
    exit()
elif '\033[1;31m普通狼人\033[0m' not in list(data.values()) and '\033[1;31m白狼王\033[0m' not in list(data.values()):
    engine.say('好人获胜，游戏结束！')
    engine.runAndWait()
    exit()
elif '\033[1;33m预言家\033[0m' not in list(data.values()) and '\033[1;33m女巫\033[0m' not in list(data.values()) \
        and '\033[1;33m猎人\033[0m' not in list(data.values()) and '\033[1;33m守卫\033[0m' not in list(data.values()):
    engine.say('狼人获胜，游戏结束！')
    engine.runAndWait()
    exit()

engine.say('请竞选警长的玩家闭眼举手！')
engine.say('3，2，1，天亮了！')
# 竞选警长发言顺序
t = datetime.datetime.now()
t = t.strftime('%M')
if int(t) % 2 == 1:
    way = '左'
else:
    way = '右'
sum1 = eval(t[0]) + eval(t[1])
if sum1 >= num_info:
    engine.say(f'现在的时间是：{t}分,从{num_info}号向{way}发言！')
    engine.runAndWait()
elif sum1 == 0:
    engine.say(f'现在的时间是：{t}分,从1号向{way}发言！')
    engine.runAndWait()
else:
    engine.say(f'现在的时间是：{t}分,从{t}号向{way}发言！')
    engine.runAndWait()
engine.say('发言结束后请给出警长人选')
engine.runAndWait()
info_police = input('发言结束后请给出警长人选:')  # 警长当选人----接口

# 公布死讯
engine.say(f'昨晚死亡的玩家是{death}，若有技能请交技能,死人发表遗言后按下回车')
engine.runAndWait()
gun = input('技能释放目标:')
if info_police in death and info_police != '':
    engine.say(f'警长请移交警徽或撕毁！')
    engine.runAndWait()
    info_police = input('继承警长人选(直接按回车视为撕毁):')
del data[eval(gun)]
input('发表遗言后按下回车')  # 发表遗言后按下回车----
engine.say('请选择发言顺序并按回车继续！')
engine.runAndWait()
input('警长决定发言顺序')  # 警长决定发言顺序----

'''
for i in range(len(data)-1):
    t1 = datetime.datetime.now().strftime('%S')
    while eval(datetime.datetime.now().strftime('%S')) + 90 <= eval(t1):
        print(f'还有{90 - (eval(datetime.datetime.now().strftime("%S")) - t1)}s发言时间')
    engine.say('发言结束！下一位！')
    engine.runAndWait()
    input('按空格继续！')
if info_police != '':
    engine.say('警长请发言并归票！')
    engine.runAndWait()
    t1 = datetime.datetime.now().strftime('%S')
    while eval(datetime.datetime.now().strftime('%S')) + 150 <= eval(t1):
        print(f'还有{150 - (eval(datetime.datetime.now().strftime("%S")) - t1)}s发言时间')
    engine.say('发言结束！请投票！')
    engine.runAndWait()
    ticket = input('被投人选：')
elif info_police == '':
    t1 = datetime.datetime.now().strftime('%S')
    while eval(datetime.datetime.now().strftime('%S')) + 90 <= eval(t1):
        print(f'还有{90 - (eval(datetime.datetime.now().strftime("%S")) - t1)}s发言时间')
    engine.say('发言结束！请投票！')
    engine.runAndWait()
    ticket = input('被投人选：')'''

input('按空格结束发言！')
ticket = input('死亡人选：')
del data[eval(ticket)]
input('发表遗言完毕请按空格!')
engine.say(f'天黑请闭眼！')
engine.runAndWait()

while True:
    # 1狼人
    engine.say('狼人请睁眼！功能狼举手示意！请在互相确认身份后给出击杀的目标！')
    engine.runAndWait()
    for a in range(10):
        print('*' * 200)
    print(f'仍存活的玩家是{list(data.keys())}')
    info_1 = input('击杀目标(不选择杀人请直接按回车):')  # 狼人击杀目标----接口(务必以功能狼最大)
    for a in range(20):
        print('*' * 200)
    engine.say('狼人，请闭眼！')
    engine.runAndWait()

    # 2预言家
    engine.say('预言家请睁眼！请给出你要查验的对象！')
    engine.runAndWait()
    for a in range(10):
        print('*' * 200)
    print(f'仍存活的玩家是{list(data.keys())}，你上局查验的对象是{info_2}')
    info_2 = input(f'查验对象:')  # 预言家查验----接口
    engine.say('他的身份是')
    engine.runAndWait()
    if data[eval(info_2)] != '\033[1;31m普通狼人\033[0m' and data[eval(info_2)] != '\033[1;31m白狼王\033[0m':
        print('\033[1;32m好人\033[0m')
    else:
        print('\033[1;31m坏人\033[0m')
    for a in range(5):
        print('*' * 200)
    engine.say('预言家，请按回车确认后闭眼！')
    engine.runAndWait()
    input('预言家确认查验对象')  # 预言家确认查验对象---接口
    for a in range(20):
        print('*' * 200)

    # 3女巫
    drug_good = 1
    drug_bad = 1
    engine.say('女巫请睁眼！今晚该号玩家死亡！你是否要使用解药！')
    engine.runAndWait()
    for a in range(10):
        print('*' * 200)
    if drug_good == 1:
        if info_1 == '':
            print('\033[1;32m无人死亡！\033[0m')
        else:
            print(f'\033[1;31m{info_1}号死亡！\033[0m')
    else:
        print('解药已用尽，不能得知死讯！')
    info_3_1 = input('你是否要使用解药！(Y/N):')  # 女巫解药----接口
    if info_3_1 == 'Y':
        drug_good -= 1
    for a in range(10):
        print('*' * 200)
    engine.say('你是否要使用毒药！')
    engine.runAndWait()
    info_3_2 = input('你是否要使用毒药！(Y/N)以及要使用的对象，并用“-”分开（例如：N;Y-3）:').split('-')  # 女巫毒药----接口
    if info_3_2[0] == 'Y':
        drug_bad -= 1
    for a in range(20):
        print('*' * 200)
    engine.say('女巫，请闭眼！')
    engine.runAndWait()

    # 4猎人
    info_4 = ''
    if '4' in info:
        engine.say('猎人请睁眼！今晚你的开枪状态是')
        engine.runAndWait()
        for a in range(10):
            print('*' * 200)
        if drug_good < 0:
            print('\033[1;32m可以开枪！\033[0m')
        elif len(info_3_2) == 1:
            print('\033[1;32m可以开枪！\033[0m')
        elif data[eval(info_3_2[0])] != '\033[1;33m猎人\033[0m':
            print('\033[1;32m可以开枪！\033[0m')
        else:
            print('\033[1;31m不可以开枪！\033[0m')
        for a in range(20):
            print('*' * 200)
        engine.say('猎人，请闭眼！')
        engine.runAndWait()

    # 5守卫
    info_5 = ''
    if '5' in info:
        engine.say('守卫请睁眼！今晚你要守卫的人是')
        engine.runAndWait()
        for a in range(10):
            print('*' * 200)
        print(f'仍存活的玩家是{list(data.keys())}！上一晚你守卫的玩家是{info_5}号，不能连续两晚守卫同一玩家')
        info_5 = input('今晚你要守卫的人是(不选择守人请直接按回车):')
        for a in range(20):
            print('*' * 200)
        engine.say('守卫，请闭眼！')
        engine.runAndWait()

    # 死亡运算
    death = []
    if info_1 != '':
        if drug_good == 0:
            if info_3_1 == 'Y':
                if info_5 == info_1:
                    del data[eval(info_1)]
                    death.append(info_1)
        if info_3_1 == 'N':
            if info_5 != info_1:
                del data[eval(info_1)]
                death.append(info_1)
    if info_3_2[0] == 'Y':
        del data[eval(info_3_2[1])]
        death.append(info_3_2[1])
    random.shuffle(death)
    if '平民' not in list(data.values()):
        engine.say('狼人获胜，游戏结束！')
        engine.runAndWait()
        exit()
    elif '\033[1;31m普通狼人\033[0m' not in list(data.values()) and '\033[1;31m白狼王\033[0m' not in list(data.values()):
        engine.say('好人获胜，游戏结束！')
        engine.runAndWait()
        exit()
    elif '\033[1;33m预言家\033[0m' not in list(data.values()) and '\033[1;33m女巫\033[0m' not in list(data.values()) \
            and '\033[1;33m猎人\033[0m' not in list(data.values()) and '\033[1;33m守卫\033[0m' not in list(data.values()):
        engine.say('狼人获胜，游戏结束！')
        engine.runAndWait()
        exit()

    engine.say('3，2，1，天亮了！')
    engine.runAndWait()

    # 公布死讯
    engine.say(f'昨晚死亡的玩家是{death}，若有技能请交技能,死人发表遗言后按下回车')
    engine.runAndWait()
    gun = input('技能释放目标:')
    if info_police in death and info_police != '':
        engine.say(f'警长请移交警徽或撕毁！')
        engine.runAndWait()
        info_police = input('继承警长人选(直接按回车视为撕毁):')
    del data[eval(gun)]
    input('发表遗言后按下回车')  # 发表遗言后按下回车----
    engine.say('请选择发言顺序并按回车继续！')
    engine.runAndWait()
    input('警长决定发言顺序')  # 警长决定发言顺序----

    input('按空格结束发言！')
    ticket = input('死亡人选：')
    del data[eval(ticket)]
    input('发表遗言完毕请按空格!')
    engine.say(f'天黑请闭眼！')
    engine.runAndWait()
