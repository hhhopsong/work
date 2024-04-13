'''
while True:
    alp = 0
    num = 0
    print('{:-^40}'.format('解释器初始化'))
    int = input('输入你的用户名：')
    info = int.lower()
    if ('a' <= info[0:1] <= 'z') and (len(info) > 6):
        for i in info:
            if '0' <= i <= '9':
                num += 1
            elif 'a' <= i <= 'z':
                alp += 1
        if (alp + num == len(info)) and (alp != 0) and (num != 0) :
            print(f'{int}可用')
        else:
            print(f'{int}不合法！')
    else:
        print(f'{int}不合法！')
'''
'''
while not __import__('re').match('^[a-zA-Z][a-zA-Z0-9]{6,}$',input('Input your username:')):continue
'''

