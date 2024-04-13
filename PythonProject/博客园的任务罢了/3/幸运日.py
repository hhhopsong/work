
import random
x = random.randint(1, 31)
count = 0
info = eval(input('猜一下2021年5月你的幸运日吧^_^\n你只有三次机会哦！猜吧（1~31）：'))
if info == x:
    print('Bingo!')
    exit()
else:
    count += 1
    while count < 3:
        if info < x:
            print('早啦早啦，填早啦！你的幸运日还早呢~')
        elif info > x:
            print('晚啦，你的幸运日比你想象的还要早哦~')
        else:
            print('Bingo!')
            exit()
        info = eval(input('再猜再猜:'))
        count += 1
    else:
        if info < x:
            print(f'早啦早啦，填早啦！你的幸运日还早呢~\n次数用完啦\n2021年5月你的幸运日是{x}号')
        elif info > x:
            print(f'晚啦，你的幸运日比你想象的还要早哦~\n次数用完啦\n2021年5月你的幸运日是{x}号')
        else:
            print('Bingo!')
            exit()


