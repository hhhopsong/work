
info = eval(input('输入一个年份：'))
if info % 400 == 0:
    print(f'{info}是闰年哦~')
else:
    if info % 4 == 0:
        if info % 100 != 0:
            print(f'{info}是闰年哦~')
        else:
            print(f'{info}不是闰年呐！')
    else:
        print(f'{info}不是闰年呐！')
