
info = input('_(:ι」∠)_输入一个多位整数：')
x1 = info
x2 = x1[::-1]
if x1 == x2:
    print('牛啊牛啊！你输入的刚好是个回文数~')
else:
    print('emmm,很可惜不是个回文数~')
