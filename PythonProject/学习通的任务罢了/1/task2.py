
exl = 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
info = eval(input('输入一个0~7的阿拉伯整数哦：'))
x = str(exl[info-1:info])[2:-3]
print(x)
