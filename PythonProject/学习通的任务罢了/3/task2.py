
info = input('输入一串字符')
x = 0
y = 0
z = 0
w = 0
for i in info:
    if 'a' <= i.lower() <= 'z':
        x += 1
    elif '0' <= i <= '9':
        y += 1
    elif i == ' ':
        z += 1
    else:
        w += 1
print(f'英文字符数：{x}')
print(f'数字字符数：{y}')
print(f'空格数：{z}')
print(f'其他字符数：{w}')
