
def code(test, key=0):
    low_code = [chr(i) for i in range(97, 97 + 26)] + [chr(i) for i in range(97, 97 + 26)]
    up_code = [chr(i).upper() for i in range(97, 97 + 26)] + [chr(i).upper() for i in range(97, 97 + 26)]
    data = []
    for i in test:
        if i in low_code:
            data.append(low_code[low_code.index(i) + key])
        elif i in up_code:
            data.append(up_code[up_code.index(i) + key])
        else:
            data.append(i)
    return ''.join(data)


info = input('输入英文文本：')
x = code(info, 5)
print(f'编码后的文本：{x}')
print(f'对编码后的文本解码：{code(x, -5)}')

