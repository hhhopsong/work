info = [0, 0, 0, 0]
pi = 3.1415926
for i in range(len(info)):
    print('请输入线径' + str(i+1) + '的值:')
    info[i] = (eval(input()) / 2) ** 2
dd = sum(info) ** 0.5 * 2
print(f'线径和为:{dd:.100f}')
