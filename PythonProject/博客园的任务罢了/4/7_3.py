
import random
# 时间模块
import datetime
t = datetime.datetime.now()
t = t.strftime('%Y%m%d')
with open('data7.txt', 'r', encoding='gbk')as f:
    data = f.read().split('\n')
info = eval(input('输入随机抽点人数：'))
# 使用集合:防止总名单重复
output = set()
while info > 0:
    in_data = set()  # 使用集合防止单次抽点出现重复
    # 随机抽取模块
    while len(in_data - output) < info and len(data) >= len(output) + info:
        rand = random.randint(0, len(data) - 1)
        in_data.add(str(data[rand]))
    if len(data) < len(output) + info and len(data) - len(output) != 0:
        print(f'抽点人数多了哟，班级内只剩{len(data) - len(output)}个人没被抽到了哦~')
    elif len(data) - len(output) == 0:
        print('班级同学已经全部被点过啦！')
        exit()
    else:
        # 写入模块
        with open(f'{t}.txt', 'a+', encoding='utf-8')as f:
            for line in in_data - output:
                f.write(line + '\n')
                print(line)
        output = output | in_data
    info = eval(input('输入随机抽点人数：'))
