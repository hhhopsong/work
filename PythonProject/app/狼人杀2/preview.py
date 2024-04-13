import pyttsx3  # 语音输出


engine = pyttsx3.init()
engine.setProperty('200', '100')
# 好人标识符为数字，坏人为字母
def preview(dead):  # 预言家（已死亡的人）
    engine.say('今晚你要查验的玩家是')
    obj = eval(input('输入号码:')) # 输入号码
    engine.say('他的身份是')
    while True:
        if obj in dead:
            continue    # 查验对象死亡
        elif 0 <= obj <= 9:
            return 1    # 好人
        else:
            return -1   # 坏人
