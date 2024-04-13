import pyttsx3  # 语音输出


engine = pyttsx3.init()
engine.setProperty('200', '100')
def magic(dieing, num): # 女巫（今晚死的人，药瓶数量[medicine, drug]）
    engine.say('今晚该号玩家死亡,你要使用解药吗?')
    if num[0] == 1:
        print(dieing)
        ans = eval(input('输入 1(使用) 或 0(不使用)'))
        if ans == 1:
            num[0] = 0
            # 复活该玩家，除非被奶死(标记法)

    else:
        print('你的解药已用尽！')