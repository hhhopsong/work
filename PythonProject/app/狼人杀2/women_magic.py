import pyttsx3  # �������


engine = pyttsx3.init()
engine.setProperty('200', '100')
def magic(dieing, num): # Ů�ף����������ˣ�ҩƿ����[medicine, drug]��
    engine.say('����ú��������,��Ҫʹ�ý�ҩ��?')
    if num[0] == 1:
        print(dieing)
        ans = eval(input('���� 1(ʹ��) �� 0(��ʹ��)'))
        if ans == 1:
            num[0] = 0
            # �������ң����Ǳ�����(��Ƿ�)

    else:
        print('��Ľ�ҩ���þ���')