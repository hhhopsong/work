import pyttsx3  # �������


engine = pyttsx3.init()
engine.setProperty('200', '100')
# ���˱�ʶ��Ϊ���֣�����Ϊ��ĸ
def preview(dead):  # Ԥ�Լң����������ˣ�
    engine.say('������Ҫ����������')
    obj = eval(input('�������:')) # �������
    engine.say('���������')
    while True:
        if obj in dead:
            continue    # �����������
        elif 0 <= obj <= 9:
            return 1    # ����
        else:
            return -1   # ����
