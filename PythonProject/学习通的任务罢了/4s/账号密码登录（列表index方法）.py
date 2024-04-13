
users = ['宋听洋', '申富明', '何驭昊', '李薪阳']
passwords = ['111111', '101010', '？？？？？？', '000000']
while True:
    user = input('输入你的用户名：')
    if user in users:
        while user in users:
            password = input('输入密码：')
            if password in passwords:
                key = users.index(user)
                if key == passwords.index(password):
                    print('登录成功！')
                    break
                else:
                    print('密码错误！请重新输入。')
            else:
                print('密码错误！请重新输入。')
        break
    else:
        print('用户名不存在！请重新输入。')


