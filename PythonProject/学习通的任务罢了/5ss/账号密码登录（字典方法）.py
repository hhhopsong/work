
accounts = {'郑文博': '213213', '王熙博': 'cccccc', '宋听洋': '111111', '耿磊': 'sbnt250'}
while True:
    info = input('输入您的用户名：')
    while info in accounts:
        password = input('输入您的密码：')
        if password == accounts[info]:
            print('登录成功！')
            exit()
        else:
            print('密码错误！请重新', end='')
    else:
        print('该用户名不存在！请重新', end='')
